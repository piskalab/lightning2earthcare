import logging
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import List
import numpy as np
import s3fs
import xarray as xr

from .token_handling import get_eumetsat_token

logger = logging.getLogger(__name__)

_LI_COLLECTION_IDS = {
    'lightning_groups': 'EO:EUM:DAT:0782',
}
_EUM_DATASTORE = None
_LI_BODY_GLOB = "*BODY*.nc"
_BUCKET_BY_PLATFORM = {
    'GOES-16': 'noaa-goes16',
    'GOES-18': 'noaa-goes18',
    'GOES-19': 'noaa-goes19',
}
_GLM_PRODUCT = 'GLM-L2-LCFA'  # Full-disk GLM L2 (LCFA) product
# Example: OR_GLM-L2-LCFA_G16_s20232400000000_e20232400000200_c20232400000223.nc
_GLM_NAME_RE = re.compile(
    r'^OR_GLM-L2-LCFA_G(?P<plat>\d{2})_s(?P<s>\d{14})_e(?P<e>\d{14})_c(?P<c>\d{14})\.nc$'
) 


def _get_datastore():
    global _EUM_DATASTORE
    if _EUM_DATASTORE is None:
        _, _EUM_DATASTORE = get_eumetsat_token()
    return _EUM_DATASTORE

def _to_datetime(ts) -> datetime:
    """
    Convert numpy.datetime64 or datetime to a timezone-aware datetime.
    """
    if isinstance(ts, np.datetime64):
        seconds = ts.astype('datetime64[s]').astype(int)
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts
    raise TypeError(f"Unsupported type for timestamp: {type(ts)}")

def _parse_glm_timefield(s: str) -> datetime:
    # Take only the first 13 digits (YYYYJJJHHMMSS)
    dt = datetime.strptime(s[:13], '%Y%j%H%M%S').replace(tzinfo=timezone.utc)
    return dt

def _iter_hours(dt_start: datetime, dt_end: datetime):
    """Yield (year, jday, hour) covering [dt_start, dt_end] inclusive."""
    current_hour = dt_start.replace(minute=0, second=0, microsecond=0)
    end_hour = dt_end.replace(minute=0, second=0, microsecond=0)
    if dt_end != end_hour:
        end_hour += timedelta(hours=1)
    while current_hour <= end_hour:
        yield current_hour.year, int(current_hour.strftime('%j')), current_hour.hour
        current_hour += timedelta(hours=1)

def _open_glm_part(fs, s3_key: str) -> xr.Dataset | None:
    """
    Open a single GLM NetCDF from NOAA S3 and normalize variables/dims.
    """
    try:
        ds = xr.open_dataset(fs.open(s3_key, mode='rb'), engine='h5netcdf', chunks="auto",)
    except Exception:
        ds = xr.open_dataset(fs.open(s3_key, mode='rb'), engine='netcdf4', chunks="auto",)

    if 'number_of_groups' in ds.dims:
        ds = ds.rename({'number_of_groups': 'groups'})

    coord_names = [c for c in ds.coords if 'groups' in getattr(ds[c], 'dims', ())]
    if coord_names:
        ds = ds.reset_coords(names=coord_names, drop=False)

    drop_names = [name for name in ds.variables if 'groups' not in getattr(ds[name], 'dims', ())]
    ds = ds.drop_vars(drop_names, errors='ignore')
    ds = ds.drop_vars(['group_time_offset', 'group_area'], errors='ignore')

    rename_map = {
        'group_frame_time_offset': 'group_time',
        'group_lat': 'latitude',
        'group_lon': 'longitude',
        'group_energy': 'radiance',
        'group_parent_flash_id': 'flash_id',
    }
    present = {k: v for k, v in rename_map.items() if k in ds.variables}
    if present:
        ds = ds.rename(present)

    return ds


def download_li(
    start_time,
    end_time,
    lightning_dir: Path,
    collection_key='lightning_groups'
) -> list[Path]:
    """
    Download Lightning data from EUMETSAT Data Store.

    Args:
        start_time: numpy.datetime64 or datetime start of range
        end_time:   numpy.datetime64 or datetime end of range
        lightning_dir: base Path to save downloads
        collection_key: key in _LI_COLLECTION_IDS

    Returns:
        List of Paths to LI files.
    """
    dt_start = _to_datetime(start_time)
    dt_end   = _to_datetime(end_time)

    coll_id = _LI_COLLECTION_IDS[collection_key]
    datastore = _get_datastore()
    collection = datastore.get_collection(coll_id)
    products = collection.search(dtstart=dt_start, dtend=dt_end)
    logger.info(f"Found {products.total_results} products in '{collection_key}'")
    if products.total_results == 0:
        logger.warning("Skipping download")
        return []

    lightning_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files: List[Path] = []

    for product in products:
        entries = list(product.entries)
        body_entries = [e for e in entries if fnmatch(e, _LI_BODY_GLOB)]
        if not body_entries:
            logger.warning(f"No {_LI_BODY_GLOB} entry found for {product._id}")
            continue

        for entry in body_entries:
            try:
                with product.open(entry=entry) as src:
                    out_path = lightning_dir / src.name

                    # If data already exist locally, reuse it
                    if out_path.exists():
                        logger.info(f"Data already exist, skipping: {out_path.name}")
                        downloaded_files.append(out_path)
                        continue

                    # Download data if missing
                    with out_path.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                    logger.info(f"Downloaded {out_path.name}")
                    downloaded_files.append(out_path)

            except Exception as e:
                logger.error(f"Error handling product {product._id} / entry {entry}: {e}")

    return downloaded_files


def load_merge_glm(
    start_time,
    end_time,
    platform: str,
    product: str = _GLM_PRODUCT,
    max_workers: int = 2,
) -> xr.Dataset | None:
    """
    Open GOES GLM L2 (LCFA) NetCDFs directly from NOAA S3 for a time window
    and return a single merged xarray.Dataset.
    """

    if platform not in _BUCKET_BY_PLATFORM:
        logger.error(f"Unsupported platform {platform!r}. Expected one of {list(_BUCKET_BY_PLATFORM)}")
        return None

    dt_start = _to_datetime(start_time)
    dt_end   = _to_datetime(end_time)
    if dt_end < dt_start:
        logger.warning("load_merge_glm: end_time < start_time; nothing to do.")
        return None

    bucket = _BUCKET_BY_PLATFORM[platform]
    fs = s3fs.S3FileSystem(anon=True)

    prefixes = [
        f"{bucket}/{product}/{year:04d}/{jday:03d}/{hour:02d}/"
        for (year, jday, hour) in _iter_hours(dt_start, dt_end)
    ]

    s3_keys = []
    for pref in prefixes:
        try:
            keys = fs.ls(pref)
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.error(f"Error listing {pref}: {e}")
            continue

        for key in keys:
            fname = key.split('/')[-1]
            m = _GLM_NAME_RE.match(fname)
            if not m:
                continue

            s_dt = _parse_glm_timefield(m.group('s'))
            e_dt = _parse_glm_timefield(m.group('e'))

            # keep only files that overlap requested time window
            if not (e_dt < dt_start or s_dt > dt_end):
                s3_keys.append(key)

    if not s3_keys:
        logger.info(f"GLM: no files found in {bucket}/{product} for {dt_start} → {dt_end}")
        return None

    s3_keys = sorted(set(s3_keys))

    # open/preprocess all files
    ds_parts: list[xr.Dataset] = []

    def _open_safe(key: str) -> xr.Dataset | None:
        try:
            return _open_glm_part(fs, key)
        except Exception as e:
            logger.error(f"Failed to open/preprocess {key}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for ds_part in ex.map(_open_safe, s3_keys):
            if ds_part is not None:
                ds_parts.append(ds_part)

    if not ds_parts:
        logger.error("No GLM datasets had groups after filtering.")
        return None

    try:
        merged = xr.concat(
            ds_parts,
            dim='groups',
            data_vars='minimal',
            coords='minimal',
            join='outer',
            combine_attrs='drop_conflicts',
            compat='override',
        )
        
        merged = merged.assign_attrs({"platform": platform})
        logger.info(f"Merged {len(ds_parts)} GLM files for platform {platform}")
        return merged

    except Exception as e:
        logger.error(f"Failed to concatenate GLM datasets: {e}")
        return None
