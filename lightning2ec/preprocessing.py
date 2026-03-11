import logging
from pathlib import Path
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from shapely import contains_xy
from shapely.geometry import Polygon
from typing import List

from .earthcare_io import fetch_earthcare_data

logger = logging.getLogger(__name__)


def _interpolate_cth(cth: np.ndarray) -> np.ndarray:
    """
    Fill missing MSI cloud top heights (NaNs) using nearest-neighbor interpolation.
    """
    rows, cols = np.indices(cth.shape)
    points = np.column_stack((rows.ravel(), cols.ravel()))
    values = cth.ravel()

    valid_mask = ~np.isnan(values)
    valid_points = points[valid_mask]
    valid_values = values[valid_mask]

    grid_rows, grid_cols = np.mgrid[0:cth.shape[0], 0:cth.shape[1]]
    interpolated = griddata(valid_points, valid_values, (grid_rows, grid_cols), method='nearest')
    return interpolated


def prepare_ec(msi_url):
    """
    Prepare EarthCARE MSI data for processing.

    Fetches the MSI dataset, extracts key fields, fills 
    missing cloud top heights using nearest-neighbor interpolation.
    """
    logger.info(f"Fetching MSI dataset from {msi_url}")

    try:
        msi_ds = fetch_earthcare_data(msi_url)

    except Exception as e:
        logger.error(f"Failed to fetch MSI data from {msi_url}: {e}")
        return None

    required_vars = ["longitude", "latitude", "cloud_top_height", "time"]
    for var in required_vars:
        if var not in msi_ds:
            logger.error(f"Missing variable '{var}' in MSI dataset")
            return None

    lon = msi_ds["longitude"].values
    lat = msi_ds["latitude"].values
    cth = msi_ds["cloud_top_height"].values
    ec_times = msi_ds["time"].values

    logger.info(
        f"Loaded MSI dataset: lon({lon.shape}), lat({lat.shape}), "
        f"cth({cth.shape}), time({len(ec_times)})"
    )

    valid_cols = ~np.isnan(lat).any(axis=0)
    lon_masked = lon[:, valid_cols]
    lat_masked = lat[:, valid_cols]
    cth_masked = cth[:, valid_cols]

    cth_interpolated = _interpolate_cth(cth_masked)
    return lon_masked, lat_masked, cth_interpolated, ec_times


def merge_li_datasets(nc_files: List[Path]) -> xr.Dataset | None:
    """
    Combine multiple LI netcdf files into a single xarray Dataset.
    """
    li_datasets = []

    for nc_path in nc_files:
        try:
            with xr.open_dataset(nc_path, engine='h5netcdf') as ds:
                ds_loaded = ds.load()
            li_datasets.append(ds_loaded)
        except Exception as e:
            logger.warning(f"Failed to open LI file {nc_path}: {e}")

    if not li_datasets:
        logger.error("No LI files could be opened successfully.")
        return None

    try:
        merged_ds = xr.concat(li_datasets, dim="groups", data_vars="all")
        logger.info(f"Successfully merged {len(li_datasets)} LI files into a single dataset.")
        return merged_ds
    except Exception as e:
        logger.error(f"Failed to concatenate LI files: {e}")
        return None


def buffer_lightning_data(
    lightning_ds: xr.Dataset,
    ec_lat: np.ndarray,
    ec_lon: np.ndarray,
    buffer_deg: float = 0.5,
) -> xr.Dataset | None:
    """
    Subset lightning groups to those within a buffered polygon around the EarthCARE swath.
    """
    nrows, _ = ec_lat.shape
    ec_valid = np.isfinite(ec_lat) & np.isfinite(ec_lon)

    left_edge, right_edge = [], []
    for i in range(nrows):
        cols = np.flatnonzero(ec_valid[i])
        if cols.size == 0:
            continue
        jL, jR = cols[0], cols[-1]
        left_edge.append((ec_lon[i, jL], ec_lat[i, jL]))
        right_edge.append((ec_lon[i, jR], ec_lat[i, jR]))

    swath_ring = left_edge + right_edge[::-1]

    try:
        swath_poly = Polygon(swath_ring)
    except ValueError as e:
        logger.warning(f"Skipping buffer_lightning_data: invalid polygon ({e})")
        return None
    buffer_poly = swath_poly.buffer(buffer_deg)

    l_lat = lightning_ds.latitude.values
    l_lon = lightning_ds.longitude.values

    minx, miny, maxx, maxy = buffer_poly.bounds
    in_bbox = (l_lon >= minx) & (l_lon <= maxx) & (l_lat >= miny) & (l_lat <= maxy)
    if not np.any(in_bbox):
        return None

    lon_bbox = l_lon[in_bbox]
    lat_bbox = l_lat[in_bbox]
    bbox_idx = np.where(in_bbox)[0]

    in_poly = contains_xy(buffer_poly, lon_bbox, lat_bbox)
    group_idx = bbox_idx[in_poly]

    if group_idx.size == 0:
        return None
    else:
        logger.info(f"Buffer LI selects {len(group_idx)} of {l_lat.size} total groups")
        return lightning_ds.isel(groups=group_idx)
