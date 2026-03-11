import logging
import time
from collections import defaultdict
from datetime import date
from typing import List
import fsspec
import xarray as xr
from pystac_client import Client

from .token_handling import get_earthcare_token

logger = logging.getLogger(__name__)


CATALOG_URL = "https://catalog.maap.eo.esa.int/catalogue/"
COLLECTION = "EarthCAREL2Products_MAAP"
ASSET_KEY = "enclosure_h5"


def _parse_orbit_frame_from_id(item_id: str):
    """Extract orbit+frame from final underscore field in STAC item id."""
    try:
        last = item_id.split("_")[-1]
        return last[:-1], last[-1].upper()
    except Exception:
        return None, None


def query_catalogue(
    products: List[str],
    frames: List[str],
    start_date: date,
    end_date: date,
    collection_id=COLLECTION,
    catalog_url=CATALOG_URL,
):
    """
    Query ESA MAAP STAC catalogue for specified products, frames, and date range.
    Returns a list of pystac.Items.
    """
    client = Client.open(catalog_url)
    datetime_str = [f"{start_date.strftime('%Y-%m-%dT00:00:00Z')}",
            f"{end_date.strftime('%Y-%m-%dT23:59:59Z')}"]

    # Build filter: (productType='X' OR productType='Y') AND (frame='A' OR frame='B')
    product_filter = " OR ".join([f"productType = '{p}'" for p in products])
    frame_filter = " OR ".join([f"frame = '{f}'" for f in frames])
    baseline_filter = "(productVersion = 'ba' OR productVersion = 'bc')"
    combined_filter = f"({product_filter}) AND ({frame_filter}) AND ({baseline_filter})"
    bboxes = [[170, -60, 180, 60], [-180, -60, 60, 60],]

    logger.info(f"Querying STAC:\n  products={products}\n  frames={frames}\n  date={datetime_str}")

    items_by_id = {}
    for bbox in bboxes:
        search = client.search(
            collections=[collection_id],
            datetime=datetime_str,
            bbox=bbox,
            filter=combined_filter,
            method="GET",
        )
        for item in search.items():
            items_by_id[item.id] = item

    return list(items_by_id.values())


def fetch_earthcare_data(ds_url, group="ScienceData", retries=5, delay=10):
    """
    Fetch EarthCARE data from a remote HTTPS URL and return it as an xarray.Dataset.

    Args:
        ds_url (str): HTTPS URL to the dataset (STAC asset)
        group (str): NetCDF group to open (default: "ScienceData")

    Returns:
        xarray.Dataset: Loaded dataset.
    """
    if not ds_url.startswith("http"):
        raise ValueError(f"Only remote HTTPS URLs are supported. Got: {ds_url}")

    logger.info(f"Opening remote EarthCARE dataset via HTTPS: {ds_url}")

    token = get_earthcare_token()

    last_exception = None

    for attempt in range(1, retries + 1):
        try:
            fs = fsspec.filesystem("https", headers={"Authorization": f"Bearer {token}"})
            with fs.open(ds_url, "rb") as f:
                ds = xr.open_dataset(f, engine="h5netcdf", group=group)
                ds.load()

            logger.info(f"Successfully loaded dataset: {ds_url}")
            return ds

        except Exception as e:
            last_exception = e
            logger.warning(
                f"Attempt {attempt} failed: {e.__class__.__name__}: {e}. "
                f"Retrying in {delay} seconds..."
            )
            time.sleep(delay)

    logger.error(f"Failed to load dataset after {retries} attempts.")
    raise last_exception


def find_ec_file_pairs(
    products: List[str],
    frames: List[str],
    start_date,
    end_date,
    collection_id=COLLECTION,
    catalog_url=CATALOG_URL,
    asset_key=ASSET_KEY,
):
    """
    Build a dict mapping orbit_frame → {product_name: remote_asset_url}.
    Only includes orbits where all requested products are available.
    """
    items = query_catalogue(products, frames, start_date, end_date,
                            collection_id=collection_id,
                            catalog_url=catalog_url)

    tmp = defaultdict(dict)
    for item in items:
        orbit, frame = _parse_orbit_frame_from_id(item.id)
        if not orbit or frame not in [f.upper() for f in frames]:
            continue

        orbit_frame = f"{orbit}{frame}"

        matched_product = next((p for p in products if p in item.id), None)
        if not matched_product:
            continue

        asset = item.assets.get(asset_key)

        if not asset or not asset.href.endswith(".h5"):
            continue

        tmp[orbit_frame][matched_product] = asset.href

    pairs = {
        k: v for k, v in tmp.items()
        if set(v.keys()) == set(products)
    }
    pairs = dict(sorted(pairs.items(), key=lambda x: x[0]))

    logger.info(f"Found {len(pairs)} complete orbit/frame pairs")
    return pairs
