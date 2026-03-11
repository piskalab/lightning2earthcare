import gc
import logging
from pathlib import Path

from .earthcare_io import find_ec_file_pairs
from .lightning_io import download_li, load_merge_glm
from .runtime import choose_lightning_sources, iter_processing_days
from .preprocessing import merge_li_datasets, buffer_lightning_data, prepare_ec
from .clustering import parent_clustering, subclustering
from .parallax import apply_parallax
from .collocation import collocate_li_to_ec, collocate_glm_to_ec, summarize_vs_cpr
from .writers import write_lightning_netcdf, write_track_netcdf

logger = logging.getLogger(__name__)


SATELLITE_GEOMETRY = {
    "MTG-I1":  {"lon": 0.0,     "lat": 0.0, "alt": 35786400.0},
    "GOES-16": {"lon": -75.2,   "lat": 0.0, "alt": 35786400.0},
    "GOES-18": {"lon": -137.0,  "lat": 0.0, "alt": 35786400.0},
    "GOES-19": {"lon": -75.2,   "lat": 0.0, "alt": 35786400.0},
}


def process_one_source(
    source_key, platform,
    start_time, end_time,
    lightning_dir,
    orbit_frame,
    ec_lon, ec_lat, ec_cth, ec_times,
    cpr_url,
    distance_threshold_km, time_threshold_s
):
    """
    Run the lightning–EarthCARE collocation pipeline for a single lightning source.

    The function performs the following steps:
        1. Load lightning data (MTG-LI or GOES-GLM) for the requested time window
        2. Filter lightning groups to the EarthCARE swath buffer
        3. Cluster lightning groups in space–time
        4. Apply parallax correction (LI only)
        5. Collocate lightning groups with EarthCARE observations
        6. Build CPR-based summary statistics
        7. Write output NetCDF files

    Parameters
    ----------
    source_key : lightning source identifier ("LI" or "GLM")
    platform : lightning satellite platform
    start_time, end_time : time window used to retrieve lightning data
    lightning_dir : directory for lightning downloads and outputs
    orbit_frame : EarthCARE orbit/frame identifier
    ec_lon, ec_lat : EarthCARE swath longitude and latitude
    ec_cth : EarthCARE cloud-top height used for parallax correction
    ec_times : EarthCARE observation timestamps
    cpr_url : URL to the CPR dataset
    distance_threshold_km : spatial matching threshold
    time_threshold_s : temporal matching threshold
    """

    try:
        # (1) Load lightning data (download/open + merge)
        if source_key == 'LI':
            paths = download_li(start_time, end_time, lightning_dir)
            if not paths:
                logger.info(f"[{source_key}] no input files downloaded; skipping.")
                return
            lightning_ds = merge_li_datasets(paths)
        elif source_key == "GLM":
            lightning_ds = load_merge_glm(start_time, end_time, platform)
        else:
            logger.error(f"Unknown source_key: {source_key}")
            return

        if lightning_ds is None:
            logger.info(f"[{source_key}] no usable files after merge; skipping.")
            return

        # (2) Spatial buffer around EarthCARE swath
        buffered_ds = buffer_lightning_data(lightning_ds, ec_lat, ec_lon)
        if buffered_ds is None:
            logger.info(f"[{source_key}] no points in buffer; skipping.")
            return
        del lightning_ds

        # Cluster lightning groups
        clustered_ds = parent_clustering(buffered_ds, eps=5.0, time_weight=0.5, min_samples=20, lat_gap=0.25)
        if clustered_ds is None:
            logger.info(f"[{source_key}] no clusters; skipping.")
            return
        del buffered_ds
        
        if source_key == 'LI':
            # (3) Parallax correction (LI only)
            geom = SATELLITE_GEOMETRY[platform]
            sat_lon, sat_lat, sat_alt = geom["lon"], geom["lat"], geom["alt"]

            shifted_lat, shifted_lon = apply_parallax(
                ec_lon, ec_lat, ec_cth,
                sat_lon, sat_lat, sat_alt
            ) 

            # (4) Collocation with EarthCARE observations
            collocated_ds = collocate_li_to_ec(
                clustered_ds, ec_cth, ec_times,
                shifted_lat, shifted_lon,
                sat_lon, sat_lat, sat_alt,
                time_threshold_s=time_threshold_s
            )
        else:
            # GLM L2 data already include lite parallax correction (implement lower ellipsoid)
            collocated_ds = collocate_glm_to_ec(
                clustered_ds, ec_times,
                ec_lat, ec_lon,
                time_threshold_s=time_threshold_s
            )
        del clustered_ds

        if collocated_ds is None:
            logger.info(f"[{source_key}] no matches found; skipping.")
            return
        
        # (5) Sub-cluster collocated lightning groups
        subclustered_ds = subclustering(collocated_ds, eps=5.0, time_weight=0.5, min_samples=20)
        del collocated_ds

        # (6) Build lightning summary along EarthCARE CPR track
        output_ds, close_count, track_counts_ds = summarize_vs_cpr(
            subclustered_ds, cpr_url,
            distance_threshold_km=distance_threshold_km,
            time_threshold_s=time_threshold_s
        )
        del subclustered_ds

        if output_ds is None:
            logger.error(f"[{orbit_frame}/{source_key}] CPR summary could not be built; skipping this orbit/source.")
            return
        
        # (7) Write output NetCDF files
        write_lightning_netcdf(output_ds, lightning_dir, orbit_frame, close_count, source_label=source_key, platform=platform)
        if (track_counts_ds["lightning_count_5"] > 0).any():
            write_track_netcdf(track_counts_ds, lightning_dir, orbit_frame, close_count, source_label=source_key, platform=platform)
        del output_ds, track_counts_ds

    finally:
        gc.collect()


def run_date_range(
    lightning_base_path: str | Path,
    log_dir: str | Path,
    start_date,
    end_date,
    products,
    frames,
    half_window_minutes: int,
    lightning_platforms,
    distance_threshold_km: float,
    time_threshold_s: int,
):
    """
    Run the EarthCARE–Lightning collocation pipeline for a range of dates.

    For each processing day the function:
        1. Queries the EarthCARE STAC catalogue for matching orbit/frame pairs
        2. Loads MSI data to obtain geolocation and time information
        3. Determines which lightning platforms cover the EarthCARE swath
        4. Runs the lightning processing pipeline for each selected source

    Results are written as NetCDF files in the lightning_base_path directory.
    """

    lightning_dir = Path(lightning_base_path)

    for current_date in iter_processing_days(start_date, end_date, log_dir):
        logger.info(f"Processing date: {current_date:%Y-%m-%d}")

        try:
            pairs = find_ec_file_pairs(
                products=products,
                frames=frames,
                start_date=current_date,
                end_date=current_date,
            )
        except Exception as e:
            logger.error(f"STAC query failed for {current_date:%Y-%m-%d}: {e}")
            continue

        if not pairs:
            logger.info(f"No matching EarthCARE orbits found for {current_date:%Y-%m-%d}")
            continue

        for orbit_frame, file_map in pairs.items():
            logger.info(f"Processing orbit frame: {orbit_frame}")

            msi_product, cpr_product = products[:2]
            msi_url = file_map[msi_product]
            cpr_url = file_map[cpr_product]

            ec_msi = prepare_ec(msi_url)
            if ec_msi is None:
                logger.error(f"Skipping orbit {orbit_frame} because MSI could not be loaded.")
                continue

            ec_lon, ec_lat, ec_cth, ec_times = ec_msi

            lightning_sources = choose_lightning_sources(
                ec_lon, ec_times, half_window_minutes,
                allowed_platforms=tuple(lightning_platforms),
            )
            if not lightning_sources:
                logger.info(f"{orbit_frame}: outside all lightning coverages, skipping")
                continue

            for lightning_source in lightning_sources:
                source_key = lightning_source["source"]
                platform   = lightning_source["platform"]
                start_time = lightning_source["start_time"]
                end_time   = lightning_source["end_time"]

                logger.info(f"{orbit_frame}: processing {source_key} ({platform}) {start_time} -> {end_time}")

                process_one_source(
                    source_key, platform,
                    start_time, end_time,
                    lightning_dir,
                    orbit_frame,
                    ec_lon, ec_lat, ec_cth, ec_times,
                    cpr_url,
                    distance_threshold_km, time_threshold_s,
                )
