import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
import pandas as pd
import xarray as xr
import numpy as np


logger = logging.getLogger(__name__)


def lightning_output_path(
    base_dir: Path,
    l_times: Sequence[np.datetime64],
    orbit_frame: str,
    close_count: int,
    source_label: str
) -> Path:
    """
    Construct an output NetCDF file path based on lightning match times and orbit frame.

    Args:
        base_dir: Directory where outputs are stored.
        l_times: Sequence of numpy.datetime64 times of matched events in time.
        orbit_frame: Identifier for orbit and frame (e.g. '12345A').
        close_count: Number of events within the nadir distance threshold.
        source_label: Label for the lightning source ('LI' or 'GLM').

    Returns:
        A Path to the new NetCDF file, with directories created as needed.
    """
    times = pd.to_datetime(l_times)
    start_str = times.min().strftime('%Y%m%dT%H%M%SZ')
    end_str   = times.max().strftime('%Y%m%dT%H%M%SZ')
    count_all = len(times)

    save_dir = base_dir / "lightning_groups"
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{source_label}_{start_str}_{end_str}_{count_all}_{close_count}_{orbit_frame}.nc"
    return save_dir / filename


def track_output_path(
    base_dir: Path,
    cpr_time: np.ndarray,
    orbit_frame: str,
    close_count: int,
    source_label: str
) -> Path:
    """
    Construct a NetCDF output file path for CPR–Lightning track summary counts.

    Args:
        base_dir : Root directory for output files.
        cpr_time : Array of CPR observation times (numpy.datetime64).
        orbit_frame : Identifier for orbit and frame (e.g. '12345A').
        source_label: Label for the lightning source ('LI' or 'GLM').

    Returns:
        Full path to the NetCDF file, with parent directories created if needed.
    """
    times = pd.to_datetime(cpr_time)
    start_str = times.min().strftime('%Y%m%dT%H%M%SZ')
    end_str   = times.max().strftime('%Y%m%dT%H%M%SZ')

    save_dir = base_dir  / "track_counts"
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f"CPR-{source_label}-sum_{start_str}_{end_str}_{close_count}_{orbit_frame}.nc"
    return save_dir / filename


def write_lightning_netcdf(
    ds: xr.Dataset,
    base_dir: Path,
    orbit_frame: str,
    close_count: int,
    source_label: str,
    platform: str
) -> None:
    """
    Save lightning dataset to NetCDF. Builds the output path using lightning_output_path().
    """
    try:
        l_times = ds["group_time"].values
        output_path = lightning_output_path(base_dir, l_times, orbit_frame, close_count, source_label)

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        lat_min = float(ds.latitude.min())
        lat_max = float(ds.latitude.max())
        lon_min = float(ds.longitude.min())
        lon_max = float(ds.longitude.max())
        time_min = np.datetime_as_string(ds.group_time.min().values, unit="s")
        time_max = np.datetime_as_string(ds.group_time.max().values, unit="s")

        ds.attrs = {}

        if source_label.upper() == "GLM":
            title = "GOES-GLM to EarthCARE collocated data"
            summary = "Subset of GOES GLM L2 (LCFA) group product, collocated to EarthCARE orbit frames"
            src_desc = f"EarthCARE MSI/CPR + NOAA GOES Geostationary Lightning Mapper (GLM) L2 LCFA (platform: {platform})"
            references = "https://www.goes-r.gov/spacesegment/glm.html, https://earth.esa.int/eogateway/missions/earthcare"
        else:
            title = "MTG-LI to EarthCARE collocated data"
            summary = "Subset of MTG-LI L2 group product, collocated to EarthCARE orbit frames, with positional corrections applied"
            src_desc = f"EarthCARE MSI/CPR + EUMETSAT MTG Lightning Imager (LI) L2 (platform: {platform})"
            references = "https://user.eumetsat.int/resources/user-guides/mtg-li-level-2-data-guide, https://earth.esa.int/eogateway/missions/earthcare"

        ds.attrs.update({
            "Conventions": "CF-1.8",
            "title": title,
            "summary": summary,
            "institution": "European Space Agency (ESA)",
            "history": f"Created on {date_str}",
            "source": src_desc,
            "references": references,
            "geospatial_lat_min": lat_min,
            "geospatial_lat_max": lat_max,
            "geospatial_lon_min": lon_min,
            "geospatial_lon_max": lon_max,
            "time_coverage_start": time_min + "Z",
            "time_coverage_end": time_max + "Z",
        })

        if "group_time" in ds:
            time_var = ds["group_time"]
            time_var.encoding.clear()
            time_var.encoding.update({
                "units": "seconds since 2000-01-01 00:00:00",
                "dtype": "float64",
            })

        ds.to_netcdf(path=str(output_path), engine="h5netcdf")
        logger.info(f"Saved NetCDF to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save NetCDF {output_path}: {e}")
        raise


def write_track_netcdf(
    summary_ds: xr.Dataset,
    base_dir: Path,
    orbit_frame: str,
    close_count: int,
    source_label: str,
    platform: str
) -> None:
    """
    Save CPR–lightning track summary to NetCDF. Builds the output path using track_output_path().
    """
    try:
        cpr_time = summary_ds["time"].values
        output_path = track_output_path(base_dir, cpr_time, orbit_frame, close_count, source_label)

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ds = summary_ds.copy()
        ds.attrs = {}

        if source_label.upper() == "GLM":
            title = "EarthCARE CPR track counts of GOES-GLM lightning groups"
            summary = "Per-CPR-point counts of GOES GLM lightning groups within distance/time thresholds"
            src_desc = f"EarthCARE CPR + NOAA GOES Geostationary Lightning Mapper (GLM) L2 LCFA (platform: {platform})"
        else:
            title = "EarthCARE CPR track counts of MTG-LI lightning groups"
            summary = "Per-CPR-point counts of MTG-LI lightning groups within distance/time thresholds"
            src_desc = f"EarthCARE CPR + EUMETSAT MTG Lightning Imager (LI) L2 (platform: {platform})"

        ds.attrs.update({
            "Conventions": "CF-1.8",
            "title": title,
            "summary": summary,
            "institution": "European Space Agency (ESA)",
            "history": f"Created on {date_str}",
            "source": src_desc,
        })

        if "time" in ds:
            time_var = ds["time"]
            time_var.encoding.clear()
            time_var.encoding.update({
                "units": "seconds since 2000-01-01 00:00:00",
                "dtype": "float64",
            })

        ds.to_netcdf(path=str(output_path), engine="h5netcdf")
        logger.info(f"Saved CPR summary NetCDF to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save CPR NetCDF {output_path}: {e}")
        raise