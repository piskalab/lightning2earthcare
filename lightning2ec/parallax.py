from datetime import datetime
import logging
import numpy as np
from pyorbital.orbital import A as EARTH_RADIUS
from pyorbital.orbital import get_observer_look
from satpy.utils import lonlat2xyz, xyz2lonlat

logger = logging.getLogger(__name__)


def _get_satellite_elevation(
    sat_lon: float,
    sat_lat: float,
    sat_alt: float,
    lon: np.ndarray,
    lat: np.ndarray,
    cth: np.ndarray
) -> np.ndarray:
    """
    Compute the elevation angle of the satellite relative to cloud-top points.

    Args:
        sat_lon, sat_lat: Satellite position in degrees.
        sat_alt: Satellite altitude in meters.
        lon, lat: Arrays of observation longitudes/latitudes (degrees).
        cth: Cloud-top heights (meters).

    Returns:
        Array of elevation angles (degrees), NaN where elevation <= 0.
    """
    placeholder = datetime(2000, 1, 1)
    # get_observer_look expects km for alt and height
    _, elevation = get_observer_look(
        sat_lon, sat_lat, sat_alt / 1e3,
        placeholder, lon, lat, cth / 1e3
    )
    elevation_masked = np.where(elevation > 0, elevation, np.nan)
    return elevation_masked


def _get_slant_cloud_distance(
    cth: np.ndarray,
    elevation: np.ndarray
) -> np.ndarray:
    """
    Calculate slant cloud-to-ground distance along the line of sight of the satellite.

    Args:
        cth: Cloud-top heights (meters).
        elevation: Elevation angles (degrees).

    Returns:
        Slant distances (meters), NaN if elevation invalid.
    """
    if np.all(np.isnan(elevation)):
        logger.warning("All elevation values are invalid (NaN)")
        return np.full_like(cth, np.nan)
    return cth / np.sin(np.deg2rad(elevation))


def _get_parallax_shift_xyz(
    sat_lon: float,
    sat_lat: float,
    sat_alt: float,
    lon: np.ndarray,
    lat: np.ndarray,
    parallax_dist: np.ndarray
) -> np.ndarray:
    """
    Compute XYZ shifts for inverse parallax correction.

    Args:
        sat_lon, sat_lat: Satellite position (degrees).
        sat_alt: Satellite altitude (meters).
        lon, lat: Arrays of original longitudes/latitudes (degrees).
        parallax_dist: Slant distances (meters).

    Returns:
        Array of shifted XYZ coordinates (meters).
    """
    sat_xyz = np.stack(lonlat2xyz(sat_lon, sat_lat), axis=-1) * sat_alt
    cth_xyz = np.stack(lonlat2xyz(lon, lat), axis=-1) * EARTH_RADIUS * 1e3
    delta = cth_xyz - sat_xyz
    sat_dist = np.linalg.norm(delta, axis=-1)
    # broadcast parallax scale factor
    scale = (parallax_dist / sat_dist)[..., np.newaxis]
    return cth_xyz + delta * scale


def apply_parallax(
    lon: np.ndarray,
    lat: np.ndarray,
    cth: np.ndarray,
    sat_lon: float,
    sat_lat: float,
    sat_alt: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parallax correction: compute shifted lat/lon arrays.

    Args:
        lon, lat: 2D arrays of original coordinates.
        cth: 2D cloud-top heights (m).
        sat_lon, sat_lat, sat_alt: Satellite parameters.

    Returns:
        2D arrays of shifted coords.
    """
    elev = _get_satellite_elevation(sat_lon, sat_lat, sat_alt, lon, lat, cth)
    slant = _get_slant_cloud_distance(cth, elev)
    xyz_shifted = _get_parallax_shift_xyz(sat_lon, sat_lat, sat_alt, lon, lat, slant)
    lon_shifted, lat_shifted = xyz2lonlat(
        xyz_shifted[..., 0], xyz_shifted[..., 1], xyz_shifted[..., 2]
    )
    return lat_shifted, lon_shifted
