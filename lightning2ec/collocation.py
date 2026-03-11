import logging
import numpy as np
import pandas as pd
import xarray as xr
from pyorbital.orbital import A as EARTH_RADIUS
from satpy.modifiers.parallax import get_parallax_corrected_lonlats
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import haversine_distances

from .earthcare_io import fetch_earthcare_data


logger = logging.getLogger(__name__)


def collocate_li_to_ec(
    li_ds: xr.Dataset,
    cth: np.ndarray,
    ec_times: np.ndarray,
    shifted_lat: np.ndarray,
    shifted_lon: np.ndarray,
    sat_lon: float,
    sat_lat: float,
    sat_alt: float,
    time_threshold_s: int = 300,
    spatial_radius_deg: float = 0.009
) -> xr.Dataset | None:
    """
    Match MTG-LI lightning groups to EarthCARE with spatial-temporal proximity.

    Args:
        li_ds: xarray Dataset of lightning groups (with coordinates latitude, longitude, group_time).
        cth: Cloud top heights (2D) from EarthCARE.
        ec_times: 1D array of EarthCARE time stamps.
        shifted_lat/lon: parallax-corrected coords, same shape as cth.
        sat_lon/lat/alt: satellite parameters for parallax computation.
        time_threshold_s: temporal threshold in seconds.
        spatial_radius_deg: spatial matching radius in degrees (~1km).

    Returns:
        matched_li_ds: subset of li_ds with only matched groups, plus new variables
    """
    try:
        li_lat = li_ds.latitude.values
        li_lon = li_ds.longitude.values
        li_time = li_ds.group_time.values

        ec_coords = np.column_stack([shifted_lat.ravel(), shifted_lon.ravel()])
        ec_coords = np.nan_to_num(ec_coords, nan=-999)
        ec_time_expanded = np.repeat(ec_times, cth.shape[1])
        cth_flat    = cth.ravel()

        # Spatial matching
        tree = cKDTree(ec_coords)
        li_points = np.column_stack([li_lat, li_lon])
        dists, idxs = tree.query(li_points, distance_upper_bound=spatial_radius_deg)
        spatial_mask = dists != np.inf

        n_lightning = li_lat.size
        parallax_lat = np.full(n_lightning, np.nan)
        parallax_lon = np.full(n_lightning, np.nan)
        ec_time_diff = np.full(n_lightning, np.timedelta64('NaT'), dtype='timedelta64[ns]')

        if not np.any(spatial_mask):
            logger.info("No spatial matches within radius; skipping.")
            return None
        else:
            matched_li_idx = np.where(spatial_mask)[0]
            matched_ec_idx  = idxs[spatial_mask]

            # Temporal filtering
            li_time_sel = li_time[matched_li_idx]
            ec_time_sel = ec_time_expanded[matched_ec_idx]
            time_diff = li_time_sel - ec_time_sel
            time_mask = np.abs(time_diff) <= np.timedelta64(time_threshold_s, 's')
            matched_li_idx_final = matched_li_idx[time_mask]
            matched_ec_idx_final  = matched_ec_idx[time_mask]

            if matched_li_idx_final.size == 0:
                logger.info("No temporal matches within threshold; skipping.")
                return None
            else:
                logger.info(f"Matched {matched_li_idx_final.size} LI groups to EarthCARE.")
                # Parallax correction on matched buffered points
                li_lon_sel = li_lon[matched_li_idx_final]
                li_lat_sel = li_lat[matched_li_idx_final]
                cth_sel = cth_flat[matched_ec_idx_final]
                parallax_lon_sel, parallax_lat_sel = get_parallax_corrected_lonlats(
                    sat_lon, sat_lat, sat_alt,
                    li_lon_sel, li_lat_sel, cth_sel
                    )
                parallax_lat[matched_li_idx_final] = parallax_lat_sel
                parallax_lon[matched_li_idx_final] = parallax_lon_sel
                ec_time_diff[matched_li_idx_final] = time_diff[time_mask]

        group_time_ns = li_time.astype("datetime64[ns]")
        group_id = li_ds["group_id"].values.astype("uint64")
        latitude = li_ds["latitude"].values.astype("float32")
        longitude = li_ds["longitude"].values.astype("float32")
        radiance = li_ds["radiance"].values.astype("float32")
        flash_id = li_ds["flash_id"].values.astype("uint32")
        parallax_lat_out = parallax_lat.astype("float32")
        parallax_lon_out = parallax_lon.astype("float32")

        matched_li_ds = xr.Dataset(
            data_vars={
                "group_id": ("groups", group_id, {"long_name": "Group ID", "units": "1"}),
                "group_time": ("groups", group_time_ns, {"long_name": "Time of lightning group", "standard_name": "time"}),
                "latitude": ("groups", latitude, {"long_name": "Latitude of group", "units": "degrees_north", "standard_name": "latitude"}),
                "longitude": ("groups", longitude, {"long_name": "Longitude of group", "units": "degrees_east", "standard_name": "longitude"}),
                "radiance": ("groups", radiance, {"long_name": "Radiance of group", "units": "mW.m-2.sr-1"}),
                "flash_id": ("groups", flash_id, {"long_name": "Parent flash ID", "units": "1"}),
                "parallax_corrected_lat": ("groups", parallax_lat_out, {"long_name": "Parallax corrected latitude", "units": "degrees_north"}),
                "parallax_corrected_lon": ("groups", parallax_lon_out, {"long_name": "Parallax corrected longitude", "units": "degrees_east"}),
                "ec_time_diff": ("groups", ec_time_diff, {"long_name": "Time difference from EarthCARE overpass"}),
            }
        )

        matched_li_ds["parent_cluster_id"] = li_ds["parent_cluster_id"].copy(deep=False)
        return matched_li_ds

    except Exception as e:
        logger.error(f"Error in collocate_li_to_ec: {e}")
        return None


def collocate_glm_to_ec(
    glm_ds: xr.Dataset,
    ec_times: np.ndarray,
    ec_lat: np.ndarray,
    ec_lon: np.ndarray,
    time_threshold_s: int = 300,
    spatial_radius_deg: float = 0.009
) -> xr.Dataset | None:
    """
    Match GOES-GLM lightning groups to EarthCARE with spatial-temporal proximity.

    Args:
        glm_ds: xarray Dataset of lightning groups (with coordinates latitude, longitude, group_time).
        ec_times: 1D array of EarthCARE time stamps.
        ec_lat/lon: EarthCARE coords.
        time_threshold_s: temporal threshold in seconds.
        spatial_radius_deg: spatial matching radius in degrees (~1km).

    Returns:
        matched_glm_ds: subset of glm_ds with only matched groups, plus new variables
    """
    try:
        glm_lat = glm_ds.latitude.values
        glm_lon = glm_ds.longitude.values
        glm_time = glm_ds.group_time.values

        ec_coords = np.column_stack([ec_lat.ravel(), ec_lon.ravel()])
        ec_coords = np.nan_to_num(ec_coords, nan=-999)
        ec_time_expanded = np.repeat(ec_times, ec_lat.shape[1])

        # Spatial matching
        tree = cKDTree(ec_coords)
        glm_points = np.column_stack([glm_lat, glm_lon])
        dists, idxs = tree.query(glm_points, distance_upper_bound=spatial_radius_deg)
        spatial_mask = dists != np.inf

        n_lightning = glm_lat.size
        ec_time_diff = np.full(n_lightning, np.timedelta64('NaT'), dtype='timedelta64[ns]')

        if not np.any(spatial_mask):
            logger.info("No spatial matches within radius; skipping.")
            return None
        else:
            matched_glm_idx = np.where(spatial_mask)[0]
            matched_ec_idx  = idxs[spatial_mask]

            # Temporal filtering
            glm_time_sel = glm_time[matched_glm_idx]
            ec_time_sel = ec_time_expanded[matched_ec_idx]
            time_diff = glm_time_sel - ec_time_sel
            time_mask = np.abs(time_diff) <= np.timedelta64(time_threshold_s, 's')
            matched_glm_idx_final = matched_glm_idx[time_mask]

            if matched_glm_idx_final.size == 0:
                logger.info("No temporal matches within threshold; skipping.")
                return None
            else:
                logger.info(f"Matched {matched_glm_idx_final.size} GLM groups to EarthCARE.")
                ec_time_diff[matched_glm_idx_final] = time_diff[time_mask]

        group_time_ns = glm_time.astype("datetime64[ns]")
        group_id = glm_ds["group_id"].values.astype("uint64")
        latitude = glm_ds["latitude"].values.astype("float32")
        longitude = glm_ds["longitude"].values.astype("float32")
        radiance = glm_ds["radiance"].values.astype("float32")
        flash_id = glm_ds["flash_id"].values.astype("uint32")

        matched_glm_ds = xr.Dataset(
            data_vars={
                "group_id": ("groups", group_id, {"long_name": "Group ID", "units": "1"}),
                "group_time": ("groups", group_time_ns, {"long_name": "Time of lightning group", "standard_name": "time"}),
                "latitude": ("groups", latitude, {"long_name": "Latitude of group", "units": "degrees_north", "standard_name": "latitude"}),
                "longitude": ("groups", longitude, {"long_name": "Longitude of group", "units": "degrees_east", "standard_name": "longitude"}),
                "radiance": ("groups", radiance, {"long_name": "Radiance of group", "units": "mW.m-2.sr-1"}),
                "flash_id": ("groups", flash_id, {"long_name": "Parent flash ID", "units": "1"}),
                "ec_time_diff": ("groups", ec_time_diff, {"long_name": "Time difference from EarthCARE overpass"}),
            }
        )

        matched_glm_ds["parent_cluster_id"] = glm_ds["parent_cluster_id"].copy(deep=False)
        return matched_glm_ds

    except Exception as e:
        logger.error(f"Error in match_li_to_ec: {e}")
        return None


def summarize_vs_cpr(
    matched_ds: xr.Dataset, cpr_url: str, distance_threshold_km=5.0, time_threshold_s=300
) -> tuple[xr.Dataset | None, int, xr.Dataset | None]:
    """
    Summarize matched lightning groups relative to the EarthCARE CPR track.

    This function:
    1. loads the CPR track from the remote CPR product,
    2. computes the distance from each lightning group to the nearest CPR sample,
    3. counts lightning groups that satisfy a nearest-CPR spatial/temporal match,
    4. builds a per-CPR, per-cluster summary dataset using loose and strict
       spatial/temporal thresholds.

    Args:
        matched_ds: Lightning dataset after collocation/clustering.
        cpr_url: Remote CPR file URL.
        distance_threshold_km: Loose spatial threshold in kilometers.
        time_threshold_s: Loose temporal threshold in seconds.

    Returns:
        A tuple of:
        - annotated_ds: input lightning dataset with distance_from_nadir added,
        - n_matched_nearest: number of lightning groups matched to their nearest
          CPR sample within the loose thresholds,
        - cpr_summary_ds: summary dataset with per-CPR, per-cluster lightning counts.
    """
    try:
        cpr = fetch_earthcare_data(cpr_url, group="ScienceData")
    except Exception as e:
        logger.error(f"Failed to fetch CPR data from {cpr_url}: {e}")
        return None, 0, None

    cpr_lat = np.asarray(cpr["latitude"].values, dtype=np.float32)
    cpr_lon = np.asarray(cpr["longitude"].values, dtype=np.float32)
    cpr_time = np.asarray(pd.to_datetime(cpr["time"].values))
    cpr_land_flag = np.asarray(cpr["land_flag"].values, dtype=np.uint8)

    # use parallax-corrected coordinates for LI, otherwise fall back (for GLM cases)
    if "parallax_corrected_lat" in matched_ds and "parallax_corrected_lon" in matched_ds:
        li_lat = np.asarray(matched_ds["parallax_corrected_lat"].values, dtype=np.float32)
        li_lon = np.asarray(matched_ds["parallax_corrected_lon"].values, dtype=np.float32)
    else:
        li_lat = np.asarray(matched_ds["latitude"].values, dtype=np.float32)
        li_lon = np.asarray(matched_ds["longitude"].values, dtype=np.float32)
    li_time = np.asarray(pd.to_datetime(matched_ds["group_time"].values))
    li_cluster = np.asarray(matched_ds["cluster_id"].values)

    n_li, n_cpr = li_lat.size, cpr_lat.size
    valid_geo_time = np.isfinite(li_lat) & np.isfinite(li_lon) & ~pd.isna(li_time)
    not_noise = np.isfinite(li_cluster) & (li_cluster != -1)
    valid_li = valid_geo_time & not_noise

    min_cpr_dist = np.full(n_li, np.nan)
    nearest_cpr_idx = np.full(n_li, -1, dtype=int)
    if valid_li.any() and n_cpr:
        dist_matrix = (
            haversine_distances(
                np.column_stack([np.radians(li_lat[valid_li]), np.radians(li_lon[valid_li])]),
                np.column_stack([np.radians(cpr_lat), np.radians(cpr_lon)])
            ) * EARTH_RADIUS
        )
        nearest_idx_valid = np.argmin(dist_matrix, axis=1)
        min_cpr_dist[valid_li] = dist_matrix[np.arange(nearest_idx_valid.size), nearest_idx_valid]
        nearest_cpr_idx[valid_li] = nearest_idx_valid

    annotated_ds = matched_ds.copy()
    min_cpr_dist = np.array(min_cpr_dist, dtype=np.float32)
    annotated_ds["distance_from_nadir"] = xr.DataArray(
        min_cpr_dist, dims=["groups"],
        attrs={"long_name": "Distance to closest EarthCARE CPR track point", "units": "km"}
    )

    loose_radius = float(distance_threshold_km)
    loose_time_threshold = int(time_threshold_s)
    cpr_time_s  = cpr_time.astype("datetime64[s]")
    li_time_s = li_time.astype("datetime64[s]")

    time_diff_nearest = np.full(n_li, np.nan)
    has_nearest = nearest_cpr_idx >= 0
    if has_nearest.any():
        time_diff_nearest[has_nearest] = np.abs(
            (li_time_s[has_nearest] - cpr_time_s[nearest_cpr_idx[has_nearest]])
            .astype("timedelta64[s]").astype(np.int64)
        )

    nearest_match_mask = valid_li & np.isfinite(min_cpr_dist) & (min_cpr_dist <= loose_radius) & np.isfinite(time_diff_nearest) & (time_diff_nearest <= loose_time_threshold)
    n_matched_nearest = int(np.count_nonzero(nearest_match_mask))

    li_coords_rad_all = np.radians(np.column_stack((li_lat, li_lon)))
    cpr_coords_rad_all = np.radians(np.column_stack((cpr_lat, cpr_lon)))

    li_valid_idx = np.flatnonzero(valid_li)
    li_coords_rad_valid = li_coords_rad_all[li_valid_idx]
    li_time_s_valid = li_time_s[li_valid_idx]
    li_cluster_valid = li_cluster[li_valid_idx]

    loose_match_count  = np.zeros(n_cpr, dtype=np.int32)
    strict_match_count = np.zeros(n_cpr, dtype=np.int32)
    loose_dicts = np.empty(n_cpr, dtype=object)
    strict_dicts = np.empty(n_cpr, dtype=object)

    strict_radius = loose_radius / 2.0
    strict_time_threshold = loose_time_threshold // 2

    # For each CPR sample, count LI groups per cluster using loose and strict thresholds
    for i in range(n_cpr):
        if li_valid_idx.size == 0:
            loose_dicts[i] = {}
            strict_dicts[i] = {}
            continue

        time_diff_s = np.abs((li_time_s_valid - cpr_time_s[i]).astype("timedelta64[s]").astype(np.int64))
        time_mask_loose = time_diff_s <= loose_time_threshold
        time_mask_strict = time_diff_s <= strict_time_threshold
        if not time_mask_loose.any() and not time_mask_strict.any():
            loose_dicts[i] = {}
            strict_dicts[i] = {}
            continue

        cpr_coord_rad = cpr_coords_rad_all[i:i+1]

        # loose mode
        if time_mask_loose.any():
            li_coords_loose = li_coords_rad_valid[time_mask_loose]
            dists_km_loose = haversine_distances(cpr_coord_rad, li_coords_loose)[0] * EARTH_RADIUS
            loose_match_mask = dists_km_loose <= loose_radius
            loose_match_count[i] = int(np.count_nonzero(loose_match_mask))

            # per-cluster dict
            if loose_match_mask.any():
                clusters_l = li_cluster_valid[time_mask_loose][loose_match_mask].astype(int)
                if clusters_l.size:
                    u, c = np.unique(clusters_l, return_counts=True)
                    loose_dicts[i] = {int(k): int(v) for k, v in zip(u, c)}
                else:
                    loose_dicts[i] = {}
            else:
                loose_dicts[i] = {}
        else:
            loose_dicts[i] = {}

        # strict mode
        if time_mask_strict.any():
            li_coords_strict = li_coords_rad_valid[time_mask_strict]
            dists_km_strict = haversine_distances(cpr_coord_rad, li_coords_strict)[0] * EARTH_RADIUS
            strict_match_mask = dists_km_strict <= strict_radius
            strict_match_count[i] = int(np.count_nonzero(strict_match_mask))

            # per-cluster dict
            if strict_match_mask.any():
                clusters_s = li_cluster_valid[time_mask_strict][strict_match_mask].astype(int)
                if clusters_s.size:
                    u, c = np.unique(clusters_s, return_counts=True)
                    strict_dicts[i] = {int(k): int(v) for k, v in zip(u, c)}
                else:
                    strict_dicts[i] = {}
            else:
                strict_dicts[i] = {}
        else:
            strict_dicts[i] = {}

    # Build CPR summary dataset with per-cluster lightning counts
    all_clusters = set()
    for d in loose_dicts:
        all_clusters.update(d.keys())
    for d in strict_dicts:
        all_clusters.update(d.keys())

    if len(all_clusters) == 0:
        unique_clusters = np.array([9999], dtype=np.uint16)
    else:
        unique_clusters = np.array(sorted(all_clusters), dtype=np.uint16)

    n_clusters = unique_clusters.size
    cluster_col_idx = {sid: i for i, sid in enumerate(unique_clusters)}

    li_count_loose_out  = np.zeros((n_cpr, n_clusters), dtype=np.uint32)
    li_count_strict_out = np.zeros((n_cpr, n_clusters), dtype=np.uint32)
    for i in range(n_cpr):
        for sid, cnt in loose_dicts[i].items():
            li_count_loose_out[i, cluster_col_idx[sid]] += cnt
        for sid, cnt in strict_dicts[i].items():
            li_count_strict_out[i, cluster_col_idx[sid]] += cnt

    cpr_index = np.arange(n_cpr, dtype=np.uint16)
    strict_count_attrs = {"long_name": f"Lightning groups count per cluster within 2.5 km radius and ±2.5 min time window around each CPR sample", "units": "1",}
    loose_count_attrs = {"long_name": f"Lightning groups count per cluster within 5 km radius and ±5 min time window around each CPR sample", "units": "1",}
    cpr_summary_ds = xr.Dataset(
        data_vars={
            "lightning_count_2p5": (("cpr", "cluster_id"), li_count_strict_out, strict_count_attrs),
            "lightning_count_5": (("cpr", "cluster_id"), li_count_loose_out, loose_count_attrs),
        },
        coords={
            "cpr": ("cpr", cpr_index, {"long_name": "EarthCARE CPR sample index"}),
            "cluster_id": ("cluster_id", unique_clusters, {"long_name": "Cluster identifier"}),
            "latitude": ("cpr", cpr_lat, {"long_name": "CPR nadir latitude", "units": "degrees_north"}),
            "longitude": ("cpr", cpr_lon, {"long_name": "CPR nadir longitude", "units": "degrees_east"}),
            "time": ("cpr", cpr_time, {"long_name": "CPR observation time"}),
            "land_flag": ("cpr", cpr_land_flag, {"long_name": "CPR land/water flag", "definition": "1 = land, 0 = not land", "units": "1"}),
        },
        attrs={},
    )

    return annotated_ds, n_matched_nearest, cpr_summary_ds
