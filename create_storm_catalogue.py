#!/usr/bin/env python3
"""
Create a lightning storm catalogue from LI / GLM lightning groups
and CPR track-count files.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics.pairwise import haversine_distances
from pyproj import Transformer

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

EARTH_RADIUS_KM = 6371.0

# ─────────────────────────────────────────────
# USER PARAMETERS
# ─────────────────────────────────────────────

LI_DIR           = Path('/home/bpiskala/Object_Data/lightning_processing/lightning_groups_20240801_20260131')
TRACK_COUNTS_DIR = Path('/home/bpiskala/Object_Data/lightning_processing/track_counts_20240801_20260131')

distance_km  = 2.5
time_s       = 150
minute_span = 60

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _infer_source_from_filename(path_nc: Path) -> str:
    """Return 'LI' or 'GLM' based on filename prefix."""
    head = path_nc.name.split('_', 1)[0].upper()
    return 'GLM' if head == 'GLM' else 'LI'


def _find_counts_sidecar_path(source: str, orbit_frame: str) -> Path | None:
    matches = sorted(TRACK_COUNTS_DIR.glob(f"*{source}*{orbit_frame}*.nc"))
    return matches[0] if matches else None


def _json_safe(obj):
    """Convert NumPy / pandas objects to JSON-safe Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return str(obj)

# ─────────────────────────────────────────────
# CORE PROCESSING
# ─────────────────────────────────────────────

def process_one_file(path_nc: Path) -> pd.DataFrame:
    """
    For each cluster defined in the sidecar counts file:
      1) Find peak CPR point by max per-cluster count.
      2) From LI data, within ±time_s of peak time: compute mean lat/lon, counts, and distances to CPR.
      3) Produce per-minute counts and per-minute mean lat/lon (relative to peak time).
    """
    orbit_frame = path_nc.name.split('_')[-1].replace('.nc', '')
    source = _infer_source_from_filename(path_nc)

    counts_path = _find_counts_sidecar_path(source, orbit_frame)
    print(f'  Found counts sidecar: {counts_path.name if counts_path else "NONE"}')
    if not counts_path:
        return None

    with xr.open_dataset(path_nc) as li, xr.open_dataset(counts_path) as counts_ds:
        c_lat = counts_ds['latitude'].values
        c_lon = counts_ds['longitude'].values
        c_time = pd.to_datetime(counts_ds['time'].values)
        land_flag = counts_ds['land_flag'].values 

        varname = f"lightning_count_{str(distance_km).replace('.', 'p')}"

        # rows = CPR index, columns = real cluster_id
        counts_df = (counts_ds[varname].astype(int).to_pandas())
        counts_df.columns = counts_ds['cluster_id'].values.astype(int)

        li_cluster = li['cluster_id'].values
        li_par_cluster = li['parent_cluster_id'].values
        li_time = pd.to_datetime(li['group_time'].values)

        # Use parallax-corrected coordinates for LI, otherwise fall back (for GLM cases)
        if "parallax_corrected_lat" in li and "parallax_corrected_lon" in li:
            li_lat = li['parallax_corrected_lat'].values
            li_lon = li['parallax_corrected_lon'].values
        else:
            li_lat = li['latitude'].values
            li_lon = li['longitude'].values

        cpr_coords_rad = np.radians(np.column_stack([c_lat, c_lon]))

        records = []
        for cid in counts_df.columns:
            series = counts_df[cid]
            if series.max() == 0:
                continue

            # Compute land_fraction and surface_type
            cpr_mask = series > 0

            if cpr_mask.sum() == 0:
                land_fraction = np.nan
                surface_type  = "unknown"
            else:
                land_fraction = land_flag[cpr_mask].mean()
                if land_fraction >= 0.9:
                    surface_type = "land"
                elif land_fraction <= 0.1:
                    surface_type = "water"
                else:
                    surface_type = "coast"

            idx_best = int(series.values.argmax())
            peak_count = int(series.iloc[idx_best])
            peak_lat = float(c_lat[idx_best])
            peak_lon = float(c_lon[idx_best])
            peak_time = c_time[idx_best]

            # LI points in this cluster within ±time_s of peak_time
            is_cluster = (li_cluster == cid)
            par_cluster_id = np.unique(li_par_cluster[is_cluster])
            is_par_cluster = (li_par_cluster == par_cluster_id)
            
            mask = is_cluster & np.isfinite(li_lat) & np.isfinite(li_lon) & ~np.isnat(li_time)
            times, lats, lons = [arr[mask] for arr in (li_time, li_lat, li_lon)]
            times_all, lats_all, lons_all = [arr[is_par_cluster] for arr in (li_time, li['latitude'].values, li['longitude'].values)]

            GRID_RES = 5000  # meters per grid cell
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            x, y = transformer.transform(lons, lats)
            grid_x = (x // GRID_RES).astype(int)
            grid_y = (y // GRID_RES).astype(int)
            n_cells = len(set(zip(grid_x, grid_y)))
            cluster_area_km2 = n_cells * (GRID_RES/1000)**2

            dt_s = np.abs((times - peak_time).astype('timedelta64[s]').astype(int))
            in_time = dt_s <= time_s

            n_in_time = int(np.count_nonzero(in_time))
            lats_t = lats[in_time]; lons_t = lons[in_time]; times_t = times[in_time]
            mean_lat = float(np.nanmean(lats_t))
            mean_lon = float(np.nanmean(lons_t))

            # distance from mean(lat,lon) to nearest CPR point
            mean_coord_rad = np.radians([[mean_lat, mean_lon]])
            dists_rad = haversine_distances(mean_coord_rad, cpr_coords_rad)[0]
            min_dist_mean_to_cpr_km = float(dists_rad.min() * EARTH_RADIUS_KM)

            # count LI points also within distance_km of *some* CPR
            li_coords_rad = np.radians(np.column_stack([lats_t, lons_t]))
            dists_rad = haversine_distances(li_coords_rad, cpr_coords_rad)
            min_dists_km = dists_rad.min(axis=1) * EARTH_RADIUS_KM
            n_in_time_and_dist = int(np.count_nonzero(min_dists_km <= distance_km))

            # per-minute bins
            minute_off = ((times_all - peak_time).astype('timedelta64[s]').astype(int) // 60)
            valid = (minute_off >= -minute_span) & (minute_off <= minute_span)
            minute_off = minute_off[valid]; lats_b = lats_all[valid]; lons_b = lons_all[valid]

            s_counts = (pd.Series(1, index=minute_off)
                        .groupby(level=0).sum()
                        .reindex(range(-minute_span, minute_span+1), fill_value=0))
            s_lat = (pd.Series(lats_b, index=minute_off)
                    .groupby(level=0).mean()
                    .reindex(range(-minute_span, minute_span+1)))
            s_lon = (pd.Series(lons_b, index=minute_off)
                    .groupby(level=0).mean()
                    .reindex(range(-minute_span, minute_span+1)))

            minute_counts = {int(k): int(v) for k, v in s_counts.items()}
            minute_mean_lat = {int(k): (round(float(v), 6) if pd.notna(v) else np.nan) for k, v in s_lat.items()}
            minute_mean_lon = {int(k): (round(float(v), 6) if pd.notna(v) else np.nan) for k, v in s_lon.items()}

            active_minutes = [k for k, v in minute_counts.items() if v > 0]
            first_lightning_min = min(active_minutes)
            last_lightning_min = max(active_minutes)
            storm_duration_min = last_lightning_min - first_lightning_min + 1

            # compute storm travel distance
            lat1 = minute_mean_lat.get(first_lightning_min, np.nan)
            lon1 = minute_mean_lon.get(first_lightning_min, np.nan)
            lat2 = minute_mean_lat.get(last_lightning_min, np.nan)
            lon2 = minute_mean_lon.get(last_lightning_min, np.nan)
            
            coords_rad = np.radians([[lat1, lon1], [lat2, lon2]])
            storm_travel_km = haversine_distances([coords_rad[0]], [coords_rad[1]])[0,0] * EARTH_RADIUS_KM

            records.append(dict(
                unique_id                 = f"{orbit_frame}_{source}_{int(cid)}",
                earthcare_id              = orbit_frame,
                source                    = source,
                parent_cluster_id         = par_cluster_id[0],
                cluster_id                = int(cid),
                surface_type              = surface_type,
                peak_datetime             = pd.to_datetime(peak_time),
                peak_lat                  = peak_lat,
                peak_lon                  = peak_lon,
                peak_lightning            = peak_count,
                nadir_lightning           = n_in_time_and_dist,
                cluster_lightning         = n_in_time,
                cluster_area_km2          = cluster_area_km2,
                cluster_mean_lat          = mean_lat,
                cluster_mean_lon          = mean_lon,
                cluster_dist_km           = min_dist_mean_to_cpr_km,
                first_lightning_min       = first_lightning_min,
                last_lightning_min        = last_lightning_min,
                duration_min              = storm_duration_min,
                travel_km                 = storm_travel_km,
                minute_counts             = minute_counts,
                minute_mean_lat           = minute_mean_lat,
                minute_mean_lon           = minute_mean_lon,
            ))

        return pd.DataFrame(records)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():

    all_frames = []

    for nc_path in sorted(list(LI_DIR.glob('LI_202*.nc')) + list(LI_DIR.glob('GLM_202*.nc'))):
        print(f'Processing {nc_path.name}')
        df = process_one_file(nc_path)
        if df is not None and not df.empty:
            all_frames.append(df)

    if not all_frames:
        print("No storms found.")
        return

    catalogue = pd.concat(all_frames, ignore_index=True)
    catalogue = catalogue.sort_values(
        by=["earthcare_id", "parent_cluster_id", "cluster_id"],
        kind="mergesort"
    ).reset_index(drop=True)

    # --- metadata ---
    field_descriptions = {
        "unique_id":            "Unique identifier of the lightning cluster (earthcare_id + source + cluster_id)",
        "earthcare_id":         "Orbit/frame identifier of the EarthCARE overpass",
        "source":               "Lightning data source: LI (MTG) or GLM (GOES)",
        "parent_cluster_id":    "Parent lightning cluster identifier",
        "cluster_id":           "Lightning cluster identifier",
        "surface_type":         "Surface classification at cluster location (land, water, coast)",
        "peak_datetime":        "Time of CPR sample with maximum lightning activity",
        "peak_lat":             "Latitude of CPR sample with maximum lightning activity",
        "peak_lon":             "Longitude of CPR sample with maximum lightning activity",
        "peak_lightning":       "Maximum lightning group count at a CPR sample",
        "nadir_lightning":      "Lightning group count within ±2.5 min and ±2.5 km of CPR nadir track",
        "cluster_lightning":    "Lightning group count within ±2.5 min of CPR peak time (any distance)",
        "cluster_area_km2":     "Area of cluster in km²",
        "cluster_mean_lat":     "Mean latitude of lightning groups within ±2.5 min of CPR peak time",
        "cluster_mean_lon":     "Mean longitude of lightning groups within ±2.5 min of CPR peak time",
        "cluster_dist_km":      "Minimum distance (km) between CPR track and mean cluster location",
        "first_lightning_min":  "Minute offset of first lightning in parent cluster relative to peak_datetime",
        "last_lightning_min":   "Minute offset of last lightning in parent cluster relative to peak_datetime",
        "duration_min":         "Parent cluster duration in minutes",
        "travel_km":            "Shortest distance (km) between first and last lightning locations in parent cluster",
        "minute_counts":        "Lightning group counts in parent cluster per minute (–60…+60) relative to peak_datetime",
        "minute_mean_lat":      "Mean lightning latitude in parent cluster per minute (–60…+60) relative to peak_datetime",
        "minute_mean_lon":      "Mean lightning longitude in parent cluster per minute (–60…+60) relative to peak_datetime",
    }

    summary_info = {
        "title": "EarthCARE lightning storm catalogue",
        "description": (
            "Each catalogue entry represents an individual lightning cluster "
            "associated with an EarthCARE CPR overpass, identified within ±2.5 minutes "
            "of the overpass and within a 2.5 km radius of the CPR nadir track."
        )}

    output = {
        "summary": summary_info,
        "metadata": field_descriptions,
        "data": catalogue.to_dict(orient="records")
    }

    out_file = f"EarthCARE_lightning_storm_catalogue_20240801_20260131.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=_json_safe)

    print(f"Saved {len(catalogue)} storms → {out_file}")


if __name__ == "__main__":
    main()
