import logging
import numpy as np
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


logger = logging.getLogger(__name__)


def _filter_by_quality(
    ds: xr.Dataset,
    cluster_bad_threshold: float = 0.25
) -> xr.Dataset:
    """
    Filters out low-quality clusters and groups based on flag variables.
      - GLM: uses 'group_quality_flag' (0 = good)
      - LI : uses the existing l1b_* warnings (0 = good)
    Drops flag and auxiliary variables after filtering.
    """
    quality_flag_candidates = [
        "group_quality_flag",
        "l1b_missing_warning",
        "l1b_geolocation_warning",
        "l1b_radiometric_warning",
    ]
    quality_flags = [f for f in quality_flag_candidates if f in ds.variables]
    auxiliary_vars = [
        "auxiliary_dataset_identifier",
        "auxiliary_dataset_status",
        "group_filter_qa",
    ]

    if not quality_flags:
        logger.info("No known quality flags found; skipping quality filtering.")
        return ds

    n_groups = ds.sizes.get("groups", None)
    if n_groups is None:
        logger.warning("Dataset missing 'groups' dimension; skipping quality filtering.")
        return ds

    valid_group_mask = np.ones(n_groups, dtype=bool)
    for flag in quality_flags:
        arr = ds[flag]
        if "groups" in getattr(arr, "dims", ()):
            valid_group_mask &= (arr.values == 0)

    parent_clusters = ds["parent_cluster_id"].values
    parent_cluster_df = ds[["parent_cluster_id"]].to_dataframe().reset_index()
    parent_cluster_df["is_bad"] = ~valid_group_mask
    parent_cluster_df = parent_cluster_df[parent_cluster_df["parent_cluster_id"] != -1]

    if not parent_cluster_df.empty:
        bad_rates = parent_cluster_df.groupby("parent_cluster_id")["is_bad"].mean()
        bad_clusters = bad_rates[bad_rates > cluster_bad_threshold].index.values
        group_in_bad_cluster = np.isin(parent_clusters, bad_clusters)
        keep_group_mask = (~group_in_bad_cluster) & valid_group_mask
        logger.info(f"Dropped {len(bad_clusters)} of {bad_rates.size} clusters for quality issues.")
    else:
        keep_group_mask = valid_group_mask
        logger.info("No non-noise clusters; applying per-group quality mask only.")

    filtered_ds = ds.isel(groups=keep_group_mask)
    vars_to_drop = [v for v in (quality_flags + auxiliary_vars) if v in filtered_ds.variables]
    if vars_to_drop:
        filtered_ds = filtered_ds.drop_vars(vars_to_drop)
    return filtered_ds


def parent_clustering(
    li_ds: xr.Dataset,
    eps: float = 5.0,
    time_weight: float = 0.5,
    min_samples: int = 20,
    lat_gap: float = 0.25
) -> xr.Dataset | None:
    """
    Cluster lightning groups in space-time using DBSCAN, after splitting into latitude chunks.
    Returns a copy of li_ds with a new variable 'parent_cluster_id' (same length as groups).

    Parameters
    ----------
    li_ds : xr.Dataset
    eps : DBSCAN eps in kilometers (applies to projected x/y). Time dimension is weighted separately.
    time_weight : Multiplier for time to balance against kms in (x,y); e.g., time_weight=0.5 means 10 minutes ~ 5 km in distance.
    min_samples : DBSCAN min_samples.
    lat_gap : Gap (deg) to split latitude chunks before clustering.

    Returns a copy of li_ds with new DataArray 'parent_cluster_id', -1 denotes noise.
    """
    MAX_GROUPS_PER_CHUNK = 100_000

    group_ids = li_ds["group_id"].values
    flash_ids = li_ds["flash_id"].values
    lat   = li_ds["latitude"].values
    lon   = li_ds["longitude"].values
    time  = li_ds["group_time"].values

    n_groups = len(group_ids)

    scale_factor_time = 1e9 * 60
    time_minutes = ((time - time.min()).astype("timedelta64[ns]").astype(np.int64)) / scale_factor_time
    time_scaled = time_minutes * float(time_weight)

    # Sort once by latitude, then process independent latitude chunks
    idx_sort = np.argsort(lat)
    flash_ids_sorted = flash_ids[idx_sort]
    lat_sorted = lat[idx_sort]
    lon_sorted = lon[idx_sort]
    time_sorted = time_scaled[idx_sort]

    unique_lats = np.unique(lat_sorted)
    lat_diffs = np.diff(unique_lats)
    gap_idx = np.where(lat_diffs >= lat_gap)[0]
    boundaries = [unique_lats[0], *[unique_lats[i+1] for i in gap_idx], unique_lats[-1] + 1e-12]

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    rng = np.random.default_rng()

    cluster_id_sorted = np.full(n_groups, -1, dtype=np.int64)
    sampled_sorted = np.zeros(n_groups, dtype=bool)
    label_offset = 0

    # Cluster each latitude chunk separately
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        chunk_mask = (lat_sorted >= lo) & (lat_sorted < hi)
        chunk_idx = np.where(chunk_mask)[0]
        n_chunk = chunk_idx.size
        if n_chunk == 0:
            continue
        
        # Sample large chunks, but keep at least one representative per flash
        if n_chunk > MAX_GROUPS_PER_CHUNK:
            sampled_chunk = np.zeros(n_chunk, dtype=bool)
            flash_ids_chunk = flash_ids_sorted[chunk_idx]
            flash_to_positions = {}
            for local_pos, fid in enumerate(flash_ids_chunk):
                if fid in flash_to_positions:
                    flash_to_positions[fid].append(local_pos)
                else:
                    flash_to_positions[fid] = [local_pos]

            n_flashes_chunk = len(flash_to_positions)

            if n_flashes_chunk > MAX_GROUPS_PER_CHUNK:
                logger.warning(
                    f"Chunk [{lo:.4f}, {hi:.4f}) has {n_flashes_chunk} flashes "
                    f"> MAX_GROUPS_PER_CHUNK={MAX_GROUPS_PER_CHUNK}. "
                    "Sampling one per flash (sample will exceed limit)."
                )
                max_target = n_flashes_chunk
            else:
                max_target = MAX_GROUPS_PER_CHUNK

            for fid, positions in flash_to_positions.items():
                positions = np.asarray(positions, dtype=int)
                chosen_local = rng.choice(positions)
                sampled_chunk[chosen_local] = True

            current_sampled = int(sampled_chunk.sum())
            remaining_capacity = max_target - current_sampled

            if remaining_capacity > 0:
                unsampled_local = np.where(~sampled_chunk)[0]
                if unsampled_local.size > 0:
                    if remaining_capacity < unsampled_local.size:
                        extra_local = rng.choice(
                            unsampled_local, size=remaining_capacity, replace=False
                        )
                    else:
                        extra_local = unsampled_local
                    sampled_chunk[extra_local] = True
        else:
            sampled_chunk = np.ones(n_chunk, dtype=bool)

        sampled_idx_global = chunk_idx[sampled_chunk]
        sampled_sorted[sampled_idx_global] = True
        if sampled_idx_global.size == 0:
            continue

        lat_sampled = lat_sorted[sampled_idx_global]
        lon_sampled = lon_sorted[sampled_idx_global]
        time_sampled = time_sorted[sampled_idx_global]

        valid_input_mask = (
            np.isfinite(lat_sampled) &
            np.isfinite(lon_sampled) &
            np.isfinite(time_sampled)
        )
        if not valid_input_mask.all():
            lat_sampled = lat_sampled[valid_input_mask]
            lon_sampled = lon_sampled[valid_input_mask]
            time_sampled = time_sampled[valid_input_mask]
            sampled_idx_global = sampled_idx_global[valid_input_mask]

        if lat_sampled.size == 0:
            continue

        x_proj_m, y_proj_m = transformer.transform(lon_sampled, lat_sampled)
        features = np.stack(
            [np.asarray(x_proj_m) / 1000.0,
             np.asarray(y_proj_m) / 1000.0,
             time_sampled],
            axis=1,
        )

        valid_feature_mask = np.isfinite(features).all(axis=1)
        if not valid_feature_mask.all():
            features = features[valid_feature_mask]
            sampled_idx_global = sampled_idx_global[valid_feature_mask]

        if features.shape[0] == 0:
            continue

        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        labels = db.fit_predict(features)

        is_clustered = labels != -1
        labels_offset = labels.copy()
        labels_offset[is_clustered] += label_offset
        if np.any(is_clustered):
            label_offset = labels_offset[is_clustered].max() + 1

        # Enforce the minimum cluster size
        if np.any(is_clustered):
            unique_labels, label_counts = np.unique(labels_offset[is_clustered], return_counts=True)
            small_clusters = unique_labels[label_counts < min_samples]
            if small_clusters.size:
                mask_small = np.isin(labels_offset, small_clusters)
                labels_offset[mask_small] = -1

        cluster_id_sorted[sampled_idx_global] = labels_offset

        # Propagate sampled cluster labels to unsampled points by nearest neighbour
        unsampled_idx_global = chunk_idx[~sampled_chunk]
        if unsampled_idx_global.size > 0 and np.any(is_clustered):
            cluster_features = features[labels_offset != -1]
            cluster_valid = labels_offset[labels_offset != -1]
            lat_unsampled = lat_sorted[unsampled_idx_global]
            lon_unsampled = lon_sorted[unsampled_idx_global]
            time_unsampled = time_sorted[unsampled_idx_global]
            x_unsampled, y_unsampled = transformer.transform(lon_unsampled, lat_unsampled)
            unsampled_features = np.stack([np.asarray(x_unsampled)/1000.0, np.asarray(y_unsampled)/1000.0, time_unsampled], axis=1)
            tree = cKDTree(cluster_features)
            dists, idxs = tree.query(unsampled_features, k=1)
            assigned_clusters = np.full(len(unsampled_idx_global), -1, dtype=np.int64)
            assigned_clusters[dists <= eps] = cluster_valid[idxs[dists <= eps]]
            cluster_id_sorted[unsampled_idx_global] = assigned_clusters

    # Map labels back to the original dataset order
    cluster_id_orig = np.empty(n_groups, dtype=np.int16)
    cluster_id_orig[idx_sort] = cluster_id_sorted

    unique_clusters = np.unique(cluster_id_orig)
    n_clusters_total = int(np.sum(unique_clusters != -1))
    if n_clusters_total == 0:
        logger.info("No clusters found (all points classified as noise).")
        return None
    else:
        logger.info(f"Total clusters found: {n_clusters_total}")

    clustered_ds = li_ds.copy()
    clustered_ds["parent_cluster_id"] = xr.DataArray(
        cluster_id_orig,
        dims=li_ds["group_id"].dims,
        attrs={
            "long_name": "Parent cluster ID",
            "description": f"DBSCAN clustering (eps={eps}, time_weight={time_weight}, min_samples={min_samples}); -1 = noise",
        },
    )

    clustered_ds = _filter_by_quality(clustered_ds)
    return clustered_ds


def subclustering(
    matched_ds: xr.Dataset,
    eps: float = 5.0,
    time_weight: float = 0.5,
    min_samples: int = 20,
) -> xr.Dataset:
    """
    Re-cluster matched lightning groups (where ec_time_diff is valid)
    within each existing parent_cluster_id, using group_time for temporal proximity.

    Adds 'cluster_id': NaN = unmatched, -1 = noise.
    """
    MAX_POINTS_PER_CLUSTER = 100_000

    lat = matched_ds["latitude"].values
    lon = matched_ds["longitude"].values
    time = matched_ds["group_time"].values
    parent_cluster = matched_ds["parent_cluster_id"].values
    flash_id = matched_ds["flash_id"].values
    valid = ~np.isnat(matched_ds["ec_time_diff"].values)

    cluster_ids = np.full(parent_cluster.shape, np.nan, dtype="float32")

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    rng = np.random.default_rng()
    label_offset = 0

    # Process each parent cluster independently and subcluster only matched groups
    for parent_id in np.unique(parent_cluster):
        if parent_id < 0:
            continue

        parent_mask = (parent_cluster == parent_id) & valid
        if not np.any(parent_mask):
            continue

        parent_idx = np.where(parent_mask)[0]
        n_parent = parent_idx.size

        # Sample very large parent clusters, while keeping at least one point per flash
        if n_parent > MAX_POINTS_PER_CLUSTER:
            sampled_local = np.zeros(n_parent, dtype=bool)
            f_c = flash_id[parent_idx]
            flash_to_positions = {}
            for local_pos, fid in enumerate(f_c):
                if fid in flash_to_positions:
                    flash_to_positions[fid].append(local_pos)
                else:
                    flash_to_positions[fid] = [local_pos]

            max_target = MAX_POINTS_PER_CLUSTER

            for positions in flash_to_positions.values():
                positions = np.asarray(positions, dtype=int)
                chosen_local = rng.choice(positions)
                sampled_local[chosen_local] = True

            current_sampled = int(sampled_local.sum())
            remaining_capacity = max_target - current_sampled

            if remaining_capacity > 0:
                unsampled_local = np.where(~sampled_local)[0]
                if unsampled_local.size > 0:
                    if remaining_capacity < unsampled_local.size:
                        extra_local = rng.choice(
                            unsampled_local,
                            size=remaining_capacity,
                            replace=False,
                        )
                    else:
                        extra_local = unsampled_local
                    sampled_local[extra_local] = True
        else:
            sampled_local = np.ones(n_parent, dtype=bool)

        sampled_idx = parent_idx[sampled_local]
        if sampled_idx.size == 0:
            continue

        # Build DBSCAN features from projected x/y coordinates and weighted time
        lat_sampled = lat[sampled_idx]
        lon_sampled = lon[sampled_idx]
        time_sampled = ((time[sampled_idx] - time[sampled_idx].min()).astype("timedelta64[ns]").astype(np.int64)/(1e9 * 60)) * time_weight

        x_proj_m, y_proj_m = transformer.transform(lon_sampled, lat_sampled)
        features = np.stack([np.asarray(x_proj_m) / 1000.0, np.asarray(y_proj_m) / 1000.0, time_sampled], axis=1)

        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        labels = db.fit_predict(features)

        # Offset labels so subcluster IDs remain unique across parent clusters
        is_clustered = labels != -1
        labels = labels.astype("float64")  # ensure float for NaN compatibility
        if np.any(is_clustered):
            labels[is_clustered] += label_offset
            label_offset = labels[is_clustered].max() + 1
        cluster_ids[sampled_idx] = labels

        # Assign unsampled points to the nearest sampled subcluster when close enough
        unsampled_idx = parent_idx[~sampled_local]
        if unsampled_idx.size > 0 and np.any(is_clustered):
            cluster_features = features[labels != -1]
            cluster_valid = labels[labels != -1]

            lat_unsampled = lat[unsampled_idx]
            lon_unsampled = lon[unsampled_idx]
            time_unsampled = ((time[unsampled_idx] - time[sampled_idx].min()).astype("timedelta64[ns]").astype(np.int64)/(1e9*60)) * time_weight
            x_unsampled, y_unsampled = transformer.transform(lon_unsampled, lat_unsampled)
            unsampled_features = np.stack([np.asarray(x_unsampled)/1000, np.asarray(y_unsampled)/1000, time_unsampled], axis=1)

            tree = cKDTree(cluster_features)
            dists, idxs = tree.query(unsampled_features, k=1)

            assigned_clusters = np.full(len(unsampled_idx), -1.0)
            assigned_clusters[dists <= eps] = cluster_valid[idxs[dists <= eps]]
            cluster_ids[unsampled_idx] = assigned_clusters

    logger.info(f"Total subclusters created: {np.nansum(np.unique(cluster_ids) != -1)}")
    
    # Attach the new subcluster labels to the matched dataset
    subclustered_ds = matched_ds.copy()
    subclustered_ds["cluster_id"] = xr.DataArray(
        cluster_ids,
        dims=matched_ds["group_id"].dims,
        attrs={
            "long_name": "Cluster ID within parent clusters",
            "description": f"DBSCAN clustering (eps={eps}, time_weight={time_weight}, min_samples={min_samples}); -1 = noise",
        },
    )
    return subclustered_ds
