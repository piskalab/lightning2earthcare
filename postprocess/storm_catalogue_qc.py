import re
import pandas as pd
from datetime import datetime, timedelta, timezone
import s3fs
import numpy as np


# --- S3 bucket mapping ---
_BUCKET_BY_PLATFORM = {
    'GOES-16': 'noaa-goes16',
    'GOES-18': 'noaa-goes18',
    'GOES-19': 'noaa-goes19',
}

# --- GLM filename regex ---
_GLM_NAME_RE = re.compile(
    r'^OR_GLM-L2-LCFA_G(?P<plat>\d{2})_s(?P<s>\d{14})_e(?P<e>\d{14})_c(?P<c>\d{14})\.nc$'
)

def _parse_glm_timefield(s: str) -> datetime:
    return datetime.strptime(s[:13], '%Y%j%H%M%S').replace(tzinfo=timezone.utc)

def _iter_hours(dt_start: datetime, dt_end: datetime):
    """Yield (year, jday, hour) covering [dt_start, dt_end] inclusive."""
    current_hour = dt_start.replace(minute=0, second=0, microsecond=0)
    end_hour = dt_end.replace(minute=0, second=0, microsecond=0)
    if dt_end != end_hour:
        end_hour += timedelta(hours=1)
    while current_hour <= end_hour:
        yield current_hour.year, int(current_hour.strftime('%j')), current_hour.hour
        current_hour += timedelta(hours=1)

def check_glm_gaps_for_window(platform, start_time, end_time, product='GLM-L2-LCFA'):
    if platform not in _BUCKET_BY_PLATFORM:
        print(f"Unsupported platform {platform}")
        return

    bucket = _BUCKET_BY_PLATFORM[platform]
    fs = s3fs.S3FileSystem(anon=True)

    # --- generate hourly prefixes ---
    prefixes = [
        f"{bucket}/{product}/{year:04d}/{jday:03d}/{hour:02d}/"
        for (year, jday, hour) in _iter_hours(start_time, end_time)
    ]

    all_files = []
    for pref in prefixes:
        try:
            keys = fs.ls(pref)
        except FileNotFoundError:
            continue
        except Exception as e:
            continue

        for key in keys:
            fname = key.split('/')[-1]
            if _GLM_NAME_RE.match(fname):
                all_files.append(fname)


    if not all_files:
        print("No GLM files found in S3 for this window.")
        return

    # --- extract timestamps ---
    timestamps = []
    for fname in all_files:
        m = _GLM_NAME_RE.match(fname)
        s_dt = _parse_glm_timefield(m.group('s'))
        e_dt = _parse_glm_timefield(m.group('e'))
        timestamps.append((s_dt, e_dt))

    timestamps.sort(key=lambda x: x[0])

    # --- check gaps ---
    gaps = []
    for i in range(len(timestamps)-1):
        current_end = timestamps[i][1]
        next_start = timestamps[i+1][0]
        if next_start > current_end:
            gaps.append((current_end, next_start))
    return gaps

def extract_missing_intervals_from_logs(log_files, storm_catalog_df):
    """
    Parse log files to detect missing timeframes for LI and GLM.
    """

    valid_orbit_frames = set(storm_catalog_df["orbit_frame"].unique())

    missing_records = []

    re_orbit = re.compile(r"Processing orbit frame:\s+(\w+)")
    re_glm_proc = re.compile(
        r"processing GLM.*?(\d{4}-\d{2}-\d{2}T[\d:.]+)\s*(?:→|->)\s*(\d{4}-\d{2}-\d{2}T[\d:.]+)"
    )
    re_li_proc = re.compile(
        r"processing LI.*?(\d{4}-\d{2}-\d{2}T[\d:.]+)\s*(?:→|->)\s*(\d{4}-\d{2}-\d{2}T[\d:.]+)"
    )
    re_glm_error = re.compile(r"s(\d{14})_e(\d{14})_c(\d{14})\.nc")
    re_li_file = re.compile(r"_L2PF_OPE_(\d{14})_(\d{14})_N__")

    for log_file in log_files:

        current_frame = None
        frame_is_valid = False
        current_source = None
        expected_start = None
        expected_end = None
        li_found_windows = []

        def evaluate_li_frame():
            nonlocal missing_records, li_found_windows
            nonlocal expected_start, expected_end

            if not frame_is_valid:
                return

            if not expected_start or not expected_end:
                return
            if not li_found_windows:
                return

            li_found_windows.sort()
            current_pointer = expected_start

            for start, end in li_found_windows:
                if start <= current_pointer:
                    current_pointer = max(current_pointer, end)
                    continue

                if start > current_pointer:
                    missing_records.append({
                        "orbit_frame": current_frame,
                        "source": "LI",
                        "missing_start": current_pointer,
                        "missing_end": start
                    })
                    current_pointer = max(current_pointer, end)

            if current_pointer < expected_end:
                missing_records.append({
                    "orbit_frame": current_frame,
                    "source": "LI",
                    "missing_start": current_pointer,
                    "missing_end": expected_end
                })

        with open(log_file) as f:
            for line in f:

                # --------------------------------------------------
                # Orbit frame detection
                # --------------------------------------------------
                m = re_orbit.search(line)
                if m:
                    if current_source == "LI" and frame_is_valid:
                        evaluate_li_frame()

                    current_frame = m.group(1)
                    frame_is_valid = current_frame in valid_orbit_frames

                    current_source = None
                    expected_start = None
                    expected_end = None
                    li_found_windows = []

                    if frame_is_valid:
                        print(f"\nProcessing VALID frame: {current_frame}")
                    else:
                        print(f"\nSkipping frame (no along-track storm): {current_frame}")

                    continue

                # HARD SKIP EVERYTHING BELOW IF FRAME NOT VALID
                if not frame_is_valid:
                    continue

                # --------------------------------------------------
                # GLM processing
                # --------------------------------------------------
                m = re_glm_proc.search(line)
                if m:
                    if current_source == "LI":
                        evaluate_li_frame()

                    current_source = "GLM"

                    try:
                        expected_start = (
                            pd.to_datetime(m.group(1), utc=True)
                            .to_pydatetime()
                        )

                        expected_end = (
                            pd.to_datetime(m.group(2), utc=True)
                            .to_pydatetime()
                        )

                        platform_match = re.search(r'\((GOES-\d+)\)', line)
                        if platform_match:
                            platform = platform_match.group(1)
                            gaps = check_glm_gaps_for_window(platform, expected_start, expected_end)

                            if gaps:
                                for gap_start, gap_end in gaps:
                                    # Crop gap to expected window
                                    cropped_start = max(gap_start, expected_start)
                                    cropped_end   = min(gap_end, expected_end)

                                    # Keep only if overlapping expected window
                                    if cropped_end > cropped_start:
                                        print(f"Detected missing GLM segment (cropped): {cropped_start} → {cropped_end}")
                                        if current_frame in valid_orbit_frames:
                                            missing_records.append({
                                                "orbit_frame": current_frame,
                                                "source": "GLM",
                                                "missing_start": cropped_start,
                                                "missing_end": cropped_end
                                            })

                    except Exception as exc:
                        print(f"Failed to parse GLM processing line in {log_file.name}: {exc}")

                    continue

                # --------------------------------------------------
                # LI processing
                # --------------------------------------------------
                m = re_li_proc.search(line)
                if m:
                    if current_source == "LI":
                        evaluate_li_frame()

                    current_source = "LI"

                    try:
                        expected_start = pd.to_datetime(m.group(1))
                        expected_end = pd.to_datetime(m.group(2))
                    except Exception as exc:
                        print(f"Failed to parse LI processing line in {log_file.name}: {exc}")

                    continue

                # --------------------------------------------------
                # GLM ERROR parsing
                # --------------------------------------------------
                if current_source == "GLM" and "ERROR" in line:
                    m = re_glm_error.search(line)
                    if m:
                        raw_start = m.group(1)
                        raw_end   = m.group(3)

                        start_str = raw_start[:-1]
                        start_frac = int(raw_start[-1])

                        end_str = raw_end[:-1]
                        end_frac = int(raw_end[-1])

                        start_time = (
                            pd.to_datetime(start_str, format="%Y%j%H%M%S")
                            + pd.to_timedelta(start_frac * 0.1, unit='s')
                        )
                        end_time = (
                            pd.to_datetime(end_str, format="%Y%j%H%M%S")
                            + pd.to_timedelta(end_frac * 0.1, unit='s')
                        )

                        missing_records.append({
                            "orbit_frame": current_frame,
                            "source": "GLM",
                            "missing_start": start_time,
                            "missing_end": end_time
                        })

                # --------------------------------------------------
                # LI file detection
                # --------------------------------------------------
                if current_source == "LI":
                    is_success_line = (
                        "Downloaded" in line or
                        "Data already exist, skipping" in line
                    )              
                    if is_success_line:
                        m = re_li_file.search(line)
                        if m:
                            start_time = pd.to_datetime(m.group(1), format="%Y%m%d%H%M%S")
                            end_time = pd.to_datetime(m.group(2), format="%Y%m%d%H%M%S")
                            li_found_windows.append((start_time, end_time))

        # Final LI evaluation
        if current_source == "LI" and frame_is_valid:
            evaluate_li_frame()

    return pd.DataFrame(missing_records)

def add_missing_peak_minutes(storm_df, df_missing, buffer_minutes=2.5):
    """
    Calculates missing time (in minutes) overlapping ±buffer_minutes around peak_datetime
    for each storm (same orbit_frame + source).
    """

    storm_df = storm_df.copy()
    storm_df["missing_peak_minutes"] = 0.0

    storm_df["peak_datetime"] = pd.to_datetime(storm_df["peak_datetime"])
    df_missing["missing_start"] = pd.to_datetime(df_missing["missing_start"])
    df_missing["missing_end"] = pd.to_datetime(df_missing["missing_end"])

    buffer = pd.to_timedelta(buffer_minutes, unit="m")

    for (orbit_frame, source), storm_group in storm_df.groupby(["orbit_frame", "source"]):

        missing_group = df_missing[
            (df_missing["orbit_frame"] == orbit_frame) &
            (df_missing["source"] == source)
        ]

        if missing_group.empty:
            continue

        for idx, storm_row in storm_group.iterrows():

            peak = storm_row["peak_datetime"]
            storm_start = peak - buffer
            storm_end   = peak + buffer

            total_overlap = 0.0
            for _, miss_row in missing_group.iterrows():
                overlap_start = max(storm_start, miss_row["missing_start"])
                overlap_end   = min(storm_end,   miss_row["missing_end"])
                overlap = (overlap_end - overlap_start).total_seconds() / 60
                if overlap > 0:
                    total_overlap += overlap

            if total_overlap > 0:
                storm_df.loc[idx, "missing_peak_minutes"] = total_overlap

    return storm_df

def mask_missing_minute_counts(storm_df, df_missing):
    """
    For each storm, checks each minute in minute_counts (-60 to +60 relative to peak)
    and masks it as NaN if it falls within any missing interval (same orbit_frame + source).
    """

    storm_df = storm_df.copy()
    storm_df["masked_minute_counts"] = storm_df["minute_counts"].copy()

    storm_df["peak_datetime"] = pd.to_datetime(storm_df["peak_datetime"])
    df_missing["missing_start"] = pd.to_datetime(df_missing["missing_start"])
    df_missing["missing_end"] = pd.to_datetime(df_missing["missing_end"])

    for (orbit_frame, source), storm_group in storm_df.groupby(["orbit_frame", "source"]):

        missing_group = df_missing[
            (df_missing["orbit_frame"] == orbit_frame) &
            (df_missing["source"] == source)
        ]

        if missing_group.empty:
            continue

        for idx, storm_row in storm_group.iterrows():

            peak = storm_row["peak_datetime"]
            minute_counts = storm_row["minute_counts"].copy()

            for minute_offset in range(-60, 61):
                minute_time = peak + pd.Timedelta(minutes=minute_offset)

                overlap = (
                    (minute_time >= missing_group["missing_start"]) &
                    (minute_time <= missing_group["missing_end"])
                )

                if overlap.any():
                    minute_counts[str(minute_offset)] = float("nan")

            storm_df.at[idx, "masked_minute_counts"] = minute_counts

    return storm_df

def json_safe(obj):
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