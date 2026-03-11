import logging
from datetime import timedelta
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


_LOG_CONTEXT = {"proc_day": "-"}
_GLM_EAST_SWITCH_DATE = np.datetime64("2025-04-07") # GLM-East platform handover date (GOES-16 -> GOES-19)


def _set_log_day(day_str: str) -> None:
    """Set the current processing day tag (YYYY-MM-DD)."""
    _LOG_CONTEXT["proc_day"] = day_str

class _ProcessingDayFilter(logging.Filter):
    def filter(self, record):
        record.proc = _LOG_CONTEXT["proc_day"]
        return True

def _set_monthly_log_file(log_dir: str | Path, year: int, month: int) -> None:
    """
    Attach a FileHandler to the ROOT logger that writes to YYYY_MM.log
    based on the *processing date*. Appends if the file already exists.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{year:04d}_{month:02d}.log"
    root = logging.getLogger()

    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    fh = logging.FileHandler(log_file, mode="a")
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [proc=%(proc)s] - %(message)s'
    )
    fh.setFormatter(formatter)
    fh.addFilter(_ProcessingDayFilter())
    root.addHandler(fh)

def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure the ROOT logger.
    File logging is attached separately per processing month.
    """
    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [proc=%(proc)s] - %(message)s'
        )

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        sh.addFilter(_ProcessingDayFilter())
        root.addHandler(sh)
    return root


def _in_lon_ranges(lon, ranges):
    """
    Return a boolean mask of longitudes that fall within any of the given ranges.
    """
    mask = np.zeros(lon.shape, dtype=bool)
    for lo, hi in ranges:
        mask |= (lon >= lo) & (lon <= hi)
    return mask

def _ec_time_window(times: np.ndarray, half_window_minutes: int):
    """
    Compute [start, end] window around EC times.
    """
    start_time = times[0] - np.timedelta64(half_window_minutes, 'm')
    end_time   = times[-1] + np.timedelta64(half_window_minutes, 'm')
    return start_time, end_time

def _glm_east_platform(at_time: np.datetime64) -> str:
    return 'GOES-16' if at_time < _GLM_EAST_SWITCH_DATE else 'GOES-19'

def _add_if_platform_allowed(lightning_sources, entry, allowed_platforms):
    if allowed_platforms is None:
        lightning_sources.append(entry)
        return

    allowed = set(allowed_platforms)
    if entry["platform"] in allowed:
        lightning_sources.append(entry)
    else:
        logger.info(
            "Skipping %s (%s) due to platform filter %s",
            entry.get("source"),
            entry.get("platform"),
            tuple(allowed_platforms),
        )


def iter_processing_days(start_date, end_date, log_dir):
    """
    Iterate over processing days while handling log context automatically.
    """
    current_date = start_date
    current_month = None

    while current_date <= end_date:
        _set_log_day(f"{current_date:%Y-%m-%d}")

        month_key = (current_date.year, current_date.month)
        if month_key != current_month:
            _set_monthly_log_file(log_dir, current_date.year, current_date.month)
            current_month = month_key

        yield current_date

        current_date += timedelta(days=1)


def choose_lightning_sources(
    ec_lon: np.ndarray,
    ec_times: np.ndarray,
    half_window_minutes: int = 60,
    allowed_platforms=None,  # e.g. ('MTG-LI','GOES-16') ; None => allow all
):
    """
    Decide which lightning providers to query based on EC longitudes and time.

    Returns a list of dicts (may be empty). Each dict has:
      - source: 'mtg_li' | 'glm_east' | 'glm_west'
      - platform: 'MTG-I1' | 'GOES-16' | 'GOES-19' | 'GOES-18'
      - start_time: np.datetime64
      - end_time: np.datetime64
      - reason: brief string for logging
    """
    lightning_sources = []

    if ec_lon.size == 0 or ec_times.size == 0:
        logger.warning("Empty ec_lon/ec_times; no lightning providers selected.")
        return lightning_sources

    lon_min_ec = float(np.nanmin(ec_lon))
    lon_max_ec = float(np.nanmax(ec_lon))
    start_time, end_time = _ec_time_window(ec_times, half_window_minutes)

    valid = np.isfinite(ec_lon)

    li_ranges = [(-60.0, 60.0)]
    east_ranges = [(-130.0, -20.0)]
    west_ranges = [(-180.0, -80.0), (170.0, 180.0)]

    mask_li = valid & _in_lon_ranges(ec_lon, li_ranges)
    mask_east = valid & _in_lon_ranges(ec_lon, east_ranges)
    mask_west = valid & _in_lon_ranges(ec_lon, west_ranges)

    # MTG-LI: lon [-60, 60]
    if np.any(mask_li):
        _add_if_platform_allowed(
            lightning_sources,
            {
                'source': 'LI',
                'platform': 'MTG-I1',
                'start_time': start_time,
                'end_time': end_time,
                'reason': f"EC lon [{lon_min_ec:.1f},{lon_max_ec:.1f}] overlaps LI lon [-60,60]"
            },
            allowed_platforms
        )

    reference_time = ec_times[len(ec_times)//2]  # midpoint time for platform selection
    denom = float(valid.sum())
    frac_east = float(mask_east.sum()) / denom
    frac_west = float(mask_west.sum()) / denom

    if frac_east > 0.0 or frac_west > 0.0:
        if frac_east >= frac_west:
            # GLM-East: platform switches by date
            platform = _glm_east_platform(reference_time)  # 'GOES-16' or 'GOES-19'
            _add_if_platform_allowed(
                lightning_sources,
                {
                    'source': 'GLM',
                    'platform': platform,
                    'start_time': start_time,
                    'end_time': end_time,
                    'reason': (f"EC lon [{lon_min_ec:.1f},{lon_max_ec:.1f}] overlaps GLM-East lon [-130,-20]; "
                           f"selected East (coverage {frac_east:.2%}) with platform {platform}")
                },
                allowed_platforms
            )
        else:
            # GLM-West: fixed GOES-18
            _add_if_platform_allowed(
                lightning_sources,
                {
                    'source': 'GLM',
                    'platform': 'GOES-18',
                    'start_time': start_time,
                    'end_time': end_time,
                    'reason': (f"EC lon [{lon_min_ec:.1f},{lon_max_ec:.1f}] overlaps GLM-West lon [-180,-80]∪[170,180]; "
                            f"selected West (coverage {frac_west:.2%}) with platform GOES-18")
                },
                allowed_platforms
            )

    if lightning_sources:
        for s in lightning_sources:
            logger.info(f"Selected {s['source']} ({s['platform']}): {s['start_time']} -> {s['end_time']} | {s['reason']}")
    else:
        logger.info(f"No lightning coverage for EC lon [{lon_min_ec:.1f},{lon_max_ec:.1f}].")

    return lightning_sources