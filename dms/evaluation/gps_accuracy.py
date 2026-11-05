"""GPS position accuracy against a reference (SO4: <= 5 m open sky, <= 15 m urban).

Accuracy needs ground truth. Two practical references:
  * static test  - park on a surveyed point (or a point read from a high-resolution
                   map / survey app) and log fixes for >= 10 minutes; error of each
                   fix = distance to that point
  * track test   - drive with a better receiver (e.g. a phone with dual-frequency GNSS
                   or an RTK unit) logging at the same time; fixes are compared with the
                   reference position interpolated at the same time
Horizontal error = great-circle distance (haversine). The thesis target is the
MEDIAN error; p95 and max are reported as well so outliers are visible.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

EARTH_RADIUS_M = 6371008.8


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def errors_to_point(fixes: Sequence[Tuple[float, float, float]], ref_lat: float, ref_lon: float) -> np.ndarray:
    """fixes: (t, lat, lon)."""
    arr = np.array(fixes, dtype=float)
    return haversine_m(arr[:, 1], arr[:, 2], ref_lat, ref_lon)


def errors_to_track(fixes: Sequence[Tuple[float, float, float]], track: Sequence[Tuple[float, float, float]],
                    time_offset_s: float = 0.0, max_gap_s: float = 2.0) -> np.ndarray:
    """Compare with a reference track (t, lat, lon), linearly interpolated at each fix time.

    ``time_offset_s`` is added to the fix times to put them on the reference clock.
    Fixes outside the track or in a reference gap longer than ``max_gap_s`` are skipped.
    """
    f = np.array(fixes, dtype=float)
    r = np.array(sorted(track), dtype=float)
    t = f[:, 0] + time_offset_s
    ok = (t >= r[0, 0]) & (t <= r[-1, 0])
    idx = np.clip(np.searchsorted(r[:, 0], t), 1, len(r) - 1)
    gap = r[idx, 0] - r[idx - 1, 0]
    ok &= gap <= max_gap_s
    lat = np.interp(t, r[:, 0], r[:, 1])
    lon = np.interp(t, r[:, 0], r[:, 2])
    return haversine_m(f[ok, 1], f[ok, 2], lat[ok], lon[ok])


def summarize_errors(err_m: np.ndarray, target_m: Optional[float] = None) -> dict:
    if err_m.size == 0:
        return {"fixes": 0}
    out = {
        "fixes": int(err_m.size),
        "median_m": round(float(np.median(err_m)), 2),
        "cep95_m": round(float(np.percentile(err_m, 95)), 2),
        "max_m": round(float(err_m.max()), 2),
        "mean_m": round(float(err_m.mean()), 2),
    }
    if target_m is not None:
        out.update(target_median_m=target_m, meets_target=bool(np.median(err_m) <= target_m),
                   share_within_target=round(float(np.mean(err_m <= target_m)), 4))
    return out


def update_rate(times: Sequence[float]) -> dict:
    t = np.sort(np.asarray(times, dtype=float))
    if t.size < 2:
        return {"fixes": int(t.size)}
    dt = np.diff(t)
    return {"median_interval_s": round(float(np.median(dt)), 3), "p95_interval_s": round(float(np.percentile(dt, 95)), 3),
            "share_within_1hz_10pct": round(float(np.mean((dt >= 0.9) & (dt <= 1.1))), 4),
            "longest_gap_s": round(float(dt.max()), 2)}
