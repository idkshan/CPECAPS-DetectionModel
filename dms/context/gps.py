"""GPS fixes: validity, staleness, update rate, outages (SO4).

Purpose   Decide whether the latest GPS reading may be used, and keep the
          numbers needed to verify SO4 (1 Hz updates, recovery <= 30 s).
Inputs    GpsFix records (from the ESP32 NEO-6M JSON or the phone's location API)
Outputs   latest_valid(t) -> GpsFix or None; stats(): update intervals, share
          of ~1 Hz updates, outages and time to re-acquire a fix
Params    context.gps_stale_s (3 s): a fix older than three missed 1-Hz updates
          is not used - the risk model then falls back to the ASSUMED speed and
          says so (ContextReading.speed_source = "assumed").

Position ACCURACY (<= 5 m open sky, <= 15 m urban) cannot be judged from the
fixes alone; it needs a reference: scripts/evaluate_gps.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class GpsFix:
    t: float  # receive time on the monitoring clock (s)
    lat: Optional[float] = None
    lon: Optional[float] = None
    speed_kmh: Optional[float] = None
    fix: bool = True  # receiver reports a valid position fix
    hdop: Optional[float] = None
    satellites: Optional[int] = None
    accuracy_m: Optional[float] = None  # Android Location.getAccuracy() if the phone is the source

    @property
    def usable(self) -> bool:
        return self.fix and self.lat is not None and self.lon is not None

    def to_dict(self) -> dict:
        return asdict(self)


class GpsTracker:
    def __init__(self, stale_s: float = 3.0):
        self.stale_s = stale_s
        self.fixes: List[GpsFix] = []
        self._last_valid: Optional[GpsFix] = None

    def add(self, fix: GpsFix) -> None:
        self.fixes.append(fix)
        if fix.usable:
            self._last_valid = fix

    def latest_valid(self, t: float) -> Optional[GpsFix]:
        f = self._last_valid
        return f if f is not None and t - f.t <= self.stale_s else None

    def outages(self) -> List[Tuple[float, float]]:
        """Intervals without a usable fix for longer than stale_s: (start, end)."""
        times = [f.t for f in self.fixes if f.usable]
        out = []
        for a, b in zip(times, times[1:]):
            if b - a > self.stale_s:
                out.append((a, b))
        return out

    def stats(self) -> dict:
        usable = [f.t for f in self.fixes if f.usable]
        if len(usable) < 2:
            return {"fixes": len(self.fixes), "usable_fixes": len(usable)}
        dt = np.diff(usable)
        outages = self.outages()
        return {
            "fixes": len(self.fixes),
            "usable_fixes": len(usable),
            "median_interval_s": round(float(np.median(dt)), 3),
            "p95_interval_s": round(float(np.percentile(dt, 95)), 3),
            "share_within_1hz_10pct": round(float(np.mean((dt >= 0.9) & (dt <= 1.1))), 3),
            "outages": len(outages),
            "longest_outage_s": round(max((b - a for a, b in outages), default=0.0), 1),
            "reacquisition_s": [round(b - a, 1) for a, b in outages],
            "target_reacquisition_s": 30.0,
        }
