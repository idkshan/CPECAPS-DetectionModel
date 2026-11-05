"""Continuous driving duration (SO4 "driving-duration logging"; fatigue input).

Purpose   How long has the driver been driving without a real break?
Inputs    time t (s), vehicle speed (km/h) or None when unknown
Outputs   continuous driving time (s), total driving time, list of breaks
Params    context.moving_speed_kmh (5)  below this the vehicle counts as stopped
          context.break_reset_s (900)  a stop this long (15 min) is a rest break:
                                        the continuous-driving clock restarts

Unknown speed (no GPS) counts as driving: missing data must not make the
driver look rested. A pause in updates longer than ``max_step_s`` (the app was
closed) is not counted as driving either.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


class DrivingDurationTracker:
    def __init__(self, moving_speed_kmh: float = 5.0, break_reset_s: float = 900.0, max_step_s: float = 5.0):
        self.moving_speed_kmh = moving_speed_kmh
        self.break_reset_s = break_reset_s
        self.max_step_s = max_step_s
        self.continuous_s = 0.0
        self.total_driving_s = 0.0
        self.stopped_s = 0.0
        self.breaks: List[Tuple[float, float]] = []  # (start, end) of stops long enough to reset
        self._last_t: Optional[float] = None
        self._stop_start: Optional[float] = None

    def update(self, t: float, speed_kmh: Optional[float]) -> float:
        if self._last_t is None:
            self._last_t = t
            return self.continuous_s
        dt = min(max(0.0, t - self._last_t), self.max_step_s)
        self._last_t = t
        moving = speed_kmh is None or speed_kmh >= self.moving_speed_kmh
        if moving:
            if self._stop_start is not None and self.stopped_s >= self.break_reset_s:
                self.breaks.append((self._stop_start, t))
            self._stop_start, self.stopped_s = None, 0.0
            self.continuous_s += dt
            self.total_driving_s += dt
        else:
            if self._stop_start is None:
                self._stop_start = t - dt
            self.stopped_s += dt
            if self.stopped_s >= self.break_reset_s:
                self.continuous_s = 0.0  # a real rest break
        return self.continuous_s

    def summary(self) -> dict:
        return {"total_driving_min": round(self.total_driving_s / 60.0, 2),
                "continuous_driving_min": round(self.continuous_s / 60.0, 2),
                "rest_breaks": [(round(a, 1), round(b, 1)) for a, b in self.breaks]}
