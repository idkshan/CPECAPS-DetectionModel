"""Driving-context providers: speed (GPS), motion (MPU6050), driving duration.

Every reading says where its speed came from (``ContextReading.speed_source``):
    gps       a fresh GPS fix (<= context.gps_stale_s old)
    fixed     a value given on the command line (bench tests, recorded videos)
    assumed   no usable speed: the risk model uses context.assumed_speed_kmh
              (moving, so a missing sensor can never suppress an alert) and the
              log shows that the speed was assumed - it is never silent.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from typing import Optional

from ..config import ContextConfig
from ..types import ContextReading
from .duration import DrivingDurationTracker
from .gps import GpsFix, GpsTracker


class ContextProvider:
    """No sensors: speed assumed, duration counted from the first frame."""

    def __init__(self, cfg: Optional[ContextConfig] = None) -> None:
        self.cfg = cfg or ContextConfig()
        self.duration = DrivingDurationTracker(self.cfg.moving_speed_kmh, self.cfg.break_reset_s)

    def read(self, t: float) -> ContextReading:
        return ContextReading(driving_s=self.duration.update(t, None), speed_source="assumed")


class StaticContext(ContextProvider):
    """Fixed speed / motion values (bench testing or recorded videos)."""

    def __init__(self, speed_kmh: Optional[float] = None, motion_g: Optional[float] = None, start_offset_s: float = 0.0,
                 cfg: Optional[ContextConfig] = None):
        super().__init__(cfg)
        self.speed_kmh = speed_kmh
        self.motion_g = motion_g
        self.start_offset_s = start_offset_s  # e.g. simulate a session that started 2 h ago

    def read(self, t: float) -> ContextReading:
        driving = self.duration.update(t, self.speed_kmh) + self.start_offset_s
        return ContextReading(self.speed_kmh, self.motion_g, driving,
                              speed_source="fixed" if self.speed_kmh is not None else "assumed")


class HttpContext(ContextProvider):
    """Polls a JSON endpoint on the ESP32 (or the phone), e.g.

        {"speed_kmh": 32.5, "motion_g": 0.08, "lat": 14.5649, "lon": 120.9932, "fix": true, "hdop": 0.9, "sats": 8}

    ``motion_g`` is the peak horizontal acceleration (in g) over the last
    second from the MPU6050. GPS readings older than context.gps_stale_s are
    not used; every fix is kept for the SO4 report (``gps.fixes``).
    """

    def __init__(self, url: str, poll_s: float = 1.0, cfg: Optional[ContextConfig] = None, clock=time.monotonic):
        super().__init__(cfg)
        self.url, self.poll_s, self.clock = url, poll_s, clock
        self.gps = GpsTracker(self.cfg.gps_stale_s)
        self._t0 = clock()
        self._motion: Optional[float] = None
        self._motion_t = -1e9
        self._stop = threading.Event()
        threading.Thread(target=self._poll, daemon=True).start()

    def ingest(self, data: dict, t: Optional[float] = None) -> None:
        """Store one JSON reading (also used by tests)."""
        t = self.clock() - self._t0 if t is None else t
        if "motion_g" in data and data["motion_g"] is not None:
            self._motion, self._motion_t = float(data["motion_g"]), t
        if any(k in data for k in ("speed_kmh", "lat", "lon")):
            self.gps.add(GpsFix(
                t=t, lat=data.get("lat"), lon=data.get("lon"),
                speed_kmh=None if data.get("speed_kmh") is None else float(data["speed_kmh"]),
                fix=bool(data.get("fix", data.get("lat") is not None or data.get("speed_kmh") is not None)),
                hdop=data.get("hdop"), satellites=data.get("sats", data.get("satellites")), accuracy_m=data.get("accuracy_m")))

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(self.url, timeout=1.0) as resp:
                    self.ingest(json.loads(resp.read().decode("utf-8")))
            except Exception:
                pass  # a missed poll simply makes the data older; staleness is checked on read
            self._stop.wait(self.poll_s)

    def read(self, t: float) -> ContextReading:
        now = self.clock() - self._t0
        fix = self.gps.latest_valid(now)
        speed = fix.speed_kmh if fix is not None else None
        motion = self._motion if now - self._motion_t <= self.cfg.gps_stale_s else None
        return ContextReading(
            speed_kmh=speed, motion_g=motion, driving_s=self.duration.update(t, speed),
            speed_source="gps" if speed is not None else "assumed",
            gps_age_s=round(now - fix.t, 2) if fix is not None else None,
            latitude=fix.lat if fix is not None else None, longitude=fix.lon if fix is not None else None)

    def close(self) -> None:
        self._stop.set()
