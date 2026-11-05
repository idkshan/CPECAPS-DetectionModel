"""Live-stream helpers: newest-frame reader and stream health (SO1, SO2, SO6).

LatestFrameReader
  Reads the source in a background thread and keeps only the NEWEST frame.
  If processing a frame takes longer than the camera's frame interval, a plain
  read loop processes older and older frames (latency keeps growing); dropping
  stale frames keeps the end-to-end latency bounded. ``dropped`` counts them.

StreamHealth
  Arrival statistics of the frames that reached the computer: frame rate,
  gaps, and the downtime intervals in which no frame arrived for longer than
  the camera timeout. Uptime = (total - downtime) / total, the SO6 measure
  "working time / total test time".
"""

from __future__ import annotations

import threading
import time
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .sources import Frame


class LatestFrameReader:
    def __init__(self, frames: Iterator[Frame]):
        self._frames = frames
        self._latest: Optional[Frame] = None
        self._cond = threading.Condition()
        self._done = False
        self._seq = 0
        self._returned_seq = 0
        self.received = 0
        self.dropped = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            for f in self._frames:
                with self._cond:
                    if self._latest is not None and self._seq > self._returned_seq:
                        self.dropped += 1  # the previous frame was never processed
                    self._latest = f
                    self._seq += 1
                    self.received += 1
                    self._cond.notify_all()
        finally:
            with self._cond:
                self._done = True
                self._cond.notify_all()

    @property
    def finished(self) -> bool:
        return self._done and self._seq == self._returned_seq

    def read(self, timeout_s: float) -> Optional[Frame]:
        """The newest frame not returned yet, or None after ``timeout_s`` without one."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while self._seq == self._returned_seq and not self._done:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            if self._seq == self._returned_seq:
                return None
            self._returned_seq = self._seq
            return self._latest


class StreamHealth:
    def __init__(self, timeout_s: float = 3.0):
        self.timeout_s = timeout_s
        self.arrivals: List[float] = []
        self.downtime: List[Tuple[float, float]] = []  # (start, end): no frame for > timeout_s
        self._down_since: Optional[float] = None
        self.start_t: Optional[float] = None
        self.end_t: Optional[float] = None

    def frame(self, t: float) -> None:
        if self.start_t is None:
            self.start_t = t
        if self._down_since is not None:
            self.downtime.append((self._down_since, t))
            self._down_since = None
        elif self.arrivals and t - self.arrivals[-1] > self.timeout_s:
            self.downtime.append((self.arrivals[-1], t))
        self.arrivals.append(t)
        self.end_t = t

    def no_frame(self, t: float) -> None:
        """Called by the watchdog when the timeout passed without a frame."""
        if self._down_since is None:
            self._down_since = self.arrivals[-1] if self.arrivals else (self.start_t if self.start_t is not None else t)
        self.end_t = t

    def stats(self) -> dict:
        if self.start_t is None or self.end_t is None:
            return {"frames": 0}
        down = list(self.downtime) + ([(self._down_since, self.end_t)] if self._down_since is not None else [])
        total = max(1e-9, self.end_t - self.start_t)
        down_s = sum(b - a for a, b in down)
        out = {"frames": len(self.arrivals), "duration_s": round(total, 2),
               "downtime_s": round(down_s, 2), "downtime_intervals": len(down),
               "uptime": round(1.0 - down_s / total, 4)}
        if len(self.arrivals) > 1:
            dt = np.diff(self.arrivals)
            out.update(mean_fps=round(1.0 / float(np.mean(dt)), 2), median_interval_ms=round(1000 * float(np.median(dt)), 1),
                       p95_interval_ms=round(1000 * float(np.percentile(dt, 95)), 1),
                       share_slower_than_10fps=round(float(np.mean(dt > 0.11)), 3))  # interval > 110 ms
        return out
