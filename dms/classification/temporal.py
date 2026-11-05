"""Stage 3: temporal persistence helpers (all in seconds, frame-rate independent)."""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Set, Tuple


class EpisodeTracker:
    """Tracks how long a boolean per-frame state has been continuously ON.

    Short OFF gaps up to ``gap_s`` (e.g. one missed YOLO detection) do not
    end an episode. Durations are measured in seconds between the first and
    the most recent ON frame, so the rule is the same at 8, 10 or 30 FPS.
    """

    def __init__(self, gap_s: float):
        self.gap_s = gap_s
        self.active = False
        self.start = 0.0
        self.last_on = 0.0
        self._fired: Set[str] = set()
        self._ended: Deque[Tuple[float, float]] = deque(maxlen=64)  # (start, end)

    def update(self, t: float, on: Optional[bool]) -> Optional[Tuple[float, float]]:
        """Feed one frame (None = state unknown, treated like OFF). Returns the episode that just ended, if any."""
        if on:
            if not self.active:
                self.active, self.start = True, t
                self._fired.clear()
            self.last_on = t
        elif self.active and t - self.last_on > self.gap_s:
            return self.force_end()
        return None

    def force_end(self) -> Optional[Tuple[float, float]]:
        """End the ongoing episode at its last ON frame (e.g. after a camera stall)."""
        if not self.active:
            return None
        ended = (self.start, self.last_on)
        self._ended.append(ended)
        self.active = False
        return ended

    def duration(self) -> float:
        return self.last_on - self.start if self.active else 0.0

    def recent_max(self, t: float, hold_s: float) -> float:
        """Longest episode that is ongoing or ended within the last ``hold_s`` seconds."""
        best = self.duration()
        for s, e in self._ended:
            if t - e <= hold_s:
                best = max(best, e - s)
        return best

    def fire_once(self, key: str, threshold_s: float) -> bool:
        """True exactly once per episode, when its duration first reaches ``threshold_s``."""
        if self.active and key not in self._fired and self.duration() >= threshold_s:
            self._fired.add(key)
            return True
        return False


class TimeWeightedRatio:
    """Fraction of time a state was ON over a sliding window (e.g. PERCLOS)."""

    def __init__(self, window_s: float, max_dt: float = 0.5):
        self.window_s = window_s
        self.max_dt = max_dt
        self._samples: Deque[Tuple[float, float, bool]] = deque()  # (t, dt, on)
        self._last_t: Optional[float] = None
        self._on = 0.0
        self._total = 0.0

    def add(self, t: float, on: bool) -> None:
        dt = 0.0 if self._last_t is None else min(self.max_dt, max(0.0, t - self._last_t))
        self._last_t = t
        self._samples.append((t, dt, on))
        self._total += dt
        self._on += dt if on else 0.0
        while self._samples and t - self._samples[0][0] > self.window_s:
            _, old_dt, old_on = self._samples.popleft()
            self._total -= old_dt
            self._on -= old_dt if old_on else 0.0

    def skip(self, t: float) -> None:
        """Advance time without a sample (state not observable on this frame)."""
        self._last_t = t

    def reset(self) -> None:
        """Forget everything (used when the drowsiness state recovers)."""
        self._samples.clear()
        self._last_t = None
        self._on = 0.0
        self._total = 0.0

    @property
    def coverage_s(self) -> float:
        return max(0.0, self._total)

    @property
    def ratio(self) -> float:
        return self._on / self._total if self._total > 1e-9 else 0.0


class EventWindow:
    """Counts events within a sliding time window (e.g. yawns in 5 minutes)."""

    def __init__(self, window_s: float):
        self.window_s = window_s
        self._times: Deque[float] = deque()

    def add(self, t: float) -> None:
        self._times.append(t)

    def count(self, t: float) -> int:
        while self._times and t - self._times[0] > self.window_s:
            self._times.popleft()
        return len(self._times)

    def count_since(self, t: float, since: float) -> int:
        """Events inside the window that happened after ``since``."""
        self.count(t)
        return sum(1 for x in self._times if x > since)

    def last(self) -> Optional[float]:
        return self._times[-1] if self._times else None
