"""End-to-end latency measurement with an on-screen marker clock (SO2).

Method ("glass-to-receiver"):
  1. This computer shows an ArUco marker on its screen. The marker's id is a
     frame counter (mod 1000); the time each counter value was drawn is stored.
  2. The IoT camera films the screen and streams to this same computer.
  3. Each received frame is searched for the marker; its id gives the time the
     picture was shown, so  latency = receive time - display time.
Both times come from ONE clock (this computer's), so no clock
synchronisation between camera and receiver is needed. The value includes the
screen refresh (~8-17 ms at 60 Hz) and the camera exposure, so it slightly
over-estimates the camera-to-app latency - a conservative measurement.

For the Android app, run the same detection in the app (OpenCV for Android has
ArUco) or film the marker screen and the phone screen together with a second
camera (classic glass-to-glass method).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

MARKER_DICT = cv2.aruco.DICT_4X4_1000 if hasattr(cv2, "aruco") else None
ID_RANGE = 1000


class MarkerClock:
    def __init__(self, size_px: int = 420):
        if MARKER_DICT is None:
            raise RuntimeError("OpenCV was built without the aruco module")
        self.dictionary = cv2.aruco.getPredefinedDictionary(MARKER_DICT)
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, cv2.aruco.DetectorParameters())
        self.size_px = size_px
        self.counter = -1
        self.shown_at: Dict[int, float] = {}  # counter -> display time

    def next_image(self, t: float) -> np.ndarray:
        """Advance the counter and draw the marker for it (call right before showing it)."""
        self.counter += 1
        self.shown_at[self.counter] = t
        if len(self.shown_at) > 5 * ID_RANGE:
            for k in list(self.shown_at)[: ID_RANGE]:
                del self.shown_at[k]
        marker = cv2.aruco.generateImageMarker(self.dictionary, self.counter % ID_RANGE, self.size_px)
        canvas = np.full((self.size_px + 120, self.size_px + 120), 255, np.uint8)
        canvas[60:60 + self.size_px, 60:60 + self.size_px] = marker
        return canvas

    def display_time(self, marker_id: int) -> Optional[float]:
        """Display time of the most recent counter value with this id."""
        if self.counter < 0:
            return None
        k = self.counter - ((self.counter - marker_id) % ID_RANGE)
        return self.shown_at.get(k)

    def latency(self, image: np.ndarray, received_t: float) -> Optional[float]:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return None
        shown = self.display_time(int(ids.flatten()[0]))
        return None if shown is None else received_t - shown


def summarize(latencies_s: Sequence[float], frames_seen: int = 0) -> dict:
    """Median / p95 / max in ms, against the SO2 targets (median <= 500 ms, max <= 1000 ms)."""
    if not latencies_s:
        return {"samples": 0, "frames": frames_seen}
    ms = np.array(latencies_s) * 1000.0
    return {
        "samples": int(ms.size),
        "frames": frames_seen,
        "median_ms": round(float(np.median(ms)), 1),
        "p95_ms": round(float(np.percentile(ms, 95)), 1),
        "max_ms": round(float(ms.max()), 1),
        "min_ms": round(float(ms.min()), 1),
        "share_le_500ms": round(float(np.mean(ms <= 500)), 4),
        "meets_so2": bool(np.median(ms) <= 500 and ms.max() <= 1000),
        "histogram_ms": {f"{int(a)}-{int(b)}": int(c) for a, b, c in zip(*_hist(ms))},
    }


def _hist(ms: np.ndarray):
    edges = np.array([0, 100, 200, 300, 400, 500, 750, 1000, max(1001.0, float(ms.max()) + 1)])
    counts, _ = np.histogram(ms, bins=edges)
    return edges[:-1], edges[1:], counts


def latencies_from(images: List[np.ndarray], received: List[float], clock: MarkerClock) -> List[float]:
    out = []
    for img, t in zip(images, received):
        lat = clock.latency(img, t)
        if lat is not None:
            out.append(lat)
    return out
