"""MJPEG-over-HTTP reader for the ESP32-CAM stream (http://<ip>:81/stream).

Why not cv2.VideoCapture? OpenCV's FFmpeg backend buffers the stream
internally, hides the per-frame headers and can block for a long time when
the Wi-Fi drops. Reading the multipart stream directly gives
  * the ESP32's own capture timestamp (``X-Timestamp`` header of the
    CameraWebServer example) next to the receive time -> SO1 capture rate and
    SO2 jitter can be measured per frame,
  * no hidden buffer (the newest frame is always the next one we read),
  * a clean timeout and automatic reconnection (uptime, SO6).

Each frame is a JPEG between multipart boundaries:
    --123456789000000000000987654321
    Content-Type: image/jpeg
    Content-Length: 12345
    X-Timestamp: 1712.123456
    <JPEG bytes>
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass
class MjpegFrame:
    image: np.ndarray
    received_t: float  # monotonic clock of this computer, seconds
    device_t: Optional[float]  # X-Timestamp from the camera (camera clock), seconds
    size_bytes: int


def _read_headers(stream) -> dict:
    headers = {}
    while True:
        line = stream.readline()
        if not line:
            raise ConnectionError("stream closed")
        line = line.strip()
        if not line:
            if headers:
                return headers
            continue  # blank lines between parts
        if line.startswith(b"--"):
            continue  # boundary
        if b":" in line:
            k, v = line.split(b":", 1)
            headers[k.strip().lower().decode("latin-1")] = v.strip().decode("latin-1")


def iter_mjpeg(url: str, timeout_s: float = 3.0, reconnect: bool = True, max_reconnects: Optional[int] = None,
               clock=time.monotonic, on_event=None) -> Iterator[MjpegFrame]:
    """Yield frames from an MJPEG stream; reconnect after errors (``on_event`` gets status strings)."""
    attempts = 0
    while True:
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                attempts = 0
                if on_event:
                    on_event("connected")
                while True:
                    headers = _read_headers(resp)
                    length = int(headers.get("content-length", "0"))
                    if length <= 0:
                        continue
                    data = resp.read(length)
                    received = clock()
                    if len(data) < length:
                        raise ConnectionError("truncated frame")
                    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        if on_event:
                            on_event("corrupt_frame")
                        continue
                    dev = headers.get("x-timestamp")
                    yield MjpegFrame(img, received, float(dev) if dev else None, length)
        except Exception as exc:  # network errors: report and retry
            if on_event:
                on_event(f"disconnected: {exc}")
            if not reconnect:
                return
            attempts += 1
            if max_reconnects is not None and attempts > max_reconnects:
                return
            time.sleep(min(2.0, 0.25 * attempts))
