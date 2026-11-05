"""Frame sources: webcam, video file, image folder, or the ESP32-CAM.

Timestamps (seconds) drive every temporal rule. Recorded videos use *video
time*, so a file can be processed faster or slower than real time without
changing the results; live sources use the wall clock.

ESP32-CAM (CameraWebServer firmware):
  * MJPEG stream:   http://<esp32-ip>:81/stream
  * single capture: http://<esp32-ip>/capture   (polled repeatedly)
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Frame:
    image: np.ndarray
    t: float  # seconds: video time for files, receive time (monotonic, from the first frame) for live sources
    index: int
    device_t: Optional[float] = None  # camera's own timestamp (ESP32 X-Timestamp), if known


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_live(spec: str) -> bool:
    """Webcam or network stream (frames must be processed as they arrive)."""
    return _is_int(spec) or spec.startswith(("http://", "https://", "rtsp://"))


def open_source(spec: str, fps: float = 10.0, max_frames: Optional[int] = None, mjpeg_reader: bool = True,
                on_event=None) -> Iterator[Frame]:
    """Yield frames from ``spec`` (camera index, path, folder or URL).

    HTTP streams use the direct MJPEG reader (camera timestamps, reconnects)
    unless ``mjpeg_reader`` is False (then OpenCV's FFmpeg backend is used).
    """
    gen: Iterator[Frame]
    if _is_int(spec):
        gen = _capture(cv2.VideoCapture(int(spec)), live=True)
    elif spec.startswith(("http://", "https://")) and (spec.rstrip("/").endswith("/capture") or spec.lower().endswith((".jpg", ".jpeg"))):
        gen = _poll_jpeg(spec, fps)
    elif spec.startswith(("http://", "https://")) and mjpeg_reader:
        gen = _mjpeg(spec, on_event)
    elif spec.startswith(("http://", "https://", "rtsp://")):
        gen = _capture(cv2.VideoCapture(spec), live=True)
    elif Path(spec).is_dir():
        gen = _folder(Path(spec), fps)
    elif Path(spec).is_file():
        gen = _capture(cv2.VideoCapture(spec), live=False)
    else:
        raise FileNotFoundError(f"Unknown source: {spec}")
    for i, frame in enumerate(gen):
        if max_frames is not None and i >= max_frames:
            break
        yield frame


def _capture(cap: cv2.VideoCapture, live: bool) -> Iterator[Frame]:
    if not cap.isOpened():
        raise RuntimeError("Could not open video source")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    t0 = time.monotonic()
    i = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if live:
                t = time.monotonic() - t0
            else:
                t = i / fps if fps > 0 else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield Frame(img, t, i)
            i += 1
    finally:
        cap.release()


def _mjpeg(url: str, on_event=None) -> Iterator[Frame]:
    from .mjpeg import iter_mjpeg

    t0 = None
    for i, f in enumerate(iter_mjpeg(url, on_event=on_event)):
        t0 = f.received_t if t0 is None else t0
        yield Frame(f.image, f.received_t - t0, i, f.device_t)


def _folder(folder: Path, fps: float) -> Iterator[Frame]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for i, p in enumerate(files):
        img = cv2.imread(str(p))
        if img is not None:
            yield Frame(img, i / fps, i)


def _poll_jpeg(url: str, fps: float) -> Iterator[Frame]:
    period = 1.0 / fps
    t0 = time.monotonic()
    i = 0
    while True:
        start = time.monotonic()
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                data = np.frombuffer(resp.read(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception as exc:
            print(f"[source] capture failed: {exc}")
            img = None
        if img is not None:
            yield Frame(img, time.monotonic() - t0, i)
            i += 1
        time.sleep(max(0.0, period - (time.monotonic() - start)))
