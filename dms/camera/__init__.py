"""Section 8 - camera and video input.

    sources.py  frames from a webcam, video file, image folder or the ESP32-CAM
    mjpeg.py    direct MJPEG reader for the ESP32 stream (camera timestamps, reconnect)
    stream.py   newest-frame reader (bounded latency) and stream health (FPS, uptime)
"""

from .mjpeg import MjpegFrame, iter_mjpeg
from .sources import Frame, open_source
from .stream import LatestFrameReader, StreamHealth

__all__ = ["Frame", "LatestFrameReader", "MjpegFrame", "StreamHealth", "iter_mjpeg", "open_source"]
