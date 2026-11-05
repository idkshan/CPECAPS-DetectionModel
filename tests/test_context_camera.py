"""Context (GPS staleness, driving duration) and camera helpers (frame reader, MJPEG, stream health)."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from dms.camera import LatestFrameReader, StreamHealth, iter_mjpeg
from dms.camera.sources import Frame
from dms.config import ContextConfig
from dms.context import DrivingDurationTracker, GpsFix, GpsTracker, HttpContext, StaticContext


def test_driving_duration_pauses_when_stopped_and_resets_after_a_break():
    d = DrivingDurationTracker(moving_speed_kmh=5.0, break_reset_s=900.0)
    t = 0.0
    for _ in range(600):  # 10 min at 40 km/h
        t += 1.0
        d.update(t, 40.0)
    assert d.continuous_s == pytest.approx(599.0)
    for _ in range(300):  # 5 min traffic-light / short stop: paused, not reset
        t += 1.0
        d.update(t, 0.0)
    assert d.continuous_s == pytest.approx(599.0)
    for _ in range(1000):  # > 15 min stop: a rest break
        t += 1.0
        d.update(t, 0.0)
    assert d.continuous_s == 0.0
    t += 1.0
    d.update(t, 30.0)
    assert len(d.breaks) == 1 and d.total_driving_s == pytest.approx(600.0)


def test_unknown_speed_counts_as_driving():
    d = DrivingDurationTracker()
    for i in range(11):
        d.update(float(i), None)
    assert d.continuous_s == pytest.approx(10.0)


def test_gps_staleness_and_update_rate():
    g = GpsTracker(stale_s=3.0)
    for i in range(10):
        g.add(GpsFix(t=float(i), lat=14.5, lon=121.0, speed_kmh=30.0))
    assert g.latest_valid(9.5) is not None and g.latest_valid(13.0) is None
    g.add(GpsFix(t=40.0, lat=14.5, lon=121.0, speed_kmh=30.0))  # 31 s outage
    st = g.stats()
    assert st["outages"] == 1 and st["longest_outage_s"] == pytest.approx(31.0)
    assert st["share_within_1hz_10pct"] == pytest.approx(0.9)


def test_http_context_flags_stale_gps_as_assumed():
    clock = {"t": 100.0}
    ctx = HttpContext("http://127.0.0.1:9/never", poll_s=3600, cfg=ContextConfig(), clock=lambda: clock["t"])
    ctx.close()
    ctx.ingest({"speed_kmh": 52.0, "lat": 14.56, "lon": 120.99, "motion_g": 0.05}, t=0.0)
    r = ctx.read(0.0)
    assert r.speed_source == "gps" and r.speed_kmh == 52.0
    clock["t"] += 10.0  # ten seconds without a fix
    r = ctx.read(10.0)
    assert r.speed_source == "assumed" and r.speed_kmh is None


def test_static_context_reports_its_source():
    assert StaticContext(40.0).read(0.0).speed_source == "fixed"
    assert StaticContext().read(0.0).speed_source == "assumed"


def test_latest_frame_reader_drops_stale_frames():
    def slow_camera():
        for i in range(20):
            yield Frame(np.zeros((2, 2, 3), np.uint8), i * 0.01, i)
            time.sleep(0.01)

    reader = LatestFrameReader(slow_camera())
    got = []
    while True:
        f = reader.read(timeout_s=1.0)
        if f is None:
            break
        got.append(f.index)
        time.sleep(0.05)  # processing is slower than the camera
    assert got[-1] == 19 and len(got) < 20 and reader.dropped > 0
    assert got == sorted(got)


def test_stream_health_uptime():
    h = StreamHealth(timeout_s=3.0)
    for i in range(100):
        h.frame(i * 0.1)
    h.no_frame(15.0)  # the link dies at ~9.9 s
    for i in range(50):
        h.frame(20.0 + i * 0.1)
    s = h.stats()
    assert s["downtime_intervals"] == 1 and s["downtime_s"] == pytest.approx(20.0 - 9.9, abs=0.01)
    assert s["uptime"] == pytest.approx(1 - 10.1 / 24.9, abs=0.01)


class _Mjpeg(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace;boundary=frame")
        self.end_headers()
        ok, jpg = cv2.imencode(".jpg", np.full((24, 32, 3), 128, np.uint8))
        for i in range(5):
            part = (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\nX-Timestamp: %d.%06d\r\n\r\n"
                    % (len(jpg), 100 + i, 500000))
            self.wfile.write(part + jpg.tobytes() + b"\r\n")
            self.wfile.flush()

    def log_message(self, *args):
        pass


def test_mjpeg_reader_parses_frames_and_camera_timestamps():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Mjpeg)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/stream"
        frames = list(iter_mjpeg(url, timeout_s=2.0, reconnect=False))
    finally:
        server.shutdown()
    assert len(frames) == 5 and frames[0].image.shape == (24, 32, 3)
    assert [f.device_t for f in frames] == pytest.approx([100.5, 101.5, 102.5, 103.5, 104.5])
