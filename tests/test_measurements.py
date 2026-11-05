"""Measurement tools for the hardware objectives: latency (SO2), session performance (SO1/SO6), GPS accuracy (SO4)."""

import cv2
import numpy as np
import pytest

from dms.evaluation.gps_accuracy import errors_to_point, errors_to_track, haversine_m, summarize_errors, update_rate
from dms.evaluation.latency import MarkerClock, summarize
from dms.evaluation.performance import session_performance, so_checks


def _camera_view(img):
    """What a camera might deliver: smaller, blurred, JPEG-compressed, colour."""
    small = cv2.resize(img, (320, 320))
    small = cv2.GaussianBlur(small, (3, 3), 0)
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 60])
    return cv2.imdecode(jpg, cv2.IMREAD_COLOR)


def test_marker_clock_measures_known_delays():
    clock = MarkerClock()
    shown = []
    for k in range(1500):  # longer than the 1000-id range: ids wrap around
        shown.append(clock.next_image(k * 0.016))
    # the camera saw the image of counter 1490 and it arrived 0.2 s after it was drawn
    lat = clock.latency(_camera_view(shown[1490]), 1490 * 0.016 + 0.2)
    assert lat == pytest.approx(0.2, abs=1e-6)
    assert clock.latency(np.full((240, 320, 3), 255, np.uint8), 30.0) is None  # no marker -> no sample


def test_latency_summary_against_so2():
    s = summarize([0.2, 0.25, 0.3, 0.9])
    assert s["median_ms"] == pytest.approx(275.0) and s["max_ms"] == pytest.approx(900.0)
    assert s["meets_so2"] is True
    assert summarize([0.2, 1.2])["meets_so2"] is False


def test_session_performance_uptime_and_checks():
    frames = [{"t": i * 0.1, "features": {"face_present": True, "eye_width_px": 30.0},
               "timings_ms": {"total_ms": 50.0, "face_ms": 10.0}} for i in range(600)]
    frames += [{"t": 70.0 + i * 0.1, "features": {"face_present": True, "eye_width_px": 30.0},
                "timings_ms": {"total_ms": 50.0}} for i in range(300)]  # 10-s outage before this
    perf = session_performance(frames, {}, [{"event": "unavailable"}])
    assert perf["downtime_intervals"] == 1 and perf["downtime_s"] == pytest.approx(10.1, abs=0.01)
    assert perf["uptime"] == pytest.approx(1 - 10.1 / 99.9, abs=1e-3)
    assert perf["processing_fps"] == pytest.approx(20.0)
    status = {(c["so"], c["metric"].split(" (")[0]): c["status"] for c in so_checks(perf)}
    assert status[("SO3/SO4", "processing rate on THIS computer")] == "met"
    assert status[("SO6", "uptime")] == "not met"  # 89.9 % < 90 %


def test_gps_errors_point_and_track():
    assert haversine_m(14.0, 121.0, 14.0, 121.0) == pytest.approx(0.0)
    assert haversine_m(0.0, 0.0, 0.0001, 0.0) == pytest.approx(11.12, abs=0.05)  # 1e-4 deg latitude ~ 11.1 m
    fixes = [(float(i), 14.0 + 1e-5, 121.0) for i in range(20)]  # ~1.1 m north of the point
    err = errors_to_point(fixes, 14.0, 121.0)
    s = summarize_errors(err, target_m=5.0)
    assert s["median_m"] == pytest.approx(1.11, abs=0.02) and s["meets_target"]
    track = [(float(i), 14.0 + i * 1e-4, 121.0) for i in range(21)]  # moving north
    fixes = [(float(i) + 0.5, 14.0 + (i + 0.5) * 1e-4, 121.0) for i in range(20)]
    assert errors_to_track(fixes, track).max() < 0.01
    rate = update_rate([float(i) for i in range(10)] + [12.0])
    assert rate["share_within_1hz_10pct"] == pytest.approx(0.9) and rate["longest_gap_s"] == pytest.approx(3.0)
