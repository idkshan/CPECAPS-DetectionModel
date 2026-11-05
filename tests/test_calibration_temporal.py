import pytest

from dms.classification.calibration import CALIBRATED, CALIBRATING, DriverCalibrator
from dms.classification.temporal import EpisodeTracker, EventWindow, TimeWeightedRatio
from dms.config import Config
from dms.types import FrameFeatures


def _feat(t, ear=0.32, mar=0.03, yaw=12.0, pitch=-6.0, blink=0.08, jaw=0.02, phone=0.0):
    return FrameFeatures(t=t, face_present=True, ear=ear, mar=mar, yaw=yaw, pitch=pitch, blink=blink, jaw_open=jaw, phone_conf=phone)


def test_calibration_learns_personal_baseline():
    cfg = Config()
    cal = DriverCalibrator(cfg.calibration, cfg.thresholds)
    assert cal.status == CALIBRATING and not cal.thresholds.calibrated
    for i in range(60):
        cal.collect(_feat(i * 0.1, ear=0.22))  # a driver with naturally narrow eyes
    assert cal.status == CALIBRATED
    thr = cal.thresholds
    assert thr.baseline.ear == pytest.approx(0.22)
    assert thr.ear_closed == pytest.approx(0.22 * cfg.thresholds.ear_closed_ratio)
    assert thr.baseline.yaw == pytest.approx(12.0)  # off-axis windshield camera absorbed


def test_calibration_rejects_unsteady_head_and_keeps_defaults():
    cfg = Config()
    cal = DriverCalibrator(cfg.calibration, cfg.thresholds)
    for i in range(60):
        cal.collect(_feat(i * 0.1, yaw=(-20 if i % 2 else 20)))
    assert cal.status == CALIBRATING
    assert "head not steady" in cal.last_failure
    assert cal.thresholds.yaw_limit == pytest.approx(cfg.thresholds.yaw_deg * cfg.thresholds.uncalibrated_pose_scale)


def test_calibration_ignores_frames_with_phone():
    cfg = Config()
    cal = DriverCalibrator(cfg.calibration, cfg.thresholds)
    for i in range(60):
        cal.collect(_feat(i * 0.1, phone=0.8))
    assert cal.status == CALIBRATING and "usable frames" in cal.last_failure


def test_adaptation_is_slow_and_clamped():
    cfg = Config()
    cal = DriverCalibrator(cfg.calibration, cfg.thresholds)
    for i in range(60):
        cal.collect(_feat(i * 0.1, ear=0.30))
    t = 6.0
    for _ in range(20000):  # a long time with much larger eyes
        t += 0.1
        cal.adapt(_feat(t, ear=0.60))
    assert cal.baseline.ear == pytest.approx(0.30 * (1 + cfg.calibration.max_ear_drift))


def test_episode_tracker_gap_tolerance_and_fire_once():
    tr = EpisodeTracker(gap_s=0.3)
    fired = []
    for i in range(40):
        t = i * 0.1
        on = not (i in (10, 11))  # a 0.2 s dropout (e.g. a missed detection)
        tr.update(t, on)
        fired.append(tr.fire_once("x", 3.0))
    assert tr.active and tr.duration() == pytest.approx(3.9)
    assert sum(fired) == 1


def test_episode_tracker_recent_max_and_reset():
    tr = EpisodeTracker(gap_s=0.1)
    for i in range(15):
        tr.update(i * 0.1, True)
    for i in range(15, 25):
        tr.update(i * 0.1, False)
    assert not tr.active
    assert tr.recent_max(2.4, hold_s=1.5) == pytest.approx(1.4)
    assert tr.recent_max(10.0, hold_s=1.5) == 0.0


def test_time_weighted_ratio_is_frame_rate_independent():
    for fps in (5, 10, 30):
        r = TimeWeightedRatio(window_s=60)
        for i in range(int(20 * fps)):
            t = i / fps
            r.add(t, (t % 10) < 1.5)  # eyes closed 15 % of the time
        assert r.ratio == pytest.approx(0.15, abs=0.02)


def test_event_window():
    w = EventWindow(window_s=300)
    for t in (0, 100, 250):
        w.add(t)
    assert w.count(260) == 3
    assert w.count(360) == 2
