"""Behaviour modules (dms/behaviors): per-frame rules, guards, events and the drowsiness state."""

import pytest

from dms.classification.calibration import DriverCalibrator
from dms.classification.classifier import BehaviorClassifier
from dms.config import Config, load_config
from dms.simulation import NEUTRAL, ScenarioBuilder, run_scenario
from dms.types import FrameFeatures


def _feat(t, **kw):
    v = {**NEUTRAL, **kw}
    ear = v.pop("ear")
    return FrameFeatures(t=t, face_present=True, ear=ear, ear_left=v.pop("ear_left", ear), ear_right=v.pop("ear_right", ear),
                         eye_width_px=v.pop("eye_width_px", 32.0), **v)


def _calibrated(cfg=None, **base):
    cfg = cfg or Config()
    cal = DriverCalibrator(cfg.calibration, cfg.thresholds)
    for i in range(60):
        cal.collect(_feat(i * 0.1, **base))
    assert cal.status == "CALIBRATED"
    return cfg, cal.thresholds


def _run_frames(cfg, thr, frames):
    clf = BehaviorClassifier(cfg)
    out = []
    for f in frames:
        st = clf.classify_frame(f, thr)
        out.append((st, clf.update(f, st)))
    return clf, out


def test_camera_stall_cannot_create_a_fake_microsleep():
    cfg, thr = _calibrated()
    frames = [_feat(10 + i * 0.1) for i in range(10)]
    frames += [_feat(11.0 + i * 0.1, ear=0.07, blink=0.9) for i in range(3)]  # eyes closing ...
    frames += [_feat(16.3 + i * 0.1, ear=0.07, blink=0.9) for i in range(3)]  # ... 5 s stall ... still closed
    _, out = _run_frames(cfg, thr, frames)
    assert out[13][0].frame_gap
    assert max(b.closure_s for _, b in out) < 0.5  # never a 5-s closure
    assert all(b.drowsiness_state == "alert" for _, b in out)


def test_per_eye_baseline_handles_an_off_axis_camera():
    # camera to one side: the near eye looks wider (EAR 0.40) than the far eye (0.22)
    cfg, thr = _calibrated(ear=0.31, ear_left=0.40, ear_right=0.22)
    assert thr.baseline.ear_left == pytest.approx(0.40) and thr.baseline.ear_right == pytest.approx(0.22)
    # the far eye alone narrows to 0.17 (77 % of ITS baseline): with one shared baseline (0.31,
    # threshold 0.217) that eye would already count as closed; per eye it does not
    _, out = _run_frames(cfg, thr, [_feat(10.0, ear=0.285, ear_left=0.40, ear_right=0.17, blink=0.12)])
    assert out[0][0].eye_sources["ear"] < 0.5


@pytest.mark.parametrize("kw,reason", [
    (dict(yaw=60.0), "head_away"),
    (dict(phone_conf=0.8), "phone"),
    (dict(mar=0.8, jaw_open=0.7), "yawn"),
    (dict(eye_width_px=8.0), "quality"),
])
def test_eyes_are_not_judged_when_another_behaviour_or_poor_quality_explains_them(kw, reason):
    cfg, thr = _calibrated()
    _, out = _run_frames(cfg, thr, [_feat(10.0, ear=0.07, blink=0.9, **kw)])
    st, b = out[0]
    assert st.eyes_not_judged == reason and st.eyes_closed is None
    assert b.drowsiness_state == "alert"


def test_look_down_guard_drops_only_ear():
    cfg = load_config(overrides={"quality": {"look_down_ear_guard": 0.5}})
    cfg, thr = _calibrated(cfg)
    _, out = _run_frames(cfg, thr, [_feat(10.0, ear=0.12, blink=0.15, look_down=0.8)])
    st = out[0][0]
    assert st.eye_sources["ear"] > 0.5  # EAR alone would say "closed" ...
    assert st.eyes_closed is False  # ... but the learned eyeBlink decides while the eyes look down


def test_face_loss_counts_as_gaze_only_after_turning():
    cfg, thr = _calibrated()
    turning = [_feat(10 + i * 0.1, yaw=NEUTRAL["yaw"] + 25) for i in range(3)] + \
        [FrameFeatures(t=10.3 + i * 0.1, face_present=False) for i in range(5)]
    _, out = _run_frames(cfg, thr, turning)
    assert out[-1][0].face_status == "lost_turning" and out[-1][0].head_away
    frontal = [_feat(10 + i * 0.1) for i in range(3)] + [FrameFeatures(t=10.3 + i * 0.1, face_present=False) for i in range(5)]
    _, out = _run_frames(cfg, thr, frontal)
    assert out[-1][0].face_status == "lost" and out[-1][0].head_away is None
    assert out[-1][1].data_valid is False


def test_blinks_long_closures_and_states():
    b = ScenarioBuilder(seed=5).alert(10)
    for _ in range(20):
        b.alert(2.5).blink()
    frames = b.build()
    out = run_scenario(frames)
    assert out[-1].behavior.long_closures == 0 and out[-1].behavior.drowsiness_state == "alert"
    assert out[-1].behavior.blink_rate_per_min and out[-1].behavior.blink_rate_per_min > 10

    b = ScenarioBuilder(seed=6).alert(10).eyes_closed(0.6).alert(5).eyes_closed(0.6).alert(5)
    out = run_scenario(b.build())
    assert out[-1].behavior.long_closures == 2
    assert out[-1].behavior.drowsiness_state == "slightly_drowsy"  # two slow blinks in a minute


def test_frequent_yawning_with_eye_signs_is_moderately_drowsy():
    b = ScenarioBuilder(seed=7).alert(10).yawn(3).alert(10).yawn(3).alert(5).eyes_closed(0.9).alert(3)
    out = run_scenario(b.build())
    assert out[-1].behavior.drowsiness_state == "moderately_drowsy"
    assert out[-1].behavior.eye_level == 1  # the risk model sees the eye level only (yawns separately)


def test_drowsy_state_recovers_step_by_step():
    cfg = load_config(overrides={"drowsiness": {"recovery_s": 20.0}})
    b = ScenarioBuilder(seed=8).alert(10).eyes_closed(2.0).alert(60)
    frames = b.build()
    out = run_scenario(frames, cfg)
    state = {round(f.t, 1): o.behavior.drowsiness_state for f, o in zip(frames, out)}
    assert state[13.0] == "moderately_drowsy"
    assert state[35.0] == "slightly_drowsy"  # 20 s without a new sign
    assert state[55.0] == "alert"


def test_calibration_flags_off_axis_camera():
    cfg = Config()
    cal = DriverCalibrator(cfg.calibration, cfg.thresholds)
    for i in range(60):
        cal.collect(_feat(i * 0.1, yaw=43.0))  # the S01 set-up
    assert cal.status == "CALIBRATED" and "43 deg" in cal.warning
