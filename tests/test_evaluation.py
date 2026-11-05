"""Evaluation package: label validation, one-to-one events, confusion matrices, alarm rates."""

import numpy as np
import pytest

from dms.evaluation.alarms import alarm_report, normalise_alert_log
from dms.evaluation.drowsiness import evaluate_states, predicted_state
from dms.evaluation.labels import load_labels
from dms.evaluation.metrics import episodes, event_metrics, frame_metrics, match_events, multiclass_metrics, segment_metrics, truth_mask


def _labels(tmp_path, text, bom=False, **kw):
    p = tmp_path / "labels.csv"
    p.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode())
    return load_labels(p, **kw)


# ------------------------------------------------------------------ labels
def test_labels_tolerate_bom_spaces_and_levels(tmp_path):
    lab = _labels(tmp_path, "Behavior , start_s, end_s\nyawning,40,44\nmoderately_drowsy,64,68\nphone,88,94\nalert,0,10\n", bom=True)
    assert lab.segments["yawning"] == [(40.0, 44.0)]
    assert lab.segments["drowsiness"] == [(64.0, 68.0)]
    assert lab.segments["phone_usage"] == [(88.0, 94.0)]
    assert lab.drowsiness_levels == [(64.0, 68.0, "moderately_drowsy")]
    assert lab.warnings == []


def test_unknown_names_are_reported_not_silently_dropped(tmp_path):
    lab = _labels(tmp_path, "behavior,start_s,end_s\nyawnig,1,3\n")
    assert lab.segments["yawning"] == [] and "yawnig" in lab.warnings[0]
    with pytest.raises(ValueError):
        _labels(tmp_path, "behavior,start_s,end_s\nyawnig,1,3\n", strict=True)


@pytest.mark.parametrize("row", ["yawning,5,5", "yawning,8,3", "yawning,-1,3", "yawning,a,3"])
def test_invalid_segments_raise(tmp_path, row):
    with pytest.raises(ValueError):
        _labels(tmp_path, f"behavior,start_s,end_s\n{row}\n")


def test_overlaps_merge_and_conflicting_levels_raise(tmp_path):
    lab = _labels(tmp_path, "behavior,start_s,end_s\nyawning,1,4\nyawning,3,6\n")
    assert lab.segments["yawning"] == [(1.0, 6.0)] and "merged" in lab.warnings[0]
    with pytest.raises(ValueError, match="overlap"):
        _labels(tmp_path, "behavior,start_s,end_s\nslightly_drowsy,10,30\nmoderately_drowsy,20,40\n")


def test_segment_after_recording_end_is_flagged(tmp_path):
    lab = _labels(tmp_path, "behavior,start_s,end_s\nphone,50,70\n", duration_s=60)
    assert any("after the recording" in w for w in lab.warnings)


# ------------------------------------------------------------------ metrics
def test_one_long_prediction_cannot_match_many_events():
    """The S01 pattern: one 180-s drowsy episode over 9 labelled closures."""
    gt = [(80 + 20 * k, 83 + 20 * k) for k in range(9)]
    pred = [(75.0, 255.0)]
    m = event_metrics(pred, gt, tol=1.0)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 8)
    assert m["recall"] == pytest.approx(1 / 9, abs=1e-3)
    assert m["gt_covered"] == 9  # the old any-overlap count, kept for reference


def test_match_events_prefers_largest_overlap():
    pred = [(0.0, 10.0), (9.0, 12.0)]
    gt = [(1.0, 9.5), (10.0, 11.0)]
    matches, fp, fn = match_events(pred, gt, tol=0.0)
    assert matches == [(0, 0), (1, 1)] and fp == [] and fn == []


def test_frame_metrics_confusion_and_rates():
    pred = np.array([1, 1, 0, 0, 1, 0], bool)
    truth = np.array([1, 0, 1, 0, 1, 0], bool)
    m = frame_metrics(pred, truth)
    assert m["confusion"]["matrix"] == [[2, 1], [1, 2]]
    assert m["fpr"] == pytest.approx(1 / 3, abs=1e-3) and m["fnr"] == pytest.approx(1 / 3, abs=1e-3)


def test_segment_metrics_independent_of_frame_rate():
    gt = [(10.0, 14.9)]
    res = []
    for fps in (10, 30):
        t = np.arange(0, 30, 1 / fps)
        pred = (t >= 10.0) & (t < 15.0)
        res.append(segment_metrics(t, pred, gt, 1.0)["f1"])
    assert res[0] == res[1] == pytest.approx(1.0)


def test_episodes_and_truth_mask():
    t = np.arange(0, 5, 0.1)
    on = np.ones_like(t, dtype=bool)
    on[20:23] = False  # 0.3 s gap -> merged
    assert len(episodes(t, on, merge_gap=0.5)) == 1
    assert truth_mask(np.array([0.0, 1.0, 2.0]), [(0.5, 1.5)]).tolist() == [False, True, False]


def test_multiclass_macro_weighted_micro():
    y_true = ["alert"] * 8 + ["slightly_drowsy"] * 2
    y_pred = ["alert"] * 7 + ["slightly_drowsy"] + ["slightly_drowsy", "alert"]
    m = multiclass_metrics(y_true, y_pred, ["alert", "slightly_drowsy", "moderately_drowsy"])
    assert m["confusion"] == [[7, 1, 0], [1, 1, 0], [0, 0, 0]]
    assert m["per_class"]["moderately_drowsy"]["support"] == 0  # absent class not in the macro mean
    assert m["macro_f1"] == pytest.approx((7 / 8 + 0.5) / 2, abs=1e-3)
    assert m["accuracy"] == m["micro_f1"] == pytest.approx(0.8)


# ------------------------------------------------------------------ drowsiness states
def test_state_metrics_three_class_and_binary(tmp_path):
    lab = _labels(tmp_path, "behavior,start_s,end_s\nslightly_drowsy,10,20\nmoderately_drowsy,30,40\n")
    t = np.arange(0, 50, 0.5)
    pred = ["moderately_drowsy" if 30 <= x <= 40 else "slightly_drowsy" if 12 <= x <= 20 else "alert" for x in t]
    rep = evaluate_states(t, pred, lab)
    assert rep["three_class"]["per_class"]["moderately_drowsy"]["f1"] == pytest.approx(1.0)
    assert rep["binary"]["fp"] == 0 and rep["binary"]["fn"] == 4  # 10.0-11.5 s missed
    assert rep["segments"]["exact"] == 2


def test_predicted_state_from_every_log_version():
    assert predicted_state({"behavior": {"drowsiness_state": "slightly_drowsy"}}) == "slightly_drowsy"
    assert predicted_state({"risk": {"drowsiness_level": "Level 3 (severely drowsy)"}}) == "moderately_drowsy"
    assert predicted_state({"behavior": {"drowsy": True}}) is None


# ------------------------------------------------------------------ alarms
def test_unnecessary_alarms_use_non_event_hours():
    times = list(np.arange(0, 3600, 1.0))
    tiers = ["NONE"] * len(times)
    events = [(100.0, 1900.0)]  # half the hour is a labelled event
    alerts = [{"t": 500.0, "tier": "HIGH", "event": "start"},  # inside the event: necessary
              {"t": 1901.5, "tier": "LOW", "event": "start"},  # within 2 s after the end: necessary
              {"t": 3000.0, "tier": "MEDIUM", "event": "start", }]  # unnecessary
    rep = alarm_report(alerts, times, tiers, events, tol_before=1.0, tol_after=2.0)
    assert rep["alarms"]["count"] == 3 and rep["alarms"]["unnecessary"] == 1
    assert rep["alarms"]["per_hour"] == pytest.approx(1.0, abs=0.01)
    assert rep["alarms"]["per_non_event_hour"] == pytest.approx(1 / ((3599 - 1803) / 3600), abs=0.01)


def test_legacy_alert_log_is_understood():
    legacy = [{"t": 1.0, "tier": "LOW", "push": False, "new_episode": True},
              {"t": 2.0, "tier": "HIGH", "push": True, "new_episode": False},
              {"t": 3.0, "tier": "HIGH", "push": False, "new_episode": False}]
    log = normalise_alert_log(legacy)
    assert len(log["start"]) == 1 and len(log["escalate"]) == 1 and len(log["push"]) == 1


def test_alarm_validation_expected_vs_triggered_tier(tmp_path):
    from dms.evaluation.alarms import alarm_validation

    lab = _labels(tmp_path, "behavior,start_s,end_s,expected_tier\nyawning,10,14,LOW\ngaze,30,35,HIGH\n"
                            "mirror_check,50,51,NONE\nphone,60,66,HIGH\n")
    assert len(lab.expected_tiers) == 4
    alerts = [{"t": 12.0, "tier": "LOW", "event": "start"},
              {"t": 32.3, "tier": "MEDIUM", "event": "start"}, {"t": 33.3, "tier": "HIGH", "event": "escalate"},
              {"t": 63.3, "tier": "MEDIUM", "event": "start"}]  # phone under-alerted
    av = alarm_validation(alerts, lab.expected_tiers)
    assert av["exact"] == 3 and av["under_alerted"] == 1 and av["over_alerted"] == 0
    assert av["confusion"][3][2] == 1  # expected HIGH, got MEDIUM
