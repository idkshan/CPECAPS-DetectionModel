"""YOLO driver-state detections -> features -> fused evidence (no real model needed)."""

import numpy as np
import pytest

from dms.classification.calibration import DriverCalibrator
from dms.classification.classifier import BehaviorClassifier
from dms.config import Config, apply_evidence_modes
from dms.features.extractor import FeatureExtractor, two_class_score
from dms.types import Detection, FaceObservation, FrameFeatures, PerceptionResult


def _face():
    pts = np.zeros((478, 3))
    pts[:, 0] = np.linspace(200, 400, 478)
    pts[:, 1] = np.linspace(100, 350, 478)
    return FaceObservation(landmarks=pts, blendshapes={"eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.1, "jawOpen": 0.02}, bbox=(200, 100, 400, 350))


def _perceive(dets, roles=frozenset({"phone", "eye_open", "eye_closed", "yawn"})):
    return PerceptionResult(frame_size=(640, 480), face=_face(), detections=dets, detector_roles=set(roles))


def test_two_class_score():
    assert two_class_score(None, None) is None
    assert two_class_score(0.8, None) == pytest.approx(0.9)
    assert two_class_score(None, 0.8) == pytest.approx(0.1)
    assert two_class_score(0.7, 0.3) == pytest.approx(0.7)


def test_yolo_eye_and_yawn_features_from_boxes_inside_face():
    ex = FeatureExtractor(Config().detector)
    dets = [
        Detection("eye_closed", 0.8, (240, 150, 280, 170), role="eye_closed"),
        Detection("eye_closed", 0.9, (600, 10, 630, 30), role="eye_closed"),  # outside the face: ignored
        Detection("yawn", 0.7, (270, 260, 330, 320), role="yawn"),
    ]
    f = ex.extract(_perceive(dets), 0.0)
    assert f.eye_closed_yolo == pytest.approx(0.9)
    assert f.yawn_yolo == pytest.approx(0.85)
    assert not dets[1].in_driver_roi


def test_no_yawn_box_means_not_yawning_and_coco_model_gives_none():
    ex = FeatureExtractor(Config().detector)
    f = ex.extract(_perceive([Detection("eye_open", 0.9, (240, 150, 280, 170), role="eye_open")]), 0.0)
    assert f.yawn_yolo == 0.0 and f.eye_closed_yolo == pytest.approx(0.05)
    f = ex.extract(_perceive([], roles={"phone"}), 0.0)  # COCO model: no driver-state classes
    assert f.eye_closed_yolo is None and f.yawn_yolo is None


def _feat(**kw):
    base = dict(t=10.0, face_present=True, ear=0.29, mar=0.02, yaw=0.0, pitch=0.0, blink=0.1, jaw_open=0.02)
    base.update(kw)
    return FrameFeatures(**base)


def _states(cfg, f):
    cal = DriverCalibrator(cfg.calibration, cfg.thresholds)
    return BehaviorClassifier(cfg).classify_frame(f, cal.thresholds)


def test_yolo_evidence_is_fused_with_thresholds():
    cfg = Config()
    s = _states(cfg, _feat(ear=0.25, eye_closed_yolo=0.95))  # EAR alone says open, YOLO says closed
    assert set(s.eye_sources) == {"ear", "blendshape", "yolo"}
    yolo_only = _states(apply_evidence_modes(Config(), "yolo", "yolo"), _feat(eye_closed_yolo=0.95, yawn_yolo=0.9))
    assert yolo_only.eyes_closed is None or yolo_only.eyes_closed  # mouth open guard may suspend eyes
    assert yolo_only.mouth_open is True
    ear_only = _states(apply_evidence_modes(Config(), "ear", "mar"), _feat(eye_closed_yolo=0.95, yawn_yolo=0.9))
    assert ear_only.eyes_closed is False and ear_only.mouth_open is False


def test_eyes_not_judged_while_phone_visible_or_during_yawn():
    cfg = Config()
    assert _states(cfg, _feat(ear=0.05, blink=0.9, phone_conf=0.8)).eyes_closed is None
    assert _states(cfg, _feat(ear=0.05, blink=0.9, mar=0.35, jaw_open=0.3)).eyes_closed is None  # mouth opening
    assert _states(cfg, _feat(ear=0.05, blink=0.9)).eyes_closed is True
    cfg.temporal.suspend_eyes_when_phone = False
    assert _states(cfg, _feat(ear=0.05, blink=0.9, phone_conf=0.8)).eyes_closed is True
