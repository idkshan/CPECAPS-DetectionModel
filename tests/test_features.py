import cv2
import numpy as np
import pytest

from dms.classification.evidence import fuse, ramp_evidence
from dms.features.geometry import MOUTH_MAR, RIGHT_EYE_EAR, eye_aspect_ratio, mouth_aspect_ratio
from dms.features.head_pose import MODEL_POINTS, POSE_LANDMARKS, camera_matrix, estimate_head_pose


def _landmarks():
    return np.zeros((478, 3))


def test_ear_formula():
    pts = _landmarks()
    p1, p2, p3, p4, p5, p6 = RIGHT_EYE_EAR
    pts[p1, :2], pts[p4, :2] = (0, 0), (10, 0)  # width 10
    pts[p2, :2], pts[p6, :2] = (3, -2), (3, 2)  # vertical 4
    pts[p3, :2], pts[p5, :2] = (7, -1), (7, 1)  # vertical 2
    assert eye_aspect_ratio(pts, RIGHT_EYE_EAR) == pytest.approx((4 + 2) / (2 * 10))


def test_mar_formula():
    pts = _landmarks()
    q1, q2, q3, q4, q5, q6, q7, q8 = MOUTH_MAR
    pts[q1, :2], pts[q5, :2] = (0, 0), (12, 0)
    pts[q2, :2], pts[q8, :2] = (3, -1), (3, 1)  # 2
    pts[q3, :2], pts[q7, :2] = (6, -3), (6, 3)  # 6
    pts[q4, :2], pts[q6, :2] = (9, -2), (9, 2)  # 4
    assert mouth_aspect_ratio(pts) == pytest.approx((2 + 6 + 4) / (3 * 12))


def test_degenerate_geometry_returns_zero():
    assert eye_aspect_ratio(_landmarks()) == 0.0
    assert mouth_aspect_ratio(_landmarks()) == 0.0


@pytest.mark.parametrize("yaw,pitch", [(0, 0), (25, 0), (-40, 0), (0, 15), (0, -25), (20, -10)])
def test_head_pose_recovers_known_rotation(yaw, pitch):
    w, h = 640, 480
    ry = cv2.Rodrigues(np.array([0.0, -np.radians(yaw), 0.0]))[0]
    rx = cv2.Rodrigues(np.array([-np.radians(pitch), 0.0, 0.0]))[0]
    rvec = cv2.Rodrigues(ry @ rx)[0]
    img, _ = cv2.projectPoints(MODEL_POINTS, rvec, np.array([[0.0], [0.0], [1500.0]]), camera_matrix(w, h), np.zeros(4))
    pts = _landmarks()
    pts[list(POSE_LANDMARKS), :2] = img.reshape(-1, 2)
    est_yaw, est_pitch, _ = estimate_head_pose(pts, w, h)
    assert est_yaw == pytest.approx(yaw, abs=0.5)
    assert est_pitch == pytest.approx(pitch, abs=0.5)


def test_ramp_evidence_both_directions():
    # EAR falls toward its threshold
    assert ramp_evidence(0.30, 0.30, 0.21) == 0.0
    assert ramp_evidence(0.21, 0.30, 0.21) == pytest.approx(0.5)
    assert ramp_evidence(0.12, 0.30, 0.21) == pytest.approx(1.0)
    # MAR rises toward its threshold
    assert ramp_evidence(0.02, 0.02, 0.42) == 0.0
    assert ramp_evidence(0.42, 0.02, 0.42) == pytest.approx(0.5)
    assert ramp_evidence(None, 0.0, 1.0) is None


def test_fuse_skips_missing_and_zero_weight_sources():
    assert fuse({"a": 0.2, "b": 0.8}, {"a": 0.5, "b": 0.5}) == pytest.approx(0.5)
    assert fuse({"a": 0.2, "b": None}, {"a": 0.5, "b": 0.5}) == pytest.approx(0.2)
    assert fuse({"a": 0.2, "b": 0.8}, {"a": 1.0, "b": 0.0}) == pytest.approx(0.2)
    assert fuse({"a": None}, {"a": 1.0}) is None
