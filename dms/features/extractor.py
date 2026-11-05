"""Stage 2: turn Stage-1 model outputs into numeric visual features.

This stage makes no decisions. It computes numbers from the AI outputs:
EAR and MAR from landmark geometry, head pose from solvePnP, the learned
blendshape scores, the optional eye-state probability, the confidence of any
YOLO phone detection located near the driver, and - with a driver-state YOLO
model - YOLO's eye-closed and yawn scores for boxes inside the driver's face.
"""

from __future__ import annotations

from typing import Iterable, Optional

from ..config import ObjectDetectorConfig
from ..types import FaceObservation, FrameFeatures, PerceptionResult
from .geometry import LEFT_EYE_EAR, RIGHT_EYE_EAR, eye_aspect_ratio, eye_width, mouth_aspect_ratio
from .head_pose import estimate_head_pose


def phone_in_driver_roi(box, face: FaceObservation | None, frame_h: int, cfg: ObjectDetectorConfig) -> bool:
    """True if a phone box is plausibly held by the driver.

    Phones far from the driver (e.g. on the passenger seat or a dashboard
    mount at the edge of the frame) are ignored. Without a face every phone
    counts, because the driver's position is unknown.
    """
    if face is None:
        return True
    fx1, fy1, fx2, fy2 = face.bbox
    fw, fh = fx2 - fx1, fy2 - fy1
    fcx = (fx1 + fx2) / 2.0
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    within_x = abs(cx - fcx) <= cfg.roi_width_factor * fw
    within_y = fy1 - cfg.roi_above_factor * fh <= cy <= frame_h
    return within_x and within_y


def inside_face(box, face: FaceObservation, margin: float) -> bool:
    """True if the centre of ``box`` lies inside the face box enlarged by ``margin``."""
    fx1, fy1, fx2, fy2 = face.bbox
    mx, my = (fx2 - fx1) * margin, (fy2 - fy1) * margin
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    return fx1 - mx <= cx <= fx2 + mx and fy1 - my <= cy <= fy2 + my


def two_class_score(positive: Optional[float], negative: Optional[float]) -> Optional[float]:
    """Combine the best confidences of a positive and a negative class into 0..1.

    0.5 means undecided. Only a positive detection -> (1 + p) / 2 >= 0.5;
    only a negative detection -> (1 - n) / 2 < 0.5; both -> (1 + p - n) / 2.
    """
    if positive is None and negative is None:
        return None
    return (1.0 + (positive or 0.0) - (negative or 0.0)) / 2.0


def _best(confs: Iterable[float]) -> Optional[float]:
    confs = list(confs)
    return max(confs) if confs else None


class FeatureExtractor:
    def __init__(self, detector_cfg: ObjectDetectorConfig):
        self.detector_cfg = detector_cfg

    def extract(self, perception: PerceptionResult, t: float) -> FrameFeatures:
        w, h = perception.frame_size
        face = perception.face

        phone_conf, phone_boxes = 0.0, 0
        for det in perception.detections:
            if det.role != "phone":
                continue
            # (the detector runs at the lower phone_stay_conf; the phone module
            #  decides which confidences start or continue an episode)
            det.in_driver_roi = phone_in_driver_roi(det.box, face, h, self.detector_cfg)
            if det.in_driver_roi:
                phone_boxes += 1
                phone_conf = max(phone_conf, det.confidence)

        if face is None:
            return FrameFeatures(t=t, face_present=False, phone_conf=phone_conf, phone_boxes=phone_boxes)

        # YOLO driver-state detections inside the driver's face
        parts = {"eye_open": [], "eye_closed": [], "yawn": [], "no_yawn": []}
        for det in perception.detections:
            if det.role in parts and det.confidence >= self.detector_cfg.conf:
                det.in_driver_roi = inside_face(det.box, face, self.detector_cfg.face_part_margin)
                if det.in_driver_roi:
                    parts[det.role].append(det.confidence)
        roles = perception.detector_roles
        eye_closed_yolo = None
        if {"eye_open", "eye_closed"} & roles:
            eye_closed_yolo = two_class_score(_best(parts["eye_closed"]), _best(parts["eye_open"]))
        yawn_yolo = None
        if "yawn" in roles:
            yawn_yolo = two_class_score(_best(parts["yawn"]), _best(parts["no_yawn"]))
            if yawn_yolo is None:
                yawn_yolo = 0.0  # face visible and no yawning mouth detected

        pts = face.landmarks
        ear_r = eye_aspect_ratio(pts, RIGHT_EYE_EAR)
        ear_l = eye_aspect_ratio(pts, LEFT_EYE_EAR)
        pose = estimate_head_pose(pts, w, h)
        bs = face.blendshapes
        blink = None
        if "eyeBlinkLeft" in bs and "eyeBlinkRight" in bs:
            blink = (bs["eyeBlinkLeft"] + bs["eyeBlinkRight"]) / 2.0
        look_down = None
        if "eyeLookDownLeft" in bs and "eyeLookDownRight" in bs:
            look_down = (bs["eyeLookDownLeft"] + bs["eyeLookDownRight"]) / 2.0
        return FrameFeatures(
            t=t,
            face_present=True,
            ear=(ear_l + ear_r) / 2.0,
            ear_left=ear_l,
            ear_right=ear_r,
            mar=mouth_aspect_ratio(pts),
            yaw=pose[0] if pose else None,
            pitch=pose[1] if pose else None,
            roll=pose[2] if pose else None,
            blink=blink,
            jaw_open=bs.get("jawOpen"),
            eye_closed_prob=perception.eye_closed_prob,
            eye_closed_yolo=eye_closed_yolo,
            yawn_yolo=yawn_yolo,
            phone_conf=phone_conf,
            phone_boxes=phone_boxes,
            eye_width_px=(eye_width(pts, LEFT_EYE_EAR) + eye_width(pts, RIGHT_EYE_EAR)) / 2.0,
            look_down=look_down,
        )
