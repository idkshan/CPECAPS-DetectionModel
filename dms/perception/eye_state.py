"""Stage 1 (AI, optional): learned eye-state classifier.

Crops both eyes using the face landmarks and classifies each crop as open or
closed with a small image classifier (for example ``yolo11n-cls`` fine-tuned
on the MRL Eye Dataset with ``scripts/prepare_mrl_eye.py`` +
``scripts/train_yolo.py --task classify``). The resulting probability of
"closed" is a third, learned source of eye-closure evidence next to EAR and
the ``eyeBlink`` blendshape.

Note: the MRL Eye Dataset contains cropped eye images only, so it cannot be
used to validate EAR (which needs a full face); it is, however, exactly the
right data to *train* this classifier.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..config import EyeStateModelConfig
from ..types import FaceObservation

# MediaPipe face-mesh eye contours (16 points each)
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


def crop_eye(frame: np.ndarray, pts: np.ndarray, scale: float, grayscale: bool) -> Optional[np.ndarray]:
    """Square crop around the eye landmarks ``pts`` (N x 2 pixels)."""
    h, w = frame.shape[:2]
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    side = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * scale
    if side < 8:
        return None
    x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x2, y2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = frame[y1:y2, x1:x2]
    if grayscale:
        crop = cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return crop


class EyeStateClassifier:
    def __init__(self, cfg: EyeStateModelConfig):
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = YOLO(cfg.model_path, task="classify")
        names = {int(k): str(v).lower() for k, v in self.model.names.items()}
        wanted = {n.lower() for n in cfg.closed_class_names}
        matches = [k for k, v in names.items() if v in wanted]
        if not matches:
            raise ValueError(f"Eye-state model classes {names} contain none of {sorted(wanted)}")
        self.closed_index = matches[0]

    def predict(self, frame_bgr: np.ndarray, face: FaceObservation) -> Optional[float]:
        """Mean probability that the eyes are closed, or None if crops are unusable."""
        crops = []
        for contour in (RIGHT_EYE_CONTOUR, LEFT_EYE_CONTOUR):
            c = crop_eye(frame_bgr, face.landmarks[contour, :2], self.cfg.crop_scale, self.cfg.grayscale)
            if c is not None:
                crops.append(c)
        if not crops:
            return None
        kwargs = {"verbose": False}
        if self.cfg.imgsz:
            kwargs["imgsz"] = self.cfg.imgsz
        results = self.model.predict(crops, **kwargs)
        probs = [float(r.probs.data[self.closed_index]) for r in results]
        return float(np.mean(probs))
