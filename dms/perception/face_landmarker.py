"""Stage 1 (AI): MediaPipe Face Landmarker wrapper.

The Face Landmarker bundle contains three neural networks:
  1. a BlazeFace-style face detector (finds the face box),
  2. a face-mesh regression network (478 3-D landmarks),
  3. a blendshape network (52 facial-expression coefficients such as
     ``eyeBlinkLeft`` and ``jawOpen``).
None of these use hand-set thresholds: they are learned models. Their
outputs become the raw material for the geometric features of Stage 2.

The same ``face_landmarker.task`` file is used by the MediaPipe Tasks SDK on
Android, so this stage ports directly to the companion app.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..config import FaceModelConfig
from ..types import FaceObservation


class FaceLandmarkerModel:
    def __init__(self, cfg: FaceModelConfig, video_mode: bool = True):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise ImportError("mediapipe is required: pip install mediapipe") from exc

        self._mp = mp
        self._video_mode = video_mode
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=cfg.model_path),
            running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
            num_faces=1,  # the driver
            min_face_detection_confidence=cfg.min_detection_confidence,
            min_face_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
            output_face_blendshapes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect(self, frame_bgr: np.ndarray, t: float) -> Optional[FaceObservation]:
        """Run the face models on one BGR frame captured at time ``t`` (seconds)."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self._video_mode:
            ts_ms = max(int(round(t * 1000.0)), self._last_ts_ms + 1)  # must strictly increase
            self._last_ts_ms = ts_ms
            result = self._landmarker.detect_for_video(image, ts_ms)
        else:
            result = self._landmarker.detect(image)

        if not result.face_landmarks:
            return None
        lms = result.face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lms], dtype=np.float64)
        blend = {}
        if result.face_blendshapes:
            blend = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        return FaceObservation(landmarks=pts, blendshapes=blend, bbox=(x1, y1, x2, y2))

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
