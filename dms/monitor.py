"""The complete detection model: Stage 1 -> Stage 5 for one frame.

    frame --(1) MediaPipe Face Landmarker + YOLO [+ eye-state CNN]
          --(2) EAR / MAR / head pose / blendshapes / phone-near-driver
          --(3) calibrated thresholds + temporal persistence
          --(4) per-behaviour severity
          --(5) hierarchical fuzzy risk -> alarm tier -> alert commands
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

import numpy as np

from .config import Config
from .context import ContextProvider
from .engine import DecisionEngine
from .features.extractor import FeatureExtractor
from .perception.face_landmarker import FaceLandmarkerModel
from .types import AlertCommand, FrameResult, PerceptionResult


class DriverMonitor:
    def __init__(
        self,
        cfg: Config,
        context: Optional[ContextProvider] = None,
        use_detector: bool = True,
        alert_sinks: Optional[List[Callable[[AlertCommand], None]]] = None,
        video_mode: bool = True,
    ):
        self.cfg = cfg
        self.context = context or ContextProvider()
        self.face_model = FaceLandmarkerModel(cfg.face, video_mode=video_mode)
        self.detector = None
        if use_detector:
            from .perception.object_detector import ObjectDetector

            self.detector = ObjectDetector(cfg.detector)
        self.eye_model = None
        if cfg.eye_model.model_path:
            from .perception.eye_state import EyeStateClassifier

            self.eye_model = EyeStateClassifier(cfg.eye_model)
        self.extractor = FeatureExtractor(cfg.detector)
        self.engine = DecisionEngine(cfg)
        self.alert_sinks = alert_sinks or []

    def perceive(self, frame: np.ndarray, t: float) -> PerceptionResult:
        """Stage 1: run the neural networks."""
        h, w = frame.shape[:2]
        timings = {}
        t0 = time.perf_counter()
        face = self.face_model.detect(frame, t)
        timings["face_ms"] = (time.perf_counter() - t0) * 1000
        dets = []
        if self.detector is not None:
            t0 = time.perf_counter()
            dets, speed = self.detector.detect(frame)
            timings["yolo_ms"] = (time.perf_counter() - t0) * 1000
            timings.update(speed)
        eye_prob = None
        if self.eye_model is not None and face is not None:
            t0 = time.perf_counter()
            eye_prob = self.eye_model.predict(frame, face)
            timings["eye_cnn_ms"] = (time.perf_counter() - t0) * 1000
        roles = self.detector.roles_available if self.detector is not None else set()
        return PerceptionResult(frame_size=(w, h), face=face, detections=dets, eye_closed_prob=eye_prob,
                                timings_ms=timings, detector_roles=roles)

    def process(self, frame: np.ndarray, t: float, index: int = 0) -> FrameResult:
        start = time.perf_counter()
        perception = self.perceive(frame, t)
        t1 = time.perf_counter()
        features = self.extractor.extract(perception, t)  # Stage 2
        t2 = time.perf_counter()
        out = self.engine.step(features, self.context.read(t))  # Stages 3-5
        t3 = time.perf_counter()
        for cmd in out.alerts:
            for sink in self.alert_sinks:
                sink(cmd)
        timings = dict(perception.timings_ms)
        timings.update(
            perception_ms=(t1 - start) * 1000,
            features_ms=(t2 - t1) * 1000,
            decision_ms=(t3 - t2) * 1000,
            total_ms=(t3 - start) * 1000,
        )
        return FrameResult(
            t=t,
            frame_index=index,
            perception=perception,
            features=features,
            states=out.states,
            behavior=out.behavior,
            severity=out.severity,
            risk=out.risk,
            alerts=out.alerts,
            active_tier=out.active_tier,
            calibration_status=out.calibration_status,
            context=out.context,
            timings_ms=timings,
            alert_state=out.alert_state,
        )

    def camera_lost(self, t: float, reason: str = "no camera frames") -> List[AlertCommand]:
        """The camera link is down: stop sounding and report "monitoring unavailable"."""
        cmds = self.engine.camera_lost(t, reason)
        for cmd in cmds:
            for sink in self.alert_sinks:
                sink(cmd)
        return cmds

    def recalibrate(self) -> None:
        self.engine.recalibrate()

    def close(self) -> None:
        self.face_model.close()
