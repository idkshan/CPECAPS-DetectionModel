"""Stage 1 (AI): YOLO object detector wrapper (YOLOv8 / YOLO11 / YOLO26).

All three families share the Ultralytics API, so the model is chosen purely by
the weights file (``yolov8n.pt``, ``yolo11n.pt``, ``yolo26n.pt`` or a
fine-tuned checkpoint / exported ``.onnx``/``.tflite``/NCNN model). This keeps
the comparison fair: everything except the network is identical.

Each detected class is mapped to a *role* by name:
  phone       -> phone-use channel
  eye_open    -> eye-state evidence (open)
  eye_closed  -> eye-state evidence (closed)
  yawn        -> mouth evidence (yawning)
  no_yawn     -> mouth evidence (not yawning, optional class)
A COCO-pretrained model only produces "phone"; a driver-state model trained on
the approved datasets produces all of them in one inference.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..config import ObjectDetectorConfig
from ..types import Detection


class ObjectDetector:
    def __init__(self, cfg: ObjectDetectorConfig):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise ImportError("ultralytics is required: pip install ultralytics") from exc

        self.cfg = cfg
        self.model = YOLO(cfg.model_path, task="detect")
        self._roles = {}
        for role, names in (
            ("phone", cfg.phone_class_names),
            ("eye_open", cfg.eye_open_class_names),
            ("eye_closed", cfg.eye_closed_class_names),
            ("yawn", cfg.yawn_class_names),
            ("no_yawn", cfg.no_yawn_class_names),
        ):
            for n in names:
                self._roles[n.lower()] = role
        self.roles_available = {r for k, r in self._roles.items() if k in {str(v).lower() for v in self.model.names.values()}}
        self._frame = 0
        self._last: Tuple[List[Detection], Dict[str, float]] = ([], {})

    def role_of(self, label: str) -> str | None:
        return self._roles.get(label.lower())

    def detect(self, frame_bgr: np.ndarray) -> Tuple[List[Detection], Dict[str, float]]:
        """Return (detections, timings_ms). Runs every ``every_n_frames`` frames."""
        run = self._frame % max(1, self.cfg.every_n_frames) == 0
        self._frame += 1
        if not run:
            return [d for d in self._last[0]], {"yolo_skipped": 1.0}

        res = self.model.predict(
            frame_bgr,
            imgsz=self.cfg.imgsz,
            # run at the lower "stay" confidence so an ongoing phone episode is not
            # lost on hard frames; eye / yawn boxes are filtered at cfg.conf later
            conf=min(self.cfg.conf, self.cfg.phone_stay_conf),
            iou=self.cfg.iou,
            device=self.cfg.device,
            verbose=False,
        )[0]
        names = res.names
        dets: List[Detection] = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            for box, c, k in zip(xyxy, conf, cls):
                label = str(names[int(k)])
                dets.append(Detection(label=label, confidence=float(c), box=tuple(float(v) for v in box), role=self.role_of(label)))
        speed = {f"yolo_{k}": float(v) for k, v in (res.speed or {}).items()}
        self._last = (dets, speed)
        return dets, speed
