"""Full pipeline with the real neural networks (skipped if models are not downloaded).

Uses the sample photo bundled with the ultralytics package: 8 s of the face
(calibration), 3.5 s of a blank frame (camera covered while the driver faced
forward), then the face again. Expected: calibration succeeds, and the blank
frames are reported as "face lost" (unknown) - not as gaze deviation or
drowsiness, so no alarm sounds.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FACE_MODEL = ROOT / "models" / "face_landmarker.task"
YOLO_MODEL = ROOT / "models" / "yolov8n.pt"

pytestmark = pytest.mark.skipif(not (FACE_MODEL.exists() and YOLO_MODEL.exists()), reason="run scripts/download_models.py first")


def _face_image():
    import ultralytics

    img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "zidane.jpg"))
    crop = img[20:407, 742:1150]  # head and shoulders of the person on the right
    return cv2.resize(crop, None, fx=1.5, fy=1.5)


def test_full_pipeline_on_real_face():
    from dms.config import load_config
    from dms.context import StaticContext
    from dms.monitor import DriverMonitor

    cfg = load_config(overrides={"face": {"model_path": str(FACE_MODEL)}, "detector": {"model_path": str(YOLO_MODEL)}})
    monitor = DriverMonitor(cfg, context=StaticContext(speed_kmh=40.0))
    face = _face_image()
    blank = np.full_like(face, 30)
    rng = np.random.default_rng(0)
    results = []
    try:
        for i in range(150):
            t = i / 10.0
            frame = blank if 8.0 <= t < 11.5 else np.clip(face.astype(np.int16) + rng.integers(-3, 4, face.shape), 0, 255).astype(np.uint8)
            results.append(monitor.process(frame, t, i))
    finally:
        monitor.close()

    first = results[5]
    assert first.features.face_present
    assert 0.2 < first.features.ear < 0.4  # open eyes
    assert first.features.mar < 0.1  # closed mouth
    assert first.features.blink is not None and first.features.jaw_open is not None  # learned blendshapes present
    assert results[79].calibration_status == "CALIBRATED"
    tiers = [r.active_tier for r in results]
    assert all(t == "NONE" for t in tiers)  # no alarm while facing forward, none for the covered camera
    assert results[100].states.face_status == "lost" and not results[100].behavior.data_valid
    assert results[100].states.eyes_not_judged == "no_face"
    assert results[140].states.face_status == "ok"
    assert results[60].features.eye_width_px > 12  # eyes large enough to be judged
    assert "total_ms" in results[-1].timings_ms
