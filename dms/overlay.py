"""Draw the pipeline's intermediate results on the frame (for demos and debugging)."""

from __future__ import annotations

import cv2
import numpy as np

from .features.geometry import LEFT_EYE_EAR, MOUTH_MAR, RIGHT_EYE_EAR
from .features.head_pose import POSE_LANDMARKS
from .types import FrameResult

TIER_COLORS = {"NONE": (80, 180, 80), "LOW": (0, 200, 255), "MEDIUM": (0, 140, 255), "HIGH": (0, 0, 255)}
LEVEL_COLORS = {"NONE": (160, 160, 160), "MILD": (0, 200, 255), "MODERATE": (0, 140, 255), "SEVERE": (0, 0, 255)}


def _text(img, s, org, color=(255, 255, 255), scale=0.5, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _bar(img, x, y, w, value, color, label):
    cv2.rectangle(img, (x, y), (x + w, y + 10), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(w * max(0.0, min(1.0, value))), y + 10), color, -1)
    cv2.line(img, (x + w // 2, y - 2), (x + w // 2, y + 12), (255, 255, 255), 1)  # decision level 0.5
    _text(img, label, (x + w + 6, y + 10), scale=0.42)


def draw(frame: np.ndarray, r: FrameResult, fps: float | None = None, warning: str = "") -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    p = r.perception

    # Stage 1 outputs
    if p is not None and p.face is not None:
        pts = p.face.landmarks
        for idx in list(RIGHT_EYE_EAR) + list(LEFT_EYE_EAR):
            cv2.circle(img, tuple(int(v) for v in pts[idx, :2]), 2, (255, 255, 0), -1)
        for idx in MOUTH_MAR:
            cv2.circle(img, tuple(int(v) for v in pts[idx, :2]), 2, (255, 0, 255), -1)
        for idx in POSE_LANDMARKS:
            cv2.circle(img, tuple(int(v) for v in pts[idx, :2]), 3, (0, 255, 0), -1)
        x1, y1, x2, y2 = (int(v) for v in p.face.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
    if p is not None:
        part_colors = {"phone": (0, 0, 255), "eye_closed": (0, 140, 255), "eye_open": (0, 200, 0), "yawn": (255, 0, 255), "no_yawn": (200, 200, 0)}
        for d in p.detections:
            if d.role not in part_colors:
                continue
            color = part_colors[d.role] if d.in_driver_roi else (128, 128, 128)
            x1, y1, x2, y2 = (int(v) for v in d.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 if d.role == "phone" else 1)
            _text(img, f"{d.role} {d.confidence:.2f}", (x1, max(12, y1 - 4)), color, 0.4 if d.role != "phone" else 0.5)

    # Side panel
    panel_w = 300
    canvas = np.zeros((max(h, 420), w + panel_w, 3), dtype=np.uint8)
    canvas[:h, :w] = img
    x, y = w + 10, 20
    f, s = r.features, r.states
    tier_color = TIER_COLORS[r.active_tier]
    cv2.rectangle(canvas, (w, 0), (w + panel_w, 34), tier_color, -1)
    _text(canvas, f"{r.alert_state} {r.active_tier if r.active_tier != 'NONE' else ''}  risk {r.risk.tier} R={r.risk.risk:.0f}",
          (x, 23), (255, 255, 255), 0.5, 2)
    y = 55
    _text(canvas, f"calibration: {r.calibration_status}", (x, y), scale=0.45)
    if fps:
        _text(canvas, f"{fps:.1f} FPS", (w + panel_w - 80, y), scale=0.45)
    y += 20
    fmt = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"
    _text(canvas, f"EAR {fmt(f.ear)}  MAR {fmt(f.mar)}", (x, y), scale=0.45)
    y += 18
    _text(canvas, f"yaw {fmt(f.yaw, 1)}  pitch {fmt(f.pitch, 1)}", (x, y), scale=0.45)
    y += 18
    _text(canvas, f"blink {fmt(f.blink, 2)}  jaw {fmt(f.jaw_open, 2)}  phone {f.phone_conf:.2f}", (x, y), scale=0.45)
    y += 22
    _text(canvas, "Stage 3 evidence (0.5 = threshold)", (x, y), (200, 200, 200), 0.42)
    y += 10
    for label, val in (("eyes closed", s.eye_evidence), ("mouth open", s.mouth_evidence), ("head away", s.pose_evidence)):
        _bar(canvas, x, y, 120, val or 0.0, (0, 200, 255), label)
        y += 18
    b = r.behavior
    _text(canvas, f"drowsiness: {b.drowsiness_state.replace('_', ' ')}", (x, y + 6), (0, 220, 255) if b.drowsy else (255, 255, 255), 0.45)
    y += 18
    _text(canvas, f"closure {b.closure_s:.1f}s  PERCLOS {fmt(b.perclos, 2)}  long {b.long_closures}", (x, y + 6), scale=0.42)
    y += 20
    _text(canvas, f"yawns {b.yawns_in_window}  away {b.gaze_away_s:.1f}s  phone {b.phone_s:.1f}s", (x, y + 6), scale=0.42)
    y += 26
    _text(canvas, "Stage 4 severity", (x, y), (200, 200, 200), 0.42)
    y += 6
    for name, sev in r.severity.as_dict().items():
        y += 16
        _bar(canvas, x, y - 9, 120, sev.score, LEVEL_COLORS[sev.level], f"{name} {sev.level}")
    y += 24
    _text(canvas, "Stage 5 fuzzy risk", (x, y), (200, 200, 200), 0.42)
    for label, val in ((f"fatigue F ({r.risk.fatigue_level})", r.risk.fatigue), ("distraction D", r.risk.distraction), ("context C", r.risk.context)):
        y += 16
        _bar(canvas, x, y - 9, 120, val, (255, 180, 0), f"{label} {val:.2f}")
    if b.no_face:
        _text(canvas, "NO FACE - monitoring degraded", (10, h - 12), (0, 0, 255), 0.6, 2)
    elif s.eyes_not_judged and s.eyes_not_judged != "no_face":
        _text(canvas, f"eyes not judged: {s.eyes_not_judged}", (10, h - 12), (0, 200, 255), 0.5, 1)
    if warning:
        _text(canvas, warning[:90], (10, 20), (0, 0, 255), 0.45, 1)
    return canvas
