"""Stage 2: Eye Aspect Ratio and Mouth Aspect Ratio (Section 1.7.15).

    EAR = (|p2 - p6| + |p3 - p5|) / (2 |p1 - p4|)
    MAR = (|q2 - q8| + |q3 - q7| + |q4 - q6|) / (3 |q1 - q5|)

Landmark indices refer to the 478-point MediaPipe face mesh.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# p1..p6 : p1/p4 are the eye corners, p2/p3 upper lid, p6/p5 lower lid
RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)  # subject's right eye (image left)
LEFT_EYE_EAR = (362, 385, 387, 263, 373, 380)  # subject's left eye (image right)
# q1..q8 : q1/q5 inner-lip corners, q2-q4 upper inner lip, q6-q8 lower inner lip
MOUTH_MAR = (78, 81, 13, 311, 308, 402, 14, 178)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(pts: np.ndarray, idx: Sequence[int] = RIGHT_EYE_EAR) -> float:
    p1, p2, p3, p4, p5, p6 = (pts[i, :2] for i in idx)
    horizontal = _dist(p1, p4)
    if horizontal <= 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horizontal)


def eye_width(pts: np.ndarray, idx: Sequence[int] = RIGHT_EYE_EAR) -> float:
    """Eye-corner distance |p1 - p4| in pixels: how large the eye is in the image."""
    return _dist(pts[idx[0], :2], pts[idx[3], :2])


def mouth_aspect_ratio(pts: np.ndarray, idx: Sequence[int] = MOUTH_MAR) -> float:
    q1, q2, q3, q4, q5, q6, q7, q8 = (pts[i, :2] for i in idx)
    horizontal = _dist(q1, q5)
    if horizontal <= 1e-6:
        return 0.0
    return (_dist(q2, q8) + _dist(q3, q7) + _dist(q4, q6)) / (3.0 * horizontal)
