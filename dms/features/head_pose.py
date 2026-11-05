"""Stage 2: head pose (yaw, pitch, roll) with OpenCV solvePnP (Section 1.7.8).

A generic 3-D face model (six points) is aligned to the matching 2-D
landmarks. The model is expressed in OpenCV camera axes (x right, y down,
z away from the camera) so that a face looking straight into the camera has
zero rotation. Absolute angles carry some bias (generic face, approximate
focal length, camera mounted off-axis on the windshield), which is why the
classifier only uses the *deviation* from the driver's calibrated baseline.

Sign convention: yaw > 0 when the face turns toward the image's right side,
pitch > 0 when the face tilts up, roll > 0 when the head tilts clockwise in
the image.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

# MediaPipe indices: nose tip, chin, image-left eye outer corner,
# image-right eye outer corner, image-left mouth corner, image-right mouth corner
POSE_LANDMARKS = (1, 152, 33, 263, 61, 291)

# Generic face model (arbitrary units, nose tip at origin), camera axes.
MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, 330.0, 65.0),
        (-225.0, -170.0, 135.0),
        (225.0, -170.0, 135.0),
        (-150.0, 150.0, 125.0),
        (150.0, 150.0, 125.0),
    ],
    dtype=np.float64,
)


def camera_matrix(width: int, height: int) -> np.ndarray:
    """Pinhole approximation: focal length ~ image width, centre of image."""
    f = float(width)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=np.float64)


def rotation_to_euler(rmat: np.ndarray) -> Tuple[float, float, float]:
    """Convert a rotation matrix to (yaw, pitch, roll) in degrees (our sign convention)."""
    angles, *_ = cv2.RQDecomp3x3(rmat)
    rx, ry, rz = (float(a) for a in angles)
    # RQDecomp3x3 returns rotations about camera x (pitch), y (yaw), z (roll).
    # In camera axes a face tilting up rotates negatively about x, and a face
    # turning to the image right rotates negatively about y.
    return -ry, -rx, rz


def estimate_head_pose(landmarks: np.ndarray, width: int, height: int) -> Optional[Tuple[float, float, float]]:
    """Return (yaw, pitch, roll) in degrees, or None if solvePnP fails."""
    image_points = landmarks[list(POSE_LANDMARKS), :2].astype(np.float64)
    cam = camera_matrix(width, height)
    dist = np.zeros((4, 1))
    ok, rvec, _tvec = cv2.solvePnP(MODEL_POINTS, image_points, cam, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        ok, rvec, _tvec = cv2.solvePnP(MODEL_POINTS, image_points, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
    rmat, _ = cv2.Rodrigues(rvec)
    return rotation_to_euler(rmat)
