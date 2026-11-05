"""Stage 3: per-driver baseline calibration and online adaptation (Section 1.7.9).

At the start of a session the driver looks ahead with a relaxed face for a
few seconds. Robust statistics (medians) of EAR, MAR, yaw, pitch and the two
blendshapes form the driver's personal baseline; all thresholds are then
expressed relative to that baseline. Until a valid baseline exists,
population defaults (with widened pose limits) are used and calibration keeps
retrying in the background.

During driving the baseline follows slow changes (lighting, posture) through
an exponential moving average that only runs while the driver is confidently
alert and facing forward, and is clamped so it can never drift far from the
calibrated values. This adaptation is statistical, not learned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Optional

import numpy as np

from ..config import CalibrationConfig, ThresholdConfig
from ..types import FrameFeatures

CALIBRATING, CALIBRATED, DEFAULT = "CALIBRATING", "CALIBRATED", "DEFAULT"


@dataclass
class Baseline:
    ear: float
    mar: float
    yaw: float
    pitch: float
    blink: float
    jaw: float
    ear_left: Optional[float] = None  # per-eye open-eye EAR (None = use the mean ``ear``)
    ear_right: Optional[float] = None


@dataclass
class Thresholds:
    """Absolute thresholds for the current driver, derived from the baseline."""

    baseline: Baseline
    ear_closed: float
    blink_closed: float
    mar_open: float
    jaw_open: float
    yaw_limit: float
    pitch_down_limit: float
    pitch_up_limit: float
    decision_level: float
    calibrated: bool
    ear_closed_ratio: float = 0.70  # per-eye thresholds are baseline_eye x this ratio

    @classmethod
    def from_baseline(cls, b: Baseline, cfg: ThresholdConfig, calibrated: bool) -> "Thresholds":
        pose_scale = 1.0 if calibrated else cfg.uncalibrated_pose_scale
        return cls(
            baseline=b,
            ear_closed=b.ear * cfg.ear_closed_ratio,
            blink_closed=min(0.95, b.blink + cfg.blink_closed_delta),
            mar_open=b.mar + cfg.mar_open_delta,
            jaw_open=min(0.95, b.jaw + cfg.jaw_open_delta),
            yaw_limit=cfg.yaw_deg * pose_scale,
            pitch_down_limit=cfg.pitch_down_deg * pose_scale,
            pitch_up_limit=cfg.pitch_up_deg * pose_scale,
            decision_level=cfg.decision_level,
            calibrated=calibrated,
            ear_closed_ratio=cfg.ear_closed_ratio,
        )


def _robust_std(values: np.ndarray) -> float:
    """Spread estimate from the inter-quartile range (IQR / 1.349 = std for normal data).

    Unlike the median absolute deviation it does not collapse to zero when the
    head alternates between two positions during the window.
    """
    if values.size < 2:
        return 0.0
    q1, q3 = np.percentile(values, [25, 75])
    return float((q3 - q1) / 1.349)


class DriverCalibrator:
    def __init__(self, cfg: CalibrationConfig, thr_cfg: ThresholdConfig, phone_conf_limit: float = 0.35):
        self.cfg = cfg
        self.thr_cfg = thr_cfg
        self.phone_conf_limit = phone_conf_limit  # frames with a confident phone box are not used
        self.warning = ""  # e.g. camera far off the face axis (calibration still accepted)
        d = cfg.defaults
        self.default_baseline = Baseline(d.ear, d.mar, d.yaw, d.pitch, d.blink, d.jaw)
        self.baseline = replace(self.default_baseline)
        self.anchor: Optional[Baseline] = None  # baseline measured at calibration
        self.status = CALIBRATING if cfg.enabled else DEFAULT
        self.last_failure = ""
        self._window_start: Optional[float] = None
        self._samples: List[FrameFeatures] = []
        self._last_adapt_t: Optional[float] = None
        self._thresholds = Thresholds.from_baseline(self.baseline, thr_cfg, calibrated=False)

    # ------------------------------------------------------------------ API
    @property
    def thresholds(self) -> Thresholds:
        return self._thresholds

    def restart(self) -> None:
        """Begin a new calibration window (e.g. new driver)."""
        if not self.cfg.enabled:
            return
        self.status = CALIBRATING
        self._window_start = None
        self._samples = []

    def progress(self, t: float) -> float:
        if self.status != CALIBRATING or self._window_start is None:
            return 1.0 if self.status == CALIBRATED else 0.0
        return min(1.0, (t - self._window_start) / self.cfg.duration_s)

    def collect(self, feats: FrameFeatures) -> None:
        """Feed a frame while CALIBRATING."""
        if self.status != CALIBRATING:
            return
        if self._window_start is None:
            self._window_start = feats.t
        usable = (
            feats.face_present
            and feats.ear is not None
            and feats.mar is not None
            and feats.yaw is not None
            and feats.phone_conf < self.phone_conf_limit
        )
        if usable:
            self._samples.append(feats)
        if feats.t - self._window_start >= self.cfg.duration_s:
            self._finish_window()

    def adapt(self, feats: FrameFeatures) -> None:
        """Slowly track the baseline on confidently alert, forward-facing frames."""
        if self.status != CALIBRATED or not self.cfg.adapt or self.anchor is None:
            self._last_adapt_t = None
            return
        if self._last_adapt_t is None:
            self._last_adapt_t = feats.t
            return
        dt = min(1.0, max(0.0, feats.t - self._last_adapt_t))
        self._last_adapt_t = feats.t
        a = 1.0 - math.exp(-dt / self.cfg.adapt_tau_s)
        b, anc, c = self.baseline, self.anchor, self.cfg

        def step(cur, x, lo, hi):
            return cur if x is None else float(min(hi, max(lo, cur + a * (x - cur))))

        b.ear = step(b.ear, feats.ear, anc.ear * (1 - c.max_ear_drift), anc.ear * (1 + c.max_ear_drift))
        if anc.ear_left and anc.ear_right:
            b.ear_left = step(b.ear_left, feats.ear_left, anc.ear_left * (1 - c.max_ear_drift), anc.ear_left * (1 + c.max_ear_drift))
            b.ear_right = step(b.ear_right, feats.ear_right, anc.ear_right * (1 - c.max_ear_drift), anc.ear_right * (1 + c.max_ear_drift))
        b.mar = step(b.mar, feats.mar, max(0.0, anc.mar - c.max_mar_drift), anc.mar + c.max_mar_drift)
        b.yaw = step(b.yaw, feats.yaw, anc.yaw - c.max_pose_drift_deg, anc.yaw + c.max_pose_drift_deg)
        b.pitch = step(b.pitch, feats.pitch, anc.pitch - c.max_pose_drift_deg, anc.pitch + c.max_pose_drift_deg)
        b.blink = step(b.blink, feats.blink, 0.0, anc.blink + 0.15)
        b.jaw = step(b.jaw, feats.jaw_open, 0.0, anc.jaw + 0.10)
        self._thresholds = Thresholds.from_baseline(b, self.thr_cfg, calibrated=True)

    def pause_adaptation(self) -> None:
        """Called whenever a drowsy/distracted state is present (no baseline drift)."""
        self._last_adapt_t = None

    # ------------------------------------------------------------ internals
    def _finish_window(self) -> None:
        samples, self._samples = self._samples, []
        self._window_start = None
        if len(samples) < self.cfg.min_samples:
            self.last_failure = f"only {len(samples)} usable frames (need {self.cfg.min_samples})"
            return
        arr = lambda name: np.array([getattr(s, name) for s in samples if getattr(s, name) is not None], dtype=float)
        ear, mar, yaw, pitch = arr("ear"), arr("mar"), arr("yaw"), arr("pitch")
        blink, jaw = arr("blink"), arr("jaw_open")
        ear_l, ear_r = arr("ear_left"), arr("ear_right")
        med = lambda a, default: float(np.median(a)) if a.size else default
        candidate = Baseline(
            ear=med(ear, self.default_baseline.ear),
            mar=med(mar, self.default_baseline.mar),
            yaw=med(yaw, 0.0),
            pitch=med(pitch, 0.0),
            blink=med(blink, self.default_baseline.blink),
            jaw=med(jaw, self.default_baseline.jaw),
            ear_left=med(ear_l, None) if ear_l.size else None,
            ear_right=med(ear_r, None) if ear_r.size else None,
        )
        pose_std = max(_robust_std(yaw), _robust_std(pitch))
        problems = []
        if pose_std > self.cfg.max_pose_std_deg:
            problems.append(f"head not steady (pose std {pose_std:.1f} deg)")
        if candidate.ear < self.cfg.min_ear:
            problems.append(f"eyes not open (EAR {candidate.ear:.3f})")
        if candidate.mar > self.cfg.max_mar:
            problems.append(f"mouth not neutral (MAR {candidate.mar:.3f})")
        if candidate.blink > self.cfg.max_blink:
            problems.append(f"eyeBlink too high ({candidate.blink:.2f})")
        if problems:
            self.last_failure = "; ".join(problems)
            return  # stay CALIBRATING with defaults; next window retries
        self.baseline = candidate
        self.anchor = replace(candidate)
        self.status = CALIBRATED
        self.last_failure = ""
        offset = float(np.hypot(candidate.yaw, candidate.pitch))
        self.warning = (f"camera is {offset:.0f} deg off the driver's face axis (limit {self.cfg.max_camera_offset_deg:.0f}): "
                        "eye/mouth measurements are unreliable - mount the camera in front of the driver"
                        if offset > self.cfg.max_camera_offset_deg else "")
        self._thresholds = Thresholds.from_baseline(self.baseline, self.thr_cfg, calibrated=True)
