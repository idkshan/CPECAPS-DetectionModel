"""Data passed between the five pipeline stages.

    Stage 1  AI perception          -> PerceptionResult (FaceObservation, Detection)
    Stage 2  visual features        -> FrameFeatures     (plain numbers)
    Stage 3  threshold classifier   -> FrameStates + BehaviorStatus
    Stage 4  severity grading       -> SeverityReport
    Stage 5  driver-risk fusion     -> RiskAssessment -> AlertCommand
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

SEVERITY_LEVELS = ("NONE", "MILD", "MODERATE", "SEVERE")
RISK_TIERS = ("NONE", "LOW", "MEDIUM", "HIGH")
BEHAVIORS = ("drowsiness", "yawning", "gaze_deviation", "phone_usage")


# ----------------------------------------------------------------- Stage 1
@dataclass
class FaceObservation:
    """Output of the MediaPipe Face Landmarker for one face."""

    landmarks: np.ndarray  # (478, 3): x, y in pixels, z relative depth (pixel scale)
    blendshapes: Dict[str, float]  # 52 learned facial-expression coefficients in [0, 1]
    bbox: Box


@dataclass
class Detection:
    """One YOLO bounding box."""

    label: str
    confidence: float
    box: Box
    role: Optional[str] = None  # phone | eye_open | eye_closed | yawn | no_yawn
    in_driver_roi: bool = True


@dataclass
class PerceptionResult:
    frame_size: Tuple[int, int]  # (width, height)
    face: Optional[FaceObservation]
    detections: List[Detection] = field(default_factory=list)
    eye_closed_prob: Optional[float] = None  # optional learned eye-state classifier
    timings_ms: Dict[str, float] = field(default_factory=dict)
    detector_roles: Set[str] = field(default_factory=set)  # roles the YOLO model can detect


# ----------------------------------------------------------------- Stage 2
@dataclass
class FrameFeatures:
    """Numeric visual features of one frame (no decisions yet)."""

    t: float
    face_present: bool
    ear: Optional[float] = None
    ear_left: Optional[float] = None
    ear_right: Optional[float] = None
    mar: Optional[float] = None
    yaw: Optional[float] = None  # degrees, + = face turned toward image right
    pitch: Optional[float] = None  # degrees, + = face tilted up
    roll: Optional[float] = None
    blink: Optional[float] = None  # mean eyeBlink blendshape (learned)
    jaw_open: Optional[float] = None  # jawOpen blendshape (learned)
    eye_closed_prob: Optional[float] = None  # learned eye-state classifier
    eye_closed_yolo: Optional[float] = None  # YOLO driver-state detector: 0..1, >= 0.5 = closed eyes found
    yawn_yolo: Optional[float] = None  # YOLO driver-state detector: 0..1, >= 0.5 = yawning mouth found
    phone_conf: float = 0.0  # best YOLO phone confidence inside the driver ROI
    phone_boxes: int = 0
    eye_width_px: Optional[float] = None  # mean eye-corner distance in pixels (landmark reliability, SO1 resolution)
    look_down: Optional[float] = None  # mean eyeLookDown blendshape (eyes looking down without closing)

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------- Stage 3
@dataclass
class FrameStates:
    """Per-frame binary states after thresholding the fused evidence."""

    eyes_closed: Optional[bool] = None
    eye_evidence: Optional[float] = None
    eye_sources: Dict[str, float] = field(default_factory=dict)
    mouth_open: Optional[bool] = None
    mouth_evidence: Optional[float] = None
    mouth_sources: Dict[str, float] = field(default_factory=dict)
    head_away: Optional[bool] = None
    pose_evidence: Optional[float] = None
    phone_visible: bool = False
    face_lost_s: float = 0.0
    face_status: str = "ok"  # ok | lost_turning (counts as gaze away) | lost (unknown) | absent (no driver)
    eyes_not_judged: Optional[str] = None  # why the eye state is unknown: head_away | phone | yawn | quality | no_face
    frame_gap: bool = False  # this frame follows a camera stall (> quality.max_frame_gap_s)


@dataclass
class BehaviorStatus:
    """Temporal interpretation of the frame states (persistence + windows)."""

    closure_s: float = 0.0  # longest eye closure within the hold window
    perclos: Optional[float] = None  # fraction of closed-eye time in the PERCLOS window
    mouth_open_s: float = 0.0
    yawns_in_window: int = 0
    gaze_away_s: float = 0.0
    offroad_ratio: float = 0.0
    phone_s: float = 0.0
    long_closures: int = 0  # closures >= temporal.closure_event_s inside the drowsiness window
    blink_rate_per_min: Optional[float] = None
    eye_level: int = 0  # eye-based drowsiness level 0/1/2 (input of the risk model)
    drowsiness_state: str = "alert"  # alert | slightly_drowsy | moderately_drowsy (thesis Table 1.11)
    # Behaviour "detected" flags (used for F1 evaluation, SO3)
    drowsy: bool = False  # drowsiness_state != alert
    eye_closure: bool = False  # eyes closed >= temporal.closure_event_s right now
    yawning: bool = False
    gaze_deviation: bool = False
    phone_usage: bool = False
    events: List[str] = field(default_factory=list)  # new events on this frame
    active: List[str] = field(default_factory=list)  # behaviours happening NOW (sustain an alert)
    no_face: bool = False  # face missing for longer than face_loss_max_s
    face_status: str = "ok"
    data_valid: bool = True  # False: this frame's eye/mouth measurements are unknown (no face, poor quality, gap)


# ----------------------------------------------------------------- Stage 4
@dataclass
class Severity:
    score: float  # 0..1, fed to the fuzzy system
    level: str  # NONE / MILD / MODERATE / SEVERE (for explanation)
    reason: str = ""


@dataclass
class SeverityReport:
    drowsiness: Severity
    yawning: Severity
    gaze_deviation: Severity
    phone_usage: Severity

    def as_dict(self) -> Dict[str, Severity]:
        return {b: getattr(self, b) for b in BEHAVIORS}

    def max_score(self) -> float:
        return max(s.score for s in self.as_dict().values())


# ----------------------------------------------------------------- Stage 5
@dataclass
class ContextReading:
    speed_kmh: Optional[float] = None
    motion_g: Optional[float] = None
    driving_s: float = 0.0  # continuous driving time (resets after a long stop)
    speed_source: str = "unknown"  # gps | fixed | assumed | unknown - "assumed" = GPS missing or stale
    gps_age_s: Optional[float] = None  # age of the GPS fix used
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@dataclass
class RiskAssessment:
    fatigue: float  # F in [0, 1]
    distraction: float  # D in [0, 1]
    context: float  # C in [0, 1]
    risk: float  # R in [0, 100]
    tier: str  # NONE / LOW / MEDIUM / HIGH (with hysteresis, see RiskConfig.tier_hysteresis)
    fatigue_level: str = "Low"  # fuzzy fatigue output: Low / Moderate / High / Severe
    raw_tier: str = "NONE"  # tier from R without hysteresis
    inputs: Dict[str, float] = field(default_factory=dict)
    top_rules: List[str] = field(default_factory=list)


@dataclass
class AlertCommand:
    """One entry of the alert stream (also written to alerts.jsonl).

    ``event`` says what happened:
      start / escalate / deescalate / stop      alert lifecycle (state changes)
      beep                                      one sound of the current pattern
      keepalive                                 "still sounding" (device fail-safe)
      push                                      push notification to the supervisor
      unavailable / available                   camera link lost / restored
    ``action`` is the buzzer mode the device should be in afterwards:
      soft_beep (LOW) | repeating_buzzer (MEDIUM) | alarm (HIGH) | off
    """

    t: float
    tier: str
    action: str
    push: bool
    reason: str
    new_episode: bool  # True on "start" (kept for older evaluation scripts)
    event: str = "beep"
    id: str = ""  # unique id (de-duplication on the phone / Firebase)
    alert_id: str = ""  # id of the alert episode this entry belongs to
    behaviors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameResult:
    t: float
    frame_index: int
    perception: Optional[PerceptionResult]
    features: FrameFeatures
    states: FrameStates
    behavior: BehaviorStatus
    severity: SeverityReport
    risk: RiskAssessment
    alerts: List[AlertCommand]
    active_tier: str
    calibration_status: str
    context: Optional[ContextReading] = None
    timings_ms: Dict[str, float] = field(default_factory=dict)
    alert_state: str = "MONITORING"
