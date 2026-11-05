"""Configuration for the driver-monitoring pipeline.

Every tunable number used by the five pipeline stages lives here so that the
thesis can cite one table of parameters and experiments can change them
through ``configs/*.yaml`` without touching code.  The dataclass defaults are
the reference values; ``configs/default.yaml`` mirrors them (a unit test keeps
the two in sync).

Piecewise-linear "anchor" lists are ``[[x0, y0], [x1, y1], ...]`` and are
evaluated with linear interpolation (values outside the range are clamped to
the first/last y).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import yaml

Anchors = List[List[float]]


# --------------------------------------------------------------------------
# Stage 1 - AI perception models
# --------------------------------------------------------------------------
@dataclass
class FaceModelConfig:
    """MediaPipe Face Landmarker (face detector + landmark/blendshape networks)."""

    model_path: str = "models/face_landmarker.task"
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class ObjectDetectorConfig:
    """YOLO object detector: phones, and (with a driver-state model) eyes and yawns.

    Any Ultralytics detection model can be used (YOLOv8, YOLO11, YOLO26, or a
    fine-tuned checkpoint). The default is the paper's baseline, YOLOv8-nano
    pretrained on COCO, which only knows "cell phone". A driver-state model
    trained with build_driver_state_dataset.py + train_yolo.py also detects
    open eyes, closed eyes and yawning mouths; those detections become AI
    evidence that the Stage-3 thresholds interpret. The default does not
    imply it is the best choice (see benchmark_yolo.py).
    """

    model_path: str = "models/yolov8n.pt"
    imgsz: int = 640
    conf: float = 0.35  # a box counts from this confidence (phone episodes start here; eye / yawn boxes)
    phone_stay_conf: float = 0.20  # an ongoing phone episode continues down to this confidence (hysteresis)
    iou: float = 0.5
    device: str = "cpu"
    every_n_frames: int = 1  # run YOLO every N frames and reuse results in between
    phone_class_names: List[str] = field(
        default_factory=lambda: ["cell phone", "phone", "mobile phone", "cellphone", "mobile_phone"]
    )
    # Driver-state classes (matched by name, case-insensitive)
    eye_open_class_names: List[str] = field(default_factory=lambda: ["eye_open", "open_eye", "open", "eyes_open"])
    eye_closed_class_names: List[str] = field(default_factory=lambda: ["eye_closed", "closed_eye", "closed", "eyes_closed"])
    yawn_class_names: List[str] = field(default_factory=lambda: ["yawn", "yawning"])
    no_yawn_class_names: List[str] = field(default_factory=lambda: ["no_yawn", "mouth_closed", "not_yawning"])
    face_part_margin: float = 0.15  # eye / mouth boxes must lie inside the face box enlarged by 15 %
    # Spatial filter: a phone only counts if its centre lies near the driver,
    # i.e. within +/- roi_width_factor face-widths of the face centre and no
    # higher than roi_above_factor face-heights above the top of the face.
    roi_width_factor: float = 2.5
    roi_above_factor: float = 0.5


@dataclass
class EyeStateModelConfig:
    """Optional learned eye-state classifier (e.g. YOLO-cls trained on MRL Eye).

    Disabled when ``model_path`` is null. When enabled, both eye regions are
    cropped using the face landmarks and classified as open/closed; the
    probability of "closed" becomes a third eye-closure evidence source.
    """

    model_path: Optional[str] = None
    closed_class_names: List[str] = field(default_factory=lambda: ["closed", "close", "closed_eye", "closed_eyes"])
    crop_scale: float = 1.8
    grayscale: bool = True  # MRL Eye images are infrared/greyscale
    imgsz: Optional[int] = None  # None -> use the size the model was trained with


# --------------------------------------------------------------------------
# Stage 3 - threshold-based classification
# --------------------------------------------------------------------------
@dataclass
class EvidenceWeights:
    """Weights used to fuse several evidence sources into one score in [0, 1].

    Sources that are unavailable on a frame (e.g. no eye-state model) are
    skipped and the remaining weights are renormalised. Setting a weight to 0
    removes that source, which is how ablation modes (EAR-only, AI-only,
    fused) are produced.
    """

    eye_ear: float = 0.5
    eye_blendshape: float = 0.5
    eye_cnn: float = 0.5
    eye_yolo: float = 0.5  # YOLO driver-state detector (eye_open / eye_closed boxes)
    mouth_mar: float = 0.5
    mouth_blendshape: float = 0.5
    mouth_yolo: float = 0.5  # YOLO driver-state detector (yawn boxes)


@dataclass
class ThresholdConfig:
    """Per-driver thresholds, expressed relative to the calibrated baseline."""

    ear_closed_ratio: float = 0.70  # eyes "closed" when EAR < 0.70 x baseline EAR
    blink_closed_delta: float = 0.35  # eyeBlink blendshape > baseline + 0.35
    mar_open_delta: float = 0.40  # mouth "wide open" when MAR > baseline + 0.40
    jaw_open_delta: float = 0.35  # jawOpen blendshape > baseline + 0.35
    yaw_deg: float = 30.0  # |yaw - baseline yaw| beyond this = head turned away
    pitch_down_deg: float = 20.0  # head lowered this far below baseline = looking down
    pitch_up_deg: float = 25.0
    decision_level: float = 0.5  # fused evidence >= this -> state is ON
    uncalibrated_pose_scale: float = 1.5  # widen pose limits while uncalibrated


@dataclass
class CalibrationDefaults:
    """Population defaults used until a valid per-driver baseline exists."""

    ear: float = 0.28
    mar: float = 0.05
    yaw: float = 0.0
    pitch: float = 0.0
    blink: float = 0.15
    jaw: float = 0.03


@dataclass
class CalibrationConfig:
    enabled: bool = True
    duration_s: float = 5.0  # length of the calibration window
    min_samples: int = 25  # minimum valid frames in the window
    max_pose_std_deg: float = 5.0  # head must be steady (robust std of yaw/pitch)
    min_ear: float = 0.18  # eyes must be open
    max_mar: float = 0.25  # mouth must be closed / neutral
    max_blink: float = 0.5
    # The camera should face the driver. A baseline head pose further than this
    # from straight-on is accepted but flagged: eye and mouth geometry become
    # unreliable (the S01 recording was calibrated at 43 deg yaw).
    max_camera_offset_deg: float = 25.0
    # Online adaptation (Section 1.7.9): slow exponential update over frames in
    # which the driver is confidently alert and facing forward.
    adapt: bool = True
    adapt_tau_s: float = 120.0  # time constant of the running baseline
    adapt_quiet_s: float = 10.0  # no behaviour episode for this long before adapting
    max_ear_drift: float = 0.20  # running EAR baseline stays within +/-20 % of calibration
    max_mar_drift: float = 0.10  # absolute
    max_pose_drift_deg: float = 10.0
    defaults: CalibrationDefaults = field(default_factory=CalibrationDefaults)


@dataclass
class TemporalConfig:
    """Persistence rules. All durations are in seconds, never in frames."""

    closure_gap_s: float = 0.15  # tolerated flicker inside one eye-closure episode
    closure_event_s: float = 0.5  # closures shorter than this are ordinary blinks
    yawn_min_s: float = 1.5  # mouth must stay wide open this long (rejects speech)
    yawn_gap_s: float = 0.3
    gaze_min_s: float = 2.0  # glances shorter than this are normal mirror checks
    gaze_gap_s: float = 0.3
    phone_confirm_s: float = 3.0  # Section 1.7.15: phone visible >= 3 s
    phone_gap_s: float = 1.0  # S01: real phone use was detected in only ~50-80 % of frames; 0.5 s split episodes
    offroad_window_s: float = 30.0
    yawn_window_s: float = 300.0
    closure_hold_s: float = 3.0  # keep reporting a long closure briefly after the eyes reopen
    gaze_hold_s: float = 1.0
    phone_hold_s: float = 1.0
    face_loss_as_gaze_away: bool = True  # face vanishing (head turned far away) counts as gaze away
    face_loss_max_s: float = 10.0  # after this the system reports "no face" instead
    # Eye-state guards: eyes are not judged while the driver looks at a phone
    # (lowered eyelids look closed), while the mouth is opening for a yawn
    # (eyes squint) and for a short time after it closes again.
    suspend_eyes_when_phone: bool = True
    yawn_eye_guard_level: float = 0.3  # mouth evidence at/above this suspends eye judgement
    yawn_eye_guard_s: float = 0.5


@dataclass
class QualityConfig:
    """When is a frame good enough to judge the eyes / mouth / head?

    Missing or unreliable measurements are treated as UNKNOWN - never as eyes
    closed or head turned - so camera problems cannot raise a drowsiness alarm.
    """

    max_frame_gap_s: float = 1.0  # a longer pause between frames (camera stall) ends every ongoing episode
    min_eye_width_px: float = 12.0  # eye narrower than this in the image -> eye state not judged (face too small/far)
    # A vanishing face counts as "head turned away" only if the head was already
    # turning (pose evidence >= this within the last face_loss_lookback_s).
    # Otherwise (hand over the camera, glare, dropout) the gaze is unknown.
    face_loss_turning_evidence: float = 0.3
    face_loss_lookback_s: float = 0.5
    # Learned eyeLookDown blendshape at/above this drops the geometric EAR from
    # the eye evidence (eyelids lower when the eyes look down at the dashboard).
    # 1.0 = off. Unvalidated: tune on the validation split before enabling.
    look_down_ear_guard: float = 1.0


@dataclass
class DrowsinessConfig:
    """Drowsiness STATE (thesis Table 1.11: alert / slightly / moderately drowsy).

    Criteria are checked over a look-back window. The defaults mirror the
    severity anchors (PERCLOS 8 % / 15 %, closure 0.8 s / 1.5 s) and the
    Table 1.11 descriptions ("frequent yawning", "slow / frequent blinks").
    They are starting points: tune them on the VALIDATION participants.
    """

    window_s: float = 60.0  # look-back for PERCLOS, long closures and the longest closure
    perclos_min_coverage_s: float = 30.0  # PERCLOS needs this much observed time since the last reset
    slight_perclos: float = 0.08
    moderate_perclos: float = 0.15
    slight_closure_s: float = 0.8  # a closure this long inside the window -> slightly drowsy
    moderate_closure_s: float = 1.5  # microsleep -> moderately drowsy
    slight_long_closures: int = 2  # this many closures >= temporal.closure_event_s ("slow blinks") in the window
    moderate_long_closures: int = 4
    slight_yawns: int = 2  # yawns within temporal.yawn_window_s ("frequent yawning")
    yawn_hold_s: float = 60.0  # ... and the last yawn no longer ago than this (frequent AND current)
    moderate_yawns: int = 0  # 0 = yawning alone never makes "moderately drowsy"
    yawns_plus_eyes_moderate: bool = True  # frequent yawning AND a slight eye sign -> moderately drowsy
    blink_rate_slight_per_min: float = 0.0  # 0 = off; frequent blinking needs validation data first
    recovery_s: float = 20.0  # step down one level after this long with no new eye sign
    state_scores: List[float] = field(default_factory=lambda: [0.0, 0.45, 0.75])  # alert / slightly / moderately -> severity


# --------------------------------------------------------------------------
# Stage 4 - severity grading
# --------------------------------------------------------------------------
@dataclass
class SeverityConfig:
    """Anchors mapping measured quantities to a severity score in [0, 1].

    The fuzzy sets of Table 1.7 cross at 1/3 (Low -> Moderate) and at 0.65
    (Moderate -> High). Those crossovers are where a behaviour's severity
    level - and, for a single behaviour, the alarm tier - changes, so every
    table places 0.34 and 0.66 at the intended limits, for example:
      gaze_s     2.0 s away   -> 0.34 (MODERATE)  3.0 s -> 0.66 (SEVERE)
      phone_s    < 3 s        -> <= 0.30          3.0 s -> 0.70 (SEVERE)
    PERCLOS no longer has its own table: it enters through the drowsiness
    state (section ``drowsiness``: 8 % slightly, 15 % moderately drowsy).
    """

    closure_s: Anchors = field(default_factory=lambda: [[0.0, 0.0], [0.4, 0.0], [0.8, 0.34], [1.5, 0.66], [2.0, 0.9], [3.0, 1.0]])
    yawn_count: Anchors = field(default_factory=lambda: [[0.0, 0.0], [1.0, 0.4], [2.0, 0.5], [3.0, 0.7], [4.0, 0.9]])
    gaze_s: Anchors = field(default_factory=lambda: [[0.0, 0.0], [1.0, 0.0], [2.0, 0.34], [3.0, 0.66], [4.0, 0.9], [5.0, 1.0]])
    offroad_ratio: Anchors = field(default_factory=lambda: [[0.0, 0.0], [0.1, 0.0], [0.25, 0.34], [0.4, 0.66], [0.6, 1.0]])
    phone_s: Anchors = field(default_factory=lambda: [[0.0, 0.0], [1.0, 0.0], [2.99, 0.3], [3.0, 0.7], [5.0, 1.0]])


# --------------------------------------------------------------------------
# Stage 5 - driver-risk fusion and alerts
# --------------------------------------------------------------------------
@dataclass
class ContextConfig:
    # stopped (< ~10 km/h) -> Low, normal driving 20-60 km/h -> Moderate, >= 80 km/h -> High
    speed_kmh: Anchors = field(
        default_factory=lambda: [[0.0, 0.0], [5.0, 0.1], [10.0, 0.34], [20.0, 0.5], [60.0, 0.5], [80.0, 0.66], [100.0, 0.9], [120.0, 1.0]]
    )
    # peak horizontal acceleration: >= 0.25 g Moderate, >= 0.45 g (harsh brake / swerve) High
    motion_g: Anchors = field(default_factory=lambda: [[0.0, 0.0], [0.1, 0.0], [0.25, 0.34], [0.45, 0.66], [0.6, 1.0]])
    # continuous driving: >= 1 h Moderate, >= 2 h High
    driving_hours: Anchors = field(default_factory=lambda: [[0.0, 0.0], [1.0, 0.34], [2.0, 0.66], [3.0, 1.0]])
    # When GPS speed is unavailable the vehicle is assumed to be moving, so a
    # missing sensor can never suppress an alert. The assumption is flagged in
    # every context reading (speed_source = "assumed") - never silent.
    assumed_speed_kmh: float = 40.0
    gps_stale_s: float = 3.0  # a fix older than this (3 missed 1-Hz updates) is not used
    moving_speed_kmh: float = 5.0  # below this the vehicle counts as stopped (driving-duration clock pauses)
    break_reset_s: float = 900.0  # a stop this long (15 min) resets the continuous-driving duration
    sudden_motion_score: float = 0.65  # context motion score (0..1) treated as an ongoing "sudden motion" event


@dataclass
class RiskConfig:
    tier_bounds: List[float] = field(default_factory=lambda: [25.0, 52.5, 77.5])  # R -> LOW / MEDIUM / HIGH
    tier_hysteresis: float = 5.0  # a tier is left only when R falls this far below its lower bound
    resolution: int = 201  # samples on each output universe for centroid defuzzification


@dataclass
class AlertConfig:
    """Alert state machine (SO5). See dms/alerts/state_machine.py for the full policy.

    An alert SOUNDS only while a behaviour is actually happening (eyes closed,
    yawning, looking away, phone, sudden motion) or shortly after it. When the
    driver is back to normal for ``clear_s`` the alert stops with an explicit
    STOP command, even if the assessed risk level is still elevated by recent
    history (PERCLOS, yawn count, off-road time). It sounds again only for a
    new behaviour or a higher risk tier.
    """

    activate_s: float = 0.3  # the risk tier must hold this long before an alert starts (filters spikes)
    clear_s: float = 2.0  # driver back to normal this long -> the alert stops
    min_alert_s: float = 2.0  # an alert lasts at least this long once started
    release_s: float = 3.0  # risk tier must stay lower this long before the alert steps down a tier
    min_interval_s: float = 5.0  # minimum gap between LOW soft beeps
    medium_repeat_s: float = 2.0  # repeating buzzer period
    medium_max_repeats: int = 3  # MEDIUM burst: buzzes per trigger when no behaviour keeps it going (0 = no minimum)
    high_repeat_s: float = 1.0  # repeated alarm period
    high_max_repeats: int = 0  # HIGH burst: alarms per trigger when no behaviour keeps it going (0 = no minimum)
    push_interval_s: float = 60.0  # at most one push notification per this many seconds
    reminder_s: float = 0.0  # >0: re-alert while the risk stays >= MEDIUM without new events (0 = off)
    escalate_closure_s: float = 0.0  # >0: eyes closed this long during an alert -> HIGH + push (0 = off)
    keepalive_s: float = 1.0  # repeat the current buzzer mode this often (device stops if keepalives stop)
    camera_timeout_s: float = 3.0  # no frame this long -> "monitoring unavailable" (alarm stops, downtime starts)


@dataclass
class Config:
    face: FaceModelConfig = field(default_factory=FaceModelConfig)
    detector: ObjectDetectorConfig = field(default_factory=ObjectDetectorConfig)
    eye_model: EyeStateModelConfig = field(default_factory=EyeStateModelConfig)
    evidence: EvidenceWeights = field(default_factory=EvidenceWeights)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    drowsiness: DrowsinessConfig = field(default_factory=DrowsinessConfig)
    severity: SeverityConfig = field(default_factory=SeverityConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


EYE_MODES = {
    "fused": {"eye_ear": 0.5, "eye_blendshape": 0.5, "eye_cnn": 0.5, "eye_yolo": 0.5},
    "ear": {"eye_ear": 1.0, "eye_blendshape": 0.0, "eye_cnn": 0.0, "eye_yolo": 0.0},
    "blendshape": {"eye_ear": 0.0, "eye_blendshape": 1.0, "eye_cnn": 0.0, "eye_yolo": 0.0},
    "cnn": {"eye_ear": 0.0, "eye_blendshape": 0.0, "eye_cnn": 1.0, "eye_yolo": 0.0},
    "yolo": {"eye_ear": 0.0, "eye_blendshape": 0.0, "eye_cnn": 0.0, "eye_yolo": 1.0},
    "ear+yolo": {"eye_ear": 0.5, "eye_blendshape": 0.0, "eye_cnn": 0.0, "eye_yolo": 0.5},
}
MOUTH_MODES = {
    "fused": {"mouth_mar": 0.5, "mouth_blendshape": 0.5, "mouth_yolo": 0.5},
    "mar": {"mouth_mar": 1.0, "mouth_blendshape": 0.0, "mouth_yolo": 0.0},
    "blendshape": {"mouth_mar": 0.0, "mouth_blendshape": 1.0, "mouth_yolo": 0.0},
    "yolo": {"mouth_mar": 0.0, "mouth_blendshape": 0.0, "mouth_yolo": 1.0},
    "mar+yolo": {"mouth_mar": 0.5, "mouth_blendshape": 0.0, "mouth_yolo": 0.5},
}


def apply_evidence_modes(cfg: "Config", eye_mode: Optional[str] = None, mouth_mode: Optional[str] = None) -> "Config":
    """Ablation presets: geometric-only, learned-only, or fused evidence."""
    if eye_mode:
        _merge(cfg.evidence, EYE_MODES[eye_mode])
    if mouth_mode:
        _merge(cfg.evidence, MOUTH_MODES[mouth_mode])
    return cfg


# Keys that existed in older versions: still accepted (and ignored) so that old
# copies of configs/default.yaml keep loading. Value = what replaced them.
DEPRECATED_KEYS = {
    "temporal.perclos_window_s": "drowsiness.window_s",
    "temporal.perclos_min_coverage_s": "drowsiness.perclos_min_coverage_s",
    "severity.perclos": "drowsiness.slight_perclos / moderate_perclos",
}


def _merge(obj: Any, overrides: dict, path: str = "") -> None:
    """Recursively apply ``overrides`` onto dataclass ``obj`` (unknown keys are errors)."""
    known = {f.name: f for f in dataclasses.fields(obj)}
    for key, value in overrides.items():
        if f"{path}{key}" in DEPRECATED_KEYS and key not in known:
            print(f"[config] '{path}{key}' is no longer used (see {DEPRECATED_KEYS[path + key]})")
            continue
        if key not in known:
            raise KeyError(f"Unknown config key '{path}{key}'. Valid keys: {sorted(known)}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"Config key '{path}{key}' must be a mapping")
            _merge(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, value)


def load_config(path: Optional[str | Path] = None, overrides: Optional[dict] = None) -> Config:
    """Build a :class:`Config` from defaults, an optional YAML file, and overrides."""
    cfg = Config()
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _merge(cfg, data)
    if overrides:
        _merge(cfg, overrides)
    return cfg
