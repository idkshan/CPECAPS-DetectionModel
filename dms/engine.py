"""Stages 3-5 chained together: FrameFeatures -> alert commands.

    Stage 3  BehaviorClassifier   behaviours (dms/behaviors/*) + drowsiness state
    Stage 4  SeverityGrader       severity score per behaviour
    Stage 5  DriverRiskModel      fuzzy risk level (with tier hysteresis)
             AlertStateMachine    start / escalate / stop the buzzer (dms/alerts)

The engine is independent of the camera and the neural networks. It can be
driven by live perception (``DriverMonitor``), by features replayed from a
session log (``dms.evaluation.session.replay``), or by synthetic scenarios
(``dms/simulation.py``) - which is how the thresholds, the fuzzy rules and the
alert policy are unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .alerts import AlertStateMachine
from .classification.calibration import CALIBRATING, DriverCalibrator
from .classification.classifier import BehaviorClassifier
from .config import Config
from .risk.driver_risk import DriverRiskModel
from .severity import SeverityGrader
from .types import AlertCommand, BehaviorStatus, ContextReading, FrameFeatures, FrameStates, RiskAssessment, SeverityReport


@dataclass
class EngineOutput:
    states: FrameStates
    behavior: BehaviorStatus
    severity: SeverityReport
    risk: RiskAssessment
    alerts: List[AlertCommand]
    active_tier: str
    calibration_status: str
    context: ContextReading
    alert_state: str = "MONITORING"


class DecisionEngine:
    def __init__(self, cfg: Config, alert_id_prefix=None):
        self.cfg = cfg
        self.calibrator = DriverCalibrator(cfg.calibration, cfg.thresholds, phone_conf_limit=cfg.detector.conf)
        self.classifier = BehaviorClassifier(cfg)
        self.grader = SeverityGrader(cfg.severity, cfg.drowsiness.state_scores)
        self.risk_model = DriverRiskModel(cfg)
        self.alerts = AlertStateMachine(cfg.alerts, id_prefix=alert_id_prefix)

    def recalibrate(self) -> None:
        self.calibrator.restart()

    def step(self, feats: FrameFeatures, ctx: ContextReading) -> EngineOutput:
        thr = self.calibrator.thresholds

        # Stage 3: what is the driver doing? (thresholds on the AI evidence + timing rules)
        states = self.classifier.classify_frame(feats, thr)
        behavior = self.classifier.update(feats, states)

        # Stage 4: how serious is each behaviour?
        severity = self.grader.grade(behavior)

        # Calibration / online adaptation of the baseline
        if self.calibrator.status == CALIBRATING:
            self.calibrator.collect(feats)
        elif self._confidently_alert(feats, states, severity):
            self.calibrator.adapt(feats)
        else:
            self.calibrator.pause_adaptation()

        # Stage 5: overall risk level -> alert state machine
        risk = self.risk_model.assess(severity, ctx)
        if risk.inputs.get("motion", 0.0) >= self.cfg.context.sudden_motion_score:
            behavior.active = behavior.active + ["sudden_motion"]
        reason = self._reason(severity, risk)
        alerts = self.alerts.update(feats.t, risk.tier, behavior.active, behavior.events, reason)
        return EngineOutput(states, behavior, severity, risk, alerts, self.alerts.active_tier, self.calibrator.status, ctx,
                            self.alerts.state)

    def camera_lost(self, t: float, reason: str = "no camera frames") -> List[AlertCommand]:
        """Call when no frame arrived for AlertConfig.camera_timeout_s (see run_monitor.py)."""
        return self.alerts.camera_lost(t, reason)

    def _confidently_alert(self, f: FrameFeatures, s: FrameStates, sev: SeverityReport) -> bool:
        return (
            f.face_present
            and not s.phone_visible
            and (s.eye_evidence or 0.0) < 0.3
            and (s.mouth_evidence or 0.0) < 0.3
            and (s.pose_evidence or 0.0) < 0.25
            and sev.max_score() < 0.1
            and not self.classifier.any_episode_active()
            and f.t - self.classifier.last_event_t >= self.cfg.calibration.adapt_quiet_s
        )

    @staticmethod
    def _reason(sev: SeverityReport, risk: RiskAssessment) -> str:
        parts = [f"{name}={s.level}({s.reason})" for name, s in sev.as_dict().items() if s.level != "NONE"]
        ctx = f"F={risk.fatigue:.2f} D={risk.distraction:.2f} C={risk.context:.2f} R={risk.risk:.0f}"
        return (", ".join(parts) + " | " if parts else "") + ctx
