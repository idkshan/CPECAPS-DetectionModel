"""Stage 5: overall driver-risk calculation (Table 1.6).

    Fatigue     F = FIS(eye_closure, yawning, driving duration)   in [0, 1]
    Distraction D = FIS(gaze deviation, phone use)                in [0, 1]
    Context     C = FIS(speed, erratic motion)                    in [0, 1]
    Overall     R = FIS(F, D, C)                                  in [0, 100]

R is mapped to a risk tier with fixed bounds (default 25 / 52.5 / 77.5,
the midpoints between the centroids of neighbouring output sets) and a
hysteresis band: a tier is entered when R reaches its lower bound and left
only when R falls ``tier_hysteresis`` (5) points below it, so R hovering
around a bound does not flip the tier (and the alarm) back and forth.
"""

from __future__ import annotations

from typing import Optional

from ..config import Config
from ..severity import piecewise
from ..types import ContextReading, RiskAssessment, SeverityReport
from .fuzzy import MamdaniSystem
from .rules import (
    FATIGUE_LEVEL_NAMES,
    context_rules,
    distraction_rules,
    fatigue_rules,
    fatigue_variable,
    overall_rules,
    risk_variable,
    unit_variable,
)


TIERS = ("NONE", "LOW", "MEDIUM", "HIGH")


def tier_for(risk: float, bounds) -> str:
    low, medium, high = bounds
    if risk >= high:
        return "HIGH"
    if risk >= medium:
        return "MEDIUM"
    if risk >= low:
        return "LOW"
    return "NONE"


class TierTracker:
    """Risk tier with hysteresis: up at the bound, down only below bound - margin."""

    def __init__(self, bounds, margin: float):
        self.bounds, self.margin = list(bounds), margin
        self.tier = "NONE"

    def update(self, risk: float) -> str:
        raw = tier_for(risk, self.bounds)
        cur = TIERS.index(self.tier)
        if TIERS.index(raw) > cur:
            self.tier = raw
        else:
            # step down while R is below the current tier's lower bound minus the margin
            while cur > 0 and risk < self.bounds[cur - 1] - self.margin:
                cur -= 1
            self.tier = TIERS[cur]
        return self.tier


class DriverRiskModel:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        res = cfg.risk.resolution
        u = unit_variable
        self.fatigue = MamdaniSystem("fatigue", [u("eye_closure"), u("yawning"), u("duration")], fatigue_variable(), fatigue_rules(), res)
        self.distraction = MamdaniSystem("distraction", [u("gaze"), u("phone")], u("distraction"), distraction_rules(), res)
        self.context = MamdaniSystem("context", [u("speed"), u("motion")], u("context"), context_rules(), res)
        self.overall = MamdaniSystem(
            "overall", [fatigue_variable(), u("distraction"), u("context")], risk_variable("risk"), overall_rules(), res
        )
        self._fatigue_var = fatigue_variable()
        self.tiers = TierTracker(cfg.risk.tier_bounds, cfg.risk.tier_hysteresis)

    def fatigue_level(self, fatigue: float) -> str:
        """Name of the fatigue set with the highest membership (Low / Moderate / High / Severe)."""
        mu = self._fatigue_var.fuzzify(fatigue)
        return FATIGUE_LEVEL_NAMES[max(mu, key=mu.get)]

    def normalise_context(self, ctx: ContextReading) -> dict:
        """Context sensors -> 0..1. Missing speed = assumed moving (flagged as speed_assumed = 1)."""
        c = self.cfg.context
        assumed = ctx.speed_kmh is None
        speed_kmh: Optional[float] = c.assumed_speed_kmh if assumed else ctx.speed_kmh
        return {
            "speed": piecewise(c.speed_kmh, speed_kmh),
            "motion": piecewise(c.motion_g, ctx.motion_g or 0.0),
            "duration": piecewise(c.driving_hours, ctx.driving_s / 3600.0),
            "speed_assumed": 1.0 if assumed else 0.0,
        }

    def assess(self, sev: SeverityReport, ctx: ContextReading) -> RiskAssessment:
        cx = self.normalise_context(ctx)
        inputs = {
            "eye_closure": sev.drowsiness.score,
            "yawning": sev.yawning.score,
            "gaze": sev.gaze_deviation.score,
            "phone": sev.phone_usage.score,
            **cx,
        }
        f = self.fatigue.infer({"eye_closure": inputs["eye_closure"], "yawning": inputs["yawning"], "duration": cx["duration"]})
        d = self.distraction.infer({"gaze": inputs["gaze"], "phone": inputs["phone"]})
        c = self.context.infer({"speed": cx["speed"], "motion": cx["motion"]})
        r = self.overall.infer({"fatigue": f.value, "distraction": d.value, "context": c.value})
        top = sorted((fr for fr in r.firing if fr[1] > 0), key=lambda fr: -fr[1])[:3]
        return RiskAssessment(
            fatigue=round(f.value, 4),
            distraction=round(d.value, 4),
            context=round(c.value, 4),
            risk=round(r.value, 2),
            tier=self.tiers.update(r.value),
            raw_tier=tier_for(r.value, self.cfg.risk.tier_bounds),
            fatigue_level=self.fatigue_level(f.value),
            inputs={k: round(v, 4) for k, v in inputs.items()},
            top_rules=[f"{rule.text('R')} [{alpha:.2f}]" for rule, alpha in top],
        )
