"""Print the fuzzy rule bases and the behaviour -> alarm-tier onset table (Markdown).

    python scripts/export_rule_tables.py > docs/generated_rule_tables.md

The onset table answers "how long / how often must a behaviour occur, on its
own, before each alarm tier is reached?" by scanning each measured quantity
through Stages 4-5 with every other behaviour at zero and normal driving
context (speed 40 km/h, no erratic motion, start of trip).
"""

import argparse
import sys

import numpy as np

import _bootstrap  # noqa: F401
from dms.config import load_config
from dms.risk.driver_risk import DriverRiskModel
from dms.risk.rules import all_rule_blocks, rule_table_markdown
from dms.severity import SeverityGrader, level_for
from dms.types import BehaviorStatus, ContextReading

OUTPUT_NAMES = {"fatigue": "F", "distraction": "D", "context": "C", "overall": "R (alarm tier)"}


def onset_table(cfg) -> str:
    grader = SeverityGrader(cfg.severity, cfg.drowsiness.state_scores)
    ctx = ContextReading(speed_kmh=40.0, motion_g=0.0, driving_s=0.0)
    scans = [
        ("Eye closure (continuous)", "closure_s", np.arange(0, 3.001, 0.05), "s"),
        ("Eye drowsiness level (1 = slightly, 2 = moderately; PERCLOS 8 % / 15 %)", "eye_level", np.arange(0, 3), "count"),
        ("Yawns in 5 min", "yawns_in_window", np.arange(0, 5), "count"),
        ("Looking away (continuous)", "gaze_away_s", np.arange(0, 5.001, 0.05), "s"),
        ("Eyes-off-road share (30 s)", "offroad_ratio", np.arange(0, 0.6001, 0.005), "ratio"),
        ("Phone visible (continuous)", "phone_s", np.arange(0, 5.001, 0.05), "s"),
    ]
    rows = ["| Behaviour (alone) | Severity MODERATE from | LOW tier from | MEDIUM tier from | HIGH tier from |", "|---|---|---|---|---|"]
    for label, attr, values, unit in scans:
        model = DriverRiskModel(cfg)  # fresh model: the risk tier has hysteresis (memory)
        first = {}
        for v in values:
            sev = grader.grade(BehaviorStatus(**{attr: int(v) if unit == "count" else float(v)}))
            first.setdefault("MODERATE" if level_for(sev.max_score()) in ("MODERATE", "SEVERE") else "", v)
            _record_tiers(first, model.assess(sev, ctx).tier, v)
        rows.append(f"| {label} | " + " | ".join(_fmt(first.get(k), unit) for k in ("MODERATE", "LOW", "MEDIUM", "HIGH")) + " |")

    first = {}
    model = DriverRiskModel(cfg)
    for g in np.arange(0, 0.8001, 0.01):
        _record_tiers(first, model.assess(grader.grade(BehaviorStatus()), ContextReading(40.0, float(g), 0.0)).tier, g)
    rows.append("| Erratic motion (alert driver) | - | " + " | ".join(_fmt(first.get(k), "g") for k in ("LOW", "MEDIUM", "HIGH")) + " |")
    return "\n".join(rows)


TIERS = ("NONE", "LOW", "MEDIUM", "HIGH")


def _record_tiers(first: dict, tier: str, value) -> None:
    """Remember the first scanned value at which each tier (or a higher one) is reached."""
    for t in TIERS[1:]:
        if TIERS.index(tier) >= TIERS.index(t):
            first.setdefault(t, value)


def _fmt(value, unit: str) -> str:
    if value is None:
        return "-"
    if unit == "count":
        return f"{value:.0f}"
    if unit == "ratio":
        return f"{100 * value:.1f} %"
    return f"{value:.2f} {unit}"


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    print("## Behaviour-to-alarm onset (single behaviour, speed 40 km/h)\n")
    print(onset_table(cfg))
    for name, rules in all_rule_blocks().items():
        print(f"\n## {name.capitalize()} block ({len(rules)} rules)\n")
        print(rule_table_markdown(rules, OUTPUT_NAMES[name]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
