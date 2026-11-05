"""Threshold tuning on the VALIDATION split (never on the test split).

Coordinate search: one parameter at a time is varied over a short list of
candidate values while the others stay fixed; the best value is kept and the
next parameter is tried; the whole round is repeated. Every candidate is
scored by replaying Stages 3-5 on the logged features of the validation
sessions (fast - the neural networks are not re-run).

Objective: the mean of the primary F1 of the chosen behaviours, among the
configurations whose unnecessary alarms stay <= the SO5 limit (0.5 per
non-event hour). Ties go to fewer alarms.

Why coordinate search and not a full grid: 5 parameters x 3 values would be
243 replays of every session; one round of coordinate search needs ~15.
The price is that interactions between parameters are explored only
partially - say so in the thesis if you use it.
"""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from ..config import Config, _merge
from .labels import LabelSet
from .session import evaluate_frames, pool, replay

# Parameters that can be tuned by replay (they act on logged Stage-2 features).
DEFAULT_SPACE: Dict[str, Dict[str, List]] = {
    "drowsiness": {
        "thresholds.ear_closed_ratio": [0.65, 0.70, 0.75],
        "drowsiness.slight_perclos": [0.06, 0.08, 0.10],
        "drowsiness.moderate_perclos": [0.12, 0.15, 0.20],
        "drowsiness.slight_long_closures": [2, 3],
        "drowsiness.recovery_s": [10.0, 20.0, 30.0],
    },
    "yawning": {
        "thresholds.mar_open_delta": [0.30, 0.40, 0.50],
        "temporal.yawn_min_s": [1.0, 1.5, 2.0],
    },
    "gaze_deviation": {
        "thresholds.yaw_deg": [25.0, 30.0, 35.0],
        "thresholds.pitch_down_deg": [15.0, 20.0, 25.0],
        "temporal.gaze_min_s": [1.5, 2.0, 2.5],
    },
    "phone_usage": {
        "temporal.phone_gap_s": [0.5, 1.0, 1.5],
        "detector.phone_stay_conf": [0.20, 0.25, 0.35],
    },
}


def nested(path: str, value) -> dict:
    out: dict = {}
    cur = out
    keys = path.split(".")
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value
    return out


def apply(cfg: Config, params: Dict[str, object]) -> Config:
    cfg = copy.deepcopy(cfg)
    for path, value in params.items():
        _merge(cfg, nested(path, value))
    return cfg


@dataclass
class Trial:
    params: Dict[str, object]
    score: float
    alarms_per_hour: float
    alarms: int
    f1: Dict[str, float] = field(default_factory=dict)
    feasible: bool = True


def _score_one(args) -> dict:
    frames, labels, cfg, tol = args
    f, a = replay(frames, cfg)
    return evaluate_frames(f, a, labels, tol)


def evaluate_config(sessions: Sequence[Tuple[list, LabelSet, str]], cfg: Config, behaviors: Sequence[str],
                    tol: float = 1.0, workers: int = 1, max_alarms_per_hour: float = 0.5) -> Trial:
    jobs = [(frames, labels, cfg, tol) for frames, labels, _ in sessions]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            reports = list(ex.map(_score_one, jobs))
    else:
        reports = [_score_one(j) for j in jobs]
    pooled = pool(reports, [p for _, _, p in sessions])
    f1 = {b: pooled[b]["primary_f1"] for b in behaviors if b in pooled}
    rate = pooled["alarms"]["alarms"]["per_non_event_hour"] or 0.0
    score = sum(f1.values()) / max(1, len(f1))
    return Trial({}, score, rate, pooled["alarms"]["alarms"]["count"], f1, rate <= max_alarms_per_hour)


def better(a: Trial, b: Trial) -> bool:
    """Is trial a better than trial b?"""
    if a.feasible != b.feasible:
        return a.feasible
    if abs(a.score - b.score) > 1e-9:
        return a.score > b.score
    return a.alarms < b.alarms


def coordinate_search(sessions, base: Config, space: Dict[str, List], behaviors: Sequence[str], rounds: int = 2,
                      tol: float = 1.0, workers: int = 1, log=print) -> Tuple[Dict[str, object], Trial, List[Trial]]:
    current: Dict[str, object] = {}
    best = evaluate_config(sessions, base, behaviors, tol, workers)
    best.params = {}
    history = [best]
    log(f"start: score {best.score:.3f} {best.f1} alarms/h {best.alarms_per_hour:.2f}")
    for r in range(rounds):
        changed = False
        for path, values in space.items():
            for v in values:
                if current.get(path, _get(base, path)) == v:
                    continue
                trial_params = {**current, path: v}
                trial = evaluate_config(sessions, apply(base, trial_params), behaviors, tol, workers)
                trial.params = trial_params
                history.append(trial)
                log(f"round {r + 1}: {path}={v}: score {trial.score:.3f} alarms/h {trial.alarms_per_hour:.2f}"
                    + ("" if trial.feasible else " (too many alarms)"))
                if better(trial, best):
                    best, current, changed = trial, trial_params, True
        if not changed:
            break
    return current, best, history


def _get(cfg: Config, path: str):
    obj = cfg
    for k in path.split("."):
        obj = getattr(obj, k)
    return obj
