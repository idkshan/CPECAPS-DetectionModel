"""Drowsiness metrics against the thesis alertness labels (Table 1.11).

The thesis labels three states: Alert, Slightly Drowsy, Moderately Drowsy.
Every frame inside a labelled drowsiness segment carries that segment's
level; every other frame of a fully labelled session is Alert.

Reported:
  * 3-class frame confusion matrix, per-class P/R/F1, macro F1 (the mean of
    the per-class F1 of the classes present) and accuracy (= micro F1)
  * binary frame metrics: drowsy (slightly or moderately) vs alert - this
    binary F1 is the SO3 drowsiness F1
  * per labelled segment: the highest level the system reported inside it
    vs the annotated level (exact / within one level)

Frames labelled only "drowsiness" (no level) enter the binary metrics but
not the 3-class ones. "severely_drowsy" labels count as Moderately Drowsy.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

import numpy as np

from .labels import LabelSet
from .metrics import frame_metrics, multiclass_metrics

STATES = ("alert", "slightly_drowsy", "moderately_drowsy")
RANK = {"alert": 0, "slightly_drowsy": 1, "moderately_drowsy": 2}
_TO_THESIS = {"alert": "alert", "slightly_drowsy": "slightly_drowsy", "moderately_drowsy": "moderately_drowsy",
              "severely_drowsy": "moderately_drowsy"}


def predicted_state(rec: dict) -> Optional[str]:
    """Drowsiness state of one logged frame, from any log version.

    New logs carry ``behavior.drowsiness_state``. Logs of the previous
    version only have the fuzzy fatigue level ("Level 2 (moderately
    drowsy)"), which is mapped Level 1 -> slightly, Level 2/3 -> moderately.
    Older logs have neither (None).
    """
    beh = rec.get("behavior") or {}
    state = beh.get("drowsiness_state")
    if state:
        return _TO_THESIS.get(state, state)
    level = (rec.get("risk") or {}).get("drowsiness_level")
    if level:
        m = re.search(r"Level\s*(\d)", str(level))
        n = int(m.group(1)) if m else 0
        return STATES[min(n, 2)]
    return None


def truth_states(times: np.ndarray, labels: LabelSet) -> List[str]:
    """Per-frame thesis class; 'unspecified' inside a drowsiness segment without a level."""
    out = ["alert"] * len(times)
    for s, e, level in labels.drowsiness_levels:
        cls = _TO_THESIS.get(level, "unspecified")
        for i in np.nonzero((times >= s) & (times <= e))[0]:
            if out[i] == "alert" or (cls in RANK and RANK.get(out[i], -1) < RANK[cls]):
                out[i] = cls
    return out


def evaluate_states(times: np.ndarray, pred_states: Sequence[Optional[str]], labels: LabelSet, tol: float = 1.0) -> dict:
    times = np.asarray(times, dtype=float)
    truth = truth_states(times, labels)
    has_pred = all(p is not None for p in pred_states) and len(pred_states) == len(times)
    report: dict = {"labelled_segments": len(labels.drowsiness_levels)}
    truth_bin = np.array([t != "alert" for t in truth])
    if has_pred:
        pred_bin = np.array([p != "alert" for p in pred_states])
        report["binary"] = frame_metrics(pred_bin, truth_bin)
        keep = [i for i, t in enumerate(truth) if t in RANK]
        report["three_class"] = multiclass_metrics([truth[i] for i in keep], [pred_states[i] for i in keep], STATES)
        report["three_class"]["note"] = "frames; macro F1 = mean F1 of the classes present in the labels"
        report["segments"] = _segment_levels(times, pred_states, labels, tol)
    else:
        report["note"] = "this log has no drowsiness state; re-run with --replay to evaluate states"
    report["class_distribution_frames"] = {c: int(sum(1 for t in truth if t == c)) for c in (*STATES, "unspecified")}
    return report


def _segment_levels(times, pred_states, labels: LabelSet, tol: float) -> dict:
    pairs = []
    for s, e, level in labels.drowsiness_levels:
        if level not in _TO_THESIS:
            continue
        inside = np.nonzero((times >= s - tol) & (times <= e + tol))[0]
        peak = max((RANK.get(pred_states[i], 0) for i in inside), default=0)
        pairs.append({"start_s": s, "end_s": e, "true": RANK[_TO_THESIS[level]], "pred": peak})
    n = len(pairs)
    exact = sum(p["true"] == p["pred"] for p in pairs)
    near = sum(abs(p["true"] - p["pred"]) <= 1 for p in pairs)
    matrix = [[0] * 3 for _ in range(3)]
    for p in pairs:
        matrix[p["true"]][p["pred"]] += 1
    return {"segments": n, "exact": exact, "within_one": near,
            "exact_acc": round(exact / n, 4) if n else None, "within_one_acc": round(near / n, 4) if n else None,
            "confusion": matrix, "classes": list(STATES), "pairs": pairs}
