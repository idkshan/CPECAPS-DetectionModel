"""Detection metrics: frame level, 1-second segment level, and event level.

Three views of the same predictions, because each answers a different question:

* frame level     - every frame is TP / FP / FN / TN. Gives the confusion
                    matrix, false-positive rate and false-negative rate. Recall
                    is limited by the confirmation delays (a yawn is confirmed
                    only after 1.5 s), so it is the strictest view.
* segment level   - the timeline is cut into 1-s segments; a segment is
                    positive if any labelled / predicted frame falls in it.
                    Insensitive to the frame rate (10 vs 20 FPS).
* event level     - labelled segments and predicted episodes are paired
                    ONE-TO-ONE (largest overlap first, within a tolerance).
                    A predicted episode can match at most one label, so one
                    long prediction that spans nine labelled events counts as
                    1 TP + 8 FN, not 9 TP (the old any-overlap rule did that).

All F1 values here are binary F1 of the positive class unless stated
otherwise; ``multiclass_metrics`` reports per-class, macro, weighted and
micro F1 explicitly.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Segment = Tuple[float, float]


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def prf(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    p, r = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
    return p, r, safe_div(2 * p * r, p + r)


def truth_mask(times: np.ndarray, segments: Sequence[Segment]) -> np.ndarray:
    truth = np.zeros(len(times), dtype=bool)
    for s, e in segments:
        truth |= (times >= s) & (times <= e)
    return truth


def frame_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Binary confusion matrix of per-frame flags."""
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    tp, fp = int((pred & truth).sum()), int((pred & ~truth).sum())
    fn, tn = int((~pred & truth).sum()), int((~pred & ~truth).sum())
    p, r, f = prf(tp, fp, fn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
        "fpr": round(safe_div(fp, fp + tn), 4),  # false-positive rate (share of normal frames flagged)
        "fnr": round(safe_div(fn, fn + tp), 4),  # false-negative rate (share of labelled frames missed)
        "positive_frames": tp + fn, "negative_frames": fp + tn,
        "confusion": {"labels": ["normal", "behaviour"], "matrix": [[tn, fp], [fn, tp]]},  # rows = truth
    }


def episodes(times: np.ndarray, on: np.ndarray, merge_gap: float = 0.5) -> List[Segment]:
    """Turn per-frame flags into (start, end) episodes; gaps <= merge_gap are bridged."""
    eps: List[Segment] = []
    start = prev = None
    for t, v in zip(times, on):
        if v:
            if start is None:
                start = t
            elif t - prev > merge_gap:
                eps.append((float(start), float(prev)))
                start = t
            prev = t
    if start is not None:
        eps.append((float(start), float(prev)))
    return eps


def _overlap(a: Segment, b: Segment, tol: float) -> float:
    """Overlap length of a and b, with b widened by ``tol`` on both sides; < 0 = no contact."""
    return min(a[1], b[1] + tol) - max(a[0], b[0] - tol)


def match_events(pred: Sequence[Segment], gt: Sequence[Segment], tol: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """One-to-one pairing of predicted and labelled events, largest overlap first."""
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            ov = _overlap(p, g, tol)
            if ov >= 0:
                pairs.append((ov, -i, -j))
    pairs.sort(reverse=True)
    used_p, used_g, matches = set(), set(), []
    for _, ni, nj in pairs:
        i, j = -ni, -nj
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append((i, j))
    return sorted(matches), [i for i in range(len(pred)) if i not in used_p], [j for j in range(len(gt)) if j not in used_g]


def event_metrics(pred: Sequence[Segment], gt: Sequence[Segment], tol: float = 1.0) -> dict:
    matches, fp_idx, fn_idx = match_events(pred, gt, tol)
    tp, fp, fn = len(matches), len(fp_idx), len(fn_idx)
    p, r, f = prf(tp, fp, fn)
    delays = [round(pred[i][0] - gt[j][0], 2) for i, j in matches]
    covered = sum(any(_overlap(p_, g, tol) >= 0 for p_ in pred) for g in gt)
    return {
        "gt_events": len(gt), "pred_events": len(pred), "tp": tp, "fp": fp, "fn": fn,
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
        "median_delay_s": round(float(np.median(delays)), 2) if delays else None,
        "delays_s": delays,
        # share of labelled events touched by ANY prediction (the old, lenient recall)
        "gt_covered": covered,
    }


def segment_metrics(times: np.ndarray, pred: np.ndarray, gt: Sequence[Segment], seg_s: float = 1.0) -> dict:
    """1-second segment grid over the session (frame-rate independent)."""
    if len(times) == 0:
        return frame_metrics(np.zeros(0, bool), np.zeros(0, bool))
    t0, t1 = float(times[0]), float(times[-1])
    n = int(np.floor((t1 - t0) / seg_s)) + 1
    pred_seg = np.zeros(n, dtype=bool)
    idx = np.clip(((np.asarray(times) - t0) / seg_s).astype(int), 0, n - 1)
    np.logical_or.at(pred_seg, idx, np.asarray(pred, bool))
    truth_seg = np.zeros(n, dtype=bool)
    for s, e in gt:
        a = int(np.floor((max(s, t0) - t0) / seg_s))
        b = int(np.floor((min(e, t1) - t0) / seg_s))
        if b >= a:
            truth_seg[max(a, 0):min(b, n - 1) + 1] = True
    out = frame_metrics(pred_seg, truth_seg)
    out["segment_s"] = seg_s
    return out


def multiclass_metrics(y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]) -> dict:
    """Confusion matrix (rows = truth, columns = prediction) and per-class / macro / weighted / micro F1."""
    index = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            cm[index[t], index[p]] += 1
    per_class = {}
    f1s, supports = [], []
    for c, i in index.items():
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p, r, f = prf(tp, fp, fn)
        support = int(cm[i, :].sum())
        per_class[c] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "support": support}
        if support > 0:  # classes absent from the ground truth do not enter the macro average
            f1s.append(f)
            supports.append(support)
    total = int(cm.sum())
    acc = safe_div(np.trace(cm), total)
    return {
        "classes": list(classes),
        "confusion": cm.tolist(),
        "per_class": per_class,
        "macro_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "weighted_f1": round(float(np.average(f1s, weights=supports)), 4) if f1s else 0.0,
        "micro_f1": round(acc, 4),  # single-label multi-class: micro F1 = accuracy
        "accuracy": round(acc, 4),
        "samples": total,
    }


def pooled_prf(counts: List[Dict[str, int]]) -> Dict[str, float]:
    """P/R/F1 from summed tp/fp/fn over several sessions (each event counts once)."""
    tp = sum(c["tp"] for c in counts)
    fp = sum(c["fp"] for c in counts)
    fn = sum(c["fn"] for c in counts)
    p, r, f = prf(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def sum_confusions(matrices: List[List[List[int]]]) -> Optional[List[List[int]]]:
    if not matrices:
        return None
    return np.sum(np.array(matrices), axis=0).tolist()
