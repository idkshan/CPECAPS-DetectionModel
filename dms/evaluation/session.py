"""Evaluate one recorded session (or many) against its label file.

A session folder written by ``run_monitor.py`` / ``process_videos.py`` holds
``frames.jsonl`` (one record per frame) and ``alerts.jsonl``. Two modes:

* as recorded - score the decisions the system made while it ran
  (= the code version that recorded the session: the BASELINE when the
  session was recorded with an older version)
* replay      - re-run Stages 3-5 of the CURRENT code on the logged Stage-2
  features (fast: the neural networks are not run again). Used for tuning on
  the validation split, for ablations and for baseline-vs-improved comparisons.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..types import ContextReading, FrameFeatures
from .alarms import alarm_report, alarm_validation, pooled_alarm_rates, tier_sequence
from .drowsiness import STATES, evaluate_states, predicted_state
from .labels import VISION_BEHAVIORS, LabelSet
from .metrics import episodes, event_metrics, frame_metrics, multiclass_metrics, pooled_prf, segment_metrics, sum_confusions, truth_mask

# per-frame flag in the log for each labelled behaviour
FLAG_OF = {"drowsiness": "drowsy", "yawning": "yawning", "gaze_deviation": "gaze_deviation", "phone_usage": "phone_usage",
           "eye_closure": "eye_closure"}
TARGET_F1 = {"drowsiness": 0.85, "gaze_deviation": 0.85, "yawning": 0.80, "phone_usage": 0.80}  # Table 1.3
# Which view is the SO3 number: states are scored per frame, discrete behaviours per event.
PRIMARY = {"drowsiness": "frame", "yawning": "event", "gaze_deviation": "event", "phone_usage": "event", "eye_closure": "event"}


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_session(folder: str | Path) -> Tuple[List[dict], List[dict], dict]:
    folder = Path(folder)
    frames = load_jsonl(folder / "frames.jsonl")
    if not frames:
        raise FileNotFoundError(f"{folder / 'frames.jsonl'} is missing or empty")
    alerts = load_jsonl(folder / "alerts.jsonl")
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return frames, alerts, summary


def replay(frames: Sequence[dict], cfg) -> Tuple[List[dict], List[dict]]:
    """Run the current Stages 3-5 over logged features; returns (frame records, alert records)."""
    from ..engine import DecisionEngine
    from ..session_log import decision_record

    engine = DecisionEngine(cfg)
    known = {f.name for f in fields(FrameFeatures)}
    t_start = float(frames[0]["t"])
    out_frames, out_alerts = [], []
    for rec in frames:
        feats = FrameFeatures(**{k: v for k, v in rec["features"].items() if k in known})
        ctx_known = {f.name for f in fields(ContextReading)}
        ctx = (ContextReading(**{k: v for k, v in rec["context"].items() if k in ctx_known}) if rec.get("context")
               else ContextReading(driving_s=feats.t - t_start))
        out = engine.step(feats, ctx)
        out_frames.append(decision_record(feats.t, out))
        out_alerts += [a.to_dict() if hasattr(a, "to_dict") else dict(a.__dict__) for a in out.alerts]
    return out_frames, out_alerts


def evaluate_frames(frames: Sequence[dict], alerts: Sequence[dict], labels: LabelSet, tolerance: float = 1.0,
                    alarm_tol_after: float = 2.0) -> dict:
    """All metrics for one session; ``frames`` are recorded or replayed records."""
    times = np.array([float(r["t"]) for r in frames])
    duration_h = (times[-1] - times[0]) / 3600.0 if len(times) > 1 else 0.0
    report: dict = {"frames": len(times), "duration_min": round(duration_h * 60, 2), "behaviors": {},
                    "label_warnings": list(labels.warnings)}
    for b, flag in FLAG_OF.items():
        if any(flag not in (r.get("behavior") or {}) for r in frames):
            continue  # older logs have no eye_closure flag
        if b == "eye_closure" and not labels.segments.get(b):
            continue  # optional label class not used in this file
        pred = np.array([bool(r["behavior"][flag]) for r in frames])
        gt = labels.segments.get(b, [])
        res = {
            "frame": frame_metrics(pred, truth_mask(times, gt)),
            "segment_1s": segment_metrics(times, pred, gt, 1.0),
            "event": event_metrics(episodes(times, pred), gt, tolerance),
            "labelled_events": len(gt),
            "labelled_seconds": round(labels.labelled_seconds(b), 1),
            "primary_view": PRIMARY[b],
        }
        if b in TARGET_F1:
            res["target_f1"] = TARGET_F1[b]
            primary_f1 = res[PRIMARY[b]]["f1"]
            res["meets_target"] = (primary_f1 >= TARGET_F1[b]) if gt else None
        report["behaviors"][b] = res

    pred_states = [predicted_state(r) for r in frames]
    report["drowsiness_states"] = evaluate_states(times, pred_states, labels, tolerance)
    t_list, tiers = tier_sequence(frames)
    report["alarms"] = alarm_report(alerts, t_list, tiers, labels.all_events(), tol_before=tolerance, tol_after=alarm_tol_after)
    if labels.expected_tiers:
        report["alarm_validation"] = alarm_validation(alerts, labels.expected_tiers)
    return report


def evaluate_session(folder: str | Path, labels: LabelSet, tolerance: float = 1.0, cfg=None) -> dict:
    """Score a session folder as recorded (cfg None) or replayed with ``cfg``."""
    frames, alerts, summary = load_session(folder)
    if cfg is not None:
        frames, alerts = replay(frames, cfg)
    report = evaluate_frames(frames, alerts, labels, tolerance)
    report.update(session=str(folder), labels=labels.source, replay=cfg is not None,
                  recorded_with=summary.get("code_version", "unknown (older than v3)"))
    return report


# ------------------------------------------------------------------ pooling
def pool(reports: Sequence[dict], participants: Sequence[str]) -> dict:
    """Pool sessions: sum TP/FP/FN and confusion matrices, then compute metrics.

    Pooled (micro) numbers count each event / frame once; the per-participant
    mean +/- SD shows how much drivers differ.
    """
    out: dict = {}
    for b in [*VISION_BEHAVIORS, "eye_closure"]:
        rs = [r["behaviors"][b] for r in reports if b in r["behaviors"]]
        if not rs:
            continue
        ev = pooled_prf([r["event"] for r in rs])
        fr = pooled_prf([r["frame"] for r in rs])
        tn = sum(r["frame"]["tn"] for r in rs)
        per_person: Dict[str, Dict[str, int]] = {}
        for r, p in zip(reports, participants):
            if b in r["behaviors"]:
                c = per_person.setdefault(p, {"tp": 0, "fp": 0, "fn": 0})
                for k in ("tp", "fp", "fn"):
                    c[k] += r["behaviors"][b]["event" if PRIMARY[b] == "event" else "frame"][k]
        f1s = [pooled_prf([c])["f1"] for c in per_person.values() if c["tp"] + c["fn"] > 0]
        delays = [d for r in rs for d in r["event"].get("delays_s", [])]
        primary = ev if PRIMARY[b] == "event" else fr
        entry = {
            "labelled_events": sum(r["labelled_events"] for r in rs),
            "labelled_seconds": round(sum(r["labelled_seconds"] for r in rs), 1),
            "event": ev,
            "frame": {**fr, "tn": tn, "fpr": round(fr["fp"] / (fr["fp"] + tn), 4) if fr["fp"] + tn else 0.0,
                      "fnr": round(fr["fn"] / (fr["fn"] + fr["tp"]), 4) if fr["fn"] + fr["tp"] else 0.0,
                      "confusion": [[tn, fr["fp"]], [fr["fn"], fr["tp"]]]},
            "primary_view": PRIMARY[b],
            "primary_f1": primary["f1"],
            "participant_f1_mean": round(float(np.mean(f1s)), 4) if f1s else None,
            "participant_f1_sd": round(float(np.std(f1s, ddof=1)), 4) if len(f1s) > 1 else None,
            "median_delay_s": round(float(np.median(delays)), 2) if delays else None,
        }
        if b in TARGET_F1:
            entry["target_f1"] = TARGET_F1[b]
            entry["meets_target"] = (primary["f1"] >= TARGET_F1[b]) if entry["labelled_events"] else None
        out[b] = entry

    three = [r["drowsiness_states"].get("three_class") for r in reports]
    three = [t for t in three if t]
    if three:
        cm = sum_confusions([t["confusion"] for t in three])
        flat_true, flat_pred = [], []
        for i, ti in enumerate(STATES):
            for j, pj in enumerate(STATES):
                flat_true += [ti] * cm[i][j]
                flat_pred += [pj] * cm[i][j]
        out["drowsiness_states"] = multiclass_metrics(flat_true, flat_pred, STATES)
        seg = [r["drowsiness_states"].get("segments") for r in reports]
        seg = [s for s in seg if s and s["segments"]]
        if seg:
            n = sum(s["segments"] for s in seg)
            out["drowsiness_states"]["segment_exact_acc"] = round(sum(s["exact"] for s in seg) / n, 4)
            out["drowsiness_states"]["segment_within_one_acc"] = round(sum(s["within_one"] for s in seg) / n, 4)
            out["drowsiness_states"]["segment_confusion"] = sum_confusions([s["confusion"] for s in seg])
    out["alarms"] = pooled_alarm_rates([r["alarms"] for r in reports])
    av = [r["alarm_validation"] for r in reports if r.get("alarm_validation")]
    if av:
        n = sum(a["segments"] for a in av)
        exact = sum(a["exact"] for a in av)
        out["alarm_validation"] = {"segments": n, "accuracy": round(exact / n, 4) if n else None,
                                   "under_alerted": sum(a["under_alerted"] for a in av),
                                   "over_alerted": sum(a["over_alerted"] for a in av),
                                   "confusion": sum_confusions([a["confusion"] for a in av])}
    return out


def _fmt(v, nd=2) -> str:
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def markdown(groups: Dict[str, dict], title: str, meta_line: str = "") -> str:
    """Thesis-ready tables for pooled results (one block per group)."""
    lines = [f"# {title}", ""]
    if meta_line:
        lines += [meta_line, ""]
    for name, g in groups.items():
        a = g["alarms"]
        lines += [f"## {name}", "",
                  f"Driving time {a['driving_hours'] * 60:.1f} min (non-event {a['non_event_hours'] * 60:.1f} min). "
                  f"Unnecessary alarms: **{_fmt(a['alarms']['per_non_event_hour'], 3)} per non-event hour** "
                  f"({a['alarms']['unnecessary']} of {a['alarms']['count']} alarms; target ≤ 0.5). "
                  f"Unnecessary pushes: {a['pushes']['unnecessary']}. Unnecessary risk escalations: "
                  f"{a['risk_escalations']['unnecessary']}.", "",
                  "| Behaviour | Labelled events | Primary view | Primary F1 | Target | Event P / R / F1 | Frame P / R / F1 | "
                  "Frame FPR / FNR | F1 per participant | Median delay (s) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for b in [*VISION_BEHAVIORS, "eye_closure"]:
            if b not in g:
                continue
            r = g[b]
            ok = "" if r.get("meets_target") is None else (" ✓" if r["meets_target"] else " ✗")
            ev, fr = r["event"], r["frame"]
            spread = "—" if r["participant_f1_mean"] is None else (
                f"{r['participant_f1_mean']:.2f}" + (f" ± {r['participant_f1_sd']:.2f}" if r["participant_f1_sd"] is not None else ""))
            lines.append(f"| {b} | {r['labelled_events']} | {r['primary_view']} | **{r['primary_f1']:.2f}**{ok} | "
                         f"{r.get('target_f1', '—')} | {ev['precision']:.2f} / {ev['recall']:.2f} / {ev['f1']:.2f} | "
                         f"{fr['precision']:.2f} / {fr['recall']:.2f} / {fr['f1']:.2f} | {fr['fpr']:.3f} / {fr['fnr']:.3f} | "
                         f"{spread} | {_fmt(r['median_delay_s'])} |")
        ds = g.get("drowsiness_states")
        if ds:
            lines += ["", f"Drowsiness states (Table 1.11), frame level: macro F1 **{ds['macro_f1']:.2f}**, "
                          f"weighted F1 {ds['weighted_f1']:.2f}, accuracy (= micro F1) {ds['accuracy']:.2f}, "
                          f"{ds['samples']} frames.", "",
                      "| Labelled \\ Predicted | " + " | ".join(STATES) + " | Recall | F1 | Frames |",
                      "|---|---|---|---|---|---|---|"]
            for i, c in enumerate(STATES):
                pc = ds["per_class"][c]
                lines.append(f"| {c} | " + " | ".join(str(v) for v in ds["confusion"][i]) +
                             f" | {pc['recall']:.2f} | {pc['f1']:.2f} | {pc['support']} |")
            if "segment_exact_acc" in ds:
                lines.append(f"\nPer labelled segment (highest reported level vs annotated): exact "
                             f"{ds['segment_exact_acc']:.2f}, within one level {ds['segment_within_one_acc']:.2f}.")
        av = g.get("alarm_validation")
        if av:
            tiers = ["NONE", "LOW", "MEDIUM", "HIGH"]
            lines += ["", f"Alarm validation (enacted behaviour at a known risk level vs tier triggered, Table 1.3): "
                          f"{av['accuracy']:.2f} exact over {av['segments']} segments; under-alerted {av['under_alerted']}, "
                          f"over-alerted {av['over_alerted']}.", "",
                      "| Expected \\ Triggered | " + " | ".join(tiers) + " |", "|---|---|---|---|---|"]
            for i, t in enumerate(tiers):
                lines.append(f"| {t} | " + " | ".join(str(v) for v in av["confusion"][i]) + " |")
        lines.append("")
    return "\n".join(lines)
