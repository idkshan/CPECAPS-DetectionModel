"""Evaluate the detection model on the participant study (validation / test splits).

The participants are NOT used for training: the YOLO models are trained on the
approved public datasets only. The participant sessions are split by person:
  * validation - used to tune thresholds / persistence times (configs/*.yaml,
                 scripts/tune_thresholds.py)
  * test       - touched once, with the frozen config, for the reported results

Manifest (CSV, one row per recorded session):
    participant,session,labels,split,condition
    P01,P01_day,labels/P01_day.csv,validation,day
    P01,P01_night,labels/P01_night.csv,validation,night
    P02,P02_day,labels/P02_day.csv,test,day
  * session   - folder name under --sessions (written by process_videos.py) or a path
  * labels    - ground-truth CSV (relative paths are read from the manifest's folder)
  * split     - validation or test (a participant must stay in ONE split)
  * condition - free text used for grouping, e.g. day / night (optional)

Examples
--------
    # tune on the validation split (replay = re-run Stages 3-5 with this config)
    python scripts/evaluate_study.py --manifest data/study/manifest.csv --split validation --config configs/tuned.yaml --tag tuned

    # final, reported numbers: the test split, once
    python scripts/evaluate_study.py --manifest data/study/manifest.csv --split test --config configs/tuned.yaml --tag final

    # ablation on the test split: AI only vs thresholds only vs both
    python scripts/evaluate_study.py --manifest data/study/manifest.csv --split test --eye-mode yolo --mouth-mode yolo --tag ai_only

    # baseline (decisions logged while recording) vs this version (replay), same sessions
    python scripts/evaluate_study.py --manifest data/study/manifest.csv --split validation --compare --tag compare

Results go to results/study_eval/<tag>/: per_session.csv, summary.csv,
summary.md (thesis-ready tables) and report.json. Pooled metrics add up
TP/FP/FN (and confusion matrices) over sessions before computing P/R/F1.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import _bootstrap  # noqa: F401
from dms.config import EYE_MODES, MOUTH_MODES, apply_evidence_modes, load_config
from dms.evaluation.labels import load_labels
from dms.evaluation.session import FLAG_OF, evaluate_session, load_jsonl, markdown, pool

SPLITS = ("validation", "test")


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def load_manifest(path: Path, sessions_dir: Path) -> List[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        reader.fieldnames = [_norm(f) for f in reader.fieldnames or []]
        missing = {"participant", "session", "labels", "split"} - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"{path}: missing column(s) {sorted(missing)}")
        for i, row in enumerate(reader, 2):
            if not (row.get("session") or "").strip():
                continue
            split = _norm(row["split"])
            split = {"val": "validation", "valid": "validation", "testing": "test"}.get(split, split)
            if split not in SPLITS:
                raise SystemExit(f"{path} line {i}: split must be validation or test, not {row['split']!r}")
            session = Path(row["session"].strip())
            if not (session / "frames.jsonl").exists():
                session = sessions_dir / row["session"].strip()
            labels = Path(row["labels"].strip())
            if not labels.is_absolute() and not labels.exists():
                labels = path.parent / labels
            rows.append({"participant": row["participant"].strip(), "session": session, "labels": labels, "split": split,
                         "condition": (row.get("condition") or "").strip() or "all"})
    leaks = defaultdict(set)
    for r in rows:
        leaks[r["participant"]].add(r["split"])
    both = sorted(p for p, s in leaks.items() if len(s) > 1)
    if both:
        raise SystemExit(f"Participant(s) in BOTH validation and test (data leakage): {both}. Keep each person in one split.")
    return rows


def group_indices(rows: List[dict], split: str) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[f"{r['split']} - all sessions"].append(i)
        groups[f"{r['split']} - {r['condition']}"].append(i)
    if split == "all":
        groups = {"all splits": list(range(len(rows))), **groups}
    conditions = {r["condition"] for r in rows}
    if len(conditions) == 1:  # the per-condition group would repeat "all sessions"
        only = next(iter(conditions))
        groups = {k: v for k, v in groups.items() if not k.endswith(f" - {only}")}
    return dict(groups)


def write_outputs(out: Path, rows, reports, summary, md: str, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_session.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["participant", "session", "split", "condition", "minutes", "behavior", "labelled_events", "event_tp",
                    "event_fp", "event_fn", "event_f1", "frame_f1", "frame_fpr", "frame_fnr", "median_delay_s",
                    "unnecessary_alarms", "non_event_minutes"])
        for r, rep in zip(rows, reports):
            for b, res in rep["behaviors"].items():
                ev, fr = res["event"], res["frame"]
                w.writerow([r["participant"], r["session"].name, r["split"], r["condition"], rep["duration_min"], b,
                            res["labelled_events"], ev["tp"], ev["fp"], ev["fn"], ev["f1"], fr["f1"], fr["fpr"], fr["fnr"],
                            ev["median_delay_s"], rep["alarms"]["alarms"]["unnecessary"],
                            round(rep["alarms"]["non_event_hours"] * 60, 2)])
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "behavior", "primary_view", "primary_f1", "target_f1", "labelled_events", "event_precision",
                    "event_recall", "event_f1", "frame_precision", "frame_recall", "frame_f1", "frame_fpr", "frame_fnr",
                    "participant_f1_mean", "participant_f1_sd", "median_delay_s"])
        for name, g in summary.items():
            for b in FLAG_OF:
                if b not in g:
                    continue
                e = g[b]
                w.writerow([name, b, e["primary_view"], e["primary_f1"], e.get("target_f1"), e["labelled_events"],
                            e["event"]["precision"], e["event"]["recall"], e["event"]["f1"], e["frame"]["precision"],
                            e["frame"]["recall"], e["frame"]["f1"], e["frame"]["fpr"], e["frame"]["fnr"],
                            e["participant_f1_mean"], e["participant_f1_sd"], e["median_delay_s"]])
            a = g["alarms"]["alarms"]
            w.writerow([name, "unnecessary_alarms_per_non_event_hour", "", a["per_non_event_hour"]])
    (out / "summary.md").write_text(md, encoding="utf-8")
    (out / "report.json").write_text(json.dumps({
        "meta": meta, "summary": summary,
        "sessions": [{**{k: str(v) for k, v in r.items()}, "report": rep} for r, rep in zip(rows, reports)]}, indent=2),
        encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sessions", default="results/study_sessions", help="folder with the session folders")
    ap.add_argument("--split", choices=["validation", "test", "all"], default="validation")
    ap.add_argument("--replay", action="store_true", help="re-run Stages 3-5 of this version on the logged features")
    ap.add_argument("--compare", action="store_true", help="report as-recorded (baseline) and replay (this version)")
    ap.add_argument("--config", default=None, help="config for the replay (implies --replay)")
    ap.add_argument("--eye-mode", choices=sorted(EYE_MODES), default=None)
    ap.add_argument("--mouth-mode", choices=sorted(MOUTH_MODES), default=None)
    ap.add_argument("--tolerance", type=float, default=1.0, help="event matching tolerance (s)")
    ap.add_argument("--tag", default=None, help="name of this run (output sub-folder)")
    ap.add_argument("--out", default="results/study_eval")
    args = ap.parse_args()
    replay = args.replay or args.compare or bool(args.config or args.eye_mode or args.mouth_mode)

    manifest = Path(args.manifest)
    rows = load_manifest(manifest, Path(args.sessions))
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]
    if not rows:
        raise SystemExit(f"No sessions for split '{args.split}' in {manifest}")
    missing = [str(r["session"]) for r in rows if not (r["session"] / "frames.jsonl").exists()]
    missing += [str(r["labels"]) for r in rows if not r["labels"].exists()]
    if missing:
        raise SystemExit("Missing files (run process_videos.py / check the label paths):\n  " + "\n  ".join(missing))

    cfg = apply_evidence_modes(load_config(args.config), args.eye_mode, args.mouth_mode) if replay else None
    variants = {"as_recorded": None, "replay": cfg} if args.compare else {("replay" if replay else "as_recorded"): cfg}
    tag = args.tag or (f"{args.split}_{'compare' if args.compare else ('replay' if replay else 'recorded')}")
    meta = {"tag": tag, "split": args.split, "sessions": len(rows), "participants": len({r["participant"] for r in rows}),
            "replay": replay, "config": args.config, "eye_mode": args.eye_mode, "mouth_mode": args.mouth_mode,
            "tolerance_s": args.tolerance}

    all_md, all_summary, all_reports = [], {}, {}
    for variant, vcfg in variants.items():
        reports = []
        for n, r in enumerate(rows, 1):
            frames = load_jsonl(r["session"] / "frames.jsonl")
            try:
                labels = load_labels(r["labels"], duration_s=float(frames[-1]["t"]))
            except ValueError as exc:
                raise SystemExit(str(exc))
            for w in labels.warnings:
                print(f"   label warning ({r['labels'].name}): {w}")
            print(f"[{variant} {n}/{len(rows)}] {r['participant']} {r['session'].name} ({r['split']}, {r['condition']})")
            reports.append(evaluate_session(r["session"], labels, args.tolerance, vcfg))
        groups = group_indices(rows, args.split)
        summary = {name: pool([reports[i] for i in idx], [rows[i]["participant"] for i in idx]) for name, idx in groups.items()}
        title = {"as_recorded": "As recorded (baseline: the version that recorded the sessions)",
                 "replay": "This version (Stages 3-5 replayed on the logged features)"}[variant]
        meta_line = (f"Split **{args.split}**, {meta['sessions']} sessions, {meta['participants']} participants"
                     + (f", config {args.config}" if args.config and variant == "replay" else "")
                     + (f", eye mode {args.eye_mode}" if args.eye_mode and variant == "replay" else "")
                     + (f", mouth mode {args.mouth_mode}" if args.mouth_mode and variant == "replay" else "") + ".")
        all_md.append(markdown(summary, f"Participant study - {title}", meta_line))
        all_summary[variant], all_reports[variant] = summary, reports

    out = Path(args.out) / tag
    md = "\n".join(all_md)
    if args.compare:
        md += "\n# Change (this version - baseline), primary F1 per group\n\n| Group | Behaviour | Baseline | This version |\n|---|---|---|---|\n"
        for name, g in all_summary["replay"].items():
            for b in FLAG_OF:
                if b in g and b in all_summary["as_recorded"].get(name, {}):
                    md += f"| {name} | {b} | {all_summary['as_recorded'][name][b]['primary_f1']:.2f} | {g[b]['primary_f1']:.2f} |\n"
    for variant in variants:
        sub = out if len(variants) == 1 else out / variant
        write_outputs(sub, rows, all_reports[variant], all_summary[variant], md, {**meta, "variant": variant})
    print("\n" + md)
    print(f"Saved to {out}/")
    if args.split == "test":
        print("Reminder: report the test split only with the frozen config; tune on the validation split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
