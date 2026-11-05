"""Tune thresholds on the VALIDATION participants and write configs/tuned.yaml.

    python scripts/tune_thresholds.py --manifest data/study/manifest.csv --behaviors drowsiness --workers 4

Only sessions whose manifest split is "validation" are used; the script
refuses to run on the test split. For each behaviour a few key parameters are
searched (dms/evaluation/tuning.py: DEFAULT_SPACE) by replaying Stages 3-5 on
the logged features. The best configuration that keeps unnecessary alarms
<= 0.5 per non-event hour is written as a small YAML file containing only the
changed values, plus a report of every configuration tried.

After tuning: FREEZE the file, then evaluate the test split once:
    python scripts/evaluate_study.py --manifest data/study/manifest.csv --split test --config configs/tuned.yaml --tag final

Search space override (YAML), e.g.:
    drowsiness.recovery_s: [15, 20, 25]
    thresholds.ear_closed_ratio: [0.68, 0.70, 0.72]
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

import _bootstrap  # noqa: F401
from dms.config import load_config
from dms.evaluation.labels import load_labels
from dms.evaluation.session import load_jsonl
from dms.evaluation.tuning import DEFAULT_SPACE, coordinate_search, nested
from evaluate_study import load_manifest

BEHAVIORS = ("drowsiness", "yawning", "gaze_deviation", "phone_usage")


def deep_update(a: dict, b: dict) -> dict:
    for k, v in b.items():
        a[k] = deep_update(a.get(k, {}), v) if isinstance(v, dict) else v
    return a


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sessions", default="results/study_sessions")
    ap.add_argument("--behaviors", nargs="+", choices=BEHAVIORS, default=list(BEHAVIORS))
    ap.add_argument("--config", default=None, help="starting config (default: built-in defaults)")
    ap.add_argument("--space", default=None, help="YAML with parameter -> candidate values (replaces the defaults)")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (one per CPU core is fine)")
    ap.add_argument("--out-config", default="configs/tuned.yaml")
    ap.add_argument("--report", default="results/tuning")
    args = ap.parse_args()

    rows = [r for r in load_manifest(Path(args.manifest), Path(args.sessions)) if r["split"] == "validation"]
    if not rows:
        raise SystemExit("No validation sessions in the manifest - tuning on test sessions is not allowed.")
    sessions = []
    for r in rows:
        frames = load_jsonl(r["session"] / "frames.jsonl")
        if not frames:
            raise SystemExit(f"missing {r['session'] / 'frames.jsonl'}")
        sessions.append((frames, load_labels(r["labels"], duration_s=float(frames[-1]["t"])), r["participant"]))
    print(f"{len(sessions)} validation sessions from {len({r['participant'] for r in rows})} participants")

    if args.space:
        with open(args.space, encoding="utf-8") as fh:
            space = yaml.safe_load(fh)
    else:
        space = {k: v for b in args.behaviors for k, v in DEFAULT_SPACE[b].items()}
    base = load_config(args.config)
    params, best, history = coordinate_search(sessions, base, space, args.behaviors, rounds=args.rounds, workers=args.workers)

    out_yaml = {}
    for path, value in params.items():
        deep_update(out_yaml, nested(path, value))
    out_cfg = Path(args.out_config)
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# Tuned on the VALIDATION split of {args.manifest} ({len(sessions)} sessions) for {', '.join(args.behaviors)}.\n"
              f"# Validation score {best.score:.3f}, F1 {best.f1}, unnecessary alarms {best.alarms_per_hour:.2f}/non-event hour.\n"
              "# Only changed values are listed; everything else is the default. Freeze this file before testing.\n")
    out_cfg.write_text(header + (yaml.safe_dump(out_yaml, sort_keys=False) if out_yaml else "{}\n"), encoding="utf-8")

    rep = Path(args.report)
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "trials.json").write_text(json.dumps([t.__dict__ for t in history], indent=2, default=str), encoding="utf-8")
    lines = ["| # | changed parameters | score | " + " | ".join(args.behaviors) + " | alarms / non-event h |",
             "|---|---|---|" + "---|" * len(args.behaviors) + "---|"]
    for i, t in enumerate(history):
        lines.append(f"| {i} | {t.params or 'defaults'} | {t.score:.3f} | " + " | ".join(f"{t.f1.get(b, 0):.2f}" for b in args.behaviors)
                     + f" | {t.alarms_per_hour:.2f}{'' if t.feasible else ' (over limit)'} |")
    (rep / "trials.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nBest (validation): score {best.score:.3f}, F1 {best.f1}, alarms/h {best.alarms_per_hour:.2f}")
    print(f"Changed: {params or 'nothing (defaults were best)'}")
    print(f"Wrote {out_cfg} and {rep / 'trials.md'}. Freeze it, then run the test split once with --config {out_cfg}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
