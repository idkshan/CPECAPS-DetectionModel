"""Score one recorded session against its human labels (SO3, SO5).

    python scripts/evaluate_behaviors.py --session results/sessions/S01 --labels labels/S01.csv

    # the same session re-decided by the CURRENT code (Stages 3-5 replayed on the
    # logged features), e.g. with a tuned config or an ablation mode
    python scripts/evaluate_behaviors.py --session results/sessions/S01 --labels labels/S01.csv \
        --replay --config configs/tuned.yaml
    python scripts/evaluate_behaviors.py ... --replay --eye-mode ear --mouth-mode mar

    # baseline vs improved on the same recording: the decisions logged while
    # recording (the version that recorded it) next to a replay with this version
    python scripts/evaluate_behaviors.py --session results/sessions/S01 --labels labels/S01.csv --compare

Label file format and names: see dms/evaluation/labels.py (or docs/EVALUATION.md).
Metric definitions (frame / 1-s segment / one-to-one event level, drowsiness
classes of Table 1.11, unnecessary alarms per non-event hour): docs/EVALUATION.md.
"""

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from dms.config import EYE_MODES, MOUTH_MODES, apply_evidence_modes, load_config
from dms.evaluation.drowsiness import STATES
from dms.evaluation.labels import load_labels
from dms.evaluation.session import evaluate_session, load_session


def _f(v, nd=2):
    return "  —  " if v is None else f"{v:.{nd}f}"


def print_report(rep: dict, title: str) -> None:
    print(f"\n{title}: {rep['session']}  ({rep['duration_min']} min, {rep['frames']} frames)")
    for w in rep.get("label_warnings", []):
        print(f"  label warning: {w}")
    print(f"{'behaviour':15s} {'GT':>4s} {'primary F1':>11s} {'target':>7s}   {'event P/R/F1':>18s}   "
          f"{'frame P/R/F1':>18s}   {'FPR':>6s} {'FNR':>6s} {'delay':>6s}")
    for b, r in rep["behaviors"].items():
        ev, fr = r["event"], r["frame"]
        primary = r[r["primary_view"]]["f1"]
        tgt = f"{r['target_f1']:.2f}" if "target_f1" in r else "  —  "
        mark = "" if r.get("meets_target") is None else (" ok" if r["meets_target"] else " LOW")
        print(f"{b:15s} {r['labelled_events']:4d} {primary:7.2f} ({r['primary_view'][0]}){mark:4s}{tgt:>5s}   "
              f"{ev['precision']:5.2f} {ev['recall']:5.2f} {ev['f1']:5.2f}   {fr['precision']:5.2f} {fr['recall']:5.2f} {fr['f1']:5.2f}   "
              f"{fr['fpr']:6.3f} {fr['fnr']:6.3f} {_f(ev['median_delay_s'], 1):>6s}")
    ds = rep["drowsiness_states"]
    if "three_class" in ds:
        tc, bn = ds["three_class"], ds["binary"]
        print(f"drowsiness states (Table 1.11): binary drowsy-vs-alert F1 {bn['f1']:.2f}; "
              f"3-class macro F1 {tc['macro_f1']:.2f}, accuracy {tc['accuracy']:.2f} ({tc['samples']} frames)")
        print("   labelled \\ predicted  " + "  ".join(f"{s[:10]:>10s}" for s in STATES))
        for i, s in enumerate(STATES):
            print(f"   {s:21s} " + "  ".join(f"{v:10d}" for v in tc["confusion"][i]))
    elif ds.get("note"):
        print(f"drowsiness states: {ds['note']}")
    a = rep["alarms"]
    print(f"alarms: {a['alarms']['count']} started, {a['alarms']['unnecessary']} unnecessary -> "
          f"{_f(a['alarms']['per_non_event_hour'], 2)} per non-event hour (target <= 0.5; non-event time "
          f"{a['non_event_hours'] * 60:.1f} min); unnecessary pushes {a['pushes']['unnecessary']}, "
          f"unnecessary risk escalations {a['risk_escalations']['unnecessary']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--session", required=True, help="folder written by run_monitor.py / process_videos.py")
    ap.add_argument("--labels", required=True, help="ground-truth CSV")
    ap.add_argument("--tolerance", type=float, default=1.0, help="event matching tolerance in seconds")
    ap.add_argument("--replay", action="store_true", help="re-run Stages 3-5 of this version on the logged features")
    ap.add_argument("--compare", action="store_true", help="as recorded (baseline) AND replay (this version)")
    ap.add_argument("--config", default=None, help="config used for --replay / --compare")
    ap.add_argument("--eye-mode", choices=sorted(EYE_MODES), default=None)
    ap.add_argument("--mouth-mode", choices=sorted(MOUTH_MODES), default=None)
    ap.add_argument("--strict-labels", action="store_true", help="treat label warnings (e.g. unknown names) as errors")
    ap.add_argument("--out", default=None, help="write the report JSON here")
    args = ap.parse_args()

    frames, _, _ = load_session(args.session)
    duration = float(frames[-1]["t"]) - float(frames[0]["t"])
    try:
        labels = load_labels(args.labels, duration_s=float(frames[-1]["t"]), strict=args.strict_labels)
    except ValueError as exc:
        raise SystemExit(str(exc))
    cfg = None
    if args.replay or args.compare or args.config or args.eye_mode or args.mouth_mode:
        cfg = apply_evidence_modes(load_config(args.config), args.eye_mode, args.mouth_mode)

    reports = {}
    if args.compare or cfg is None:
        reports["as_recorded"] = evaluate_session(args.session, labels, args.tolerance, None)
        print_report(reports["as_recorded"], "AS RECORDED (version that recorded the session)")
    if cfg is not None:
        reports["replay"] = evaluate_session(args.session, labels, args.tolerance, cfg)
        print_report(reports["replay"], "REPLAY with this version" + (f" + {args.config}" if args.config else ""))
    if args.compare:
        print("\nchange (replay - recorded), primary F1:")
        for b, r in reports["replay"]["behaviors"].items():
            old = reports["as_recorded"]["behaviors"].get(b)
            if old:
                print(f"   {b:15s} {old[old['primary_view']]['f1']:.2f} -> {r[r['primary_view']]['f1']:.2f}")
        o, n = reports["as_recorded"]["alarms"]["alarms"], reports["replay"]["alarms"]["alarms"]
        print(f"   unnecessary alarms per non-event hour: {_f(o['per_non_event_hour'])} -> {_f(n['per_non_event_hour'])}")
    print(f"\n(session duration {duration / 60:.1f} min)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(reports if len(reports) > 1 else next(iter(reports.values())), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
