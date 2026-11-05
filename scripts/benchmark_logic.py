"""Logic benchmark: baseline vs this version on identical synthetic drives.

    python scripts/benchmark_logic.py --sessions 12 --baseline-root ../baseline_checkout

What it does
  1. generates labelled 20-minute synthetic drives (dms.simulation.synthetic_drive):
     normal blinking, mirror and dashboard glances, talking, face dropouts and a
     camera stall (never labelled), plus labelled yawns, gaze deviations, phone
     use, a slightly-drowsy and a moderately-drowsy phase
  2. replays the SAME features through each code version (Stages 3-5 only)
  3. scores every version with the SAME evaluation code (dms.evaluation):
     one-to-one event F1, frame F1, drowsiness states (Table 1.11), unnecessary
     alarms per non-event hour, and how long the buzzer keeps sounding after
     each labelled event has ended ("tail")

What it is NOT: an accuracy estimate. The drives are a model of a driver
(values from typical literature ranges), so the numbers show whether the
rules and the alert policy behave as designed and how two versions differ on
the same inputs - real accuracy needs labelled recordings
(scripts/evaluate_study.py on the participant sessions).

To get a baseline checkout:  git worktree add ../baseline_checkout <commit>
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from dms.evaluation.alarms import sound_report
from dms.evaluation.labels import load_labels
from dms.evaluation.session import evaluate_frames, load_jsonl, pool
from dms.simulation import synthetic_drive

ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).with_name("_replay_runner.py")


def generate(out: Path, sessions: int, minutes: float, seed: int) -> list:
    names = []
    for k in range(sessions):
        name = f"drive_{k:02d}"
        folder = out / "data" / name
        folder.mkdir(parents=True, exist_ok=True)
        frames, labels = synthetic_drive(seed + k, minutes)
        with open(folder / "features.jsonl", "w", encoding="utf-8") as fh:
            for f in frames:
                fh.write(json.dumps(f.to_dict()) + "\n")
        with open(folder / "labels.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["behavior", "start_s", "end_s"])
            w.writerows(labels)
        names.append(name)
    return names


def run_version(code_root: Path, out: Path, names: list, tag: str, config=None, min_phone_conf: float = 0.0) -> None:
    for name in names:
        cmd = [sys.executable, str(RUNNER), "--code-root", str(code_root), "--features",
               str(out / "data" / name / "features.jsonl"), "--out", str(out / "runs" / tag / name),
               "--min-phone-conf", str(min_phone_conf)]
        if config:
            cmd += ["--config", str(Path(config).resolve())]
        subprocess.run(cmd, check=True)


def evaluate(tag: str, out: Path, names: list) -> dict:
    reports, sounds = [], []
    for name in names:
        run = out / "runs" / tag / name
        frames, alerts = load_jsonl(run / "frames.jsonl"), load_jsonl(run / "alerts.jsonl")
        labels = load_labels(out / "data" / name / "labels.csv")
        reports.append(evaluate_frames(frames, alerts, labels))
        sounds.append(sound_report(alerts, [f["t"] for f in frames], labels.all_events()))
    pooled = pool(reports, names)
    pooled["sound"] = {
        "sounds_outside_events_per_non_event_hour": round(
            sum(s["sounds_outside_events"] for s in sounds) / max(1e-9, pooled["alarms"]["non_event_hours"]), 1),
        "tail_mean_s": round(sum(s["tail_mean_s"] or 0 for s in sounds) / len(sounds), 2),
        "tail_max_s": max(s["tail_max_s"] or 0 for s in sounds),
        "events_with_tail_over_5s": sum(s["events_with_tail_over_5s"] for s in sounds),
        "events": sum(s["events"] for s in sounds),
    }
    return pooled


def row(label, fn, results):
    cells = []
    for r in results.values():
        try:
            v = fn(r)
        except (KeyError, TypeError):
            v = None
        cells.append("—" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v)))
    return f"| {label} | " + " | ".join(cells) + " |"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--baseline-root", default=None, help="checkout of the baseline version (git worktree)")
    ap.add_argument("--baseline-phone-conf", type=float, default=0.35,
                    help="the baseline's detector ran at this confidence: weaker phone boxes never reached it")
    ap.add_argument("--variant", nargs=2, action="append", default=[], metavar=("NAME", "CONFIG_YAML"),
                    help="extra run of this version with a config (ablation), e.g. --variant no_recovery cfg.yaml")
    ap.add_argument("--out", default="results/logic_benchmark")
    args = ap.parse_args()

    out = Path(args.out)
    names = generate(out, args.sessions, args.minutes, args.seed)
    versions = {}  # label -> (code root, config)
    if args.baseline_root:
        versions["baseline"] = (Path(args.baseline_root).resolve(), None)
    versions["this version"] = (ROOT, None)
    for name, cfg in args.variant:
        versions[name] = (ROOT, cfg)
    results = {}
    for k, (label, (root, cfg)) in enumerate(versions.items()):
        print(f"running {label} ({root}{', ' + cfg if cfg else ''}) on {len(names)} drives ...", flush=True)
        tag = f"v{k}_{label.replace(' ', '_')}"
        run_version(root, out, names, tag, cfg, args.baseline_phone_conf if label == "baseline" else 0.0)
        results[label] = evaluate(tag, out, names)

    hours = next(iter(results.values()))["alarms"]["driving_hours"]
    lines = [f"# Logic benchmark: {len(names)} synthetic drives x {args.minutes:g} min ({hours:.1f} h)", "",
             "Synthetic drives (dms.simulation.synthetic_drive) - this checks the rules and the alert policy; "
             "it is NOT an accuracy estimate.", "",
             "| Metric | " + " | ".join(results) + " |", "|---|" + "---|" * len(results)]
    for b, view in (("drowsiness", "frame"), ("yawning", "event"), ("gaze_deviation", "event"), ("phone_usage", "event")):
        lines.append(row(f"{b} F1 ({view}, primary)", lambda r, b=b, v=view: r[b][v]["f1"], results))
        lines.append(row(f"{b} precision / recall ({view})",
                         lambda r, b=b, v=view: f"{r[b][v]['precision']:.2f} / {r[b][v]['recall']:.2f}", results))
    lines.append(row("eye_closure (microsleep) event F1", lambda r: r["eye_closure"]["event"]["f1"], results))
    lines.append(row("drowsiness 3-class macro F1 (Table 1.11)", lambda r: r["drowsiness_states"]["macro_f1"], results))
    lines.append(row("drowsiness frame FPR", lambda r: r["drowsiness"]["frame"]["fpr"], results))
    lines.append(row("unnecessary alarms / non-event hour (SO5 <= 0.5)",
                     lambda r: r["alarms"]["alarms"]["per_non_event_hour"], results))
    lines.append(row("alarms started (total)", lambda r: r["alarms"]["alarms"]["count"], results))
    lines.append(row("unnecessary pushes", lambda r: r["alarms"]["pushes"]["unnecessary"], results))
    lines.append(row("buzzer sounds outside events / non-event hour", lambda r: r["sound"]["sounds_outside_events_per_non_event_hour"], results))
    lines.append(row("sound after an event ended: mean tail (s)", lambda r: r["sound"]["tail_mean_s"], results))
    lines.append(row("sound after an event ended: max tail (s)", lambda r: r["sound"]["tail_max_s"], results))
    lines.append(row("events followed by > 5 s of sound", lambda r: f"{r['sound']['events_with_tail_over_5s']} / {r['sound']['events']}", results))
    md = "\n".join(lines) + "\n"
    (out / "results.md").write_text(md, encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print("\n" + md)
    print(f"Saved {out / 'results.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
