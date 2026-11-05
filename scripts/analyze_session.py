"""Performance report of recorded sessions: FPS, stage latency, uptime, GPS rate (SO1, SO3/SO4, SO6).

    python scripts/analyze_session.py results/sessions/S01 [results/sessions/S02 ...]

Prints and saves <session>/performance.json and performance.md. What these
numbers can and cannot show is listed in dms/evaluation/performance.py and
docs/SO_COMPLIANCE.md (e.g. desktop processing FPS is not Android FPS).
"""

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from dms.evaluation.performance import session_performance, so_checks
from dms.evaluation.session import load_session


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--camera-timeout", type=float, default=3.0)
    args = ap.parse_args()
    for s in args.sessions:
        folder = Path(s)
        frames, alerts, summary = load_session(folder)
        perf = session_performance(frames, summary, alerts, args.camera_timeout)
        checks = so_checks(perf)
        uptime = f"{100 * perf['uptime']:.1f} %" if perf.get("uptime") is not None else "—"
        lines = [f"# Performance: {folder.name}", "",
                 f"{perf['frames']} frames, {perf['duration_min']} min; input {perf['input_fps']} FPS; "
                 f"processing {perf.get('processing_fps', '—')} FPS (this computer); uptime {uptime}", "",
                 "| SO | Metric | Target | Measured | Status |", "|---|---|---|---|---|"]
        lines += [f"| {c['so']} | {c['metric']} | {c['target']} | {c['measured'] if c['measured'] is not None else '—'} | {c['status']} |"
                  for c in checks]
        if "stage_latency_ms" in perf:
            lines += ["", "| Stage | mean ms | p50 | p95 |", "|---|---|---|---|"]
            lines += [f"| {k} | {v['mean']} | {v['p50']} | {v['p95']} |" for k, v in perf["stage_latency_ms"].items()]
        md = "\n".join(lines) + "\n"
        (folder / "performance.json").write_text(json.dumps({"performance": perf, "so_checks": checks}, indent=2, default=str))
        (folder / "performance.md").write_text(md)
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
