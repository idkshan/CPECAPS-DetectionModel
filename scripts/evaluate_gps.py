"""GPS accuracy and update rate against a reference (SO4).

    # static test: the receiver sat on a known point for >= 10 minutes
    python scripts/evaluate_gps.py --fixes results/sessions/G01/gps.csv --ref-point 14.564912 120.993215 --condition open_sky

    # driving test with a reference receiver logging at the same time
    python scripts/evaluate_gps.py --fixes gps.csv --ref-track reference.csv --time-offset 0 --condition urban

--fixes: CSV with columns t, lat, lon (run_monitor.py writes gps.csv; the
Android app can export the same). --ref-track: CSV with t, lat, lon on a clock
that equals fix time + --time-offset. Targets: median error <= 5 m (open_sky)
or <= 15 m (urban); updates at 1 Hz. Results: printed and saved as JSON.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from dms.evaluation.gps_accuracy import errors_to_point, errors_to_track, summarize_errors, update_rate

TARGETS = {"open_sky": 5.0, "urban": 15.0}


def read_points(path):
    out = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                if row.get("fix", "True") in ("False", "false", "0"):
                    continue
                out.append((float(row["t"]), float(row["lat"]), float(row["lon"])))
            except (TypeError, ValueError, KeyError):
                continue  # rows without a position
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--fixes", required=True)
    ref = ap.add_mutually_exclusive_group(required=True)
    ref.add_argument("--ref-point", nargs=2, type=float, metavar=("LAT", "LON"))
    ref.add_argument("--ref-track")
    ap.add_argument("--time-offset", type=float, default=0.0, help="seconds added to fix times to match the reference clock")
    ap.add_argument("--condition", choices=sorted(TARGETS), default="open_sky")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fixes = read_points(args.fixes)
    if not fixes:
        raise SystemExit(f"No usable fixes (t, lat, lon) in {args.fixes}")
    if args.ref_point:
        err = errors_to_point(fixes, *args.ref_point)
    else:
        err = errors_to_track(fixes, read_points(args.ref_track), args.time_offset)
    report = {"condition": args.condition, "reference": "point" if args.ref_point else "track",
              "accuracy": summarize_errors(err, TARGETS[args.condition]), "update_rate": update_rate([f[0] for f in fixes])}
    print(json.dumps(report, indent=2))
    out = Path(args.out or Path(args.fixes).with_suffix(".accuracy.json"))
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
