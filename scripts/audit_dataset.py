"""Audit a dataset BEFORE training or reporting results on it.

Finds corrupt images, malformed labels, class imbalance, identical or
near-identical images in different splits, and drivers / persons that appear
in more than one split (leakage that inflates test scores).

    # driver-state dataset built by build_driver_state_dataset.py (its split
    # manifest says which person / driver every image came from)
    python scripts/audit_dataset.py --data datasets/driver_state --groups datasets/driver_state/split_manifest.csv

    # a Roboflow re-export: map files back to State Farm drivers
    python scripts/audit_dataset.py --data datasets/phone_roboflow --groups data/state-farm/driver_imgs_list.csv

    # MRL Eye classification dataset (subject = sXXXX in the file name)
    python scripts/audit_dataset.py --data datasets/mrl_eye --classification

Built-in group rules (used when a file matches): YawDD frames (yawdd_<n>-<Female|Male>...),
MRL Eye crops (sXXXX_...). Add more with --group-regex (first capture group = group).
Writes <out>/audit.json and <out>/audit.md; exit code 1 when leakage is found.
"""

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from dms.data.audit import audit_classification, audit_detection, markdown
from dms.data.splits import group_from_name, load_group_map, original_name

BUILTIN_RULES = [
    r"^yawdd_(\d+-?(?:female|male))",  # frames written by build_driver_state_dataset.py
    r"^(s\d{4})_\d+_",                  # MRL Eye file names
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--data", required=True, help="dataset root (folder with data.yaml, or <split>/<class>/ for --classification)")
    ap.add_argument("--classification", action="store_true", help="classification layout (e.g. MRL Eye)")
    ap.add_argument("--groups", nargs="*", default=[], help="CSV(s) mapping file -> group (driver_imgs_list.csv, split_manifest.csv)")
    ap.add_argument("--group-regex", nargs="*", default=[], help="regex(es) on the file name; first capture group = group")
    ap.add_argument("--near-threshold", type=int, default=4, help="max differing hash bits for near-duplicates (0-4)")
    ap.add_argument("--no-duplicates", action="store_true", help="skip image decoding / duplicate search (fast)")
    ap.add_argument("--out", default=None, help="output folder (default results/dataset_audit/<dataset name>)")
    args = ap.parse_args()
    if not 0 <= args.near_threshold <= 4:
        raise SystemExit("--near-threshold must be 0..4 (the band search is exact only up to 4 bits)")

    root = Path(args.data)
    group_map = {}
    for csv_path in args.groups:
        group_map.update(load_group_map(Path(csv_path)))
    rules = [re.compile(r, re.IGNORECASE) for r in [*args.group_regex, *BUILTIN_RULES]]

    def group_of(name: str):
        if group_map:
            g = group_from_name(name, group_map)
            if g is not None:
                return g
        base = original_name(name)
        for rule in rules:
            m = rule.search(base)
            if m:
                return (m.group(1) if m.groups() else m.group(0)).lower()
        return None

    sample = [p.name for p in itertools.islice((q for q in root.rglob("*") if q.is_file()), 500)]
    use_groups = bool(group_map or args.group_regex) or any(r.search(n) for n in sample for r in rules)
    kwargs = dict(group_of=group_of if use_groups else None, near_threshold=args.near_threshold,
                  check_duplicates=not args.no_duplicates)
    if args.classification:
        report = audit_classification(root, **kwargs)
    else:
        report = audit_detection(root, progress=lambda m: print(f"  {m}", flush=True), **kwargs)
    out = Path(args.out or f"results/dataset_audit/{root.name}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md = markdown(report)
    (out / "audit.md").write_text(md, encoding="utf-8")
    print(md)
    if not use_groups:
        print("NOTE: no group information - pass --groups / --group-regex to check driver/person leakage.")
    print(f"Saved {out / 'audit.md'} and audit.json")
    leakage = any(word in p for p in report["problems"] for word in ("leakage",))
    return 1 if leakage else 0


if __name__ == "__main__":
    sys.exit(main())
