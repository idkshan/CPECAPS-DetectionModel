"""Replay Stage-2 features through the decision engine of ANY code version.

Used by benchmark_logic.py to run the baseline (an older checkout) and this
version on identical inputs. Standalone on purpose: it imports ``dms`` from
--code-root, not from this repository.

    python scripts/_replay_runner.py --code-root <checkout> --features drive/features.jsonl --out <folder>
"""

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed", type=float, default=40.0)
    ap.add_argument("--config", default=None, help="YAML config (only for code versions that support load_config)")
    ap.add_argument("--min-phone-conf", type=float, default=0.0,
                    help="drop phone confidences below this (what an older detector running at this threshold would log)")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.code_root).resolve()))
    from dms.config import Config, load_config
    from dms.engine import DecisionEngine
    from dms.types import ContextReading, FrameFeatures

    known = {f.name for f in fields(FrameFeatures)}
    engine = DecisionEngine(load_config(args.config) if args.config else Config())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = None
    with open(args.features, encoding="utf-8") as fin, open(out / "frames.jsonl", "w", encoding="utf-8") as ff, \
            open(out / "alerts.jsonl", "w", encoding="utf-8") as fa:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("phone_conf", 0.0) < args.min_phone_conf:
                rec["phone_conf"], rec["phone_boxes"] = 0.0, 0
            feats = FrameFeatures(**{k: v for k, v in rec.items() if k in known})
            t0 = feats.t if t0 is None else t0
            res = engine.step(feats, ContextReading(speed_kmh=args.speed, motion_g=None, driving_s=feats.t - t0))
            ff.write(json.dumps({"t": feats.t, "behavior": asdict(res.behavior),
                                 "risk": {"tier": res.risk.tier, "drowsiness_level": res.risk.fatigue_level}}) + "\n")
            for a in res.alerts:
                fa.write(json.dumps(a.to_dict() if hasattr(a, "to_dict") else asdict(a)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
