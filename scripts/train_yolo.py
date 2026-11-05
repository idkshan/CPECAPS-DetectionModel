"""Fine-tune several YOLO variants with *identical* settings (fair comparison).

Detection (phone detector, Section 1.7.8):
    python scripts/train_yolo.py --data datasets/phone/data.yaml \
        --models models/yolov8n.pt models/yolo11n.pt models/yolo26n.pt --imgsz 640 --epochs 100

Classification (optional eye-state model on MRL Eye, see prepare_mrl_eye.py):
    python scripts/train_yolo.py --task classify --data datasets/mrl_eye \
        --models models/yolov8n-cls.pt models/yolo11n-cls.pt models/yolo26n-cls.pt --imgsz 96 --epochs 30

A GPU is strongly recommended (``--device 0``); Google Colab's free GPU is
enough for nano models. Results for every model are collected in
``<project>/training_summary.json`` so the thesis can report them side by side.
"""

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--data", required=True, help="data.yaml (detect) or dataset folder (classify)")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--task", choices=["detect", "classify"], default="detect")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="0 for first GPU, cpu for CPU")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", default=None, help="output folder (default runs/<task>)")
    args = ap.parse_args()

    from ultralytics import YOLO

    project = Path(args.project or f"runs/{args.task}").resolve()
    summary = {"settings": vars(args), "models": {}}
    for weights in args.models:
        name = Path(weights).stem
        print(f"\n========== training {name} ==========")
        model = YOLO(weights, task=args.task)
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            seed=args.seed,
            deterministic=True,
            device=args.device,
            workers=args.workers,
            project=str(project),
            name=name,
            exist_ok=True,
            plots=True,
        )
        best = project / name / "weights" / "best.pt"
        metrics = YOLO(str(best)).val(data=args.data, imgsz=args.imgsz, device=args.device, plots=False, verbose=False)
        if args.task == "detect":
            p, r = float(metrics.box.mp), float(metrics.box.mr)
            result = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                      "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)}
        else:
            result = {"top1": float(metrics.top1), "top5": float(metrics.top5)}
        summary["models"][name] = {"best_weights": str(best), "val": {k: round(v, 4) for k, v in result.items()}}
        print(json.dumps(summary["models"][name], indent=2))

    project.mkdir(parents=True, exist_ok=True)
    (project / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {project / 'training_summary.json'}")
    print("Next: python scripts/benchmark_yolo.py --models <best.pt files> --data <data.yaml> --split test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
