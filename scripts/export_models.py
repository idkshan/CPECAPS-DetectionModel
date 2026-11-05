"""Export trained YOLO models to Android formats (TFLite FP32 / FP16 / INT8, NCNN).

    python scripts/export_models.py \
        --models runs/detect/yolov8n/weights/best.pt runs/detect/yolo11n/weights/best.pt runs/detect/yolo26n/weights/best.pt \
        --imgsz 320 640 --formats tflite tflite_fp16 tflite_int8 --data datasets/driver_state/data.yaml

Files are written to exports/android/ as <model>_<imgsz>_<format>.tflite
(e.g. yolo11n_320_tflite_fp16.tflite), the names android_benchmark.py expects.
INT8 needs --data: its images calibrate the quantisation.
TFLite export needs TensorFlow (pip install tensorflow); Google Colab has it.
"""

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from benchmark_yolo import EXPORT_ARGS, export, model_name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--imgsz", nargs="+", type=int, default=[320])
    ap.add_argument("--formats", nargs="+", default=["tflite", "tflite_fp16", "tflite_int8"], choices=sorted(EXPORT_ARGS))
    ap.add_argument("--data", default=None, help="data.yaml for INT8 calibration")
    ap.add_argument("--out", default="exports/android")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done, failed = [], []
    for m in args.models:
        for imgsz in args.imgsz:
            for fmt in args.formats:
                label = f"{model_name(Path(m))} @{imgsz} [{fmt}]"
                if fmt == "tflite_int8" and not args.data:
                    print(f"SKIP {label}: INT8 needs --data")
                    continue
                try:
                    path = export(Path(m), fmt, imgsz, out, args.data)
                    done.append(path)
                    print(f"OK   {label} -> {path}")
                except Exception as exc:
                    failed.append(label)
                    print(f"FAIL {label}: {exc}")
    print(f"\n{len(done)} exported to {out}, {len(failed)} failed")
    print("Next: python scripts/android_benchmark.py --models exports/android")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
