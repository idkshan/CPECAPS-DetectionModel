"""Measure exported YOLO models ON an Android phone (speed, memory, temperature).

Uses Google's official TensorFlow Lite benchmark tool, run on the phone over
USB with ADB, so every model is measured the same way on the real hardware:

  * inference time (median of repeats, average over --runs inferences)
  * initialisation time and peak memory
  * CPU (--threads), GPU delegate and NNAPI (phone NPU/DSP) where supported
  * battery temperature before / after each model (thermal load)

Requirements
------------
  1. Android phone with Developer options -> USB debugging ON, connected by USB
  2. ADB (Android SDK Platform-Tools) installed and on PATH (`adb devices` lists the phone)
  3. Exported .tflite files (scripts/export_models.py)

Example
-------
    python scripts/android_benchmark.py --models exports/android --delegates cpu gpu nnapi \
        --merge results/yolo_benchmark/results.csv

The benchmark tool is downloaded automatically the first time (--download-tool).

Note: this measures the network only. YOLOv8 / YOLO11 still need NMS after the
network (in the app), while YOLO26 is NMS-free; add the desktop post_ms column
or measure the app end-to-end for the complete comparison (see the guide).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import _bootstrap  # noqa: F401

TOOL_URL = "https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model"
REMOTE_DIR = "/data/local/tmp"
NAME_RE = re.compile(r"^(?P<model>.+)_(?P<imgsz>\d+)_(?P<fmt>tflite(?:_fp16|_int8)?)$")


def adb(args: List[str], serial: Optional[str], timeout: int = 600) -> str:
    cmd = ["adb"] + (["-s", serial] if serial else []) + args
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return (res.stdout or "") + (res.stderr or "")


def device_info(serial: Optional[str]) -> Dict[str, str]:
    prop = lambda k: adb(["shell", "getprop", k], serial).strip()
    return {
        "phone": f"{prop('ro.product.manufacturer')} {prop('ro.product.model')}".strip(),
        "soc": prop("ro.soc.model") or prop("ro.board.platform") or prop("ro.hardware"),
        "android": prop("ro.build.version.release"),
        "abi": prop("ro.product.cpu.abi"),
    }


def battery_temp_c(serial: Optional[str]) -> Optional[float]:
    m = re.search(r"temperature:\s*(\d+)", adb(["shell", "dumpsys", "battery"], serial))
    return int(m.group(1)) / 10.0 if m else None


def parse_benchmark(text: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"inference_ms": None, "init_ms": None, "first_ms": None, "peak_mem_mb": None}
    m = re.search(r"Inference timings in us:\s*Init:\s*([\d.]+),\s*First inference:\s*([\d.]+),.*?Inference \(avg\):\s*([\d.e+]+)", text, re.S)
    if m:
        out["init_ms"] = float(m.group(1)) / 1000
        out["first_ms"] = float(m.group(2)) / 1000
        out["inference_ms"] = float(m.group(3)) / 1000
    m = re.search(r"Peak memory footprint \(MB\):\s*init=([\d.]+)\s+overall=([\d.]+)", text)
    if m:
        out["peak_mem_mb"] = float(m.group(2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--models", nargs="+", default=["exports/android"], help=".tflite files or folders")
    ap.add_argument("--delegates", nargs="+", default=["cpu", "gpu"], choices=["cpu", "gpu", "nnapi"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3, help="repeat each measurement; the median is reported")
    ap.add_argument("--tool", default="tools/android_aarch64_benchmark_model", help="local benchmark tool binary")
    ap.add_argument("--serial", default=None, help="adb device serial when several phones are connected")
    ap.add_argument("--merge", default=None, help="desktop results.csv (benchmark_yolo.py) to add accuracy columns")
    ap.add_argument("--out", default="results/android_benchmark")
    args = ap.parse_args()

    devices = adb(["devices"], None)
    if "\tdevice" not in devices:
        print(devices)
        raise SystemExit("No phone found. Enable USB debugging, connect the phone, accept the prompt, and check `adb devices`.")

    tool = Path(args.tool)
    if not tool.exists():
        tool.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading the TFLite benchmark tool -> {tool}")
        urllib.request.urlretrieve(TOOL_URL, tool)
    adb(["push", str(tool), f"{REMOTE_DIR}/benchmark_model"], args.serial)
    adb(["shell", "chmod", "+x", f"{REMOTE_DIR}/benchmark_model"], args.serial)
    info = device_info(args.serial)
    print("Phone:", info)

    files: List[Path] = []
    for m in args.models:
        p = Path(m)
        files += sorted(p.glob("*.tflite")) if p.is_dir() else [p]
    if not files:
        raise SystemExit("No .tflite files found. Run scripts/export_models.py first.")

    accuracy: Dict[tuple, dict] = {}
    if args.merge and Path(args.merge).exists():
        for row in csv.DictReader(open(args.merge, newline="", encoding="utf-8")):
            accuracy[(row.get("model"), str(row.get("imgsz")), row.get("format"))] = row

    rows = []
    for f in files:
        m = NAME_RE.match(f.stem)
        meta = m.groupdict() if m else {"model": f.stem, "imgsz": "", "fmt": "tflite"}
        remote = f"{REMOTE_DIR}/{f.name}"
        adb(["push", str(f), remote], args.serial)
        for delegate in args.delegates:
            t_before = battery_temp_c(args.serial)
            results = []
            for _ in range(args.repeats):
                cmd = ["shell", f"{REMOTE_DIR}/benchmark_model", f"--graph={remote}", f"--num_threads={args.threads}",
                       f"--warmup_runs={args.warmup}", f"--num_runs={args.runs}",
                       f"--use_gpu={'true' if delegate == 'gpu' else 'false'}",
                       f"--use_nnapi={'true' if delegate == 'nnapi' else 'false'}"]
                results.append(parse_benchmark(adb(cmd, args.serial, timeout=1800)))
            ok = [r for r in results if r["inference_ms"] is not None]
            t_after = battery_temp_c(args.serial)
            row = {
                "model": meta["model"], "imgsz": meta["imgsz"], "format": meta["fmt"], "delegate": delegate,
                "threads": args.threads, "size_mb": round(f.stat().st_size / 1e6, 2),
                "inference_ms_median": round(statistics.median(r["inference_ms"] for r in ok), 2) if ok else None,
                "inference_ms_min": round(min(r["inference_ms"] for r in ok), 2) if ok else None,
                "inference_ms_max": round(max(r["inference_ms"] for r in ok), 2) if ok else None,
                "init_ms": round(statistics.median(r["init_ms"] for r in ok), 1) if ok else None,
                "peak_mem_mb": max((r["peak_mem_mb"] for r in ok if r["peak_mem_mb"] is not None), default=None),
                "fps_network": round(1000 / statistics.median(r["inference_ms"] for r in ok), 1) if ok else None,
                "battery_temp_before_c": t_before, "battery_temp_after_c": t_after,
                "status": "ok" if ok else "failed (delegate not supported?)",
            }
            # accuracy of this exact export if benchmark_yolo.py measured it, else of the PyTorch model (same weights)
            acc = accuracy.get((meta["model"], meta["imgsz"], meta["fmt"])) or accuracy.get((meta["model"], meta["imgsz"], "pytorch"))
            if acc:
                row["desktop_accuracy_from"] = acc.get("format")
                for k in ("precision", "recall", "f1", "ap50", "post_ms", "eye_open_f1", "eye_closed_f1", "yawn_f1", "phone_f1"):
                    if acc.get(k) not in (None, ""):
                        row[f"desktop_{k}"] = acc[k]
            rows.append(row)
            print(json.dumps(row))
        adb(["shell", "rm", "-f", remote], args.serial)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    cols = [c for c in ["model", "imgsz", "format", "delegate", "size_mb", "inference_ms_median", "fps_network", "init_ms",
                        "peak_mem_mb", "battery_temp_after_c", "desktop_f1", "desktop_ap50", "status"] if c in keys]
    md = [f"Phone: {info['phone']} | SoC: {info['soc']} | Android {info['android']} | threads {args.threads} | "
          f"{args.runs} runs x {args.repeats} repeats\n", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    (out / "results.md").write_text("\n".join(md) + "\n")
    (out / "device.json").write_text(json.dumps(info, indent=2))
    print("\n" + "\n".join(md))
    print(f"\nSaved to {out}/results.(csv|md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
