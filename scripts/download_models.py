"""Download the pretrained models used by the detection pipeline into models/.

    python scripts/download_models.py            # face landmarker + 3 YOLO nano detectors
    python scripts/download_models.py --cls      # also the YOLO nano classifiers (eye-state training)
    python scripts/download_models.py --sizes n s  # nano and small variants
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401

FACE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/{name}.pt"
FAMILIES = ("yolov8", "yolo11", "yolo26")


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists  {dest}")
        return
    print(f"  get     {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f"  saved   {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--out", default="models")
    ap.add_argument("--sizes", nargs="+", default=["n"], help="YOLO sizes: n s m l x")
    ap.add_argument("--cls", action="store_true", help="also download -cls classification weights")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fetch(FACE_URL, out / "face_landmarker.task")
    for fam in FAMILIES:
        for size in args.sizes:
            fetch(YOLO_URL.format(name=f"{fam}{size}"), out / f"{fam}{size}.pt")
            if args.cls:
                fetch(YOLO_URL.format(name=f"{fam}{size}-cls"), out / f"{fam}{size}-cls.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
