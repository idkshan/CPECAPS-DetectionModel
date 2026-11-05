"""Save still frames from videos, e.g. SynDD1 dashboard videos for phone labelling.

State Farm / AUC show the driver from the passenger side; the ESP32-CAM looks at
the driver from the dashboard. Adding phone frames from an approved dashboard-view
dataset (SynDD1 "Dashboard" videos) to the phone dataset reduces this camera-angle
gap. Upload the saved frames to the same Roboflow project as the State Farm
phones, draw/check the phone boxes there, and export everything together.

    # every 2 s of every video
    python scripts/extract_frames.py --videos data/syndd1/Dashboard --out data/syndd1_frames --every-sec 2

    # only inside listed time ranges (e.g. the phone-call / texting segments)
    python scripts/extract_frames.py --videos data/syndd1/Dashboard --out data/syndd1_phone_frames \
        --ranges data/syndd1_phone_ranges.csv --every-sec 1

Ranges CSV: video,start_s,end_s  (video = file name without extension). Frames are
named <video>_<seconds>.jpg so the driver/video can still be traced (split by person!).
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2

import _bootstrap  # noqa: F401
from process_videos import find_videos


def load_ranges(path: str):
    ranges = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            ranges[Path(row["video"].strip()).stem].append((float(row["start_s"]), float(row["end_s"])))
    return ranges


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--videos", required=True, help="folder with videos (searched recursively) or one video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every-sec", type=float, default=2.0, help="keep one frame every N seconds")
    ap.add_argument("--ranges", default=None, help="CSV video,start_s,end_s: only keep frames inside these ranges")
    ap.add_argument("--max-per-video", type=int, default=None)
    args = ap.parse_args()

    src = Path(args.videos)
    videos = [src] if src.is_file() else find_videos(src)
    if not videos:
        raise SystemExit(f"No videos found in {src}")
    ranges = load_ranges(args.ranges) if args.ranges else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for video in videos:
        if ranges is not None and video.stem not in ranges:
            continue
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps * args.every_sec)))
        i = kept = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            t = i / fps
            inside = ranges is None or any(s <= t <= e for s, e in ranges[video.stem])
            if i % step == 0 and inside:
                ok, img = cap.retrieve()
                if ok:
                    cv2.imwrite(str(out / f"{video.stem}_{t:07.1f}.jpg"), img)
                    kept += 1
                    if args.max_per_video and kept >= args.max_per_video:
                        break
            i += 1
        cap.release()
        total += kept
        print(f"{video.name}: {kept} frames")
    print(f"\n{total} frames saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
