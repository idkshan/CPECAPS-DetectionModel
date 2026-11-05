"""Run the detection model on every recorded participant video in a folder.

    python scripts/process_videos.py --videos data/study/videos --yolo models/driver_state_best.pt

Each video becomes one session folder results/study_sessions/<video name>/
(frames.jsonl, alerts.jsonl, summary.json), the input evaluate_study.py needs.
Name the videos after the session, e.g. P01_day.mp4, P01_night.mp4.

Videos that already have a finished session (summary.json) are skipped, so the
command can be stopped and restarted; use --overwrite to redo them.
Every option after "--" is passed on to run_monitor.py unchanged, e.g.
    python scripts/process_videos.py --videos data/study/videos -- --imgsz 320 --conf 0.4
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
RUN_MONITOR = Path(__file__).with_name("run_monitor.py")


def find_videos(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def main() -> int:
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--videos", required=True, help="folder with the recorded videos (searched recursively)")
    ap.add_argument("--log-dir", default="results/study_sessions", help="where the session folders are written")
    ap.add_argument("--yolo", default=None, help="YOLO weights (the trained driver-state model)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--overwrite", action="store_true", help="re-process videos that already have a session")
    ap.add_argument("--save-video", action="store_true", help="also write an annotated copy next to each session")
    args = ap.parse_args(argv)

    videos = find_videos(Path(args.videos))
    if not videos:
        raise SystemExit(f"No videos found in {args.videos}")
    names = [v.stem for v in videos]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise SystemExit(f"Two videos share a name (session names must be unique): {dupes}")

    done, skipped, failed = 0, 0, []
    for n, video in enumerate(videos, 1):
        session = Path(args.log_dir) / video.stem
        if (session / "summary.json").exists() and not args.overwrite:
            print(f"[{n}/{len(videos)}] skip {video.name} (already processed)")
            skipped += 1
            continue
        cmd = [sys.executable, str(RUN_MONITOR), "--source", str(video), "--no-show", "--quiet",
               "--log-dir", args.log_dir, "--session-name", video.stem]
        if args.yolo:
            cmd += ["--yolo", args.yolo]
        if args.config:
            cmd += ["--config", args.config]
        if args.save_video:
            cmd += ["--save-video", str(session / f"{video.stem}_annotated.mp4")]
        cmd += passthrough
        print(f"[{n}/{len(videos)}] {video.name} -> {session}")
        t0 = time.time()
        res = subprocess.run(cmd)
        if res.returncode != 0 or not (session / "summary.json").exists():
            failed.append(video.name)
            print(f"   FAILED ({video.name})")
        else:
            done += 1
            print(f"   done in {time.time() - t0:.0f} s")
    print(f"\n{done} processed, {skipped} skipped, {len(failed)} failed{': ' + ', '.join(failed) if failed else ''}")
    print("Next: python scripts/evaluate_study.py --manifest data/study/manifest.csv")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
