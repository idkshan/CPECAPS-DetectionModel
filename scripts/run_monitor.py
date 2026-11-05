"""Run the driver-monitoring detection model on a camera, video, folder or ESP32-CAM.

Examples
--------
    # laptop webcam, COCO-pretrained YOLOv8n (phones only)
    python scripts/run_monitor.py --source 0

    # your trained driver-state model (eyes, yawns, phones)
    python scripts/run_monitor.py --source 0 --yolo runs/detect/yolo11n/weights/best.pt

    # ESP32-CAM stream, context from the ESP32, alerts to its buzzer
    python scripts/run_monitor.py --source http://192.168.4.1:81/stream \\
        --context-url http://192.168.4.1/sensors --alert-url http://192.168.4.1/alert

    # recorded video, no window, save annotated copy
    python scripts/run_monitor.py --source drive.mp4 --no-show --save-video out.mp4

Live sources are read in a background thread and only the newest frame is
processed (stale frames are dropped, so latency cannot build up). If no frame
arrives for alerts.camera_timeout_s (3 s), any alert is stopped and a
"monitoring unavailable" event is logged; that time counts as downtime (SO6).

Keys (window mode): q / Esc = quit, c = recalibrate (new driver).
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2

import _bootstrap  # noqa: F401
from dms import __version__
from dms.alerts import ConsoleAlertSink, HttpAlertSink
from dms.camera import LatestFrameReader, StreamHealth, open_source
from dms.camera.sources import is_live
from dms.config import EYE_MODES, MOUTH_MODES, apply_evidence_modes, load_config
from dms.context import HttpContext, StaticContext
from dms.monitor import DriverMonitor
from dms.overlay import draw
from dms.session_log import SessionLogger


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--source", default="0", help="camera index, video file, image folder, or http(s) URL")
    ap.add_argument("--config", default=None, help="YAML config (defaults: configs/default.yaml values)")
    ap.add_argument("--yolo", default=None, help="YOLO weights/export (e.g. runs/detect/yolo11n/weights/best.pt)")
    ap.add_argument("--imgsz", type=int, default=None, help="YOLO inference size (e.g. 320 for mobile-like speed)")
    ap.add_argument("--conf", type=float, default=None, help="YOLO confidence threshold (default 0.35)")
    ap.add_argument("--yolo-every", type=int, default=None, help="run YOLO every N frames")
    ap.add_argument("--no-yolo", action="store_true", help="disable the YOLO detector")
    ap.add_argument("--eye-model", default=None, help="optional eye-state classifier (YOLO-cls weights)")
    ap.add_argument("--eye-mode", choices=sorted(EYE_MODES), default=None, help="eye evidence sources (ablation)")
    ap.add_argument("--mouth-mode", choices=sorted(MOUTH_MODES), default=None, help="mouth evidence sources (ablation)")
    ap.add_argument("--speed", type=float, default=None, help="fixed speed in km/h (default: unknown -> assumed moving)")
    ap.add_argument("--motion", type=float, default=None, help="fixed motion level in g")
    ap.add_argument("--driving-minutes", type=float, default=0.0, help="pretend the trip started this many minutes ago")
    ap.add_argument("--context-url", default=None, help="ESP32/phone JSON endpoint with speed_kmh / motion_g / lat / lon")
    ap.add_argument("--alert-url", default=None, help="ESP32 endpoint that drives the buzzer")
    ap.add_argument("--alert-style", choices=["beeps", "state"], default="beeps", help="buzzer firmware style (see dms/alerts/sinks.py)")
    ap.add_argument("--no-calibration", action="store_true", help="use population default thresholds")
    ap.add_argument("--fps", type=float, default=10.0, help="frame rate for image folders / JPEG polling")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--log-dir", default="results/sessions")
    ap.add_argument("--session-name", default=None)
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--show", dest="show", action="store_true", default=True)
    ap.add_argument("--no-show", dest="show", action="store_false")
    ap.add_argument("--save-video", default=None, help="write the annotated video here")
    ap.add_argument("--quiet", action="store_true", help="do not print alerts to the console")
    return ap.parse_args()


def fingerprint(path) -> dict:
    """Model file identity for the session log (which weights produced these results?)."""
    if not path:
        return {"path": None}
    p = Path(path)
    if not p.is_file():
        return {"path": str(path), "exists": False}
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(path), "sha256_16": h.hexdigest()[:16], "size_mb": round(p.stat().st_size / 1e6, 2)}


def main() -> int:
    args = parse_args()
    overrides = {}
    if args.yolo:
        overrides.setdefault("detector", {})["model_path"] = args.yolo
    if args.imgsz:
        overrides.setdefault("detector", {})["imgsz"] = args.imgsz
    if args.conf is not None:
        overrides.setdefault("detector", {})["conf"] = args.conf
    if args.yolo_every:
        overrides.setdefault("detector", {})["every_n_frames"] = args.yolo_every
    if args.eye_model:
        overrides["eye_model"] = {"model_path": args.eye_model}
    if args.no_calibration:
        overrides["calibration"] = {"enabled": False}
    cfg = load_config(args.config, overrides)
    apply_evidence_modes(cfg, args.eye_mode, args.mouth_mode)

    context = (HttpContext(args.context_url, cfg=cfg.context) if args.context_url
               else StaticContext(args.speed, args.motion, args.driving_minutes * 60, cfg=cfg.context))
    sinks = [] if args.quiet else [ConsoleAlertSink()]
    if args.alert_url:
        sinks.append(HttpAlertSink(args.alert_url, style=args.alert_style))

    monitor = DriverMonitor(cfg, context=context, use_detector=not args.no_yolo, alert_sinks=sinks)
    logger = None
    if not args.no_log:
        detector_names = sorted(str(v) for v in monitor.detector.model.names.values()) if monitor.detector else []
        meta = {
            "source": args.source,
            "models": {"face": fingerprint(cfg.face.model_path),
                       "detector": None if args.no_yolo else {**fingerprint(cfg.detector.model_path), "classes": detector_names},
                       "eye_model": fingerprint(cfg.eye_model.model_path)},
            "config": cfg.to_dict(),  # the exact settings of this run (replay with the same config)
            "command": " ".join(sys.argv),
        }
        logger = SessionLogger(args.log_dir, args.session_name, meta)
        print(f"Logging to {logger.dir}")

    live = is_live(args.source)
    health = StreamHealth(cfg.alerts.camera_timeout_s)
    stream_events = []
    source = open_source(args.source, fps=args.fps, max_frames=args.max_frames,
                         on_event=lambda e: stream_events.append({"t": round(time.monotonic(), 3), "event": e}))
    reader = LatestFrameReader(source) if live else None
    writer = None
    fps_ema = None
    last = time.perf_counter()
    t_last_frame = 0.0
    warned = False

    print("Calibrating: look straight ahead with a relaxed face for a few seconds..." if cfg.calibration.enabled else "Calibration disabled")
    try:
        frames = iter(source) if reader is None else None
        while True:
            if reader is not None:
                frame = reader.read(timeout_s=cfg.alerts.camera_timeout_s)
                if frame is None:
                    if reader.finished:
                        break
                    t_now = t_last_frame + cfg.alerts.camera_timeout_s
                    health.no_frame(t_now)
                    cmds = monitor.camera_lost(t_now, f"no frame for {cfg.alerts.camera_timeout_s:g} s")
                    if logger:
                        logger.log_alerts(cmds)
                    continue
            else:
                frame = next(frames, None)
                if frame is None:
                    break
            t_last_frame = frame.t
            health.frame(frame.t)
            result = monitor.process(frame.image, frame.t, frame.index)
            if logger:
                logger.log(result)
            cal = monitor.engine.calibrator
            if cal.warning and not warned:
                print(f"WARNING: {cal.warning}")
                warned = True
            now = time.perf_counter()
            inst = 1.0 / max(1e-6, now - last)
            last = now
            fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst

            if args.show or args.save_video:
                vis = draw(frame.image, result, fps_ema, warning=cal.warning)
                if args.save_video:
                    if writer is None:
                        h, w = vis.shape[:2]
                        writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), args.fps if args.fps else 10, (w, h))
                    writer.write(vis)
                if args.show:
                    cv2.imshow("Driver Monitor v3", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("c"):
                        monitor.recalibrate()
                        print("Recalibrating...")
    except KeyboardInterrupt:
        pass
    finally:
        cal = monitor.engine.calibrator
        calibration = {"status": cal.status, "baseline": cal.baseline.__dict__, "last_failure": cal.last_failure,
                       "warning": cal.warning}
        monitor.close()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        if logger:
            extra = {"stream_health": health.stats(), "stream_events": stream_events[-50:],
                     "dropped_stale_frames": reader.dropped if reader else 0,
                     "driving_duration": context.duration.summary(), "alert_state": monitor.engine.alerts.summary()}
            if isinstance(context, HttpContext):
                extra["gps"] = context.gps.stats()
                with open(logger.dir / "gps.csv", "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=["t", "lat", "lon", "speed_kmh", "fix", "hdop", "satellites", "accuracy_m"])
                    w.writeheader()
                    w.writerows(f.to_dict() for f in context.gps.fixes)
            summary = logger.close(calibration, extra)
            print(json.dumps({k: summary.get(k) for k in ("frames", "processing_fps", "alert_episodes", "stream_health", "calibration")},
                             indent=2, default=str))
        if isinstance(context, HttpContext):
            context.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
