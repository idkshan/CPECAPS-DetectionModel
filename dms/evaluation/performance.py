"""Performance of a recorded session: frame rates, stage latency, uptime (SO1, SO3/SO4 speed, SO6).

Definitions used in the report
  input FPS        frames that reached the processing loop per second of session
                   time (camera capture + transmission rate seen by the app, SO1)
  processing FPS   1000 / mean processing time per frame (how many frames per
                   second the model COULD process on this computer, SO3/SO4)
  effective FPS    frames actually processed per second of session time
  downtime         intervals without any frame for longer than the camera
                   timeout (3 s) - "monitoring unavailable" (thesis 1.7.11)
  uptime           (session time - downtime) / session time  (SO6: >= 90 %)
These come from software logs. Hardware stability under driving vibration,
the mount and RA 10913 placement must be verified physically.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def _pct(a: np.ndarray, q: float) -> float:
    return round(float(np.percentile(a, q)), 2)


def session_performance(frames: Sequence[dict], summary: Optional[dict] = None, alerts: Sequence[dict] = (),
                        camera_timeout_s: float = 3.0) -> dict:
    summary = summary or {}
    times = np.array([float(f["t"]) for f in frames])
    if times.size < 2:
        return {"frames": int(times.size)}
    duration = float(times[-1] - times[0])
    dt = np.diff(times)
    gaps = [(float(times[i]), float(times[i + 1])) for i in np.nonzero(dt > camera_timeout_s)[0]]
    unavailable = [a for a in alerts if a.get("event") == "unavailable"]
    down_s = sum(b - a for a, b in gaps)
    out = {
        "frames": int(times.size),
        "duration_min": round(duration / 60, 2),
        "input_fps": round(times.size / duration, 2) if duration > 0 else None,
        "frame_interval_ms": {"median": round(1000 * float(np.median(dt)), 1), "p95": round(1000 * _pct(dt, 95), 1),
                              "max": round(1000 * float(dt.max()), 1)},
        "share_intervals_slower_than_10fps": round(float(np.mean(dt > 0.11)), 4),
        "downtime_intervals": len(gaps),
        "downtime_s": round(down_s, 2),
        "uptime": round(1.0 - down_s / duration, 4) if duration > 0 else None,
        "monitoring_unavailable_events": len(unavailable),
    }
    stage = {}
    for key in ("face_ms", "yolo_ms", "eye_cnn_ms", "perception_ms", "features_ms", "decision_ms", "total_ms"):
        vals = np.array([f["timings_ms"][key] for f in frames if key in (f.get("timings_ms") or {})], dtype=float)
        if vals.size:
            stage[key] = {"mean": round(float(vals.mean()), 2), "p50": _pct(vals, 50), "p95": _pct(vals, 95)}
    if stage:
        out["stage_latency_ms"] = stage
        if "total_ms" in stage:
            out["processing_fps"] = round(1000.0 / stage["total_ms"]["mean"], 2)
    face = [bool((f.get("features") or {}).get("face_present")) for f in frames if f.get("features") is not None]
    if face:
        out["face_detected_ratio"] = round(float(np.mean(face)), 4)
    widths = np.array([f["features"]["eye_width_px"] for f in frames
                       if (f.get("features") or {}).get("eye_width_px") is not None], dtype=float)
    if widths.size:  # SO1: is the face large enough in the image for landmarks?
        out["eye_width_px"] = {"p5": _pct(widths, 5), "median": _pct(widths, 50)}
    for key in ("stream_health", "dropped_stale_frames", "gps", "driving_duration"):
        if key in summary:
            out[key] = summary[key]
    events: dict = {}
    for a in alerts:
        events[a.get("event", "command")] = events.get(a.get("event", "command"), 0) + 1
    out["alert_events"] = events
    return out


def so_checks(perf: dict) -> List[dict]:
    """Software-measurable SO targets for this session (hardware items need field tests)."""
    rows = []

    def add(so, metric, target, value, ok):
        rows.append({"so": so, "metric": metric, "target": target, "measured": value,
                     "status": "n/a" if value is None else ("met" if ok else "not met")})

    fps_in = perf.get("input_fps")
    add("SO1", "capture + transmission rate (input FPS)", ">= 10 FPS", fps_in, fps_in is not None and fps_in >= 10)
    pf = perf.get("processing_fps")
    add("SO3/SO4", "processing rate on THIS computer", ">= 10 FPS", pf, pf is not None and pf >= 10)
    up = perf.get("uptime")
    add("SO6", "uptime (software view: frames arriving)", ">= 90 %", None if up is None else round(100 * up, 1), up is not None and up >= 0.9)
    ew = perf.get("eye_width_px", {}).get("p5")
    add("SO1", "eye width in image, 5th percentile (landmark resolution)", ">= 12 px (quality.min_eye_width_px)", ew,
        ew is not None and ew >= 12)
    gps = perf.get("gps") or {}
    rate = gps.get("share_within_1hz_10pct")
    add("SO4", "GPS updates at 1 Hz (share of 0.9-1.1 s intervals)", "~1.0", rate, rate is not None and rate >= 0.9)
    reacq = gps.get("longest_outage_s")
    add("SO4", "GPS re-acquisition (longest outage)", "<= 30 s", reacq, reacq is not None and reacq <= 30)
    return rows
