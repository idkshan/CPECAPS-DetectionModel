"""Measure camera-to-computer latency with an on-screen marker clock (SO2).

    python scripts/measure_latency.py --source http://192.168.4.1:81/stream --seconds 120

Set-up: point the ESP32-CAM at THIS computer's screen (the window that opens
shows a changing marker), keep the Wi-Fi conditions you want to report
("good Wi-Fi": phone/laptop near the router, little other traffic) and let it
run. Each received frame that shows the marker gives one latency sample:
receive time - time the marker was drawn, both on this computer's clock.

Report: number of samples, median / p95 / max (ms), share <= 500 ms, and a
histogram -> SO2 targets: median <= 500 ms, maximum <= 1000 ms.
Saved to results/latency/<name>/samples.csv and summary.json.

This measures camera -> Wi-Fi -> computer. For camera -> Android app, run the
same marker detection inside the app, or film the marker screen and the phone
screen together (see dms/evaluation/latency.py).
"""

import argparse
import csv
import json
import sys
import threading
import time
from pathlib import Path

import cv2

import _bootstrap  # noqa: F401
from dms.camera import iter_mjpeg
from dms.evaluation.latency import MarkerClock, summarize


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--source", required=True, help="ESP32 stream URL (http://<ip>:81/stream) or webcam index")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--name", default=None, help="name of this measurement (e.g. good_wifi_1)")
    ap.add_argument("--out", default="results/latency")
    args = ap.parse_args()

    clock = MarkerClock()
    samples, lock = [], threading.Lock()
    frames_seen = [0]
    stop = threading.Event()

    def receive():
        if args.source.isdigit():
            cap = cv2.VideoCapture(int(args.source))
            while not stop.is_set():
                ok, img = cap.read()
                t = time.monotonic()
                if ok:
                    handle(img, t)
            cap.release()
        else:
            for f in iter_mjpeg(args.source):
                if stop.is_set():
                    break
                handle(f.image, f.received_t)

    def handle(img, t):
        frames_seen[0] += 1
        lat = clock.latency(img, t)
        if lat is not None:
            with lock:
                samples.append((t, lat))

    threading.Thread(target=receive, daemon=True).start()
    end = time.monotonic() + args.seconds
    print("Point the camera at the marker window. Measuring ... (q to stop early)")
    while time.monotonic() < end:
        img = clock.next_image(time.monotonic())
        with lock:
            n = len(samples)
            last = samples[-1][1] * 1000 if samples else float("nan")
        cv2.putText(img, f"samples {n}  last {last:.0f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
        cv2.imshow("latency marker (film me)", img)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    stop.set()
    cv2.destroyAllWindows()

    lat = [s[1] for s in samples]
    summary = summarize(lat, frames_seen[0])
    summary.update(source=args.source, seconds=args.seconds, method="on-screen ArUco marker clock (same computer clock)")
    out = Path(args.out) / (args.name or time.strftime("latency_%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "samples.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["receive_t", "latency_ms"])
        w.writerows((round(t, 4), round(x * 1000, 1)) for t, x in samples)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
