"""Save the eye crops exactly as the pipeline feeds them to the eye-state classifier.

Use it to check that training and inference see the same kind of image
(Section 4 of the evaluation guide): MRL Eye crops are tight, near-infrared,
greyscale eye images; the pipeline crops eyes from your camera with the face
landmarks (crop_scale, greyscale conversion in configs/default.yaml).

    python scripts/dump_eye_crops.py --source data/study/videos/P01_day.mp4 --out results/eye_crops/P01 --every-sec 2

Open a few crops next to MRL samples. If they look different (much larger
margin, colour, blur), adjust eye_model.crop_scale / grayscale - or label some
of these crops (open / closed folders) and use them as a domain test set:
    python scripts/train_yolo.py ... then  yolo classify val model=<best.pt> data=<folder with these crops>
"""

import argparse
import sys
from pathlib import Path

import cv2

import _bootstrap  # noqa: F401
from dms.config import load_config
from dms.perception.eye_state import LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR, crop_eye
from dms.perception.face_landmarker import FaceLandmarkerModel
from dms.sources import open_source


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--source", required=True, help="video file, image folder or camera index")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--every-sec", type=float, default=2.0, help="save crops every N seconds")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = FaceLandmarkerModel(cfg.face, video_mode=True)
    last, saved = -1e9, 0
    for frame in open_source(args.source, max_frames=args.max_frames):
        if frame.t - last < args.every_sec:
            continue
        face = model.detect(frame.image, frame.t)
        if face is None:
            continue
        last = frame.t
        for side, contour in (("right", RIGHT_EYE_CONTOUR), ("left", LEFT_EYE_CONTOUR)):
            crop = crop_eye(frame.image, face.landmarks[contour, :2], cfg.eye_model.crop_scale, cfg.eye_model.grayscale)
            if crop is not None:
                cv2.imwrite(str(out / f"{Path(str(args.source)).stem}_{frame.t:08.2f}_{side}.png"), crop)
                saved += 1
    model.close()
    print(f"{saved} eye crops saved to {out} (crop_scale {cfg.eye_model.crop_scale}, greyscale {cfg.eye_model.grayscale})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
