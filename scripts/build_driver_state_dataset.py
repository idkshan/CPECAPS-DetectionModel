"""Build the YOLO *driver-state* dataset from the approved datasets.

One YOLO model then detects four classes in a single inference:

    0 eye_closed     1 eye_open     2 phone     3 yawn

Sources
-------
* ``--phone-dataset``: your reviewed phone dataset (Roboflow "YOLOv8" export
  of datasets/phone, built from State Farm with autolabel_phones.py). Its
  phone boxes become class 2 (phone). Its existing train/val/test split is kept.
* ``--yawdd``: the YawDD videos (folder searched recursively for .avi/.mp4).
  Frames are sampled, and the split is made *by person* (e.g. "12-Female").
  YawDD's Mirror and Dash videos are grouped by the same number + gender, so
  one person never spans two splits under that naming; if you cannot rule out
  that a Dash person appears under another number in Mirror, use
  ``--exclude-dash``.

Leakage checks
--------------
* ``--phone-groups driver_imgs_list.csv`` maps every phone photo (also after
  Roboflow renaming) to its State Farm driver and reports drivers that sit in
  more than one split of your Roboflow export; ``--resplit-phone-by-group``
  then re-splits the phone photos by driver instead of keeping Roboflow's
  (often random) split.
* split_manifest.csv lists every output image with its source, group and
  split; run scripts/audit_dataset.py on the result.

Automatic labelling (to be checked by a person in Roboflow afterwards)
--------------------------------------------------------------------
For every image MediaPipe finds the face; then
* each eye gets a box, labelled eye_closed / eye_open from EAR and the
  eyeBlink score compared with the person's normal values (per-video median;
  population defaults for still photos). Unclear eyes are not guessed:
  video frames with an unclear eye are skipped, still photos are listed
  in review_eyes.txt.
* the mouth gets a "yawn" box when MAR and jawOpen are clearly wide open.
  In YawDD videos named "Normal" or "Talking" a yawn is never labelled.
* closed eyes and yawns are rare, so video frames that contain them are
  sampled more densely than ordinary frames.

Phone photos also contain eyes; they are labelled too, otherwise YOLO would
learn that eyes in those photos are "background".

Important: the eye / yawn labels are produced by the same EAR / eyeBlink /
MAR / jawOpen measurements that the pipeline's thresholds use. Until a person
has corrected them, a YOLO model trained on them mostly *imitates* those
thresholds, and a test score on uncorrected labels measures agreement with
the auto-labeller, not accuracy. Correct at least the test split by hand.

Example
-------
    python scripts/build_driver_state_dataset.py --phone-dataset datasets/phone_roboflow \
        --yawdd data/yawdd --out datasets/driver_state
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

import _bootstrap  # noqa: F401
from dms.config import FaceModelConfig
from dms.data.splits import group_from_name, group_split, load_group_map, write_split_manifest
from dms.features.geometry import LEFT_EYE_EAR, RIGHT_EYE_EAR, eye_aspect_ratio, mouth_aspect_ratio
from dms.perception.eye_state import LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR

# Alphabetical, the order Roboflow uses in its exports, so a split re-exported
# from Roboflow keeps the same class ids. The pipeline reads classes by NAME.
CLASSES = ["eye_closed", "eye_open", "phone", "yawn"]
EYE_CLOSED, EYE_OPEN, PHONE, YAWN = range(4)
PHONE_ALIASES = {"cell phone", "phone", "mobile phone", "cellphone", "mobile_phone", "smartphone"}
OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".avi", ".mp4", ".mov", ".mkv"}

# population defaults for still photos (no per-person baseline available)
DEFAULT_BASE = {"ear": 0.28, "blink": 0.15, "mar": 0.05, "jaw": 0.03}


# ------------------------------------------------------------- measurement
def measure(face) -> Dict[str, float]:
    pts, bs = face.landmarks, face.blendshapes
    return {
        "ear_r": eye_aspect_ratio(pts, RIGHT_EYE_EAR),
        "ear_l": eye_aspect_ratio(pts, LEFT_EYE_EAR),
        "blink_r": bs.get("eyeBlinkRight", 0.0),
        "blink_l": bs.get("eyeBlinkLeft", 0.0),
        "mar": mouth_aspect_ratio(pts),
        "jaw": bs.get("jawOpen", 0.0),
    }


def ramp(x: float, zero_at: float, half_at: float) -> float:
    span = half_at - zero_at
    return float(np.clip(0.5 * (x - zero_at) / span, 0.0, 1.0)) if abs(span) > 1e-9 else 0.0


def eye_state(ear: float, blink: float, base: dict, args) -> Optional[int]:
    """EYE_CLOSED / EYE_OPEN, or None when unclear."""
    score = (ramp(ear, base["ear"], base["ear"] * args.ear_ratio) + ramp(blink, base["blink"], base["blink"] + args.blink_delta)) / 2
    if score >= args.sure_closed:
        return EYE_CLOSED
    if score <= args.sure_open:
        return EYE_OPEN
    return None


def mouth_state(mar: float, jaw: float, base: dict, args) -> Optional[bool]:
    """True = yawn, False = not yawning, None = unclear."""
    score = (ramp(mar, base["mar"], base["mar"] + args.mar_delta) + ramp(jaw, base["jaw"], base["jaw"] + args.jaw_delta)) / 2
    if score >= args.sure_yawn:
        return True
    if score <= args.sure_no_yawn:
        return False
    return None


def box_from_points(pts: np.ndarray, sx: float, sy: float, min_h_ratio: float, w: int, h: int) -> Tuple[float, float, float, float]:
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw = (x2 - x1) * sx
    bh = max((y2 - y1) * sy, bw * min_h_ratio)
    x1, x2 = max(0, cx - bw / 2), min(w, cx + bw / 2)
    y1, y2 = max(0, cy - bh / 2), min(h, cy + bh / 2)
    return (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h


def face_labels(face, m: dict, base: dict, args, w: int, h: int, allow_yawn: bool) -> Tuple[List[str], bool]:
    """YOLO label lines for eyes and mouth; second value False if something was unclear."""
    lines, clear = [], True
    for contour, ear, blink in ((RIGHT_EYE_CONTOUR, m["ear_r"], m["blink_r"]), (LEFT_EYE_CONTOUR, m["ear_l"], m["blink_l"])):
        st = eye_state(ear, blink, base, args)
        if st is None:
            clear = False
            continue
        cx, cy, bw, bh = box_from_points(face.landmarks[contour, :2], 1.5, 2.2, 0.45, w, h)
        lines.append(f"{st} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    ms = mouth_state(m["mar"], m["jaw"], base, args)
    if ms is None or (ms and not allow_yawn):
        clear = False
    elif ms:
        cx, cy, bw, bh = box_from_points(face.landmarks[OUTER_LIPS, :2], 1.2, 1.2, 0.0, w, h)
        lines.append(f"{YAWN} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines, clear


# ------------------------------------------------------------- phone dataset
def find_splits(root: Path) -> Dict[str, Path]:
    """Image folders of a YOLO dataset in either layout (Roboflow or Ultralytics)."""
    found = {}
    for split, names in (("train", ["train"]), ("val", ["valid", "val"]), ("test", ["test"])):
        for n in names:
            for cand in (root / n / "images", root / "images" / n):
                if cand.is_dir():
                    found[split] = cand
                    break
            if split in found:
                break
    return found


def phone_class_ids(root: Path) -> set:
    for y in list(root.glob("*.yaml")):
        names = yaml.safe_load(open(y)).get("names")
        if names is None:
            continue
        names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
        ids = {k for k, v in names.items() if str(v).lower() in PHONE_ALIASES}
        if ids:
            return ids
    return {0}


def label_of(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


# ------------------------------------------------------------- YawDD
def subject_of(video: Path) -> str:
    m = re.match(r"(\d+)-?(Female|Male)", video.stem, re.IGNORECASE)
    if m:
        return f"{m.group(2).lower()}{int(m.group(1))}"
    gender = next((p.lower() for p in video.parts if p.lower() in ("female", "male")), "")
    num = re.match(r"(\d+)", video.stem)
    return f"{gender}{num.group(1)}" if num else video.stem


def video_frames(path: Path, stride: int):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            yield i / fps, frame
        i += 1
    cap.release()


# ------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--phone-dataset", default=None, help="reviewed phone dataset (Roboflow YOLOv8 export)")
    ap.add_argument("--yawdd", default=None, help="folder with YawDD videos")
    ap.add_argument("--out", default="datasets/driver_state")
    ap.add_argument("--face-model", default="models/face_landmarker.task")
    ap.add_argument("--scan-stride", type=int, default=3, help="analyse every Nth video frame")
    ap.add_argument("--every-sec", type=float, default=1.0, help="keep an ordinary frame every N seconds")
    ap.add_argument("--event-every-sec", type=float, default=0.2, help="keep closed-eye / yawn frames every N seconds")
    ap.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15])
    ap.add_argument("--exclude-dash", action="store_true", help="skip YawDD 'Dash' videos (see the leakage note above)")
    ap.add_argument("--phone-groups", default=None, help="CSV mapping phone photos to drivers (driver_imgs_list.csv)")
    ap.add_argument("--resplit-phone-by-group", action="store_true", help="re-split the phone photos by driver")
    ap.add_argument("--seed", type=int, default=0)
    # labelling thresholds (same meaning as the live model's thresholds)
    ap.add_argument("--ear-ratio", type=float, default=0.70)
    ap.add_argument("--blink-delta", type=float, default=0.35)
    ap.add_argument("--mar-delta", type=float, default=0.40)
    ap.add_argument("--jaw-delta", type=float, default=0.35)
    ap.add_argument("--sure-closed", type=float, default=0.65)
    ap.add_argument("--sure-open", type=float, default=0.35)
    ap.add_argument("--sure-yawn", type=float, default=0.65)
    ap.add_argument("--sure-no-yawn", type=float, default=0.35)
    args = ap.parse_args()
    if not args.phone_dataset and not args.yawdd:
        ap.error("give --phone-dataset and/or --yawdd")

    from dms.perception.face_landmarker import FaceLandmarkerModel

    out = Path(args.out)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    counts = defaultdict(lambda: np.zeros(len(CLASSES) + 1, dtype=int))  # per split: classes + images
    review: List[str] = []

    def save(split: str, name: str, image: Optional[np.ndarray], src: Optional[Path], lines: List[str]) -> None:
        dest = out / "images" / split / name
        if src is not None:
            shutil.copy2(src, dest)
        else:
            cv2.imwrite(str(dest), image)
        (out / "labels" / split / (Path(name).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        counts[split][-1] += 1
        for ln in lines:
            counts[split][int(ln.split()[0])] += 1

    landmarker = FaceLandmarkerModel(FaceModelConfig(model_path=args.face_model), video_mode=False)

    manifest: List[dict] = []

    # ---- phone dataset (still photos): keep (or re-split by driver), add eyes/mouth
    if args.phone_dataset:
        root = Path(args.phone_dataset)
        splits = find_splits(root)
        if not splits:
            raise SystemExit(f"No train/valid/test image folders found in {root}")
        phone_ids = phone_class_ids(root)
        print(f"Phone dataset: splits {sorted(splits)}, phone class ids {sorted(phone_ids)}")
        photos = [(split, p) for split, img_dir in splits.items()
                  for p in sorted(q for q in img_dir.iterdir() if q.suffix.lower() in IMG_EXTS)]
        group_map = load_group_map(Path(args.phone_groups)) if args.phone_groups else {}
        driver_of = {p: (group_from_name(p.name, group_map) if group_map else None) for _, p in photos}
        if group_map:
            where = defaultdict(set)
            for split, p in photos:
                if driver_of[p] is not None:
                    where[driver_of[p]].add(split)
            crossing = {g: sorted(v) for g, v in where.items() if len(v) > 1}
            unknown = sum(1 for v in driver_of.values() if v is None)
            print(f"  phone photos: {len(where)} drivers, {unknown} photos without a known driver")
            if crossing:
                print(f"  LEAKAGE: {len(crossing)} drivers appear in more than one split of {root.name}, e.g. "
                      + ", ".join(f"{g}: {v}" for g, v in list(crossing.items())[:5]))
                if not args.resplit_phone_by_group:
                    print("  -> add --resplit-phone-by-group to split the phone photos by driver instead")
        if args.resplit_phone_by_group:
            if not group_map:
                raise SystemExit("--resplit-phone-by-group needs --phone-groups (e.g. data/state-farm/driver_imgs_list.csv)")
            keys = {driver_of[p] or f"image:{p.name}" for _, p in photos}
            new_split = group_split(keys, args.split, args.seed)
            photos = [(new_split[driver_of[p] or f"image:{p.name}"], p) for _, p in photos]
            print("  phone photos re-split by driver")
        for split, img_path in photos:
            lines = []
            lp = label_of(img_path)
            if lp.exists():
                for ln in lp.read_text().splitlines():
                    parts = ln.split()
                    if len(parts) >= 5 and int(float(parts[0])) in phone_ids:
                        lines.append(" ".join([str(PHONE)] + parts[1:5]))
            img = cv2.imread(str(img_path))
            if img is None:
                review.append(f"{split}/{img_path.name}: unreadable image skipped")
                continue
            h, w = img.shape[:2]
            face = landmarker.detect(img, 0.0)
            if face is not None:
                face_lines, clear = face_labels(face, measure(face), DEFAULT_BASE, args, w, h, allow_yawn=True)
                lines += face_lines
                if not clear:
                    review.append(f"{split}/{img_path.name}")
            name = f"sf_{img_path.name}"
            save(split, name, None, img_path, lines)
            manifest.append({"file": name, "source": str(img_path), "group": driver_of.get(img_path) or "", "split": split})
        print("  phone photos done")

    # ---- YawDD videos: per-person split, per-video baseline, dense sampling of events
    if args.yawdd:
        videos = sorted(p for p in Path(args.yawdd).rglob("*") if p.suffix.lower() in VID_EXTS)
        if args.exclude_dash:
            videos = [v for v in videos if "dash" not in {part.lower() for part in v.parts}]
        groups = defaultdict(list)
        for v in videos:
            groups[subject_of(v)].append(v)
        split_of = group_split(groups, args.split, args.seed)
        print(f"YawDD: {len(videos)} videos from {len(groups)} people")
        for vi, video in enumerate(videos, 1):
            split = split_of[subject_of(video)]
            name_l = video.stem.lower()
            allow_yawn = not ("normal" in name_l or "talking" in name_l)
            # pass 1: measurements only (keeping whole frames in memory does not scale to long videos)
            measured = []  # (t, landmarks, blendshapes, measurements)
            for t, frame in video_frames(video, args.scan_stride):
                face = landmarker.detect(frame, 0.0)
                if face is not None:
                    measured.append((t, face, measure(face), frame.shape[1], frame.shape[0]))
            if len(measured) < 10:
                continue
            base = {
                "ear": float(np.median([x[2]["ear_r"] + x[2]["ear_l"] for x in measured]) / 2),
                "blink": float(np.median([x[2]["blink_r"] + x[2]["blink_l"] for x in measured]) / 2),
                "mar": float(np.median([x[2]["mar"] for x in measured])),
                "jaw": float(np.median([x[2]["jaw"] for x in measured])),
            }
            selected: Dict[float, List[str]] = {}
            last_kept = -1e9
            for t, face, m, w, h in measured:
                lines, clear = face_labels(face, m, base, args, w, h, allow_yawn)
                if not clear:
                    continue  # plenty of frames: never guess on video
                event = any(ln.split()[0] in (str(EYE_CLOSED), str(YAWN)) for ln in lines)
                if t - last_kept < (args.event_every_sec if event else args.every_sec):
                    continue
                last_kept = t
                selected[round(t, 3)] = lines
            # pass 2: read the video again and save only the selected frames
            for t, frame in video_frames(video, args.scan_stride):
                lines = selected.get(round(t, 3))
                if lines is None:
                    continue
                name = f"yawdd_{video.stem}_{int(t * 1000):07d}.jpg"
                save(split, name, frame, None, lines)
                manifest.append({"file": name, "source": str(video), "group": subject_of(video), "split": split})
            print(f"  [{vi}/{len(videos)}] {video.name}: {len(selected)} frames -> {split}")

    landmarker.close()
    yaml.safe_dump({"path": str(out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
                    "names": dict(enumerate(CLASSES))}, open(out / "data.yaml", "w"), sort_keys=False)
    (out / "review_eyes.txt").write_text("\n".join(review) + "\n")
    write_split_manifest(out / "split_manifest.csv", manifest)
    print("\nsplit   images  " + "  ".join(f"{c:>10s}" for c in CLASSES))
    for split in ("train", "val", "test"):
        c = counts[split]
        print(f"{split:6s} {c[-1]:7d}  " + "  ".join(f"{v:10d}" for v in c[:-1]))
    print(f"\n{len(review)} phone photos with an unclear eye or mouth -> {out / 'review_eyes.txt'}")
    print(f"Dataset config: {out / 'data.yaml'}")
    print(f"Split manifest: {out / 'split_manifest.csv'}  ->  python scripts/audit_dataset.py --data {out} --groups {out / 'split_manifest.csv'}")
    print("NEXT: upload to Roboflow, check the boxes (at least the test split), export YOLOv8, then train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
