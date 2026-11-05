"""Build a YOLO phone-detection dataset from State Farm / AUC distracted-driver images.

Both datasets are *image-classification* datasets (folders c0..c9); they have
no bounding boxes, which a YOLO detector needs. This script pseudo-labels them:

  * images from the phone classes (default c1-c4: texting / talking, right and
    left hand) are passed through a large pretrained "teacher" detector; every
    COCO "cell phone" box above --teacher-conf becomes a label (class 0 = phone)
  * images from the negative classes (default c0: safe driving) get empty
    label files, teaching the model what "no phone" looks like
  * phone-class images where the teacher finds NO phone are NOT saved as
    negatives (that would teach the model that a phone at the ear is
    background). They are listed in needs_review.txt; with --keep-unlabeled
    they are copied to <out>/unlabeled/ for manual labelling.

The split is done by GROUP (driver), never by image, because neighbouring
frames of one driver are near-identical:
  * State Farm: --driver-csv data/state-farm/driver_imgs_list.csv
  * other data: --group-csv FILE (columns file/img + group/subject/driver) or
                --group-regex REGEX (first capture group of the file name)
  * a dataset that comes with its own driver-disjoint split (e.g. AUC train /
    test): run once per official subset with --assign-split train|val|test
A random per-image split is refused unless --allow-image-split is given.

Every run writes split_manifest.csv (source, group, split, boxes) for
scripts/audit_dataset.py.

IMPORTANT: pseudo-labels must be checked by a person (e.g. in CVAT, Label
Studio or Roboflow) before the test split is used to report results;
otherwise the evaluation measures agreement with the teacher model.

Example
-------
    python scripts/autolabel_phones.py --src data/state-farm/imgs/train \
        --driver-csv data/state-farm/driver_imgs_list.csv --out datasets/phone --teacher models/yolo11x.pt
"""

import argparse
import csv
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

import _bootstrap  # noqa: F401
from dms.data.splits import group_split, load_group_map, write_split_manifest

PHONE_ALIASES = {"cell phone", "phone", "mobile phone", "cellphone"}
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--src", required=True, help="folder containing class sub-folders c0..c9")
    ap.add_argument("--out", default="datasets/phone")
    ap.add_argument("--teacher", default="models/yolo11x.pt", help="large pretrained detector used as labeller")
    ap.add_argument("--teacher-conf", type=float, default=0.30)
    ap.add_argument("--phone-classes", nargs="+", default=["c1", "c2", "c3", "c4"])
    ap.add_argument("--negative-classes", nargs="+", default=["c0"])
    ap.add_argument("--negatives-per-class", type=int, default=1500, help="cap on negative images per class")
    ap.add_argument("--driver-csv", default=None, help="State Farm driver_imgs_list.csv (driver-wise split)")
    ap.add_argument("--group-csv", default=None, help="any CSV mapping file name -> group (driver / person)")
    ap.add_argument("--group-regex", default=None, help="regex on the file name; its first group is the driver")
    ap.add_argument("--assign-split", choices=["train", "val", "test"], default=None,
                    help="put every image of this run into one split (use the dataset's own driver-disjoint split)")
    ap.add_argument("--allow-image-split", action="store_true", help="accept a random per-image split (leaks drivers)")
    ap.add_argument("--keep-unlabeled", action="store_true", help="copy phone images without a teacher box to <out>/unlabeled/")
    ap.add_argument("--prefix", default="", help="file-name prefix (e.g. auc_) when several sources share --out")
    ap.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    group_of = {}
    if args.driver_csv:
        group_of.update(load_group_map(Path(args.driver_csv)))
    if args.group_csv:
        group_of.update(load_group_map(Path(args.group_csv)))
    regex = re.compile(args.group_regex) if args.group_regex else None
    if not (group_of or regex or args.assign_split or args.allow_image_split):
        raise SystemExit("Refusing a per-image split: frames of one driver would end up in train AND test.\n"
                         "Give --driver-csv / --group-csv / --group-regex, or --assign-split for a dataset with its own\n"
                         "driver-disjoint split, or --allow-image-split if you accept the leakage.")

    from ultralytics import YOLO

    src, out = Path(args.src), Path(args.out)
    teacher = YOLO(args.teacher)
    phone_ids = [k for k, v in teacher.names.items() if str(v).lower() in PHONE_ALIASES]

    import random
    rng = random.Random(args.seed)
    items = []  # (path, class_folder)
    for cls in args.phone_classes:
        items += [(p, cls) for p in sorted((src / cls).iterdir()) if p.suffix.lower() in IMG_EXTS]
    for cls in args.negative_classes:
        files = sorted(p for p in (src / cls).iterdir() if p.suffix.lower() in IMG_EXTS)
        rng.shuffle(files)
        items += [(p, cls) for p in files[: args.negatives_per_class]]
    if not items:
        raise SystemExit(f"No images found in {src}/<{','.join(args.phone_classes + args.negative_classes)}>")

    def group(p: Path) -> str:
        if p.name in group_of:
            return group_of[p.name]
        if regex:
            m = regex.search(p.name)
            if m:
                return m.group(1) if m.groups() else m.group(0)
        return f"image:{p.name}"  # unknown -> its own group (= per-image split for this file)

    groups = defaultdict(list)
    for p, cls in items:
        groups[group(p)].append((p, cls))
    unknown = sum(1 for g in groups if g.startswith("image:"))
    if unknown and not (args.allow_image_split or args.assign_split):
        print(f"WARNING: {unknown} images have no driver/group; each becomes its own group.")
    split_of = ({g: args.assign_split for g in groups} if args.assign_split else group_split(groups, args.split, args.seed))

    review, manifest = [], []
    counts = defaultdict(lambda: defaultdict(int))
    for key, members in groups.items():
        split = split_of[key]
        for p, cls in members:
            name = f"{args.prefix}{cls}_{p.name}"
            lines = []
            if cls in args.phone_classes:
                res = teacher.predict(str(p), conf=args.teacher_conf, classes=phone_ids, device=args.device, verbose=False)[0]
                for (cx, cy, w, h) in res.boxes.xywhn.cpu().numpy():
                    lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                if not lines:
                    review.append(str(p))
                    counts[split]["phone_images_without_box"] += 1
                    if args.keep_unlabeled:
                        (out / "unlabeled").mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, out / "unlabeled" / name)
                    manifest.append({"source": str(p), "file": name, "group": key, "split": "unlabeled", "class_folder": cls, "boxes": 0})
                    continue  # never saved as a negative example
            img_dir, lbl_dir = out / "images" / split, out / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, img_dir / name)
            (lbl_dir / (Path(name).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            counts[split]["images"] += 1
            counts[split]["boxes"] += len(lines)
            manifest.append({"source": str(p), "file": name, "group": key, "split": split, "class_folder": cls, "boxes": len(lines)})

    with open(out / "data.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump({"path": str(out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
                        "names": {0: "phone"}}, fh)
    manifest_path = out / "split_manifest.csv"
    if manifest_path.exists():  # several runs (e.g. State Farm + AUC) append to one manifest
        with open(manifest_path, newline="", encoding="utf-8") as fh:
            old = [r for r in csv.DictReader(fh) if r["source"] not in {m["source"] for m in manifest}]
        manifest = old + manifest
    write_split_manifest(manifest_path, manifest)
    with open(out / "needs_review.txt", "a", encoding="utf-8") as fh:
        fh.write("\n".join(review) + ("\n" if review else ""))
    for split in ("train", "val", "test"):
        c = counts[split]
        print(f"{split:5s}: {c['images']} images, {c['boxes']} phone boxes, {c['phone_images_without_box']} phone images skipped (no box)")
    print(f"{len(review)} phone-class images without a teacher box -> {out / 'needs_review.txt'}"
          + (f" (copied to {out / 'unlabeled'})" if args.keep_unlabeled else ""))
    print(f"Dataset config: {out / 'data.yaml'}; split manifest: {manifest_path}  (review labels before reporting test results!)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
