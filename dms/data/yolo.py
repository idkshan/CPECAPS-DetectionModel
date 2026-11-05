"""Read YOLO datasets in both folder layouts.

    Ultralytics:  <root>/images/{train,val,test}/*.jpg   <root>/labels/{...}/*.txt
    Roboflow:     <root>/{train,valid,test}/images/*.jpg <root>/{...}/labels/*.txt

A label file has one line per box: ``class cx cy w h`` (normalised 0..1).
An image without a label file (or with an empty one) is a background image.
Classification datasets (MRL Eye) use <root>/{train,val,test}/<class>/*.png.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_FOLDERS = {"train": ("train",), "val": ("valid", "val"), "test": ("test",)}


def find_splits(root: Path) -> Dict[str, Path]:
    """Image folder of each split, in either layout."""
    found = {}
    for split, names in SPLIT_FOLDERS.items():
        for n in names:
            for cand in (root / n / "images", root / "images" / n):
                if cand.is_dir():
                    found[split] = cand
                    break
            if split in found:
                break
    return found


def find_classification_splits(root: Path) -> Dict[str, Path]:
    found = {}
    for split, names in SPLIT_FOLDERS.items():
        for n in names:
            if (root / n).is_dir() and any(p.is_dir() for p in (root / n).iterdir()):
                found[split] = root / n
                break
    return found


def class_names(root: Path) -> Dict[int, str]:
    """Class names from the dataset's YAML (data.yaml or any *.yaml in the root)."""
    for y in sorted(root.glob("*.yaml")):
        with open(y, encoding="utf-8") as fh:
            names = (yaml.safe_load(fh) or {}).get("names")
        if names is None:
            continue
        return dict(enumerate(names)) if isinstance(names, list) else {int(k): str(v) for k, v in names.items()}
    return {}


def label_path(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def list_images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


@dataclass
class LabelFile:
    boxes: List[Tuple[int, float, float, float, float]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    exists: bool = True


def parse_label_file(path: Path, n_classes: Optional[int]) -> LabelFile:
    """Parse and validate one YOLO label file (missing file = background image)."""
    out = LabelFile()
    if not path.exists():
        out.exists = False
        return out
    seen = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            out.errors.append(f"line {line_no}: expected 5 values (class cx cy w h), got {len(parts)}"
                              + (" - polygon/segment labels are not boxes" if len(parts) > 5 and len(parts) % 2 == 1 else ""))
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            out.errors.append(f"line {line_no}: not numeric")
            continue
        if n_classes is not None and not 0 <= cls < n_classes:
            out.errors.append(f"line {line_no}: class id {cls} outside 0..{n_classes - 1}")
        if w <= 0 or h <= 0:
            out.errors.append(f"line {line_no}: zero or negative box size")
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)) or cx - w / 2 < -0.01 or cx + w / 2 > 1.01 \
                or cy - h / 2 < -0.01 or cy + h / 2 > 1.01:
            out.errors.append(f"line {line_no}: box outside the image (values must be normalised 0..1)")
        key = (cls, round(cx, 4), round(cy, 4), round(w, 4), round(h, 4))
        if key in seen:
            out.errors.append(f"line {line_no}: duplicate box")
        seen.add(key)
        out.boxes.append((cls, cx, cy, w, h))
    return out
