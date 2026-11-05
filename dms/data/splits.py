"""Group-wise dataset splits.

Frames from one video, photos of one driver, or eye crops of one MRL subject
are near-identical. If they land in both training and test, the test score
measures memory, not generalisation. Every builder therefore splits by GROUP
(driver / person / video): whole groups go to train, val or test.
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Optional, Sequence

SPLIT_NAMES = ("train", "val", "test")


def group_split(groups: Iterable[Hashable], ratios: Sequence[float] = (0.7, 0.15, 0.15), seed: int = 0) -> Dict[Hashable, str]:
    """Assign every group to train / val / test.

    Groups are shuffled with ``seed`` (reproducible) and cut by ``ratios`` of
    the NUMBER OF GROUPS. Every split with a ratio above zero gets at least
    one group when there are enough groups.
    """
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6 or min(ratios) < 0:
        raise ValueError(f"split ratios must be three non-negative numbers that sum to 1, got {list(ratios)}")
    keys = sorted(set(groups), key=str)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    cut1, cut2 = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    counts = [cut1, cut2 - cut1, n - cut2]
    wanted = [i for i, r in enumerate(ratios) if r > 0]
    if n >= len(wanted):
        for i in wanted:  # an empty split borrows a group from the largest one
            if counts[i] == 0:
                j = max(range(3), key=lambda k: counts[k])
                counts[j] -= 1
                counts[i] += 1
    for i, r in enumerate(ratios):
        if r == 0 and counts[i]:  # rounding must not put groups into a split that was switched off
            counts[0] += counts[i]
            counts[i] = 0
    cut1, cut2 = counts[0], counts[0] + counts[1]
    return {k: ("train" if i < cut1 else "val" if i < cut2 else "test") for i, k in enumerate(keys)}


def write_split_manifest(path: Path, rows: List[dict]) -> None:
    """CSV with one row per source item: source, group, split (+ any extra columns).

    Keep it next to the dataset; ``audit_dataset.py --groups`` can read it
    back to prove that no group crosses splits.
    """
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


_RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)


def original_name(file_name: str) -> str:
    """Undo Roboflow renaming: 'c1_img_123_jpg.rf.ab12cd.jpg' -> 'c1_img_123.jpg'.

    Roboflow replaces the extension dot with '_' and appends '.rf.<hash>'.
    Names that were not renamed are returned unchanged.
    """
    stem, dot, ext = file_name.rpartition(".")
    if not dot:
        return file_name
    base = _RF_SUFFIX.sub("", stem)
    if base == stem:
        return file_name
    m = re.match(r"^(.*)_(jpe?g|png|bmp|webp)$", base, re.IGNORECASE)
    return f"{m.group(1)}.{m.group(2)}" if m else f"{base}.{ext}"


def load_group_map(path: Path, key_column: Optional[str] = None, group_column: Optional[str] = None) -> Dict[str, str]:
    """file name -> group from a CSV.

    Understands State Farm's ``driver_imgs_list.csv`` (columns subject, img),
    split manifests written by the builders (source, group) and any CSV with
    explicit ``key_column`` / ``group_column``.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        cols = [c.strip() for c in reader.fieldnames or []]
        key_column = key_column or next((c for c in ("img", "file", "image", "source", "filename") if c in cols), None)
        group_column = group_column or next((c for c in ("subject", "group", "driver", "person", "participant") if c in cols), None)
        if not key_column or not group_column:
            raise ValueError(f"{path}: cannot tell which columns hold the file name and the group (columns: {cols})")
        out = {}
        for row in reader:
            key = Path(str(row[key_column]).strip()).name
            out[key] = str(row[group_column]).strip()
        return out


def group_from_name(file_name: str, group_map: Optional[Dict[str, str]] = None, regex: Optional[str] = None) -> Optional[str]:
    """Group of one dataset file: via a map (original or Roboflow-renamed name) or a regex."""
    name = Path(file_name).name
    if group_map:
        for cand in (name, original_name(name)):
            if cand in group_map:
                return group_map[cand]
            # builders prefix files (e.g. 'c1_img_1.jpg', 'sf_c1_img_1.jpg'); try without prefixes
            parts = cand.split("_")
            for i in range(1, min(4, len(parts))):
                stripped = "_".join(parts[i:])
                if stripped in group_map:
                    return group_map[stripped]
    if regex:
        m = re.search(regex, original_name(name))
        if m:
            return m.group(1) if m.groups() else m.group(0)
    return None
