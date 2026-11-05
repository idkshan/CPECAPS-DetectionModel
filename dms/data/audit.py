"""Dataset audit: problems that silently inflate or corrupt evaluation results.

Checks
  * unreadable / corrupt images, orphan label files, malformed label lines
  * class distribution per split (boxes and images per class, background images)
  * exact duplicates (same bytes) and near-duplicates (64-bit difference hash,
    Hamming distance <= ``near_threshold``) ACROSS splits = leakage
  * group leakage: one driver / person / video in more than one split
    (groups come from a CSV such as State Farm's driver_imgs_list.csv, a
    builder's split_manifest.csv, or a regex on the file name)

Near-duplicate search: the 64-bit hash is cut into 5 bands; two hashes that
differ in at most 4 bits must agree on at least one band (pigeonhole), so only
images sharing a band are compared.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .yolo import class_names, find_classification_splits, find_splits, label_path, list_images, parse_label_file

BANDS = ((0, 13), (13, 26), (26, 39), (39, 52), (52, 64))


def dhash(img_gray: np.ndarray) -> int:
    """64-bit difference hash: is each pixel brighter than its right neighbour (9x8 thumbnail)."""
    small = cv2.resize(img_gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0]) if bits.size == 64 else 0


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _band(h: int, lo: int, hi: int) -> int:
    return (h >> lo) & ((1 << (hi - lo)) - 1)


def read_image(path: Path) -> Tuple[Optional[np.ndarray], str]:
    """Reduced-size greyscale decode (fast) + md5 of the bytes; None if unreadable."""
    data = path.read_bytes()
    md5 = hashlib.md5(data).hexdigest()
    if not data:
        return None, md5
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_REDUCED_GRAYSCALE_2)
    if img is None or img.size == 0:
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return (img if img is not None and img.size else None), md5


def near_duplicate_pairs(items: List[Tuple[str, Path, int]], threshold: int = 4, cross_split_only: bool = True,
                         limit: int = 200) -> Tuple[int, List[dict]]:
    """Pairs of images whose hashes differ in <= threshold bits (different splits by default)."""
    buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, (_, _, h) in enumerate(items):
        for b, (lo, hi) in enumerate(BANDS):
            buckets[(b, _band(h, lo, hi))].append(idx)
    seen = set()
    examples, count = [], 0
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i_pos, i in enumerate(members):
            for j in members[i_pos + 1:]:
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                seen.add(key)
                si, pi, hi_ = items[i]
                sj, pj, hj = items[j]
                if cross_split_only and si == sj:
                    continue
                d = hamming(hi_, hj)
                if d <= threshold:
                    count += 1
                    if len(examples) < limit:
                        examples.append({"a": f"{si}/{pi.name}", "b": f"{sj}/{pj.name}", "bits": d})
    return count, examples


def audit_detection(root: Path, group_of: Optional[Callable[[str], Optional[str]]] = None, near_threshold: int = 4,
                    check_duplicates: bool = True, progress: Optional[Callable[[str], None]] = None) -> dict:
    """Audit a YOLO detection dataset; returns a JSON-able report."""
    splits = find_splits(root)
    if not splits:
        raise FileNotFoundError(f"no train/val/test image folders found in {root}")
    names = class_names(root)
    n_classes = len(names) if names else None
    report: dict = {"root": str(root), "layout_splits": {k: str(v) for k, v in splits.items()}, "classes": names,
                    "splits": {}, "problems": []}
    hashes: List[Tuple[str, Path, int]] = []
    md5_seen: Dict[str, List[str]] = defaultdict(list)
    groups_by_split: Dict[str, Counter] = {}
    unknown_group = 0
    for split, folder in splits.items():
        images = list_images(folder)
        stats = {"images": len(images), "background_images": 0, "corrupt_images": [], "label_errors": {},
                 "boxes_per_class": Counter(), "images_per_class": Counter(), "image_sizes": Counter()}
        groups = Counter()
        for k, img_path in enumerate(images):
            if progress and k % 2000 == 0:
                progress(f"{split}: {k}/{len(images)}")
            lf = parse_label_file(label_path(img_path), n_classes)
            if lf.errors:
                stats["label_errors"][img_path.name] = lf.errors[:5]
            if not lf.boxes:
                stats["background_images"] += 1
            for cls in {b[0] for b in lf.boxes}:
                stats["images_per_class"][names.get(cls, str(cls))] += 1
            for b in lf.boxes:
                stats["boxes_per_class"][names.get(b[0], str(b[0]))] += 1
            if check_duplicates:
                img, md5 = read_image(img_path)
                if img is None:
                    stats["corrupt_images"].append(img_path.name)
                else:
                    hashes.append((split, img_path, dhash(img)))
                    md5_seen[md5].append(f"{split}/{img_path.name}")
                    if k < 300:  # sizes from a sample (reduced decode = half size)
                        stats["image_sizes"][f"{img.shape[1] * 2}x{img.shape[0] * 2}"] += 1
            if group_of is not None:
                g = group_of(img_path.name)
                if g is None:
                    unknown_group += 1
                else:
                    groups[g] += 1
        label_dir = label_path(images[0]).parent if images else None
        if label_dir is not None and label_dir.is_dir():
            stems = {p.stem for p in images}
            stats["orphan_label_files"] = sorted(p.name for p in label_dir.glob("*.txt") if p.stem not in stems)[:50]
        for key in ("boxes_per_class", "images_per_class", "image_sizes"):
            stats[key] = dict(stats[key].most_common())
        stats["label_error_files"] = len(stats["label_errors"])
        stats["label_errors"] = dict(list(stats["label_errors"].items())[:30])
        report["splits"][split] = stats
        groups_by_split[split] = groups

    # class balance (train split)
    train = report["splits"].get("train", {})
    boxes = train.get("boxes_per_class", {})
    for cls in (names or {}).values():
        if cls not in boxes:
            report["problems"].append(f"class '{cls}' has no boxes in the training split")
    if len(boxes) >= 2:
        ratio = max(boxes.values()) / max(1, min(boxes.values()))
        report["train_imbalance_ratio"] = round(ratio, 1)
        if ratio > 10:
            report["problems"].append(f"strong class imbalance in train (largest/smallest class = {ratio:.0f}x)")

    if check_duplicates:
        cross = {m: locs for m, locs in md5_seen.items() if len({loc.split('/')[0] for loc in locs}) > 1}
        within = {m: locs for m, locs in md5_seen.items() if len(locs) > 1 and m not in cross}
        report["exact_duplicates"] = {"across_splits": len(cross), "within_split": len(within),
                                      "examples_across": list(cross.values())[:20]}
        n_near, examples = near_duplicate_pairs(hashes, near_threshold, cross_split_only=True)
        report["near_duplicates_across_splits"] = {"pairs": n_near, "threshold_bits": near_threshold, "examples": examples[:50]}
        if cross:
            report["problems"].append(f"{len(cross)} identical images appear in more than one split (leakage)")
        if n_near:
            report["problems"].append(f"{n_near} near-identical image pairs across splits (possible leakage: "
                                      "frames of the same video / driver)")
        corrupt = sum(len(s["corrupt_images"]) for s in report["splits"].values())
        if corrupt:
            report["problems"].append(f"{corrupt} unreadable / corrupt images")
    label_errs = sum(s["label_error_files"] for s in report["splits"].values())
    if label_errs:
        report["problems"].append(f"{label_errs} label files with errors")

    if group_of is not None:
        report["groups"] = _group_report(groups_by_split, unknown_group)
        if report["groups"]["groups_in_several_splits"]:
            report["problems"].append(f"{len(report['groups']['groups_in_several_splits'])} groups (drivers / persons) "
                                      "appear in more than one split (leakage)")
    return report


def _group_report(groups_by_split: Dict[str, Counter], unknown: int) -> dict:
    where: Dict[str, Dict[str, int]] = defaultdict(dict)
    for split, counter in groups_by_split.items():
        for g, n in counter.items():
            where[g][split] = n
    crossing = {g: s for g, s in where.items() if len(s) > 1}
    return {"groups_per_split": {s: len(c) for s, c in groups_by_split.items()},
            "images_with_unknown_group": unknown,
            "groups_in_several_splits": dict(list(crossing.items())[:50]),
            "n_groups_in_several_splits": len(crossing)}


def audit_classification(root: Path, group_of: Optional[Callable[[str], Optional[str]]] = None,
                         near_threshold: int = 4, check_duplicates: bool = True) -> dict:
    """Audit a classification dataset laid out as <split>/<class>/<image>."""
    splits = find_classification_splits(root)
    if not splits:
        raise FileNotFoundError(f"no train/val/test class folders found in {root}")
    report: dict = {"root": str(root), "splits": {}, "problems": []}
    hashes, md5_seen = [], defaultdict(list)
    groups_by_split: Dict[str, Counter] = {}
    unknown = 0
    for split, folder in splits.items():
        per_class = Counter()
        corrupt, groups = [], Counter()
        for cls_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            for img_path in list_images(cls_dir):
                per_class[cls_dir.name] += 1
                if check_duplicates:
                    img, md5 = read_image(img_path)
                    if img is None:
                        corrupt.append(img_path.name)
                    else:
                        hashes.append((split, img_path, dhash(img)))
                        md5_seen[md5].append(f"{split}/{img_path.name}")
                if group_of is not None:
                    g = group_of(img_path.name)
                    if g is None:
                        unknown += 1
                    else:
                        groups[g] += 1
        report["splits"][split] = {"images_per_class": dict(per_class), "corrupt_images": corrupt[:50]}
        groups_by_split[split] = groups
    if check_duplicates:
        cross = {m: locs for m, locs in md5_seen.items() if len({loc.split('/')[0] for loc in locs}) > 1}
        n_near, examples = near_duplicate_pairs(hashes, near_threshold, cross_split_only=True)
        report["exact_duplicates_across_splits"] = len(cross)
        report["near_duplicates_across_splits"] = {"pairs": n_near, "threshold_bits": near_threshold, "examples": examples[:50]}
        if cross:
            report["problems"].append(f"{len(cross)} identical images appear in more than one split (leakage)")
    if group_of is not None:
        report["groups"] = _group_report(groups_by_split, unknown)
        if report["groups"]["n_groups_in_several_splits"]:
            report["problems"].append("subjects appear in more than one split (leakage)")
    train = report["splits"].get("train", {}).get("images_per_class", {})
    if len(train) >= 2:
        ratio = max(train.values()) / max(1, min(train.values()))
        report["train_imbalance_ratio"] = round(ratio, 2)
    return report


def markdown(report: dict) -> str:
    lines = [f"# Dataset audit: {report['root']}", ""]
    if report["problems"]:
        lines += ["## Problems", ""] + [f"- **{p}**" for p in report["problems"]] + [""]
    else:
        lines += ["No problems found.", ""]
    lines += ["## Splits", "", "| Split | Images | Background | Boxes per class | Corrupt | Label-error files |",
              "|---|---|---|---|---|---|"]
    for split, s in report["splits"].items():
        if "images" in s:
            boxes = ", ".join(f"{k}: {v}" for k, v in s["boxes_per_class"].items()) or "—"
            lines.append(f"| {split} | {s['images']} | {s['background_images']} | {boxes} | {len(s['corrupt_images'])} | "
                         f"{s['label_error_files']} |")
        else:
            per = ", ".join(f"{k}: {v}" for k, v in s["images_per_class"].items())
            lines.append(f"| {split} | {sum(s['images_per_class'].values())} | — | {per} | {len(s['corrupt_images'])} | — |")
    if "near_duplicates_across_splits" in report:
        nd = report["near_duplicates_across_splits"]
        lines += ["", f"Near-duplicate pairs across splits (<= {nd['threshold_bits']} bits): **{nd['pairs']}**"]
        for e in nd["examples"][:10]:
            lines.append(f"- {e['a']}  ~  {e['b']} ({e['bits']} bits)")
    if "groups" in report:
        g = report["groups"]
        lines += ["", f"Groups per split: {g['groups_per_split']}; images with unknown group: {g['images_with_unknown_group']}; "
                      f"groups in several splits: **{g['n_groups_in_several_splits']}**"]
        for name, where in list(g["groups_in_several_splits"].items())[:10]:
            lines.append(f"- {name}: {where}")
    return "\n".join(lines) + "\n"
