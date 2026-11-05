"""Dataset tooling: group splits, Roboflow name mapping, label validation, leakage audit."""

import cv2
import numpy as np
import pytest

from dms.data.audit import audit_classification, audit_detection, dhash, hamming
from dms.data.splits import group_from_name, group_split, load_group_map, original_name
from dms.data.yolo import parse_label_file


def test_group_split_keeps_groups_whole_and_is_reproducible():
    groups = [f"driver{i}" for i in range(26)]
    a = group_split(groups, (0.7, 0.15, 0.15), seed=3)
    assert a == group_split(groups, (0.7, 0.15, 0.15), seed=3)
    assert set(a.values()) == {"train", "val", "test"}
    assert sum(v == "train" for v in a.values()) == 18


def test_group_split_respects_disabled_splits_and_small_counts():
    assert set(group_split(["a", "b", "c"], (0.7, 0.15, 0.15)).values()) == {"train", "val", "test"}
    two = group_split([f"g{i}" for i in range(10)], (0.85, 0.15, 0.0))
    assert "test" not in two.values() and "val" in two.values()
    with pytest.raises(ValueError):
        group_split(["a"], (0.5, 0.5, 0.5))


def test_roboflow_names_map_back_to_original_files(tmp_path):
    assert original_name("c1_img_123_jpg.rf.0a1b2c3d4e.jpg") == "c1_img_123.jpg"
    assert original_name("img_9.jpg") == "img_9.jpg"
    csv_path = tmp_path / "driver_imgs_list.csv"
    csv_path.write_text("subject,classname,img\np002,c1,img_123.jpg\np015,c0,img_7.jpg\n")
    gm = load_group_map(csv_path)
    assert group_from_name("sf_c1_img_123_jpg.rf.ffff.jpg", gm) == "p002"
    assert group_from_name("c0_img_7.jpg", gm) == "p015"
    assert group_from_name("other.jpg", gm) is None


def test_label_file_errors_are_reported(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("0 0.5 0.5 0.2 0.2\n7 0.5 0.5 0.1 0.1\n0 1.2 0.5 0.1 0.1\n0 0.5 0.5 0.2\n0 0.5 0.5 0.2 0.2\n")
    lf = parse_label_file(p, n_classes=4)
    joined = " ".join(lf.errors)
    assert "class id 7" in joined and "outside the image" in joined and "expected 5 values" in joined and "duplicate" in joined
    assert not parse_label_file(tmp_path / "missing.txt", 4).exists


def _img(seed, size=64):
    rng = np.random.default_rng(seed)
    return (rng.random((size, size)) * 255).astype(np.uint8)


def test_dhash_detects_near_duplicates():
    a = _img(1)
    b = np.clip(a.astype(int) + 3, 0, 255).astype(np.uint8)  # brightness shift
    assert hamming(dhash(a), dhash(b)) <= 4
    assert hamming(dhash(a), dhash(_img(2))) > 10


def _write_detection_dataset(root, leak=True):
    (root / "data.yaml").write_text("names: [eye_closed, eye_open, phone, yawn]\n")
    for split, seeds in (("train", [1, 2, 3]), ("val", [4]), ("test", [5, 1 if leak else 6])):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for k, seed in enumerate(seeds):
            name = f"img_{split}_{k}"
            cv2.imwrite(str(root / "images" / split / f"{name}.png"), _img(seed))
            (root / "labels" / split / f"{name}.txt").write_text("2 0.5 0.5 0.2 0.2\n")
    (root / "labels" / "train" / "img_train_0.txt").write_text("9 0.5 0.5 0.2 0.2\n")  # bad class id


def test_audit_finds_cross_split_duplicates_label_errors_and_group_leakage(tmp_path):
    _write_detection_dataset(tmp_path)
    groups = {"img_train_0.png": "p1", "img_train_1.png": "p2", "img_train_2.png": "p3", "img_val_0.png": "p4",
              "img_test_0.png": "p5", "img_test_1.png": "p1"}
    rep = audit_detection(tmp_path, group_of=groups.get)
    assert rep["exact_duplicates"]["across_splits"] == 1
    assert rep["near_duplicates_across_splits"]["pairs"] >= 1
    assert rep["splits"]["train"]["label_error_files"] == 1
    assert rep["groups"]["n_groups_in_several_splits"] == 1
    assert any("leakage" in p for p in rep["problems"])


def test_audit_clean_dataset_has_no_leakage(tmp_path):
    _write_detection_dataset(tmp_path, leak=False)
    rep = audit_detection(tmp_path)
    assert rep["exact_duplicates"]["across_splits"] == 0
    assert rep["near_duplicates_across_splits"]["pairs"] == 0


def test_audit_classification_subject_leakage(tmp_path):
    for k, (split, subj) in enumerate((("train", "s0001"), ("val", "s0002"), ("test", "s0001"))):
        for cls, seed in (("open", 10 + k), ("closed", 20 + k)):
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            cv2.imwrite(str(d / f"{subj}_0000{seed}_0_0_{int(cls == 'open')}_0_0_01.png"), _img(seed))
    import re
    rep = audit_classification(tmp_path, group_of=lambda n: re.match(r"(s\d{4})_", n).group(1))
    assert rep["groups"]["n_groups_in_several_splits"] == 1
