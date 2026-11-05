"""Section 7 - dataset loading, splitting and auditing (training side).

    splits.py  group-wise train/val/test split (a person / driver / video never
               spans two splits) and the split manifest written next to a dataset
    yolo.py    read a YOLO dataset in either folder layout; parse label files
    audit.py   find corrupt images, label errors, class imbalance, exact and
               near-duplicate images across splits, and group leakage

The dataset *builders* (scripts/autolabel_phones.py, prepare_mrl_eye.py,
build_driver_state_dataset.py) use splits.py; scripts/audit_dataset.py runs
audit.py on any finished dataset - including one re-exported from Roboflow,
whose own random split can silently undo a subject-wise split.
"""
