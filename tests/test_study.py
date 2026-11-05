"""Participant-study evaluation: manifest, replay, pooling, leakage check."""

import csv
import json
import sys

import pytest

import evaluate_study
from dms.config import Config
from dms.evaluation.labels import load_labels
from dms.evaluation.session import evaluate_session
from dms.simulation import ScenarioBuilder


def write_session(folder, seed=0):
    """A synthetic session folder holding only Stage-2 features (evaluated with replay)."""
    b = ScenarioBuilder(fps=10, seed=seed)
    b.alert(40).yawn(4).alert(20).eyes_closed(4).alert(20).phone(6).alert(20)
    folder.mkdir(parents=True)
    with open(folder / "frames.jsonl", "w", encoding="utf-8") as fh:
        for f in b.build():
            fh.write(json.dumps({"t": f.t, "features": f.to_dict()}) + "\n")
    (folder / "alerts.jsonl").write_text("")
    return folder


def write_labels(path, bom=False):
    text = "behavior,start_s,end_s\nyawning,40,44\nmoderately_drowsy,64,68\nphone,88,94\n"
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode())
    return path


def test_session_replay_detects_labelled_behaviours(tmp_path):
    session = write_session(tmp_path / "S")
    labels = load_labels(write_labels(tmp_path / "l.csv"))
    rep = evaluate_session(session, labels, cfg=Config())
    assert rep["behaviors"]["yawning"]["event"]["tp"] == 1
    assert rep["behaviors"]["phone_usage"]["event"]["tp"] == 1
    assert rep["drowsiness_states"]["labelled_segments"] == 1
    assert "three_class" in rep["drowsiness_states"]
    assert rep["alarms"]["non_event_hours"] < rep["alarms"]["driving_hours"]


def test_study_pools_participants_and_blocks_leakage(tmp_path, monkeypatch):
    rows = []
    for i, split in enumerate(["validation", "validation", "test"]):
        write_session(tmp_path / "sessions" / f"P{i}_day", seed=i)
        write_labels(tmp_path / f"P{i}_day.csv")
        rows.append([f"P{i}", f"P{i}_day", f"P{i}_day.csv", split, "day"])
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="") as fh:
        csv.writer(fh).writerows([["participant", "session", "labels", "split", "condition"]] + rows)
    out = tmp_path / "eval"
    monkeypatch.setattr(sys, "argv", ["x", "--manifest", str(manifest), "--sessions", str(tmp_path / "sessions"),
                                      "--split", "validation", "--replay", "--tag", "t", "--out", str(out)])
    assert evaluate_study.main() == 0
    report = json.loads((out / "t" / "report.json").read_text())
    assert report["meta"]["sessions"] == 2 and report["meta"]["participants"] == 2
    yawn = report["summary"]["validation - all sessions"]["yawning"]
    assert yawn["labelled_events"] == 2 and yawn["event"]["tp"] == 2
    assert (out / "t" / "summary.md").exists() and (out / "t" / "per_session.csv").exists()

    # the same participant in both splits is refused
    with open(manifest, "a", newline="") as fh:
        csv.writer(fh).writerow(["P0", "P2_day", "P2_day.csv", "test", "night"])
    with pytest.raises(SystemExit, match="leakage"):
        evaluate_study.load_manifest(manifest, tmp_path / "sessions")


def test_tuning_uses_validation_only_and_writes_a_config(tmp_path, monkeypatch):
    import tune_thresholds

    rows = []
    for i, split in enumerate(["validation", "validation", "test"]):
        write_session(tmp_path / "sessions" / f"P{i}_day", seed=i)
        write_labels(tmp_path / f"P{i}_day.csv")
        rows.append([f"P{i}", f"P{i}_day", f"P{i}_day.csv", split, "day"])
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="") as fh:
        csv.writer(fh).writerows([["participant", "session", "labels", "split", "condition"]] + rows)
    space = tmp_path / "space.yaml"
    space.write_text("drowsiness.recovery_s: [10.0, 20.0]\n")
    out_cfg = tmp_path / "tuned.yaml"
    monkeypatch.setattr(sys, "argv", ["x", "--manifest", str(manifest), "--sessions", str(tmp_path / "sessions"),
                                      "--behaviors", "drowsiness", "--space", str(space), "--rounds", "1",
                                      "--out-config", str(out_cfg), "--report", str(tmp_path / "rep")])
    assert tune_thresholds.main() == 0
    assert out_cfg.exists() and "VALIDATION" in out_cfg.read_text()
    from dms.config import load_config
    load_config(out_cfg)  # the written file is a valid config
    trials = json.loads((tmp_path / "rep" / "trials.json").read_text())
    assert 2 <= len(trials) <= 3  # defaults + 10 s (+ 20 s again if 10 s became the current best)
    assert all(set(t["params"]) <= {"drowsiness.recovery_s"} for t in trials)
