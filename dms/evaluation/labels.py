"""Ground-truth labels: one CSV per recording, one row per labelled segment.

    behavior,start_s,end_s
    yawning,63.2,68.9
    gaze_deviation,120.0,124.5
    phone_usage,300.1,309.8
    slightly_drowsy,900.0,960.0
    moderately_drowsy,1500.0,1560.0

Times are seconds from the start of the video. Frames outside every segment
are "normal" for that behaviour (the thesis labels every frame: Section
1.7.21), so a session must be labelled completely, not just partly.

Behaviour names and the classes they belong to:
  drowsiness       drowsiness, drowsy                  (state, level not given)
                   slightly_drowsy, level1, mild       (Table 1.11 "Slightly Drowsy")
                   moderately_drowsy, level2, moderate (Table 1.11 "Moderately Drowsy")
                   severely_drowsy, level3, severe     (counted as Moderately Drowsy
                                                        in the 3-class thesis metrics)
  yawning          yawning, yawn
  gaze_deviation   gaze_deviation, gaze, looking_away
  phone_usage      phone_usage, phone, phone_use
  eye_closure      eye_closure, microsleep, long_closure   (optional: every
                   eye closure >= 0.5 s, scored against the eye-closure detector)
  sudden_motion    sudden_motion, harsh_braking, swerve   (context event: an
                   alarm here is expected, Section 1.7.1; not a vision metric)
Recognised but not a positive class: alert, normal, talking, mirror_check, blink.

Optional 4th column ``expected_tier`` (NONE / LOW / MEDIUM / HIGH): the alarm
tier the enacted behaviour SHOULD trigger (thesis Table 1.3 "alarm
validation"); evaluated by dms.evaluation.alarms.alarm_validation.

Anything else is reported as a warning (typos used to be dropped silently).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

Segment = Tuple[float, float]

VISION_BEHAVIORS = ("drowsiness", "yawning", "gaze_deviation", "phone_usage")
OPTIONAL_BEHAVIORS = ("eye_closure",)
CONTEXT_EVENTS = ("sudden_motion",)
ALL_LABEL_CLASSES = VISION_BEHAVIORS + OPTIONAL_BEHAVIORS + CONTEXT_EVENTS

DROWSINESS_LEVELS = {
    "drowsiness": "unspecified", "drowsy": "unspecified",
    "slightly_drowsy": "slightly_drowsy", "level1": "slightly_drowsy", "level_1": "slightly_drowsy", "mild": "slightly_drowsy",
    "moderately_drowsy": "moderately_drowsy", "level2": "moderately_drowsy", "level_2": "moderately_drowsy",
    "moderate": "moderately_drowsy",
    "severely_drowsy": "severely_drowsy", "level3": "severely_drowsy", "level_3": "severely_drowsy", "severe": "severely_drowsy",
}
ALIASES = {
    **{k: "drowsiness" for k in DROWSINESS_LEVELS},
    "yawn": "yawning", "yawning": "yawning",
    "gaze": "gaze_deviation", "gaze_deviation": "gaze_deviation", "looking_away": "gaze_deviation", "gaze_away": "gaze_deviation",
    "phone": "phone_usage", "phone_use": "phone_usage", "phone_usage": "phone_usage",
    "eye_closure": "eye_closure", "microsleep": "eye_closure", "long_closure": "eye_closure",
    "sudden_motion": "sudden_motion", "harsh_braking": "sudden_motion", "harsh_brake": "sudden_motion", "swerve": "sudden_motion",
}
NOT_POSITIVE = {"alert", "normal", "none", "talking", "mirror_check", "blink", "singing"}


@dataclass
class LabelSet:
    segments: Dict[str, List[Segment]] = field(default_factory=lambda: {k: [] for k in ALL_LABEL_CLASSES})
    drowsiness_levels: List[Tuple[float, float, str]] = field(default_factory=list)  # (start, end, level)
    expected_tiers: List[Tuple[float, float, str, str]] = field(default_factory=list)  # (start, end, behaviour, tier)
    warnings: List[str] = field(default_factory=list)
    source: str = ""

    def all_events(self) -> List[Segment]:
        """Every labelled positive segment of any class (used to judge alarms)."""
        return merge_segments([s for segs in self.segments.values() for s in segs])

    def labelled_seconds(self, behavior: str) -> float:
        return sum(e - s for s, e in self.segments.get(behavior, []))


def _norm(text: str) -> str:
    return text.strip().lower().replace(" ", "_").replace("-", "_")


def merge_segments(segs: List[Segment]) -> List[Segment]:
    """Sort and merge overlapping/touching segments."""
    out: List[Segment] = []
    for s, e in sorted(segs):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def load_labels(path: str | Path, duration_s: Optional[float] = None, strict: bool = False) -> LabelSet:
    """Read and validate a label CSV.

    Errors (raise ValueError): missing columns, unreadable numbers, end <= start,
    negative times, two different drowsiness levels overlapping.
    Warnings (kept in ``LabelSet.warnings``; errors when ``strict``): unknown
    behaviour names, overlapping segments of one behaviour (merged), segments
    that end after the recording.
    """
    labels = LabelSet(source=str(path))
    raw: Dict[str, List[Segment]] = {k: [] for k in ALL_LABEL_CLASSES}
    with open(path, newline="", encoding="utf-8-sig") as fh:  # utf-8-sig: Excel writes a BOM
        reader = csv.DictReader(fh)
        reader.fieldnames = [_norm(f) for f in reader.fieldnames or []]
        missing = {"behavior", "start_s", "end_s"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing column(s) {sorted(missing)}; the header must be behavior,start_s,end_s")
        for line_no, row in enumerate(reader, start=2):
            name = _norm(row.get("behavior") or "")
            if not name:
                continue
            try:
                start, end = float(row["start_s"]), float(row["end_s"])
            except (TypeError, ValueError):
                raise ValueError(f"{path} line {line_no}: start_s/end_s must be numbers, got {row.get('start_s')!r}, {row.get('end_s')!r}")
            if start < 0 or end <= start:
                raise ValueError(f"{path} line {line_no}: invalid segment {start}-{end} (need 0 <= start < end)")
            if duration_s is not None and end > duration_s + 1.0:
                labels.warnings.append(f"line {line_no}: segment ends at {end:.1f} s, after the recording ({duration_s:.1f} s)")
            if name in NOT_POSITIVE:
                tier = (row.get("expected_tier") or "").strip().upper()
                if tier:  # e.g. "mirror_check ... NONE": the alarm must stay silent
                    labels.expected_tiers.append((start, end, name, tier))
                continue
            cls = ALIASES.get(name)
            if cls is None:
                labels.warnings.append(f"line {line_no}: unknown behaviour '{row['behavior'].strip()}' ignored "
                                       f"(use one of: {', '.join(sorted(set(ALIASES)))})")
                continue
            raw[cls].append((start, end))
            tier = (row.get("expected_tier") or "").strip().upper()
            if tier:
                if tier not in ("NONE", "LOW", "MEDIUM", "HIGH"):
                    raise ValueError(f"{path} line {line_no}: expected_tier must be NONE, LOW, MEDIUM or HIGH, not {tier!r}")
                labels.expected_tiers.append((start, end, cls, tier))
            if cls == "drowsiness":
                labels.drowsiness_levels.append((start, end, DROWSINESS_LEVELS[name]))

    for cls, segs in raw.items():
        merged = merge_segments(segs)
        if len(merged) < len(segs):
            labels.warnings.append(f"{cls}: {len(segs) - len(merged)} overlapping segment(s) merged")
        labels.segments[cls] = merged

    levels = sorted(labels.drowsiness_levels)
    for (s1, e1, l1), (s2, e2, l2) in zip(levels, levels[1:]):
        if s2 < e1 and l1 != l2:
            raise ValueError(f"{path}: drowsiness levels '{l1}' ({s1}-{e1}) and '{l2}' ({s2}-{e2}) overlap")
    labels.drowsiness_levels = levels

    if strict and labels.warnings:
        raise ValueError(f"{path}: " + "; ".join(labels.warnings))
    return labels
