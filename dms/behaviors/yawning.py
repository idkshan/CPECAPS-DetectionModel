"""Yawning: the mouth stays wide open for >= 1.5 s.

Purpose   Detect yawns (fatigue sign; thesis Table 1.11 "frequent yawning").
Inputs    FrameFeatures.mar (geometric Mouth Aspect Ratio), jaw_open (learned
          jawOpen blendshape), yawn_yolo (YOLO "yawn" boxes, if the driver-state
          model is used); the driver's calibrated neutral mouth (Thresholds)
Outputs   FrameStates.mouth_open, mouth_evidence, mouth_sources
          YawnStatus: yawning flag, mouth_open_s, yawns_in_window, events
Params    thresholds.mar_open_delta (0.40), jaw_open_delta (0.35), decision_level (0.5)
          evidence.mouth_mar / mouth_blendshape / mouth_yolo (fusion weights)
          temporal.yawn_min_s (1.5 s), yawn_gap_s (0.3 s), yawn_window_s (300 s)

Algorithm
  Each source becomes evidence 0..1 (0 = this driver's neutral mouth, 0.5 =
  their threshold); the weighted mean >= 0.5 means "mouth wide open" on this
  frame. Speech opens the mouth for a fraction of a second at a time, a yawn
  keeps it wide open for several seconds, so an open-mouth episode becomes a
  yawn only once it has lasted 1.5 s (short closures <= 0.3 s are bridged).
  Yawns are counted over the last 5 minutes for the fatigue level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from ..classification.calibration import Thresholds
from ..classification.evidence import fuse, ramp_evidence
from ..classification.temporal import EpisodeTracker, EventWindow
from ..config import Config
from ..types import FrameFeatures, FrameStates


@dataclass
class YawnStatus:
    mouth_open_s: float = 0.0
    yawns_in_window: int = 0
    yawning: bool = False
    events: List[str] = field(default_factory=list)


class YawningDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        tc = cfg.temporal
        self.episode = EpisodeTracker(tc.yawn_gap_s)
        self.yawns = EventWindow(tc.yawn_window_s)
        self._last_mouth_active_t = -math.inf

    def frame_state(self, f: FrameFeatures, thr: Thresholds, st: FrameStates) -> None:
        if not f.face_present:
            return
        ev, b = self.cfg.evidence, thr.baseline
        sources = {
            "mar": ramp_evidence(f.mar, b.mar, thr.mar_open),
            "blendshape": ramp_evidence(f.jaw_open, b.jaw, thr.jaw_open),
            "yolo": f.yawn_yolo,
        }
        st.mouth_sources = {k: v for k, v in sources.items() if v is not None}
        st.mouth_evidence = fuse(sources, {"mar": ev.mouth_mar, "blendshape": ev.mouth_blendshape, "yolo": ev.mouth_yolo})
        if st.mouth_evidence is not None:
            st.mouth_open = st.mouth_evidence >= thr.decision_level
            if st.mouth_evidence >= self.cfg.temporal.yawn_eye_guard_level:
                self._last_mouth_active_t = f.t

    def eye_guard(self, t: float, st: FrameStates) -> bool:
        """True while the mouth is opening for a yawn or just closed (eyes squint then)."""
        return bool(st.mouth_open) or (t - self._last_mouth_active_t) <= self.cfg.temporal.yawn_eye_guard_s

    def update(self, t: float, st: FrameStates, gap: bool) -> YawnStatus:
        tc = self.cfg.temporal
        if gap:
            self.episode.force_end()
        self.episode.update(t, st.mouth_open)
        events = []
        if self.episode.fire_once("yawn", tc.yawn_min_s):
            self.yawns.add(t)
            events.append("yawn")
        return YawnStatus(
            mouth_open_s=self.episode.duration(),
            yawns_in_window=self.yawns.count(t),
            yawning=self.episode.active and self.episode.duration() >= tc.yawn_min_s,
            events=events,
        )

    @property
    def active(self) -> bool:
        return self.episode.active
