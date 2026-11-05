"""Mobile-phone use: a phone held near the driver for >= 3 s.

Purpose   Detect handheld phone use (thesis Section 1.7.15: confirmed after 3 s).
Inputs    FrameFeatures.phone_conf = best YOLO "phone" confidence among boxes
          near the driver (Stage 2 drops phones far from the face)
Outputs   FrameStates.phone_visible; PhoneStatus: phone_s, phone_usage, events
Params    detector.conf (0.35)           a phone episode STARTS at this confidence
          detector.phone_stay_conf (0.20) an ongoing episode CONTINUES down to this
          temporal.phone_confirm_s (3 s), phone_gap_s (1.0 s), phone_hold_s (1 s)

Why the two confidence levels (hysteresis): a held phone is partly hidden by
the hand and moves, so YOLO's confidence drops below 0.35 on many frames
(in S01 the phone was detected in only ~50-80 % of the frames of real use,
and every episode broke before reaching 3 s). Starting strict but continuing
leniently - the way object trackers keep a track alive - and bridging gaps
up to 1 s keeps one phone use as one episode, without letting weak
detections START an episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..classification.temporal import EpisodeTracker
from ..config import Config
from ..types import FrameFeatures, FrameStates


@dataclass
class PhoneStatus:
    phone_s: float = 0.0
    phone_usage: bool = False
    events: List[str] = field(default_factory=list)


class PhoneDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.episode = EpisodeTracker(cfg.temporal.phone_gap_s)

    def frame_state(self, f: FrameFeatures, st: FrameStates) -> None:
        d = self.cfg.detector
        threshold = min(d.phone_stay_conf, d.conf) if self.episode.active else d.conf
        st.phone_visible = f.phone_conf > 0.0 and f.phone_conf >= threshold

    def update(self, t: float, st: FrameStates, gap: bool) -> PhoneStatus:
        tc = self.cfg.temporal
        if gap:
            self.episode.force_end()
        self.episode.update(t, st.phone_visible)
        events = ["phone_usage"] if self.episode.fire_once("phone", tc.phone_confirm_s) else []
        return PhoneStatus(
            phone_s=self.episode.recent_max(t, tc.phone_hold_s),
            phone_usage=self.episode.active and self.episode.duration() >= tc.phone_confirm_s,
            events=events,
        )

    @property
    def active(self) -> bool:
        return self.episode.active
