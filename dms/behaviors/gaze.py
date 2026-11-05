"""Gaze deviation: the head is turned away from the driver's own forward pose.

Purpose   Detect looking away from the road (thesis: via head pose, Section 1.7.8).
Inputs    FrameFeatures.yaw / pitch (solvePnP head pose, degrees),
          the calibrated baseline pose and limits (Thresholds),
          face presence (the face model loses the face when the head turns far)
Outputs   FrameStates.head_away, pose_evidence, face_status
          GazeStatus: gaze_away_s, offroad_ratio, gaze_deviation flag, events
Params    thresholds.yaw_deg (30), pitch_down_deg (20), pitch_up_deg (25)
          temporal.gaze_min_s (2.0 s: shorter glances are mirror checks)
          temporal.gaze_gap_s, gaze_hold_s, offroad_window_s (30 s)
          temporal.face_loss_as_gaze_away, face_loss_max_s (10 s)
          quality.face_loss_turning_evidence (0.3), face_loss_lookback_s (0.5 s)

Algorithm
  pose evidence = max over yaw, pitch-down, pitch-up of a ramp that is 0 at
  the baseline pose and 0.5 at the limit (e.g. 30 deg of yaw). Evidence >= 0.5
  -> head away on this frame. A gaze deviation is an episode of head-away
  frames lasting >= 2 s (gaps up to 0.3 s bridged).

Face lost: if the face disappears while the head was turning (pose evidence
>= 0.3 within the last 0.5 s), the head most likely turned beyond the
landmark model's range -> counted as head away for up to 10 s. If it
disappears while the driver faced forward (hand over the camera, glare,
dropout), the gaze is UNKNOWN - previously this also counted as looking away.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

from ..classification.calibration import Thresholds
from ..classification.evidence import ramp_evidence
from ..classification.temporal import EpisodeTracker, TimeWeightedRatio
from ..config import Config
from ..types import FrameFeatures, FrameStates


@dataclass
class GazeStatus:
    gaze_away_s: float = 0.0
    offroad_ratio: float = 0.0
    gaze_deviation: bool = False
    no_face: bool = False
    events: List[str] = field(default_factory=list)


class GazeDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        tc = cfg.temporal
        self.episode = EpisodeTracker(tc.gaze_gap_s)
        self.offroad = TimeWeightedRatio(tc.offroad_window_s)
        self._last_face_t: Optional[float] = None
        self._recent_pose: Deque[Tuple[float, float]] = deque()  # (t, pose evidence) of recent face frames

    def frame_state(self, f: FrameFeatures, thr: Thresholds, st: FrameStates) -> None:
        tc, q = self.cfg.temporal, self.cfg.quality
        if not f.face_present:
            if self._last_face_t is None:
                st.face_status = "absent"  # never seen a face yet
                return
            st.face_lost_s = f.t - self._last_face_t
            was_turning = any(ev >= q.face_loss_turning_evidence for _, ev in self._recent_pose)
            if st.face_lost_s > tc.face_loss_max_s:
                st.face_status = "absent"
            elif tc.face_loss_as_gaze_away and was_turning:
                st.face_status = "lost_turning"
                st.head_away = True
                st.pose_evidence = 1.0
            else:
                st.face_status = "lost"  # gaze unknown
            return

        self._last_face_t = f.t
        if f.yaw is None or f.pitch is None:
            return
        b = thr.baseline
        dp = f.pitch - b.pitch
        st.pose_evidence = max(
            ramp_evidence(abs(f.yaw - b.yaw), 0.0, thr.yaw_limit),
            ramp_evidence(-dp, 0.0, thr.pitch_down_limit),
            ramp_evidence(dp, 0.0, thr.pitch_up_limit),
        )
        st.head_away = st.pose_evidence >= thr.decision_level
        self._recent_pose.append((f.t, st.pose_evidence))
        while self._recent_pose and f.t - self._recent_pose[0][0] > q.face_loss_lookback_s:
            self._recent_pose.popleft()

    def update(self, t: float, st: FrameStates, gap: bool) -> GazeStatus:
        tc = self.cfg.temporal
        if gap:
            self.episode.force_end()
        self.episode.update(t, st.head_away)
        if st.head_away is None:
            self.offroad.skip(t)
        else:
            self.offroad.add(t, st.head_away)
        events = ["gaze_deviation"] if self.episode.fire_once("gaze", tc.gaze_min_s) else []
        return GazeStatus(
            gaze_away_s=self.episode.recent_max(t, tc.gaze_hold_s),
            # off-road time over a fixed window length (not over observed time),
            # so a single glance early in a session is not over-weighted
            offroad_ratio=self.offroad.ratio * min(1.0, self.offroad.coverage_s / tc.offroad_window_s),
            gaze_deviation=self.episode.active and self.episode.duration() >= tc.gaze_min_s,
            no_face=st.face_status == "absent",
            events=events,
        )

    @property
    def active(self) -> bool:
        return self.episode.active
