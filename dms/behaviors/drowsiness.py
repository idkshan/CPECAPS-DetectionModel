"""Drowsiness: eye closure events and the drowsiness STATE of thesis Table 1.11.

Purpose   1. per frame: are the eyes closed?            (AI evidence + driver thresholds)
          2. events:    blinks, long closures (>= 0.5 s, "slow blinks"), microsleeps
          3. state:     Alert / Slightly Drowsy / Moderately Drowsy (Table 1.11)
Inputs    FrameFeatures: ear_left / ear_right (geometric Eye Aspect Ratio),
          blink (learned eyeBlink blendshape), eye_closed_prob (optional eye-state
          CNN), eye_closed_yolo (YOLO driver-state boxes), look_down, eye_width_px;
          the driver's calibrated baseline (Thresholds); the other behaviours'
          frame states (head away, phone, yawning) and the yawn count
Outputs   FrameStates.eyes_closed / eye_evidence / eye_sources / eyes_not_judged
          DrowsinessStatus (closure_s, perclos, long closures, state, events)
Params    thresholds.ear_closed_ratio (0.70), blink_closed_delta (0.35), decision_level
          evidence.eye_* weights; temporal.closure_gap_s, closure_event_s (0.5 s),
          closure_hold_s; quality.*; drowsiness.* (state criteria, recovery_s)

Frame level - eyes closed?
  Each available source becomes evidence 0..1 (0 = this driver's open eye,
  0.5 = their threshold). EAR is normalised PER EYE: with a camera off to one
  side the two eyes have different open-eye EARs, so one shared baseline
  would make the threshold too strict for one eye and too loose for the other.
  The eyes are NOT judged (state unknown, not "closed") when the head is
  turned away, a phone is in view, the mouth is yawning, the face is missing
  or the eyes are too small in the image.

Events - blink vs long closure vs microsleep
  An episode of closed-eye frames shorter than 0.5 s is an ordinary blink.
  >= 0.5 s is a long closure (slow blink). >= 1.5 s is a microsleep.

State - Table 1.11 (criteria in configs/default.yaml, section ``drowsiness``)
  Moderately drowsy  any of: PERCLOS >= 15 %; a closure >= 1.5 s; >= 4 long
                     closures in the last 60 s; frequent yawning together with
                     a slight eye sign
  Slightly drowsy    any of: PERCLOS >= 8 %; a closure >= 0.8 s; >= 2 long
                     closures in 60 s; >= 2 yawns in 5 min with the last one
                     within 60 s ("frequent yawning", still going on)
  Alert              otherwise
  Eye signs RECOVER: after ``recovery_s`` (20 s) without a new long closure
  the eye level steps down one level and the eye windows restart. Without
  this, PERCLOS (a 60-s window) kept the driver "drowsy" for up to a minute
  after the eyes were normal again (S01: drowsy flag on 75-255 s, frame
  precision 0.08). Yawning counts while yawns are frequent (>= 2 in 5 min)
  AND current (last one <= 60 s ago); without the second condition one burst
  of yawns kept the driver "slightly drowsy" for 5 minutes.

The RISK model (Stage 5) gets the eye part only (closure length and the eye
level); yawning reaches it through its own channel, so it is not counted twice.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

from ..classification.calibration import Thresholds
from ..classification.evidence import fuse, ramp_evidence
from ..classification.temporal import EpisodeTracker, EventWindow, TimeWeightedRatio
from ..config import Config
from ..types import FrameFeatures, FrameStates
from .quality import FrameQuality

STATES = ("alert", "slightly_drowsy", "moderately_drowsy")
BLINK_MIN_S = 0.0  # a single closed frame counts as a blink


@dataclass
class DrowsinessStatus:
    closure_s: float = 0.0  # longest closure ongoing or ended within temporal.closure_hold_s
    eye_closure: bool = False  # a long closure (>= closure_event_s) is happening now
    perclos: Optional[float] = None  # since the last recovery, None until enough coverage
    long_closures: int = 0
    longest_closure_s: float = 0.0
    blink_rate_per_min: Optional[float] = None
    eye_level: int = 0  # 0 alert, 1 slight, 2 moderate - eye signs only (goes to the risk model)
    state: str = "alert"  # combined with yawning frequency (Table 1.11)
    events: List[str] = field(default_factory=list)


class DrowsinessDetector:
    def __init__(self, cfg: Config, quality: Optional[FrameQuality] = None):
        self.cfg = cfg
        tc, dc = cfg.temporal, cfg.drowsiness
        self.quality = quality or FrameQuality(cfg.quality)
        self.closure = EpisodeTracker(tc.closure_gap_s)
        self.perclos = TimeWeightedRatio(dc.window_s)
        self.long_closures = EventWindow(dc.window_s)  # end times of closures >= closure_event_s
        self.blinks = EventWindow(60.0)
        self._closures: Deque[Tuple[float, float]] = deque()  # (end time, duration) of long closures
        self.eye_level = 0
        self._reset_t = -math.inf  # eye signs before this time no longer count (after a recovery)
        self._last_sign_t = -math.inf
        self._last_change_t = -math.inf
        self._first_t: Optional[float] = None

    # ------------------------------------------------------------ per frame
    def frame_state(self, f: FrameFeatures, thr: Thresholds, st: FrameStates, mouth_guard: bool) -> None:
        tc, q, ev = self.cfg.temporal, self.cfg.quality, self.cfg.evidence
        if not f.face_present:
            st.eyes_not_judged = "no_face"
            return
        b = thr.baseline
        ear = None
        if f.ear_left is not None and f.ear_right is not None and b.ear_left and b.ear_right:
            left = ramp_evidence(f.ear_left, b.ear_left, b.ear_left * thr.ear_closed_ratio)
            right = ramp_evidence(f.ear_right, b.ear_right, b.ear_right * thr.ear_closed_ratio)
            ear = (left + right) / 2.0
        elif f.ear is not None:
            ear = ramp_evidence(f.ear, b.ear, thr.ear_closed)
        sources = {
            "ear": ear,
            "blendshape": ramp_evidence(f.blink, b.blink, thr.blink_closed),
            "cnn": f.eye_closed_prob,
            "yolo": f.eye_closed_yolo,
        }
        weights = {"ear": ev.eye_ear, "blendshape": ev.eye_blendshape, "cnn": ev.eye_cnn, "yolo": ev.eye_yolo}
        if f.look_down is not None and f.look_down >= q.look_down_ear_guard and sources["blendshape"] is not None:
            weights["ear"] = 0.0  # eyes looking down: lowered lids shrink EAR without closing
        st.eye_sources = {k: v for k, v in sources.items() if v is not None}
        st.eye_evidence = fuse(sources, weights)

        if st.head_away:
            st.eyes_not_judged = "head_away"
        elif tc.suspend_eyes_when_phone and st.phone_visible:
            st.eyes_not_judged = "phone"
        elif mouth_guard:
            st.eyes_not_judged = "yawn"
        elif not self.quality.eyes_measurable(f):
            st.eyes_not_judged = "quality"
        elif st.eye_evidence is not None:
            st.eyes_closed = st.eye_evidence >= thr.decision_level

    # ------------------------------------------------------------ over time
    def _closure_ended(self, ended: Optional[Tuple[float, float]], t: float) -> bool:
        """Book an ended closure episode; True if it was a long closure."""
        if ended is None:
            return False
        duration = ended[1] - ended[0]
        if duration >= self.cfg.temporal.closure_event_s:
            self.long_closures.add(ended[1])
            self._closures.append((ended[1], duration))
            return True
        if duration >= BLINK_MIN_S:
            self.blinks.add(ended[1])
        return False

    def update(self, t: float, st: FrameStates, gap: bool, yawns: EventWindow) -> DrowsinessStatus:
        tc, dc = self.cfg.temporal, self.cfg.drowsiness
        if self._first_t is None:
            self._first_t = t
        long_ended = False
        if gap:  # a camera stall must not stretch a closure into a fake microsleep
            long_ended |= self._closure_ended(self.closure.force_end(), t)
        long_ended |= self._closure_ended(self.closure.update(t, st.eyes_closed), t)
        if st.eyes_closed is None:
            self.perclos.skip(t)
        else:
            self.perclos.add(t, st.eyes_closed)
        events = ["eye_closure"] if self.closure.fire_once("closure", tc.closure_event_s) else []

        current = self.closure.duration() if self.closure.active else 0.0
        closing_now = self.closure.active and current >= tc.closure_event_s
        while self._closures and t - self._closures[0][0] > dc.window_s:
            self._closures.popleft()
        since = self._reset_t
        n_long = self.long_closures.count_since(t, since) + int(closing_now)
        longest = max([d for end, d in self._closures if end > since] + [current])
        perclos = self.perclos.ratio if self.perclos.coverage_s >= dc.perclos_min_coverage_s else None
        blink_rate = None
        if t - self._first_t >= 60.0:
            blink_rate = float(self.blinks.count(t))  # blinks in the last 60 s = per minute

        if (perclos is not None and perclos >= dc.moderate_perclos) or longest >= dc.moderate_closure_s \
                or n_long >= dc.moderate_long_closures:
            eye_sign = 2
        elif (perclos is not None and perclos >= dc.slight_perclos) or longest >= dc.slight_closure_s \
                or n_long >= dc.slight_long_closures \
                or (dc.blink_rate_slight_per_min > 0 and blink_rate is not None and blink_rate >= dc.blink_rate_slight_per_min):
            eye_sign = 1
        else:
            eye_sign = 0

        if closing_now or long_ended:
            self._last_sign_t = t
        if eye_sign > self.eye_level:
            self.eye_level = eye_sign
            self._last_sign_t = self._last_change_t = t
        elif self.eye_level > 0 and t - self._last_sign_t >= dc.recovery_s and t - self._last_change_t >= dc.recovery_s:
            self.eye_level -= 1  # recovery: step down one level and restart the eye windows
            self._last_change_t = self._reset_t = t
            self.perclos.reset()

        n_yawns = yawns.count(t)
        last_yawn = yawns.last()
        recent = last_yawn is not None and t - last_yawn <= dc.yawn_hold_s
        yawn_level = 0
        if recent:
            yawn_level = 2 if dc.moderate_yawns and n_yawns >= dc.moderate_yawns else \
                1 if dc.slight_yawns and n_yawns >= dc.slight_yawns else 0
        level = max(self.eye_level, yawn_level)
        if dc.yawns_plus_eyes_moderate and yawn_level >= 1 and self.eye_level >= 1:
            level = 2
        return DrowsinessStatus(
            closure_s=self.closure.recent_max(t, tc.closure_hold_s),
            eye_closure=closing_now,
            perclos=perclos,
            long_closures=n_long,
            longest_closure_s=longest,
            blink_rate_per_min=blink_rate,
            eye_level=self.eye_level,
            state=STATES[level],
            events=events,
        )

    @property
    def active(self) -> bool:
        return self.closure.active
