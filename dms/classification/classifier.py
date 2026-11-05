"""Stage 3: threshold-based classification of the AI-derived visual features.

This is the rule-based layer. It does not look at pixels; it *interprets*
what the neural networks of Stage 1 measured. The actual rules for each
behaviour live in ``dms/behaviors/`` (one module per behaviour); this class
runs them in the right order for every frame and merges their results:

    1. classify_frame(): per-frame states    quality -> gaze -> phone -> yawning -> drowsiness
       (drowsiness last: the eyes are not judged while the head is turned away,
       a phone is in view or the driver is yawning)
    2. update(): temporal behaviour          episodes, events, windows, drowsiness state

The output separates what is happening NOW (``BehaviorStatus.active``: the
behaviours that may keep an alert sounding) from what happened RECENTLY
(windows such as PERCLOS, yawn count and off-road ratio, which raise the
assessed risk level but must not keep an alarm going on their own).
"""

from __future__ import annotations

import math

from ..behaviors import DrowsinessDetector, FrameQuality, GazeDetector, PhoneDetector, YawningDetector
from ..config import Config
from ..types import BehaviorStatus, FrameFeatures, FrameStates
from .calibration import Thresholds


class BehaviorClassifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.quality = FrameQuality(cfg.quality)
        self.gaze = GazeDetector(cfg)
        self.phone = PhoneDetector(cfg)
        self.yawning = YawningDetector(cfg)
        self.drowsiness = DrowsinessDetector(cfg, self.quality)
        self.last_event_t = -math.inf

    # ------------------------------------------------------ per-frame states
    def classify_frame(self, f: FrameFeatures, thr: Thresholds) -> FrameStates:
        st = FrameStates()
        st.frame_gap = self.quality.check_gap(f.t)
        self.gaze.frame_state(f, thr, st)
        self.phone.frame_state(f, st)
        self.yawning.frame_state(f, thr, st)
        self.drowsiness.frame_state(f, thr, st, mouth_guard=self.yawning.eye_guard(f.t, st))
        return st

    # ---------------------------------------------------- temporal behaviour
    def update(self, f: FrameFeatures, st: FrameStates) -> BehaviorStatus:
        t, gap = f.t, st.frame_gap
        gaze = self.gaze.update(t, st, gap)
        phone = self.phone.update(t, st, gap)
        yawn = self.yawning.update(t, st, gap)
        drowsy = self.drowsiness.update(t, st, gap, self.yawning.yawns)

        events = drowsy.events + yawn.events + gaze.events + phone.events
        if events:
            self.last_event_t = t
        active = [name for name, on in (("eye_closure", drowsy.eye_closure), ("yawning", yawn.yawning),
                                         ("gaze_deviation", gaze.gaze_deviation), ("phone_usage", phone.phone_usage)) if on]
        return BehaviorStatus(
            closure_s=drowsy.closure_s,
            perclos=drowsy.perclos,
            mouth_open_s=yawn.mouth_open_s,
            yawns_in_window=yawn.yawns_in_window,
            gaze_away_s=gaze.gaze_away_s,
            offroad_ratio=gaze.offroad_ratio,
            phone_s=phone.phone_s,
            long_closures=drowsy.long_closures,
            blink_rate_per_min=drowsy.blink_rate_per_min,
            eye_level=drowsy.eye_level,
            drowsiness_state=drowsy.state,
            drowsy=drowsy.state != "alert",
            eye_closure=drowsy.eye_closure,
            yawning=yawn.yawning,
            gaze_deviation=gaze.gaze_deviation,
            phone_usage=phone.phone_usage,
            events=events,
            active=active,
            no_face=gaze.no_face,
            face_status=st.face_status,
            data_valid=f.face_present and not gap and st.eyes_not_judged != "quality",
        )

    def any_episode_active(self) -> bool:
        return self.drowsiness.active or self.yawning.active or self.gaze.active or self.phone.active
