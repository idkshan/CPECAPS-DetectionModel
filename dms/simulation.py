"""Synthetic feature timelines for testing Stages 3-5 without a camera.

Values mimic what the Stage-1/2 models produce for a real driver (open-eye
EAR ~0.29, closed mouth MAR ~0.02, ...), with small seeded noise. The
scenarios reproduce the behaviour-to-alarm mapping of Section 1.7.1 and are
used by the unit tests and by ``scripts/simulate_scenarios.py``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

import numpy as np

from .config import Config
from .context import StaticContext
from .engine import DecisionEngine, EngineOutput
from .types import FrameFeatures

NEUTRAL = dict(ear=0.29, mar=0.02, yaw=8.0, pitch=-4.0, roll=0.0, blink=0.10, jaw_open=0.02, look_down=0.05)
EYE_WIDTH_PX = 32.0  # a 640x480 camera about 60 cm from the face


class ScenarioBuilder:
    def __init__(self, fps: float = 10.0, seed: int = 0, noise: bool = True, fps_jitter: float = 0.0):
        self.dt = 1.0 / fps
        self.t = 0.0
        self.frames: List[FrameFeatures] = []
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.fps_jitter = fps_jitter  # +/- share of the frame interval (Wi-Fi cameras are not perfectly regular)

    def _jitter(self, value: float, sd: float) -> float:
        return float(value + (self.rng.normal(0, sd) if self.noise else 0.0))

    def _emit(self, seconds: float, face: bool = True, phone_conf: float = 0.0, **overrides) -> "ScenarioBuilder":
        n = max(1, int(round(seconds / self.dt)))
        for _ in range(n):
            if face:
                v = {**NEUTRAL, **overrides}
                ear = max(0.0, self._jitter(v["ear"], 0.008))
                f = FrameFeatures(
                    t=round(self.t, 4),
                    face_present=True,
                    ear=ear,
                    ear_left=ear,
                    ear_right=ear,
                    eye_width_px=EYE_WIDTH_PX,
                    look_down=v["look_down"],
                    mar=max(0.0, self._jitter(v["mar"], 0.01)),
                    yaw=self._jitter(v["yaw"], 1.0),
                    pitch=self._jitter(v["pitch"], 1.0),
                    roll=v["roll"],
                    blink=min(1.0, max(0.0, self._jitter(v["blink"], 0.02))),
                    jaw_open=min(1.0, max(0.0, self._jitter(v["jaw_open"], 0.01))),
                    phone_conf=phone_conf,
                    phone_boxes=int(phone_conf > 0),
                )
            else:
                f = FrameFeatures(t=round(self.t, 4), face_present=False, phone_conf=phone_conf, phone_boxes=int(phone_conf > 0))
            self.frames.append(f)
            self.t += self.dt * (1.0 + (self.rng.uniform(-self.fps_jitter, self.fps_jitter) if self.fps_jitter else 0.0))
        return self

    # Behaviours ---------------------------------------------------------
    def alert(self, seconds: float):
        return self._emit(seconds)

    def blink(self, seconds: float = 0.2):
        self._emit(seconds, ear=0.08, blink=0.85)
        return self._emit(0.1)

    def eyes_closed(self, seconds: float):
        return self._emit(seconds, ear=0.07, blink=0.9)

    def yawn(self, seconds: float = 3.0):
        return self._emit(seconds, mar=0.85, jaw_open=0.75, ear=0.22, blink=0.45)

    def talk(self, seconds: float):
        # speech: the mouth opens and closes quickly and never stays wide open
        for _ in range(int(seconds / 0.4)):
            self._emit(0.2, mar=0.3, jaw_open=0.25)
            self._emit(0.2, mar=0.08, jaw_open=0.06)
        return self

    def look_away(self, seconds: float, yaw: float = 50.0):
        return self._emit(seconds, yaw=NEUTRAL["yaw"] + yaw)

    def look_down(self, seconds: float, pitch: float = -30.0):
        return self._emit(seconds, pitch=NEUTRAL["pitch"] + pitch, ear=0.2, look_down=0.6)

    def glance_down(self, seconds: float = 0.8):
        """Eyes (not head) look down at the dashboard: lids lower, EAR drops, head pitch barely moves."""
        return self._emit(seconds, pitch=NEUTRAL["pitch"] - 6.0, ear=0.2, blink=0.3, look_down=0.7)

    def phone(self, seconds: float, conf: float = 0.7):
        return self._emit(seconds, phone_conf=conf)

    def face_lost(self, seconds: float):
        """The face disappears while the driver faces forward (hand over the camera, glare, dropout)."""
        return self._emit(seconds, face=False)

    def turn_away_and_lose_face(self, seconds: float, yaw: float = 45.0):
        """The head turns beyond the landmark model's range: a short turn, then no face."""
        self._emit(0.3, yaw=NEUTRAL["yaw"] + yaw)
        return self._emit(max(0.0, seconds - 0.3), face=False)

    def camera_stall(self, seconds: float):
        """No frames at all (stream stall): time jumps forward."""
        self.t += seconds
        return self

    def build(self) -> List[FrameFeatures]:
        return list(self.frames)


def run_scenario(
    frames: List[FrameFeatures],
    cfg: Optional[Config] = None,
    speed_kmh: Optional[float] = 40.0,
    motion_g: Optional[float] = None,
    motion_at: Optional[tuple] = None,
) -> List[EngineOutput]:
    """Run frames through a fresh DecisionEngine.

    ``motion_at=(start_s, end_s, g)`` injects an erratic-motion burst.
    """
    engine = DecisionEngine(cfg or Config())
    ctx = StaticContext(speed_kmh=speed_kmh, motion_g=motion_g)
    out = []
    for f in frames:
        reading = ctx.read(f.t)
        if motion_at and motion_at[0] <= f.t < motion_at[1]:
            reading = replace(reading, motion_g=motion_at[2])
        out.append(engine.step(f, reading))
    return out


def max_tier(outputs: List[EngineOutput], start_s: float = 0.0, frames: Optional[List[FrameFeatures]] = None) -> str:
    order = ("NONE", "LOW", "MEDIUM", "HIGH")
    tiers = [o.active_tier for i, o in enumerate(outputs) if frames is None or frames[i].t >= start_s]
    return max(tiers, key=order.index) if tiers else "NONE"


# ---------------------------------------------------------------------------
# Synthetic 20-minute drives for the logic benchmark (scripts/benchmark_logic.py)
# ---------------------------------------------------------------------------
def synthetic_drive(seed: int, minutes: float = 20.0, fps: float = 10.0):
    """A labelled synthetic drive: realistic nuisance behaviour + labelled events.

    Returns (frames, label_rows) where label_rows are (behavior, start_s, end_s).
    Nuisance (never labelled): blinks every 2-5 s (0.1-0.3 s), mirror glances
    (0.5-1.4 s), dashboard glances, talking, short face dropouts, a camera
    stall. Labelled: single yawns, gaze deviations (2.5-6 s), phone use
    (5-15 s, YOLO misses the phone in bursts of 0.2-0.9 s), a 2-min "slightly drowsy" phase
    (3 yawns, slow blinks) and a 2-min "moderately drowsy" phase (long
    closures every 5-9 s and one microsleep).

    The numbers come from typical values in the drowsiness / distraction
    literature, NOT from the detector's thresholds - but they are still a
    model of a driver: use this to test the LOGIC (alert lifecycle, false
    alarms from nuisance behaviour), not as an accuracy estimate.
    """
    rng = np.random.default_rng(seed)
    b = ScenarioBuilder(fps=fps, seed=seed, fps_jitter=0.1)
    labels: List[tuple] = []

    def label(name, start):
        labels.append((name, round(start, 2), round(b.t, 2)))

    def filler(seconds: float, drowsy: bool = False):
        end = b.t + seconds
        while b.t < end:
            r = rng.random()
            gap = rng.uniform(2.0, 5.0)
            b.alert(min(gap, max(0.1, end - b.t)))
            if b.t >= end:
                break
            if r < 0.70:
                b.blink(rng.uniform(0.1, 0.3) if not drowsy else rng.uniform(0.2, 0.45))
            elif r < 0.82 and not drowsy:
                b.look_away(rng.uniform(0.5, 1.4), yaw=rng.choice([-1, 1]) * rng.uniform(25, 40))
            elif r < 0.90 and not drowsy:
                b.glance_down(rng.uniform(0.4, 1.0))
            elif r < 0.97:
                b.talk(rng.uniform(2.0, 6.0))
            else:
                b.face_lost(rng.uniform(0.1, 0.4))

    def yawn():
        start = b.t
        b.yawn(rng.uniform(3.0, 6.0))
        label("yawning", start)

    def gaze():
        start = b.t
        if rng.random() < 0.6:
            b.look_away(rng.uniform(2.5, 6.0), yaw=rng.choice([-1, 1]) * rng.uniform(45, 60))
        else:
            b.look_down(rng.uniform(2.5, 5.0))
        label("gaze_deviation", start)

    def phone():
        # YOLO misses the phone in BURSTS (hand covers it, motion blur): in S01 the
        # phone was found in only 55-80 % of the frames of real use
        start = b.t
        end = b.t + rng.uniform(5.0, 15.0)
        while b.t < end:
            if rng.random() < 0.12:
                miss_end = b.t + rng.uniform(0.2, 0.9)
                while b.t < min(miss_end, end):
                    b.phone(0.1, conf=float(rng.uniform(0.2, 0.34)) if rng.random() < 0.5 else 0.0)
            else:
                b.phone(0.1, conf=float(rng.uniform(0.45, 0.85)))
        label("phone_usage", start)

    total = minutes * 60.0
    b.alert(12.0)  # calibration
    filler(60)
    yawn()
    filler(50)
    gaze()
    filler(40)
    phone()
    filler(45)

    start = b.t  # ---- slightly drowsy: frequent yawning, slower blinks
    for _ in range(3):
        filler(rng.uniform(15, 25), drowsy=True)
        yawn()
        if rng.random() < 0.6:
            b.alert(rng.uniform(3, 6))
            c0 = b.t
            b.eyes_closed(rng.uniform(0.5, 0.7))
            labels.append(("eye_closure", round(c0, 2), round(b.t, 2)))
    filler(20, drowsy=True)
    label("slightly_drowsy", start)

    filler(60)
    gaze()
    filler(40)
    phone()
    filler(40)

    start = b.t  # ---- moderately drowsy: long closures, one microsleep, fewer glances
    sleep_at = rng.integers(4, 8)
    for k in range(12):
        b.alert(rng.uniform(5.0, 9.0))
        dur = rng.uniform(1.6, 2.4) if k == sleep_at else rng.uniform(0.6, 1.2)
        c0 = b.t
        b.eyes_closed(dur)
        labels.append(("eye_closure", round(c0, 2), round(b.t, 2)))  # every closure >= 0.5 s
    b.alert(5.0)
    label("moderately_drowsy", start)

    filler(90)
    b.camera_stall(rng.uniform(2.0, 4.0))
    filler(40)
    yawn()
    filler(60)
    gaze()
    filler(50)
    phone()
    filler(max(10.0, total - b.t))
    return b.build(), labels
