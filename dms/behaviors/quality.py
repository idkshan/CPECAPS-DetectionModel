"""Frame quality: can this frame be judged at all?

Purpose   Keep camera problems from looking like driver behaviour.
Inputs    frame time t, FrameFeatures.eye_width_px
Outputs   gap (bool): this frame follows a camera stall
          eyes_measurable (bool): the eyes are large enough in the image
Params    quality.max_frame_gap_s   (1.0 s)  longer pauses end every episode
          quality.min_eye_width_px  (12 px)  smaller eyes -> eye state unknown

Why: at 10 FPS a frame arrives every 0.1 s. If the ESP32 stream stalls for
5 s while the eyes were half-closed, the next frame must not be read as
"eyes closed for 5 s" (a fake microsleep). And an eye only ~10 px wide moves
by 1-2 px when it closes - below the landmark noise - so EAR is meaningless.
"""

from __future__ import annotations

from typing import Optional

from ..config import QualityConfig
from ..types import FrameFeatures


class FrameQuality:
    def __init__(self, cfg: QualityConfig):
        self.cfg = cfg
        self._last_t: Optional[float] = None
        self.gaps = 0

    def check_gap(self, t: float) -> bool:
        gap = self._last_t is not None and (t - self._last_t) > self.cfg.max_frame_gap_s
        self._last_t = t
        self.gaps += int(gap)
        return gap

    def eyes_measurable(self, f: FrameFeatures) -> bool:
        # Older logs have no eye_width_px: assume measurable (previous behaviour).
        return f.eye_width_px is None or f.eye_width_px >= self.cfg.min_eye_width_px
