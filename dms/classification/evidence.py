"""Stage 3: convert features into evidence scores and fuse them.

Every signal (EAR, eyeBlink blendshape, eye-state CNN, MAR, jawOpen, head
pose) is mapped to an *evidence score* in [0, 1] with the same convention:

    0.0  = the driver's own neutral baseline
    0.5  = exactly at the per-driver threshold
    1.0  = clearly past the threshold (twice as far from baseline)

Because all sources share this scale, they can be averaged (weighted) and the
state is ON when the fused evidence reaches the decision level (0.5).
"""

from __future__ import annotations

from typing import Dict, Optional


def ramp_evidence(x: Optional[float], zero_at: float, half_at: float) -> Optional[float]:
    """Linear evidence: 0 at ``zero_at`` (baseline), 0.5 at ``half_at`` (threshold).

    Works in either direction (e.g. EAR decreases toward its threshold, MAR
    increases toward it).
    """
    if x is None:
        return None
    span = half_at - zero_at
    if abs(span) < 1e-9:
        return 1.0 if (x - zero_at) * (1 if span >= 0 else -1) > 0 else 0.0
    return float(min(1.0, max(0.0, 0.5 * (x - zero_at) / span)))


def fuse(sources: Dict[str, Optional[float]], weights: Dict[str, float]) -> Optional[float]:
    """Weighted mean of the available (non-None, weight > 0) evidence sources."""
    total, weight_sum = 0.0, 0.0
    for name, value in sources.items():
        w = weights.get(name, 0.0)
        if value is None or w <= 0:
            continue
        total += w * value
        weight_sum += w
    if weight_sum == 0:
        return None
    return total / weight_sum
