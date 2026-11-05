"""False alarms (SO5): four different things that must not be mixed up.

1. False-positive DETECTION  - a predicted behaviour episode that matches no
   labelled episode of the same behaviour (see metrics.event_metrics).
2. Unnecessary RISK ESCALATION - the assessed risk tier rises (e.g. NONE ->
   LOW) while no behaviour of any kind is labelled.
3. Unnecessary ALARM - an audible alert episode starts while no behaviour
   (and no sudden-motion event) is labelled. This is the SO5 metric:
   target <= 0.5 per NON-EVENT driving hour (thesis Section 2, after Deans et
   al. 2025), i.e. per hour of driving outside all labelled events.
4. Unnecessary PUSH - a push notification sent under the same condition.

"While labelled" allows for the system's built-in confirmation delays: an
alarm is necessary if it starts between ``tol_before`` seconds before a
labelled segment and ``tol_after`` seconds after it ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ..types import RISK_TIERS
from .labels import Segment, merge_segments


@dataclass
class AlarmEvent:
    t: float
    tier: str
    kind: str  # start | escalate | push
    reason: str = ""


def _rank(tier: str) -> int:
    return RISK_TIERS.index(tier) if tier in RISK_TIERS else 0


def normalise_alert_log(records: Sequence[dict]) -> Dict[str, List[AlarmEvent]]:
    """Alarm starts, escalations and pushes from either alert-log format.

    Current logs (alert state machine) have an ``event`` field
    ("start", "escalate", "push", "stop", ...). Older logs only have the
    command stream with ``new_episode`` and ``push`` flags.
    """
    out: Dict[str, List[AlarmEvent]] = {"start": [], "escalate": [], "push": []}
    legacy_tier = "NONE"
    for r in records:
        t, tier, reason = float(r.get("t", 0.0)), str(r.get("tier", "NONE")), str(r.get("reason", ""))
        if "event" in r:
            kind = r["event"]
            if kind in out:
                out[kind].append(AlarmEvent(t, tier, kind, reason))
            continue
        if r.get("new_episode"):
            out["start"].append(AlarmEvent(t, tier, "start", reason))
            legacy_tier = tier
        elif _rank(tier) > _rank(legacy_tier):
            out["escalate"].append(AlarmEvent(t, tier, "escalate", reason))
            legacy_tier = tier
        if r.get("push"):
            out["push"].append(AlarmEvent(t, tier, "push", reason))
    return out


def risk_escalations(times: Sequence[float], tiers: Sequence[str]) -> List[AlarmEvent]:
    """Every frame where the assessed risk tier rises."""
    out, prev = [], "NONE"
    for t, tier in zip(times, tiers):
        if _rank(tier) > _rank(prev):
            out.append(AlarmEvent(float(t), tier, "escalate"))
        prev = tier
    return out


def necessary_windows(events: Sequence[Segment], tol_before: float, tol_after: float) -> List[Segment]:
    return merge_segments([(max(0.0, s - tol_before), e + tol_after) for s, e in events])


def is_necessary(t: float, windows: Sequence[Segment]) -> bool:
    return any(s <= t <= e for s, e in windows)


def non_event_seconds(t0: float, t1: float, windows: Sequence[Segment]) -> float:
    covered = sum(max(0.0, min(e, t1) - max(s, t0)) for s, e in windows)
    return max(0.0, (t1 - t0) - covered)


def alarm_report(alerts: Sequence[dict], times: Sequence[float], tiers: Sequence[str], events: Sequence[Segment],
                 tol_before: float = 1.0, tol_after: float = 2.0) -> dict:
    """Unnecessary escalations / alarms / pushes, per total hour and per non-event hour."""
    log = normalise_alert_log(alerts)
    windows = necessary_windows(events, tol_before, tol_after)
    t0, t1 = (float(times[0]), float(times[-1])) if len(times) else (0.0, 0.0)
    total_h = (t1 - t0) / 3600.0
    quiet_h = non_event_seconds(t0, t1, windows) / 3600.0

    def summarise(items: List[AlarmEvent]) -> dict:
        bad = [a for a in items if not is_necessary(a.t, windows)]
        by_tier = {tier: sum(1 for a in bad if a.tier == tier) for tier in RISK_TIERS[1:]}
        return {
            "count": len(items), "unnecessary": len(bad), "unnecessary_by_tier": by_tier,
            "per_hour": round(len(bad) / total_h, 3) if total_h > 0 else None,
            "per_non_event_hour": round(len(bad) / quiet_h, 3) if quiet_h > 0 else None,
            "unnecessary_times_s": [round(a.t, 2) for a in bad][:50],
        }

    return {
        "definitions": "unnecessary = starts outside every labelled event "
                       f"(window {tol_before:g} s before to {tol_after:g} s after)",
        "driving_hours": round(total_h, 4),
        "non_event_hours": round(quiet_h, 4),
        "risk_escalations": summarise(risk_escalations(times, tiers)),
        "alarms": summarise(log["start"]),
        "alarm_escalations": summarise(log["escalate"]),
        "pushes": summarise(log["push"]),
        "target_per_non_event_hour": 0.5,
    }


def pooled_alarm_rates(reports: Sequence[dict]) -> Dict[str, dict]:
    """Sum unnecessary counts and hours over sessions, then divide (not a mean of rates)."""
    total_h = sum(r["driving_hours"] for r in reports)
    quiet_h = sum(r["non_event_hours"] for r in reports)
    out = {"driving_hours": round(total_h, 3), "non_event_hours": round(quiet_h, 3)}
    for key in ("risk_escalations", "alarms", "alarm_escalations", "pushes"):
        count = sum(r[key]["count"] for r in reports)
        bad = sum(r[key]["unnecessary"] for r in reports)
        out[key] = {"count": count, "unnecessary": bad,
                    "per_hour": round(bad / total_h, 3) if total_h > 0 else None,
                    "per_non_event_hour": round(bad / quiet_h, 3) if quiet_h > 0 else None}
    return out


def tier_sequence(frames: Sequence[dict]) -> Tuple[List[float], List[str]]:
    times = [float(f["t"]) for f in frames]
    tiers = [str((f.get("risk") or {}).get("tier", "NONE")) for f in frames]
    return times, tiers


SOUND_EVENTS = {"start", "escalate", "deescalate", "beep"}


def sound_times(records: Sequence[dict]) -> List[float]:
    """Times at which the buzzer made a sound (older logs: every command was a sound)."""
    out = []
    for r in records:
        if "event" in r:
            if r["event"] in SOUND_EVENTS:
                out.append(float(r["t"]))
        else:
            out.append(float(r["t"]))
    return out


def sound_report(alerts: Sequence[dict], times: Sequence[float], events: Sequence[Segment],
                 tol_before: float = 1.0, tol_after: float = 2.0) -> dict:
    """How much the buzzer sounds when nothing is happening, and how long after an event it goes on.

    ``tail`` = time from the end of a labelled event to the last sound before the
    next event - the "alerts continue after the driver is normal" problem.
    """
    sounds = sound_times(alerts)
    windows = necessary_windows(events, tol_before, tol_after)
    t0, t1 = (float(times[0]), float(times[-1])) if len(times) else (0.0, 0.0)
    quiet_h = non_event_seconds(t0, t1, windows) / 3600.0
    bad = [t for t in sounds if not is_necessary(t, windows)]
    merged = merge_segments(list(events))
    tails = []
    for i, (_, end) in enumerate(merged):
        nxt = merged[i + 1][0] if i + 1 < len(merged) else t1
        after = [t for t in sounds if end < t < nxt]
        tails.append(max(after) - end if after else 0.0)
    return {
        "sounds": len(sounds),
        "sounds_outside_events": len(bad),
        "sounds_outside_events_per_non_event_hour": round(len(bad) / quiet_h, 1) if quiet_h > 0 else None,
        "tail_mean_s": round(sum(tails) / len(tails), 2) if tails else None,
        "tail_max_s": round(max(tails), 2) if tails else None,
        "events_with_tail_over_5s": sum(1 for x in tails if x > 5.0),
        "events": len(merged),
    }


def alarm_validation(alerts: Sequence[dict], expected: Sequence[tuple], tol_after: float = 3.0) -> dict:
    """Thesis Table 1.3 "alarm validation": enacted behaviour at a known risk level vs the tier triggered.

    ``expected``: (start, end, behaviour, expected tier). Triggered tier = the
    highest tier of any sounding alert command between start and end + tol_after.
    """
    rows, matrix = [], [[0] * 4 for _ in range(4)]
    sounds = []
    for r in alerts:
        if r.get("event", "beep") in SOUND_EVENTS or "event" not in r:
            sounds.append((float(r["t"]), str(r.get("tier", "NONE"))))
    for start, end, behaviour, tier in expected:
        hit = [tr for t, tr in sounds if start <= t <= end + tol_after]
        got = max(hit, key=_rank) if hit else "NONE"
        matrix[_rank(tier)][_rank(got)] += 1
        rows.append({"start_s": start, "end_s": end, "behavior": behaviour, "expected": tier, "triggered": got})
    n = len(rows)
    exact = sum(1 for r in rows if r["expected"] == r["triggered"])
    return {"segments": n, "exact": exact, "accuracy": round(exact / n, 4) if n else None,
            "under_alerted": sum(1 for r in rows if _rank(r["triggered"]) < _rank(r["expected"])),
            "over_alerted": sum(1 for r in rows if _rank(r["triggered"]) > _rank(r["expected"])),
            "confusion": matrix, "tiers": list(RISK_TIERS), "rows": rows}
