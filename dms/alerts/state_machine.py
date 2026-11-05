"""Section 6 - alert state machine (SO5: three-tier adaptive alarm).

Four things that used to be mixed up are kept apart:

    detected behaviour   what the driver is doing NOW (eyes closed, yawning, looking
                         away, phone, sudden motion)          - BehaviorStatus.active
    assessed risk level  NONE / LOW / MEDIUM / HIGH from the fuzzy model; it also
                         remembers recent history (PERCLOS, yawns per 5 min, off-road
                         time), so it stays elevated for a while - RiskAssessment.tier
    active alert         the buzzer is sounding in one of three patterns
    cleared alert        the alert has stopped although the risk may still be elevated

States
    MONITORING   no alert, risk NONE
    ALERTING     an alert is sounding (tier LOW / MEDIUM / HIGH)
    CLEARED      the alert has stopped, the risk level is still elevated (silent;
                 the app shows the risk level)
    UNAVAILABLE  no camera frames: monitoring is not possible (thesis 1.7.11);
                 this is reported separately and is never treated as drowsiness

Policy (every number is in AlertConfig / configs/default.yaml)
  START     risk tier >= LOW, held for ``activate_s`` (0.3 s: filters one-frame
            spikes), AND one of: a behaviour is happening / a behaviour event just
            fired / the risk rose above the highest tier already alerted since the
            risk was last NONE / (optional) reminder interval elapsed.
  PATTERN   LOW = one soft beep (another one for each new event, >= 5 s apart)
            MEDIUM = repeating buzzer every 2 s; HIGH = alarm every 1 s + push
  SUSTAIN   while a behaviour is happening the pattern continues without limit
            (a driver whose eyes stay closed keeps getting the buzzer).
  ESCALATE  immediately when the risk tier rises (push when entering HIGH).
  STEP DOWN when the risk tier stays lower for ``release_s`` (3 s).
  STOP      when NO behaviour has happened for ``clear_s`` (2 s), the alert has
            lasted ``min_alert_s`` and a MEDIUM burst (``medium_max_repeats``
            buzzes) is complete -> explicit STOP command (buzzer off), state
            CLEARED. Old evidence in the risk level cannot keep it sounding:
            that was the bug where alarms continued after the driver was normal.
            Also when the risk tier falls to NONE for ``release_s``.
  PUSH      once per alert that reaches HIGH, at most one per ``push_interval_s``
            (later ones are logged as push_suppressed, never sent twice).
  KEEPALIVE while MEDIUM/HIGH sounds, the current mode is re-sent every
            ``keepalive_s``; a device that hears nothing for ~3 s should switch
            its buzzer off by itself (fail-safe if the phone or the link dies).

Every command has a unique id (de-duplication in the app / Firebase) and the
id of its alert episode; all of them are written to alerts.jsonl.
"""

from __future__ import annotations

import math
import uuid
from typing import Dict, List, Optional

from ..config import AlertConfig
from ..types import RISK_TIERS, AlertCommand

ACTIONS = {"NONE": "off", "LOW": "soft_beep", "MEDIUM": "repeating_buzzer", "HIGH": "alarm"}
MONITORING, ALERTING, CLEARED, UNAVAILABLE = "MONITORING", "ALERTING", "CLEARED", "UNAVAILABLE"


def _rank(tier: str) -> int:
    return RISK_TIERS.index(tier)


class AlertStateMachine:
    def __init__(self, cfg: AlertConfig, id_prefix: Optional[str] = None):
        self.cfg = cfg
        self.state = MONITORING
        self.tier = "NONE"  # tier of the sounding alert
        self.alert_id = ""
        self.episodes = {t: 0 for t in RISK_TIERS[1:]}  # alerts started, by starting tier
        self.transitions: List[dict] = []  # (t, from, to, reason) - also visible in alerts.jsonl
        self._prefix = id_prefix if id_prefix is not None else uuid.uuid4().hex[:8]
        self._seq = 0
        # risk-tier debounce
        self._pending_tier, self._pending_since, self._eligible = "NONE", -math.inf, "NONE"
        # alert bookkeeping
        self._started_t = self._last_beep_t = self._last_keepalive_t = -math.inf
        self._last_push_t = self._last_active_t = self._last_event_t = -math.inf
        self._last_low_beep_t = self._cleared_t = -math.inf
        self._beeps_since_trigger = 0
        self._below_since: Optional[float] = None
        self._last_alerted_tier = "NONE"
        self._closure_since: Optional[float] = None
        self._pushed_this_alert = False

    # ------------------------------------------------------------- helpers
    @property
    def active_tier(self) -> str:
        """Tier of the alert that is sounding now ("NONE" when silent)."""
        return self.tier if self.state == ALERTING else "NONE"

    def _cmd(self, t: float, event: str, tier: str, reason: str, behaviors: List[str], push: bool = False) -> AlertCommand:
        self._seq += 1
        action = ACTIONS["NONE"] if event in ("stop", "unavailable", "available", "push_suppressed") else ACTIONS[tier]
        return AlertCommand(t=round(t, 3), tier=tier, action=action, push=push, reason=reason, new_episode=event == "start",
                            event=event, id=f"{self._prefix}-{self._seq:06d}", alert_id=self.alert_id, behaviors=list(behaviors))

    def _transition(self, t: float, new_state: str, reason: str) -> None:
        if new_state != self.state:
            self.transitions.append({"t": round(t, 3), "from": self.state, "to": new_state, "reason": reason})
            self.state = new_state

    def _push(self, t: float, reason: str, behaviors: List[str]) -> List[AlertCommand]:
        if self._pushed_this_alert:
            return []
        self._pushed_this_alert = True
        if t - self._last_push_t < self.cfg.push_interval_s:
            return [self._cmd(t, "push_suppressed", "HIGH", f"push cooldown ({self.cfg.push_interval_s:.0f} s): {reason}", behaviors)]
        self._last_push_t = t
        return [self._cmd(t, "push", "HIGH", reason, behaviors, push=True)]

    def _start(self, t: float, tier: str, reason: str, behaviors: List[str]) -> List[AlertCommand]:
        self.alert_id = f"{self._prefix}-A{self._seq + 1:05d}"
        self.tier = tier
        self.episodes[tier] += 1
        self._started_t = self._last_beep_t = self._last_keepalive_t = t
        self._beeps_since_trigger = 1
        self._below_since = None
        self._pushed_this_alert = False
        if tier == "LOW":
            self._last_low_beep_t = t
        self._last_alerted_tier = max(self._last_alerted_tier, tier, key=_rank)
        self._transition(t, ALERTING, reason)
        cmds = [self._cmd(t, "start", tier, reason, behaviors)]
        if tier == "HIGH":
            cmds += self._push(t, reason, behaviors)
        return cmds

    def _stop(self, t: float, reason: str, behaviors: List[str], risk_tier: str) -> List[AlertCommand]:
        cmd = self._cmd(t, "stop", self.tier, reason, behaviors)
        self.tier = "NONE"
        self._cleared_t = t
        self._transition(t, CLEARED if _rank(risk_tier) > 0 else MONITORING, reason)
        return [cmd]

    # ------------------------------------------------------------- main step
    def update(self, t: float, risk_tier: str, active: List[str], events: List[str], reason: str) -> List[AlertCommand]:
        cfg, cmds = self.cfg, []
        if self.state == UNAVAILABLE:
            self._transition(t, MONITORING, "camera frames are back")
            cmds.append(self._cmd(t, "available", "NONE", "camera frames are back", active))

        # risk tier debounce: a higher tier must hold activate_s; a lower one counts at once
        if risk_tier != self._pending_tier:
            self._pending_tier, self._pending_since = risk_tier, t
        if _rank(risk_tier) < _rank(self._eligible) or t - self._pending_since >= cfg.activate_s:
            self._eligible = risk_tier
        tier = self._eligible

        if active:
            self._last_active_t = t
        if events:
            self._last_event_t = t
        recent_event = t - self._last_event_t <= cfg.activate_s + 1e-6
        self._closure_since = (self._closure_since if self._closure_since is not None else t) if "eye_closure" in active else None
        if _rank(tier) == 0:
            self._last_alerted_tier = "NONE"

        if self.state in (MONITORING, CLEARED):
            if _rank(tier) == 0:
                self._transition(t, MONITORING, "risk back to NONE")
                return cmds
            start_reason = None
            if active or recent_event:
                start_reason = reason or f"behaviour at risk {tier}"
            elif _rank(tier) > _rank(self._last_alerted_tier):
                start_reason = f"risk rose to {tier}: {reason}"
            elif cfg.reminder_s > 0 and _rank(tier) >= _rank("MEDIUM") and t - self._cleared_t >= cfg.reminder_s:
                start_reason = f"reminder (risk still {tier}): {reason}"
            if start_reason is not None:
                cmds += self._start(t, tier, start_reason, active)
            else:
                self._transition(t, CLEARED, "risk elevated, driver currently normal")
            return cmds

        # ---- ALERTING
        if _rank(tier) > _rank(self.tier):
            self.tier = tier
            self._beeps_since_trigger = 1
            self._below_since = None
            self._last_beep_t = self._last_keepalive_t = t
            self._last_alerted_tier = max(self._last_alerted_tier, tier, key=_rank)
            cmds.append(self._cmd(t, "escalate", tier, reason, active))
            if tier == "HIGH":
                cmds += self._push(t, reason, active)
        elif _rank(tier) < _rank(self.tier):
            if self._below_since is None:
                self._below_since = t
            elif t - self._below_since >= cfg.release_s:
                if _rank(tier) == 0:
                    return cmds + self._stop(t, "risk back to NONE", active, tier)
                self.tier = tier
                self._below_since = None
                self._beeps_since_trigger = 1
                cmds.append(self._cmd(t, "deescalate", tier, reason, active))
        else:
            self._below_since = None

        # optional safety escalation: eyes closed for a long time during an alert
        if (cfg.escalate_closure_s > 0 and self.tier != "HIGH" and self._closure_since is not None
                and t - self._closure_since >= cfg.escalate_closure_s):
            self.tier = "HIGH"
            self._last_beep_t = self._last_keepalive_t = t
            cmds.append(self._cmd(t, "escalate", "HIGH", f"eyes closed >= {cfg.escalate_closure_s:g} s", active))
            cmds += self._push(t, f"eyes closed >= {cfg.escalate_closure_s:g} s", active)

        if events:
            self._beeps_since_trigger = 0  # a new behaviour event re-arms the burst

        # stop when the driver has been normal for clear_s
        burst = {"LOW": 0, "MEDIUM": cfg.medium_max_repeats, "HIGH": cfg.high_max_repeats}[self.tier]  # LOW = single beep
        burst_done = burst <= 0 or self._beeps_since_trigger >= burst
        if (not active and t - self._last_active_t >= cfg.clear_s and t - self._started_t >= cfg.min_alert_s
                and burst_done and not recent_event):
            return cmds + self._stop(t, f"driver back to normal for {cfg.clear_s:g} s", active, tier)

        # sound pattern
        since_beep = t - self._last_beep_t
        if self.tier == "LOW":
            if events and t - self._last_low_beep_t >= cfg.min_interval_s:
                self._last_low_beep_t = self._last_beep_t = t
                cmds.append(self._cmd(t, "beep", "LOW", reason, active))
        else:
            period = cfg.high_repeat_s if self.tier == "HIGH" else cfg.medium_repeat_s
            allowed = bool(active) or burst <= 0 or self._beeps_since_trigger < burst
            if since_beep >= period and allowed:
                self._last_beep_t = t
                self._beeps_since_trigger += 1
                cmds.append(self._cmd(t, "beep", self.tier, reason, active))
            if t - self._last_keepalive_t >= cfg.keepalive_s:
                self._last_keepalive_t = t
                cmds.append(self._cmd(t, "keepalive", self.tier, reason, active))
        return cmds

    def camera_lost(self, t: float, reason: str = "no camera frames") -> List[AlertCommand]:
        """No frames for AlertConfig.camera_timeout_s: stop sounding, report monitoring unavailable."""
        if self.state == UNAVAILABLE:
            return []
        cmds = []
        if self.state == ALERTING:
            cmds += self._stop(t, f"monitoring unavailable: {reason}", [], "NONE")
        self.tier = "NONE"
        self._transition(t, UNAVAILABLE, reason)
        cmds.append(self._cmd(t, "unavailable", "NONE", reason, []))
        self._pending_tier, self._eligible, self._last_alerted_tier = "NONE", "NONE", "NONE"
        return cmds

    def summary(self) -> Dict[str, object]:
        return {"state": self.state, "episodes": dict(self.episodes), "transitions": len(self.transitions)}


# Name used by version 1/2 code
AlertManager = AlertStateMachine
