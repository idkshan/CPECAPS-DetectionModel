"""Where alert commands go: console, the ESP32 buzzer (HTTP), later the Android app.

Device protocol (HttpAlertSink): one GET per command
    <url>?event=start&tier=HIGH&mode=alarm&push=1&id=ab12cd34-000017&alert=ab12cd34-A00016
  event  start | escalate | deescalate | beep | keepalive | stop | unavailable | available | push
  mode   soft_beep | repeating_buzzer | alarm | off   (what the buzzer should do now)
Two firmware styles are supported:
  * "beeps" (default, like version 1/2): the firmware sounds once per request;
    every start / escalate / beep request makes one sound, stop/off silences.
  * "state": the firmware runs the pattern itself after start/escalate/deescalate
    and needs the keepalive requests; it must switch itself off when it gets
    "stop" OR when no request arrives for ~3 s (fail-safe if the phone dies).
The ESP32 firmware is outside this repository; adjust the query format to it.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
import urllib.request

from ..types import AlertCommand

DEVICE_EVENTS = {
    "beeps": {"start", "escalate", "deescalate", "beep", "stop", "unavailable"},
    "state": {"start", "escalate", "deescalate", "keepalive", "stop", "unavailable"},
}


class ConsoleAlertSink:
    """Prints the alert lifecycle (start / escalate / step down / stop / push) - not every keepalive."""

    def __init__(self, show_beeps: bool = False):
        self.show_beeps = show_beeps

    def __call__(self, cmd: AlertCommand) -> None:
        if cmd.event == "keepalive" or (cmd.event == "beep" and not self.show_beeps):
            return
        bell = "\a" if cmd.event in ("start", "escalate") and cmd.tier in ("MEDIUM", "HIGH") else ""
        push = " + PUSH" if cmd.push else ""
        print(f"{bell}[{cmd.t:8.2f}s] {cmd.event.upper():<11s} {cmd.tier:<6} {cmd.action}{push} :: {cmd.reason}", flush=True)


class HttpAlertSink:
    """Forward alert commands to the ESP32 buzzer with non-blocking GET requests."""

    def __init__(self, url: str, timeout_s: float = 0.5, style: str = "beeps"):
        if style not in DEVICE_EVENTS:
            raise ValueError(f"style must be one of {sorted(DEVICE_EVENTS)}")
        self.url, self.timeout_s, self.style = url, timeout_s, style

    def __call__(self, cmd: AlertCommand) -> None:
        if cmd.event not in DEVICE_EVENTS[self.style]:
            return
        query = urllib.parse.urlencode({"event": cmd.event, "tier": cmd.tier, "mode": cmd.action, "action": cmd.action,
                                        "push": int(cmd.push), "id": cmd.id, "alert": cmd.alert_id})
        target = f"{self.url}?{query}"

        def send():
            try:
                urllib.request.urlopen(target, timeout=self.timeout_s).read()
            except Exception as exc:  # network errors must never stop monitoring
                print(f"[alert-sink] could not reach {self.url}: {exc}")

        threading.Thread(target=send, daemon=True).start()


def alert_to_json(cmd: AlertCommand) -> str:
    return json.dumps(cmd.to_dict())
