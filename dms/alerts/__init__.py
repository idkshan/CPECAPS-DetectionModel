"""Section 6 - alert and notification system.

    state_machine.py  when an alert starts, escalates, steps down and STOPS (SO5)
    sinks.py          where the commands go (console, ESP32 buzzer over HTTP)
"""

from .sinks import ConsoleAlertSink, HttpAlertSink, alert_to_json
from .state_machine import ACTIONS, ALERTING, CLEARED, MONITORING, UNAVAILABLE, AlertManager, AlertStateMachine

__all__ = ["ACTIONS", "ALERTING", "CLEARED", "MONITORING", "UNAVAILABLE", "AlertManager", "AlertStateMachine",
           "ConsoleAlertSink", "HttpAlertSink", "alert_to_json"]
