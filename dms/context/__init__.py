"""Section 9 - driving context: GPS speed and position, motion, driving duration.

    providers.py  where readings come from (none / fixed values / ESP32 or phone JSON)
    gps.py        GPS fix validity, staleness, 1-Hz update rate, outages (SO4)
    duration.py   continuous driving time with rest-break reset (SO4, fatigue input)
"""

from .duration import DrivingDurationTracker
from .gps import GpsFix, GpsTracker
from .providers import ContextProvider, HttpContext, StaticContext

__all__ = ["ContextProvider", "DrivingDurationTracker", "GpsFix", "GpsTracker", "HttpContext", "StaticContext"]
