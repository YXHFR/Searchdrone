"""Signals that a mode change came from one of the flight controller's own failsafes.

The onboard executor uses them to tell a failsafe apart from a person taking over, and
the ground station's emergency path lets a failsafe's mode stand. Three sources:

* a ``STATUSTEXT`` whose text matches one of :data:`PATTERNS` (case-insensitive),
* ``BATTERY_STATUS.charge_state`` LOW, CRITICAL or EMERGENCY (the battery failsafe),
* ``HEARTBEAT.system_status`` CRITICAL.

The exact ArduCopter texts are checked against SITL by the failsafe scenario tests; the
list is adjusted there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal

from pymavlink.dialects.v20 import ardupilotmega as mavlink

__all__ = ["BATTERY_FAILSAFE_STATES", "PATTERNS", "FailsafeSignal", "signal_from", "text_signal"]

PATTERNS: Final = MappingProxyType(
    {
        name: re.compile(pattern, re.IGNORECASE)
        for name, pattern in {
            "failsafe": r"failsafe",
            "fence_breach": r"fence breach",
            "battery": r"battery.*(low|critical)",
            "gcs": r"gcs.*(lost|failsafe)",
            "ekf": r"ekf.*(failsafe|variance)",
            "crash": r"crash",
            "srtl_unavailable": r"smartrtl.*(unavailable|deactivated|full)",
        }.items()
    }
)

BATTERY_FAILSAFE_STATES: Final = frozenset(
    {
        mavlink.MAV_BATTERY_CHARGE_STATE_LOW,
        mavlink.MAV_BATTERY_CHARGE_STATE_CRITICAL,
        mavlink.MAV_BATTERY_CHARGE_STATE_EMERGENCY,
    }
)

SignalSource = Literal["statustext", "battery", "heartbeat"]


@dataclass(frozen=True, slots=True)
class FailsafeSignal:
    source: SignalSource
    name: str  # pattern name, "charge_state_<n>" or "system_status_critical"
    text: str | None = None


def text_signal(text: str) -> str | None:
    """The name of the first pattern ``text`` matches, or ``None``."""
    for name, pattern in PATTERNS.items():
        if pattern.search(text):
            return name
    return None


def _text_of(msg: Any) -> str:
    text = msg.text
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return str(text).rstrip("\x00")


def signal_from(msg: Any) -> FailsafeSignal | None:
    """The failsafe signal in a message from the flight controller, if it carries one."""
    kind = msg.get_type()
    if kind == "STATUSTEXT":
        text = _text_of(msg)
        name = text_signal(text)
        return FailsafeSignal("statustext", name, text) if name else None
    if kind == "BATTERY_STATUS":
        state = int(getattr(msg, "charge_state", 0))
        if state in BATTERY_FAILSAFE_STATES:
            return FailsafeSignal("battery", f"charge_state_{state}")
        return None
    if kind == "HEARTBEAT" and int(msg.system_status) == mavlink.MAV_STATE_CRITICAL:
        return FailsafeSignal("heartbeat", "system_status_critical")
    return None
