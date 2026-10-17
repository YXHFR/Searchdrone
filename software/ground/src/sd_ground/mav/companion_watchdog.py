"""``CompanionWatchdog``: the ground station's guard for a dead or hung onboard computer.

It follows the HEARTBEAT from ``(fc_sysid, 191)`` on the ground's own MAVLink channel.
Its ``tick`` runs in the MAVLink thread (never the main loop, never the database) and
says what to do; the thread sends it the same way as STOP and FLY HOME.

* Step 1: no onboard heartbeat for ``COMPANION_WATCHDOG_S`` (3 s), the watchdog idle and
  the flight controller in GUIDED: STOP's steps, latch ``watchdog_stop``, phase
  ``stopped``.
* Step 2: no onboard heartbeat for ``COMPANION_WATCHDOG_RTL_S`` (20 s) and either the
  vehicle is airborne in GUIDED, AUTO or LOITER with the watchdog idle, or the watchdog
  stopped it itself and it still stands in BRAKE or LOITER: FLY HOME's steps, latch
  ``watchdog_rtl``, phase ``rtl``.
* It never acts while the operator's STOP or FLY HOME stands (then only the alarm
  ``ONBOARD_SILENT``), while the battery latch stands, on the ground, or in a mode it
  did not set itself (BRAKE, LOITER, RTL, SMART_RTL, LAND, ALT_HOLD, STABILIZE, ...).
* The phase goes back to idle when the heartbeat returns or the vehicle has landed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from sd_mav.modes import CopterMode
from sd_safety.constants import COMPANION_WATCHDOG_RTL_S, COMPANION_WATCHDOG_S

__all__ = ["CompanionWatchdog", "WatchInput", "WatchdogAction", "WatchdogPhase"]

WatchdogPhase = Literal["idle", "stopped", "rtl"]
WatchdogAction = Literal["watchdog_stop", "watchdog_rtl"]
_STOP_MS: Final = int(COMPANION_WATCHDOG_S * 1000)
_RTL_MS: Final = int(COMPANION_WATCHDOG_RTL_S * 1000)
_STEP2_IDLE_MODES: Final = frozenset({CopterMode.GUIDED, CopterMode.AUTO, CopterMode.LOITER})
_OWN_STOP_MODES: Final = frozenset({CopterMode.BRAKE, CopterMode.LOITER})


@dataclass(frozen=True, slots=True)
class WatchInput:
    now_ms: int
    onboard_seen_ms: int | None  # last HEARTBEAT from (fc_sysid, 191)
    fc_mode: int | None  # from a fresh HEARTBEAT
    in_flight: bool
    operator_override: bool  # the operator's STOP or FLY HOME stands
    battery_latch: bool


class CompanionWatchdog:
    def __init__(self, start_ms: int) -> None:
        self.phase: WatchdogPhase = "idle"
        self._start_ms = start_ms  # counts as the last heartbeat until the first one
        self.alarm_active = False  # ONBOARD_SILENT is shown

    def silent_ms(self, inp: WatchInput) -> int:
        last = inp.onboard_seen_ms if inp.onboard_seen_ms is not None else self._start_ms
        return inp.now_ms - last

    def tick(self, inp: WatchInput) -> WatchdogAction | None:
        silent = self.silent_ms(inp)
        if silent < _STOP_MS or not inp.in_flight:
            self.phase = "idle"
            self.alarm_active = False
            return None
        self.alarm_active = True
        if inp.operator_override or inp.battery_latch or inp.fc_mode is None:
            return None
        mode = inp.fc_mode
        if self.phase == "idle" and mode == CopterMode.GUIDED and silent < _RTL_MS:
            self.phase = "stopped"
            return "watchdog_stop"
        step2 = (self.phase == "idle" and mode in _STEP2_IDLE_MODES) or (
            self.phase == "stopped" and mode in _OWN_STOP_MODES
        )
        if silent >= _RTL_MS and step2:
            self.phase = "rtl"
            return "watchdog_rtl"
        return None
