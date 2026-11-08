"""``CompanionWatchdog``: the ground station's guard for a dead or hung onboard computer.

It follows the HEARTBEAT from ``(fc_sysid, 191)`` on the ground's own MAVLink channel.
Its ``tick`` runs in the MAVLink thread (never the main loop, never the database) and
says what to do; the thread sends it the same way as STOP and FLY HOME.

The onboard heartbeat goes out only while the executor's loop runs
(``Executor.loop_alive``), so a hung executor falls silent here as a dead one does.

* No onboard heartbeat for ``COMPANION_WATCHDOG_S`` (3 s), the watchdog idle and the
  flight controller airborne in GUIDED or AUTO: STOP's steps, latch ``watchdog_stop``,
  phase ``stopped``. Nobody steers any more: in GUIDED the drone would fly on to a spot
  nobody checks, and in AUTO it would fly the rest of the search through keep-out
  cylinders that came meanwhile. In LOITER it already hovers, so only the alarm.
* It never flies home (owner decision, option A): the drone hovers until the operator's
  STOP or FLY HOME, the pilot's RC or the battery failsafe brings it down or home, or
  the radio link to the ground is lost as well and the flight controller's own
  ground-station failsafe does.
* It never acts while the operator's STOP or FLY HOME stands or the pilot steers by hand
  (then only the alarm ``ONBOARD_SILENT``), while the battery latch stands, or on the
  ground. Besides GUIDED and AUTO it never acts in a mode someone else chose (LOITER,
  RTL, SMART_RTL, LAND, ALT_HOLD, STABILIZE, ...).
* The phase goes back to idle when the heartbeat returns or the vehicle has landed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from sd_mav.modes import CopterMode
from sd_safety.constants import COMPANION_WATCHDOG_S

__all__ = ["CompanionWatchdog", "WatchInput", "WatchdogAction", "WatchdogPhase"]

WatchdogPhase = Literal["idle", "stopped"]
WatchdogAction = Literal["watchdog_stop"]
_STOP_MS: Final = int(COMPANION_WATCHDOG_S * 1000)
_STEERED_MODES: Final = frozenset({CopterMode.GUIDED, CopterMode.AUTO})


@dataclass(frozen=True, slots=True)
class WatchInput:
    now_ms: int
    onboard_seen_ms: int | None  # last HEARTBEAT from (fc_sysid, 191)
    fc_mode: int | None  # from a fresh HEARTBEAT
    in_flight: bool
    operator_override: bool  # the operator's STOP or FLY HOME stands
    battery_latch: bool
    manual: bool = False  # the pilot steers by hand (Ta över)


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
        if inp.operator_override or inp.battery_latch or inp.manual or inp.fc_mode is None:
            return None
        if self.phase == "idle" and inp.fc_mode in _STEERED_MODES:
            self.phase = "stopped"
            return "watchdog_stop"
        return None
