"""The fallback for STOP and FLY HOME when the ground's MAVLink does not reach the flight
controller.

STOP and FLY HOME go as MAVLink straight from the ground station; the
``OperatorOverrideNotice`` over SD-link only makes the executor let go. If the flight
controller's HEARTBEAT does not show a mode in the target set (``STOP_MODES`` for the
STOP steps, ``HOME_MODES`` for the FLY HOME steps) within
``OVERRIDE_ONBOARD_FALLBACK_S`` of the notice, this sends the same steps itself, from
1/191 with ``SendOrigin.OVERRIDE_FALLBACK`` (the only user of that origin), with the same
rules as the ground (``sd_mav.chain``).

* Only while the flight controller still flies a mode this computer can command
  (``sd_mav.chain.late_order``) and the RC mode switch has not moved: a mode chosen
  with an RC transmitter, or a failsafe that just began, is left alone.
* Only when ``override_phase(...)`` says airborne. On the ground nothing is sent: the
  onboard computer never disarms, and ArduPilot disarms by itself after landing.
* The newest action wins: before every send the chain checks that its notice is still
  the newest one; a newer notice starts its own chain with its own timer. An accepted
  Resume (no latch kind left, ``still_active``) or landing stops it; a refused Resume
  does not, so the STOP still stands.
* The executor stays in HELD. The event ``override_fallback_used`` says that it ran.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any, Final

from sd_mav.chain import (
    failsafe_began_recently,
    heartbeat_fresh,
    late_order,
    mode_shown,
    rc_switch_moved,
    set_mode_with_fallback,
)
from sd_mav.modes import (
    HOME_MODES,
    HOME_STEPS,
    STOP_MODES,
    STOP_STEPS,
    OverridePhase,
    override_phase,
)
from sd_mav.vehicle import SendOrigin, Vehicle
from sd_obs.clock import Clock
from sd_safety.constants import OVERRIDE_ONBOARD_FALLBACK_S

__all__ = ["FALLBACK_DELAY_S", "OverrideFallback"]

FALLBACK_DELAY_S: Final = OVERRIDE_ONBOARD_FALLBACK_S


class OverrideFallback:
    def __init__(
        self,
        vehicle: Vehicle,
        clock: Clock,
        *,
        still_active: Callable[[], bool],
        emit: Callable[[str, dict[str, Any]], None],
    ) -> None:
        self.vehicle = vehicle
        self.clock = clock
        self._still_active = still_active
        self._seq: int | None = None
        self._emit = emit
        self._task: asyncio.Task[None] | None = None
        self.used = 0

    def start(self, kind: str, action_seq: int, steps: str) -> None:
        """A new notice: cancel an older chain and start this one's timer."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._seq = action_seq
        self._task = asyncio.ensure_future(self._run(kind, action_seq, steps))

    def stop(self) -> None:
        """An accepted Resume: the chain of the notice it answers never sends again."""
        self._seq = None
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task

    def _current(self, action_seq: int) -> bool:
        return action_seq == self._seq and self._still_active()

    def _airborne(self) -> bool:
        t = self.vehicle.telemetry
        if not heartbeat_fresh(self.vehicle):
            return False  # nothing to send to: the flight controller is not heard
        return override_phase(t.armed, t.landed_state, t.alt_rel_home_m) is OverridePhase.AIRBORNE

    async def _run(self, kind: str, action_seq: int, steps: str) -> None:
        chain = STOP_STEPS if steps == "stop" else HOME_STEPS
        targets = STOP_MODES if steps == "stop" else HOME_MODES
        rc_before = self.vehicle.telemetry.rc_mode_pwm
        await self.clock.sleep(FALLBACK_DELAY_S)
        if not self._current(action_seq):
            return
        shown = mode_shown(self.vehicle)
        if shown is None or shown in targets:
            return  # the ground's MAVLink got through (or nothing to judge from)
        if (
            not late_order(shown, targets)
            or rc_switch_moved(self.vehicle, rc_before)
            or failsafe_began_recently(self.vehicle)
        ):
            return  # a person with an RC transmitter, or the flight controller, decided
        if not self._airborne():
            return
        self.used += 1
        self._emit(
            "override_fallback_used",
            {"kind": kind, "action_seq": action_seq, "fc_mode": self.vehicle.telemetry.fc_mode},
        )
        result = await set_mode_with_fallback(
            self.vehicle,
            chain,
            origin=SendOrigin.OVERRIDE_FALLBACK,
            aborted=lambda: not self._current(action_seq) or not self._airborne(),
        )
        self._emit(
            "override_fallback_result",
            {
                "kind": kind,
                "action_seq": action_seq,
                "result": result.result,
                "mode": result.confirmed.name if result.confirmed is not None else None,
            },
        )
