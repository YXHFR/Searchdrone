"""A ground emergency path on a fake flight controller, for STOP / FLY HOME tests.

``Stack`` wires the real parts together: ``GroundMav`` (the MAVLink thread's core) on a
``FakeFlightController``, ``Latches`` and ``VehicleActions``, all on the test's own loop
with a ``FakeClock``. A second ``FakeVehicle`` as 1/191 plays the onboard computer's
heartbeat, which the companion watchdog follows.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_ground.auth import Principal
from sd_ground.bus import EventBus
from sd_ground.mav.thread import GroundMav, ThreadSafeMav
from sd_ground.services.latches import Latches
from sd_ground.services.vehicle_actions import NoticeResult, VehicleActions
from sd_mav.connection import Role
from sd_mav.fake import FakeFlightController, FakeLink, FakeVehicle
from sd_mav.modes import CopterMode
from sd_schemas.base import SdModel, new_id
from sd_testing import FakeClock

VID = "sim-01"
PILOT = Principal(role="pilot", issuer_ref="operator:demo", auth_mode="local")
OBSERVER = Principal(role="observer", issuer_ref="operator:guest", auth_mode="local")


class TimedFc(FakeFlightController):
    """Records when (real ``time.monotonic``) each message reached the flight controller."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.arrivals: list[tuple[float, int, int, Any]] = []

    def deliver(self, link: FakeLink, data: bytes) -> None:
        before = len(self.received)
        super().deliver(link, data)
        now = time.monotonic()
        for r in self.received[before:]:
            self.arrivals.append((now, r.sysid, r.compid, r.msg))


def ground_modes(fc: FakeFlightController, since: int = 0) -> list[CopterMode]:
    """Every DO_SET_MODE from the ground station (250/190), in order."""
    return [
        CopterMode(int(r.msg.param2))
        for r in fc.received[since:]
        if (r.sysid, r.compid) == (250, 190)
        and r.msg.get_type() == "COMMAND_LONG"
        and int(r.msg.command) == mavlink.MAV_CMD_DO_SET_MODE
    ]


def ground_disarms(fc: FakeFlightController, since: int = 0) -> int:
    return sum(
        1
        for r in fc.received[since:]
        if (r.sysid, r.compid) == (250, 190)
        and r.msg.get_type() == "COMMAND_LONG"
        and int(r.msg.command) == mavlink.MAV_CMD_COMPONENT_ARM_DISARM
    )


@dataclass
class StubGround:
    """What ``VehicleActions.drain`` and ``OperatorPresence`` need from ``Ground``."""

    clock: FakeClock
    bus: EventBus = field(default_factory=EventBus)
    events: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    operator_seen_mono: float | None = None

    def add_event(
        self,
        type_: str,
        *,
        severity: str = "info",
        mission_id: str | None = None,
        vehicle_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.events.append((type_, severity, payload or {}))

    def types(self) -> list[str]:
        return [t for t, _, _ in self.events]


@dataclass
class RecordingSink:
    """SD-link towards the onboard computer, as a list."""

    is_up: bool = True
    sent: list[SdModel] = field(default_factory=list[SdModel])
    # How the onboard side answers a take-over notice: at once, or never (a timeout); or
    # the send itself hangs, or raises.
    acks: bool = True
    hangs: bool = False
    raises: bool = False
    waits: list[float] = field(default_factory=list[float])
    clock: FakeClock | None = None  # the stack's clock, for the time a missing ack takes

    def up(self, vehicle_id: str) -> bool:
        return self.is_up

    async def send(self, vehicle_id: str, message: SdModel) -> bool:
        if not self.is_up:
            return False
        self.sent.append(message)
        return True

    async def send_acked(self, vehicle_id: str, message: SdModel, timeout_s: float) -> NoticeResult:
        if self.raises:
            raise ConnectionResetError("link went away mid-send")
        if self.hangs:
            await asyncio.Event().wait()
        if not await self.send(vehicle_id, message):
            return "down"
        if self.acks:
            return "acked"
        self.waits.append(timeout_s)
        await (self.clock.sleep(timeout_s) if self.clock else asyncio.sleep(timeout_s))
        return "timeout"


@dataclass
class Stack:
    clock: FakeClock
    fc: FakeFlightController
    ground: FakeVehicle
    onboard: FakeVehicle
    core: GroundMav
    latches: Latches
    actions: VehicleActions
    g: StubGround
    sink: RecordingSink
    onboard_alive: list[bool]

    async def run(self, seconds: float) -> None:
        await self.actions.drain(self.g)  # the main loop's pump, as often as it runs there
        await self.clock.run_for(seconds, step=0.01)
        await self.actions.drain(self.g)

    def fly(self, alt: float = 20.0, mode: CopterMode = CopterMode.GUIDED) -> None:
        """Put the fake in the air at once (no take-off needed for these tests)."""
        self.fc.armed = True
        self.fc.mode = mode
        self.fc.alt_rel = alt
        self.fc.target = None

    def stop(self, who: Principal = PILOT) -> Any:
        return self.actions.dispatch(
            "stop", VID, client_action_id=str(new_id()), via="button", by=who
        )

    def home(self, who: Principal = PILOT) -> Any:
        return self.actions.dispatch(
            "return_home", VID, client_action_id=str(new_id()), via="button", by=who
        )

    def resume(self, who: Principal = PILOT) -> Any:
        return self.actions.resume(VID, client_action_id=str(new_id()), by=who)

    def outcomes(self) -> list[dict[str, Any]]:
        return [
            e.data
            for e in self.g.bus.replay(self.g.bus.server_epoch, 0) or []
            if e.type == "vehicle.action"
        ]


@asynccontextmanager
async def stack(*, sink_up: bool = True, rtl_pct: float = 15.0) -> AsyncIterator[Stack]:
    clock = FakeClock()
    fc = FakeFlightController(clock)
    alive = [True]
    ground = FakeVehicle(clock, fc=fc, role=Role.GROUND)
    onboard = FakeVehicle(clock, fc=fc, heartbeat_allowed=lambda: alive[0])
    latches = Latches(clock)
    sink = RecordingSink(is_up=sink_up, clock=clock)
    actions = VehicleActions(clock, latches, vehicle_ids=[VID], notices=sink)
    core = GroundMav(
        ground,
        clock,
        vehicle_id=VID,
        latches=latches,
        on_report=actions.on_report,
        on_event=actions.on_event,
        system_action=actions.system,
        rtl_pct=lambda: rtl_pct,
    )
    actions.mav = ThreadSafeMav(core, asyncio.get_running_loop())
    await onboard.start()
    await core.start()
    s = Stack(clock, fc, ground, onboard, core, latches, actions, StubGround(clock), sink, alive)
    await s.run(2.0)  # connected, streams asked for
    try:
        yield s
    finally:
        await core.close()
        await onboard.close()
