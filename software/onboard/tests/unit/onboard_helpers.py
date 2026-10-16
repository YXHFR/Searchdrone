"""Shared helpers for the onboard unit tests: a fake flight controller that also flies
AUTO missions, a recording uplink, a charter near the SITL site and order builders."""

from __future__ import annotations

import asyncio
import datetime as dt
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_mav.connection import Role
from sd_mav.fake import CMAC_HOME, FakeFlightController, FakeVehicle
from sd_mav.mission import FENCE, MISSION, fence_hash, fence_items
from sd_mav.modes import CopterMode
from sd_onboard.executor.executor import Executor, ExecutorOptions
from sd_safety.constants import FC_SAFETY_EXPECTED, GOTO_ARRIVE_RADIUS_M
from sd_safety.targets import Plane
from sd_schemas.base import LatLonAlt, new_id
from sd_schemas.charter import AutonomyMode, MissionCharter
from sd_schemas.gate import StateRef
from sd_schemas.link import (
    ApprovalRef,
    CharterAck,
    CharterInstall,
    ClearanceConfirmation,
    CommanderStatus,
    OperatorOverrideNotice,
    SkillCommand,
)
from sd_testing import FakeClock

HOME_LAT, HOME_LON, HOME_ALT = CMAC_HOME
PLANE = Plane(HOME_LAT, HOME_LON)
ONBOARD_SKILLS = ["takeoff", "goto", "survey_sector", "hold", "resume", "return_home"]
_M_PER_DEG = 111_320.0


def at(east: float, north: float) -> tuple[float, float]:
    """(lat, lon) of a point east/north of home, in metres."""
    return PLANE.latlon(east, north)


def square(half: float, east: float = 0.0, north: float = 0.0) -> dict[str, Any]:
    corners = [(-half, -half), (half, -half), (half, half), (-half, half), (-half, -half)]
    ring = []
    for x, y in corners:
        lat, lon = at(x + east, y + north)
        ring.append([lon, lat])
    return {"type": "Polygon", "coordinates": [ring]}


AREA = square(200.0)


class AutoFc(FakeFlightController):
    """The fake flight controller, plus AUTO missions, MISSION_CURRENT and
    MISSION_ITEM_REACHED, and a battery that can be set."""

    def __init__(self, clock: FakeClock, **kw: Any) -> None:
        super().__init__(clock, **kw)
        self.loiter_until: float | None = None
        self.track: list[tuple[float, float, float]] = []  # lat, lon, alt while armed

    def step(self, dt_s: float) -> None:
        if self.armed and self.mode == CopterMode.AUTO:
            self._auto(dt_s)
        elif self.mode == CopterMode.LOITER and self.armed:
            self.target = None
        super().step(dt_s)
        if self.armed:
            self.track.append((self.lat, self.lon, self.alt_rel))
        if self.missions[MISSION]:
            self._send(mavlink.MAVLink_mission_current_message(self.mission_current))

    def _auto(self, dt_s: float) -> None:
        items = self.missions[MISSION]
        if not 0 < self.mission_current < len(items):
            self.target = None
            return
        item = items[self.mission_current]
        if item.command == mavlink.MAV_CMD_DO_CHANGE_SPEED:
            self.speed_ms = float(item.param2)
            self.mission_current += 1
            return
        if item.command == mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH:
            self.mode = CopterMode.RTL
            return
        lat, lon = item.x / 1e7, item.y / 1e7
        self.target = (lat, lon, float(item.z))
        north = (lat - self.lat) * _M_PER_DEG
        east = (lon - self.lon) * _M_PER_DEG * math.cos(math.radians(self.lat))
        close = math.hypot(north, east) <= GOTO_ARRIVE_RADIUS_M and abs(self.alt_rel - item.z) < 1
        if not close:
            return
        if item.command == mavlink.MAV_CMD_NAV_LOITER_TIME:
            now = self.clock.monotonic()
            if self.loiter_until is None:
                self.loiter_until = now + float(item.param1)
            if now < self.loiter_until:
                return
            self.loiter_until = None
        self._send(mavlink.MAVLink_mission_item_reached_message(self.mission_current))
        self.mission_current += 1


@dataclass
class Uplink:
    acks: list[tuple[UUID, str, tuple[str, ...]]] = field(default_factory=list)
    events: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    charter_acks: list[CharterAck] = field(default_factory=list)

    def ack(
        self, ref_id: UUID, stage: str, *, reason_codes: tuple[str, ...] = (), **_: Any
    ) -> None:
        self.acks.append((ref_id, stage, tuple(reason_codes)))

    def event(self, type: str, severity: str, data: dict[str, Any]) -> None:
        self.events.append((type, severity, data))

    def charter_ack(self, ack: CharterAck) -> None:
        self.charter_acks.append(ack)

    def stages(self, ref: UUID) -> list[str]:
        return [stage for r, stage, _ in self.acks if r == ref]

    def codes(self, ref: UUID) -> tuple[str, ...]:
        found = [
            codes for r, stage, codes in self.acks if r == ref and stage in ("rejected", "failed")
        ]
        return found[-1] if found else ()

    def kinds(self) -> list[str]:
        return [t for t, _, _ in self.events]


def charter(**overrides: Any) -> MissionCharter:
    data: dict[str, Any] = {
        "charter_id": new_id(),
        "mission_id": new_id(),
        "vehicle_id": "sim-01",
        "version": 1,
        "area": AREA,
        "home": LatLonAlt(lat=HOME_LAT, lon=HOME_LON, alt_amsl_m=HOME_ALT),
        "pilot_position": {"lat": HOME_LAT, "lon": HOME_LON},
        "allowed_skills": ONBOARD_SKILLS,
        "autonomy_mode": AutonomyMode.WITHIN_CHARTER,
        "sectors": [],
        "fence_hash": None,
        "approved_by": "operator:pilot",
        "approved_at": dt.datetime(2026, 10, 15, 12, 0, tzinfo=dt.UTC),
    }
    data.update(overrides)
    return MissionCharter.model_validate(data)


EXPECTED_PARAMS = {k: float(v) for k, v in FC_SAFETY_EXPECTED.items()}


@dataclass
class Rig:
    clock: FakeClock
    fc: AutoFc
    vehicle: FakeVehicle
    ground: FakeVehicle  # the ground station's MAVLink channel (GCS heartbeat)
    ex: Executor
    up: Uplink
    stop: asyncio.Event
    charter: MissionCharter | None = None
    background: list[asyncio.Task[None]] = field(default_factory=list)
    link: bool = True
    _seq: int = 1000

    async def run(self, seconds: float, step: float = 0.02) -> None:
        await self.clock.run_for(seconds, step)

    async def until(self, cond: Any, limit_s: float, step: float = 0.02) -> None:
        end = self.clock.monotonic() + limit_s
        while not cond():
            if self.clock.monotonic() > end:
                raise AssertionError(f"not reached within {limit_s} s (state {self.ex.state})")
            await self.clock.run_for(step, step)

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def install(self, c: MissionCharter | None = None) -> MissionCharter:
        c = c or charter()
        polygon = [(lat, lon) for lon, lat in c.area["coordinates"][0][:-1]]
        items = fence_items(polygon)
        self.fc.missions[FENCE] = items
        c = c.model_copy(update={"fence_hash": fence_hash(items)})
        await self.ex.on_charter(CharterInstall(charter=c))
        self.charter = c
        await self.run(3.0)
        return c

    def clearance(self, **overrides: Any) -> UUID:
        c = self.charter
        assert c is not None
        data: dict[str, Any] = {
            "clearance_id": new_id(),
            "mission_id": c.mission_id,
            "vehicle_id": "sim-01",
            "charter_id": c.charter_id,
            "charter_version": c.version,
            "valid_for_ms": 120_000,
            "confirmed_by": "operator:pilot",
            "origin": "local",
        }
        data.update(overrides)
        ref = new_id()
        self.ex.on_clearance(ClearanceConfirmation.model_validate(data), ref)
        return ref

    def order(
        self,
        skill: str,
        params: dict[str, Any] | None = None,
        *,
        issuer: str = "operator:pilot",
        approval: ApprovalRef | None = None,
        charter_version: int | None = None,
        based_on: StateRef | None = None,
    ) -> SkillCommand:
        snap = self.ex.snapshot()
        version = charter_version
        if version is None:
            version = self.ex.charter.version if self.ex.charter else 1
        return SkillCommand(
            command_id=new_id(),
            skill=skill,
            params=params or {},
            issuer=issuer,
            rationale=None,
            charter_version=version,
            based_on=based_on or StateRef(state_epoch=snap.state_epoch, state_seq=snap.state_seq),
            approval=approval,
        )

    async def send(self, cmd: SkillCommand, settle_s: float = 0.3) -> SkillCommand:
        await self.ex.on_skill_command(cmd, deadline_ms=None)
        await self.run(settle_s)
        return cmd

    async def takeoff(self, alt: float = 40.0) -> SkillCommand:
        if self.charter is None:
            await self.install()
        self.clearance()
        cmd = await self.send(self.order("takeoff", {"alt_m": alt}))
        await self.until(lambda: self.ex.state.name == "HOLD", 60)
        return cmd

    def notice(self, kind: str, seq: int | None = None) -> OperatorOverrideNotice:
        return OperatorOverrideNotice(
            override_id=new_id(),
            kind=kind,  # type: ignore[arg-type]
            action_seq=self.next_seq() if seq is None else seq,
            issuer_kind="operator",
            operator_id="operator:pilot",
        )

    def steering_sent(self, since: int = 0) -> list[Any]:
        """Messages from 1/191 that steer (everything but startup traffic)."""
        out = []
        for msg in self.vehicle.sent[since:]:
            kind = msg.get_type()
            if kind in (
                "HEARTBEAT",
                "PARAM_REQUEST_READ",
                "MISSION_REQUEST_LIST",
                "MISSION_REQUEST_INT",
                "MISSION_ACK",
            ):
                continue
            if kind == "COMMAND_LONG" and int(msg.command) in (
                mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                mavlink.MAV_CMD_REQUEST_MESSAGE,
            ):
                continue
            out.append(msg)
        return out


@asynccontextmanager
async def rig(
    *,
    link_up: bool = True,
    options: dict[str, Any] | None = None,
    start: bool = True,
) -> AsyncIterator[Rig]:
    clock = FakeClock()
    fc = AutoFc(clock)
    fc.params.update({"FENCE_ALT_MAX": 125.0, "FENCE_RADIUS": 530.0, "LOIT_SPEED_MS": 15.0})
    up = Uplink()
    holder: list[Executor] = []
    vehicle = FakeVehicle(
        clock,
        fc=fc,
        on_event=lambda k, d: holder[0].on_vehicle_event(k, d) if holder else None,
        heartbeat_allowed=lambda: bool(holder) and holder[0].loop_alive(),
        send_gate=lambda o: holder[0].send_gate(o) if holder else "starting",
    )
    ground = FakeVehicle(clock, role=Role.GROUND, fc=fc)
    opts = {"vehicle_id": "sim-01", "expected_params": EXPECTED_PARAMS, **(options or {})}
    ex = Executor(vehicle, clock, up, ExecutorOptions(**opts))
    holder.append(ex)
    stop = asyncio.Event()
    await vehicle.start()
    await ground.start()
    r = Rig(clock, fc, vehicle, ground, ex, up, stop)
    task: asyncio.Task[None] | None = None
    if start:
        task = asyncio.ensure_future(ex.run(stop))
        await r.until(lambda: ex._started, 10)
        if link_up:
            ex.on_link(True, clock.monotonic_ms())
            _keep_link(r)
    try:
        yield r
    finally:
        stop.set()
        for t in [task, *r.background]:
            if t is not None:
                t.cancel()
                await asyncio.gather(t, return_exceptions=True)
        await ex.close()
        await vehicle.close()
        await ground.close()


def _keep_link(r: Rig) -> None:
    """SD-link up, with traffic from the ground every 0.5 s, while ``r.link`` is set."""

    async def beat() -> None:
        while not r.stop.is_set():
            if r.link:
                r.ex.on_link(True, r.clock.monotonic_ms())
                r.ex.on_commander_status(CommanderStatus(connected=True, state="ready"))
            else:
                r.ex.on_link(False, None)
            await r.clock.sleep(0.5)

    r.background.append(asyncio.ensure_future(beat()))
