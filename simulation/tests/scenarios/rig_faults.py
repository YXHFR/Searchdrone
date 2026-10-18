"""Extra pieces for the scenario tests that go past the gate on purpose, or watch more.

* ``BusTap``: everything the ground station publishes (alerts, alarms, chat status), as
  the panel's live connection would see it.
* ``presence``: the ground's operator-presence watch (no pilot WebSocket for 30 s during
  an active mission: the alarm and a hold), as ``sd-ground serve`` runs it.
* The harness flying past the gate (253/190): GUIDED with a velocity or a position
  target, straight to the flight controller. Only these tests do it, and everything the
  harness sends is recorded on the timeline (sender ``harness``).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from pymavlink import mavutil
from scenario_rig import Scenario  # pyright: ignore[reportMissingImports]

from sd_ground.mav.thread import vehicle_events
from sd_ground.services.vehicle_actions import OperatorPresence
from sd_mav import CopterMode

__all__ = [
    "GUIDED_VELOCITY",
    "HARNESS_GOTO",
    "BusTap",
    "brain_started",
    "finished",
    "ground_vehicle_events",
    "harness_goto",
    "harness_guided_velocity",
    "presence",
    "tap",
]

mavlink: Any = mavutil.mavlink

# What the harness may send in these tests, beyond SIM_* and the battery (INV5).
GUIDED_VELOCITY = ("cmd:176", "SET_POSITION_TARGET_LOCAL_NED")
HARNESS_GOTO = ("cmd:176", "SET_POSITION_TARGET_GLOBAL_INT")
VELOCITY_ONLY = 0x0DC7  # ignore position, acceleration, yaw and yaw rate
SEND_HZ = 2.0
GUIDED_SEEN_S = 3.0
FC = (1, 1)


@dataclass
class BusTap:
    """Every bus event after it started: (host time, topic, type, data)."""

    seen: list[tuple[float, str, str, dict[str, Any]]] = field(
        default_factory=list[tuple[float, str, str, dict[str, Any]]]
    )

    def of(self, type_: str) -> list[tuple[float, dict[str, Any]]]:
        return [(t, d) for t, _, kind, d in self.seen if kind == type_]

    def alerts(self, code: str) -> list[float]:
        return [t for t, d in self.of("alert") if d.get("code") == code]

    def first(self, type_: str, after: float = 0.0) -> float | None:
        return next((t for t, d in self.of(type_) if t >= after), None)


@contextlib.asynccontextmanager
async def tap(sc: Scenario) -> AsyncIterator[BusTap]:
    out = BusTap()
    sub = sc.r.g.bus.subscribe()

    async def drain() -> None:
        while True:
            event = await sub.queue.get()
            if event is None:
                return
            if event.topic != "telemetry":
                out.seen.append((time.monotonic(), event.topic, event.type, dict(event.data)))

    task = asyncio.ensure_future(drain())
    try:
        yield out
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        sc.r.g.bus.unsubscribe(sub)


@contextlib.asynccontextmanager
async def presence(sc: Scenario) -> AsyncIterator[OperatorPresence]:
    watch = OperatorPresence(sc.r.actions, mission_active=sc.r.flow.mission_active)
    task = asyncio.ensure_future(watch.run(sc.r.g))
    try:
        yield watch
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def brain_started(sc: Scenario, skill: str, limit_s: float = 40) -> dict[str, Any]:
    """The AI's first order with ``skill`` that runs (or already ran)."""
    found: list[dict[str, Any]] = []

    async def running() -> bool:
        found[:] = [
            c
            for c in await sc.commands()
            if c["skill"] == skill
            and c["issuer_kind"] == "brain"
            and c["status"] in ("running", "done")
        ]
        return bool(found)

    try:
        await sc.until(running, limit_s, f"{skill} kör")
    except AssertionError as exc:
        raise AssertionError(f"{exc}; {await sc.describe()}") from None
    return found[0]


async def finished(sc: Scenario, command_id: str, limit_s: float) -> dict[str, Any]:
    """Until the order is ``done``; any other end fails at once."""
    found: list[dict[str, Any]] = []

    async def done() -> bool:
        cmd = await sc.command(command_id)
        if cmd["status"] not in ("done", "accepted", "running", "dispatched", "received"):
            raise AssertionError(f"{cmd['skill']}: {cmd['status']} {cmd['status_reason']}")
        found[:] = [cmd]
        return cmd["status"] == "done"

    await sc.until(done, limit_s, "ordern klar")
    return found[0]


def ground_vehicle_events(sc: Scenario) -> list[tuple[float, int, str]]:
    """The ground's MAVLink events (flight controller texts, ...) into its audit log, as
    ``sd-ground serve`` wires them. Returns every text the flight controller sent on the
    ground's channel: (host time, severity, text)."""
    vehicle = sc.r.core.vehicle
    keep = vehicle_events(sc.r.actions.on_event, now=time.monotonic)
    texts: list[tuple[float, int, str]] = []

    def on_event(kind: str, details: dict[str, Any]) -> None:
        if kind == "mav.statustext":
            texts.append(
                (time.monotonic(), int(details.get("severity", 7)), str(details.get("text")))
            )
        keep(kind, details)

    vehicle._on_event = on_event  # pyright: ignore[reportPrivateUsage]
    return texts


def _record(sc: Scenario, type_: str, **kw: Any) -> None:
    sc._harness_sent(type_, **kw)  # pyright: ignore[reportPrivateUsage]


async def _guided(sc: Scenario) -> None:
    mode = CopterMode.GUIDED
    _record(
        sc,
        "COMMAND_LONG",
        command=176,
        params=(float(mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED), mode),
    )
    await asyncio.to_thread(sc.sim.harness.set_mode, "GUIDED")
    # The ground's telemetry shows the mode a moment after the harness saw it.
    await sc.until(lambda: sc.mode() == mode, GUIDED_SEEN_S, "GUIDED i markens telemetri")


async def harness_guided_velocity(
    sc: Scenario, north_ms: float, east_ms: float, down_ms: float, duration_s: float
) -> float:
    """GUIDED and a velocity target from the harness, sent again at 2 Hz until
    ``duration_s`` or until the flight controller leaves GUIDED; returns the host time
    the flight controller left GUIDED (or the end)."""
    await _guided(sc)
    await asyncio.to_thread(sc.sim.harness.drain)  # the backlog queued while the rig flew
    conn = sc.sim.harness._conn  # pyright: ignore[reportPrivateUsage]
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        if sc.mode() != CopterMode.GUIDED:
            return time.monotonic()
        _record(sc, "SET_POSITION_TARGET_LOCAL_NED")
        conn.mav.set_position_target_local_ned_send(
            0,
            *FC,
            mavlink.MAV_FRAME_LOCAL_NED,
            VELOCITY_ONLY,
            0,
            0,
            0,
            north_ms,
            east_ms,
            down_ms,
            0,
            0,
            0,
            0,
            0,
        )
        await asyncio.sleep(1 / SEND_HZ)
    return time.monotonic()


async def harness_goto(
    sc: Scenario, lat: float, lon: float, alt_m: float, duration_s: float
) -> float:
    """GUIDED and a position target straight to the flight controller (past the gate),
    sent again at 2 Hz until ``duration_s`` or until the flight controller leaves GUIDED;
    returns the host time it left GUIDED (or the end)."""
    await _guided(sc)
    await asyncio.to_thread(sc.sim.harness.drain)  # the backlog queued while the rig flew
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        if sc.mode() != CopterMode.GUIDED:
            return time.monotonic()
        _record(sc, "SET_POSITION_TARGET_GLOBAL_INT")
        await asyncio.to_thread(sc.sim.harness.goto, lat, lon, alt_m)
        await asyncio.sleep(1 / SEND_HZ)
    return time.monotonic()
