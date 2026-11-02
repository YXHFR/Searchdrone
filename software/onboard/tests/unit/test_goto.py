"""``goto`` against a stub flight controller: the stop before a keep-out climb
(``from_stop``) and which end a leg arrived at."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
from onboard_helpers import PLANE, at

from sd_mav.modes import CopterMode
from sd_onboard.skills._common import STOP_WAIT_S
from sd_onboard.skills.goto import TICK_S, GotoHandler
from sd_schemas.base import LatLonAlt, new_id

ALT = 40.0


class StubIO:
    """A drone flying east at ``speed_ms`` that slows by ``brake_ms2`` (0: never slows),
    at the height last sent; ``on_sleep``/``on_mode`` run as the handler awaits them."""

    def __init__(self, speed_ms: float, brake_ms2: float) -> None:
        self.t = 0.0
        self.east = 0.0
        self.speed = speed_ms
        self.brake = brake_ms2
        self.sent: list[tuple[float, float, float]] = []
        self.on_sleep: Any = None
        self.on_mode: Any = None
        self.clock = SimpleNamespace(monotonic=lambda: self.t)
        self.alt = ALT
        self.mode = CopterMode.GUIDED.name

    @property
    def vehicle(self) -> Any:
        lat, lon = at(self.east, 0.0)
        return SimpleNamespace(
            lat=lat,
            lon=lon,
            alt_rel_home_m=self.alt,
            groundspeed_ms=self.speed,
            fc_mode=self.mode,
        )

    async def set_mode(self, mode: CopterMode) -> bool:
        self.mode = mode.name
        if self.on_mode is not None:
            self.on_mode()
        return True

    async def change_speed(self, _speed: float) -> None:
        return None

    async def set_position_target(
        self, lat: float, lon: float, alt: float, _yaw: float | None
    ) -> None:
        self.sent.append((lat, lon, alt))
        self.alt = alt

    async def sleep(self, dt: float) -> None:
        self.t += dt
        self.east += self.speed * dt
        self.speed = max(0.0, self.speed - self.brake * dt)
        if self.on_sleep is not None:
            self.on_sleep()


def ctx_of(io: StubIO) -> Any:
    v = io.vehicle
    return SimpleNamespace(
        command_id=new_id(),
        vehicle=v,
        plane=PLANE,
        clock=io.clock,
        alt_amsl=lambda a: a,
        here=lambda a=None: LatLonAlt(
            lat=v.lat, lon=v.lon, alt_amsl_m=v.alt_rel_home_m if a is None else a
        ),
    )


def east_of(lat: float, lon: float) -> float:
    return PLANE.xy(lat, lon)[0]


async def test_a_drone_never_still_climbs_from_where_it_is_after_the_stop_wait() -> None:
    """The flight controller never reports the drone still: after ``STOP_WAIT_S`` the
    climb's spot is where the drone is then, and the leg goes on."""
    io = StubIO(speed_ms=1.0, brake_ms2=0.0)
    goto = GotoHandler()
    v = io.vehicle
    goto.prepare(ctx_of(io), v.lat, v.lon, ALT, 3.0, from_stop=True)
    goto._deadline = 3 * STOP_WAIT_S  # pyright: ignore[reportPrivateUsage]  # never: fails
    await goto._stop_first(io)  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    assert abs(io.t - STOP_WAIT_S) <= TICK_S + 1e-9
    assert goto.target is not None
    assert math.isclose(east_of(*goto.target[:2]), io.east, abs_tol=1e-6)
    assert not goto._stopping  # pyright: ignore[reportPrivateUsage]


async def test_an_end_set_while_stopping_is_flown_as_given() -> None:
    """A new end that is no climb comes while the drone stops: the stop ends, and the new
    end is kept as it was given (not replaced by where the drone stopped)."""
    io = StubIO(speed_ms=4.0, brake_ms2=1.0)
    goto = GotoHandler()
    v = io.vehicle
    ctx = ctx_of(io)
    goto.prepare(ctx, v.lat, v.lon, ALT, 3.0, from_stop=True)
    far = at(100.0, 50.0)

    def retarget() -> None:
        if io.t >= 1.0 - 1e-9 and goto.target is not None and goto.target[:2] != far:
            goto.retarget(ctx, far[0], far[1], ALT, 3.0)

    io.on_sleep = retarget
    await goto._stop_first(io)  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    assert goto.target == (far[0], far[1], ALT)
    assert not goto._stopping  # pyright: ignore[reportPrivateUsage]
    assert io.t < 2.0  # no stop waited for after the new end


async def test_a_climb_set_while_stopping_stops_afresh_then_climbs_from_there() -> None:
    """A new climb (``from_stop``, higher) comes while the drone stops for the first: the
    stop goes on for the new one, and only once the drone stands still is the spot fixed
    there, at the new height; ``_stopping`` is never left set (a moving spot fixed)."""
    io = StubIO(speed_ms=4.0, brake_ms2=1.0)
    goto = GotoHandler()
    v = io.vehicle
    ctx = ctx_of(io)
    goto.prepare(ctx, v.lat, v.lon, ALT, 3.0, from_stop=True)
    high = ALT + 20.0

    def climb() -> None:
        if io.t >= 1.0 - 1e-9 and goto.target is not None and goto.target[2] != high:
            w = io.vehicle
            goto.retarget(ctx_of(io), w.lat, w.lon, high, 3.0, from_stop=True)

    io.on_sleep = climb
    await goto._stop_first(io)  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    assert io.speed < 0.5
    assert goto.target is not None and goto.target[2] == high
    assert math.isclose(east_of(*goto.target[:2]), io.east, abs_tol=1e-6)
    assert not goto._stopping  # pyright: ignore[reportPrivateUsage]
    sent_east = [east_of(lat, lon) for lat, lon, _ in io.sent]
    assert sent_east == sorted(sent_east)  # every spot sent follows the drone, none behind


@pytest.mark.parametrize("late", [False, True])
async def test_a_leg_with_a_new_end_as_it_switches_to_loiter_has_not_arrived(late: bool) -> None:
    """At the end, still, as ``start`` switches to LOITER a new end comes (``retarget``):
    the leg did not arrive at it (without one it did)."""
    io = StubIO(speed_ms=0.0, brake_ms2=0.0)
    goto = GotoHandler()
    v = io.vehicle
    ctx = ctx_of(io)
    goto.prepare(ctx, v.lat, v.lon, ALT, 3.0)
    far = at(100.0, 0.0)

    def new_end() -> None:
        if late and io.mode == CopterMode.LOITER.name:
            goto.retarget(ctx, far[0], far[1], ALT, 3.0)

    io.on_mode = new_end
    plan = SimpleNamespace(expected_duration_s=10.0)
    await goto.start(plan, io)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    assert goto.arrived is not late
