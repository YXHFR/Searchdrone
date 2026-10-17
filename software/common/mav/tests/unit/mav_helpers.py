"""Shared helpers for the sd_mav unit tests."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable
from typing import Any

from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_mav.allowlist import Profile, SendContext, SendEnvelope
from sd_mav.fake import CMAC_HOME
from sd_mav.modes import CopterMode
from sd_safety.targets import distance_m
from sd_testing import FakeClock

HOME = (CMAC_HOME[0], CMAC_HOME[1])
FC = 1


def ctx(
    profile: Profile = Profile.ONBOARD,
    origin: str = "skill",
    *,
    home: tuple[float, float] | None = HOME,
    fc_mode: int | None = CopterMode.GUIDED,
    on_ground: bool = False,
    envelope: SendEnvelope | None = None,
    armed: bool | None = None,
    airborne: bool = False,
) -> SendContext:
    return SendContext(
        profile=profile,
        origin=origin,
        target_system=FC,
        envelope=envelope or SendEnvelope(),
        home=home,
        fc_mode=fc_mode,
        on_ground=on_ground,
        armed=armed,
        airborne=airborne,
    )


def command(cmd: int, *params: float, target: int = FC, compid: int = 1) -> Any:
    p = [*params, *([0.0] * (7 - len(params)))]
    return mavlink.MAVLink_command_long_message(target, compid, cmd, 0, *p[:7])


def set_mode(mode: int) -> Any:
    return command(mavlink.MAV_CMD_DO_SET_MODE, 1, mode)


def offset(lat: float, lon: float, north_m: float, east_m: float = 0.0) -> tuple[float, float]:
    """A point ``north_m`` north and ``east_m`` east of (lat, lon), geodesically exact.

    Scales a first spherical guess until ``distance_m`` (the same formula as the
    allowlist) gives the asked distance to within a millimetre.
    """
    want = math.hypot(north_m, east_m)
    if want == 0:
        return lat, lon
    dlat = north_m / 111_320.0
    dlon = east_m / (111_320.0 * math.cos(math.radians(lat)))
    for _ in range(50):
        got = distance_m(lat, lon, lat + dlat, lon + dlon)
        if abs(got - want) < 1e-3:
            break
        dlat, dlon = dlat * want / got, dlon * want / got
    return lat + dlat, lon + dlon


def position_target(
    lat: float, lon: float, alt: float, *, frame: int = 6, mask: int = 0x0DF8
) -> Any:
    return mavlink.MAVLink_set_position_target_global_int_message(
        0, FC, 1, frame, mask, round(lat * 1e7), round(lon * 1e7), alt, 0, 0, 0, 0, 0, 0, 0, 0
    )


def mission_item(
    seq: int,
    cmd: int,
    frame: int,
    *params: float,
    lat: float = 0.0,
    lon: float = 0.0,
    z: float = 0.0,
    mission_type: int = 0,
) -> Any:
    p1, p2, p3, p4 = [*params, *([0.0] * (4 - len(params)))][:4]
    return mavlink.MAVLink_mission_item_int_message(
        FC,
        1,
        seq,
        frame,
        cmd,
        0,
        1,
        p1,
        p2,
        p3,
        p4,
        round(lat * 1e7),
        round(lon * 1e7),
        z,
        mission_type,
    )


def _zero(fieldtype: str, length: int) -> Any:
    if fieldtype == "char":
        return b""
    value: Any = 0.0 if fieldtype in ("float", "double") else 0
    return [value] * length if length else value


def zero_message(cls: Any) -> Any:
    """An instance of a pymavlink message class with every field zero."""
    lengths = dict(zip(cls.ordered_fieldnames, cls.array_lengths, strict=True))
    args = [
        _zero(ftype, lengths[name])
        for name, ftype in zip(cls.fieldnames, cls.fieldtypes, strict=True)
    ]
    return cls(*args)


def all_message_classes() -> list[type]:
    return sorted(
        set(mavlink.mavlink_map.values()),
        key=lambda c: c.msgname,  # type: ignore[attr-defined]
    )


def all_commands() -> list[int]:
    return sorted(
        int(v) for k, v in vars(mavlink).items() if k.startswith("MAV_CMD_") and isinstance(v, int)
    )


async def run_with_clock[T](clock: FakeClock, coro: Awaitable[T], *, max_s: float = 60.0) -> T:
    task = asyncio.ensure_future(coro)
    waited = 0.0
    while not task.done():
        if waited > max_s:
            task.cancel()
            raise TimeoutError("the coroutine did not finish in simulated time")
        await clock.run_for(0.05, step=0.05)
        waited += 0.05
    return task.result()
