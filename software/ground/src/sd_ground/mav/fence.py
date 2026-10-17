"""The fence in the flight controller: upload, read back, read and move its limits.

These coroutines run in the MAVLink thread's loop on the ground station's own channel
(250/190); the main loop reaches them through ``ThreadSafeMav.call``. Every message
passes the ground allowlist, which allows the fence upload and the fence limits only
while the vehicle stands disarmed on the ground. The AI has no way to start them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Final

from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_mav.allowlist import FC_COMPID, FENCE_GROUND_STATE
from sd_mav.mission import FENCE, MissionError, download, fence_hash, fence_items, upload
from sd_mav.vehicle import ForbiddenMavlinkError, SendOrigin, Vehicle

__all__ = [
    "FENCE_ATTEMPTS",
    "FENCE_STATE_WAIT_S",
    "fence_allowed_again",
    "read_fence_hash",
    "read_params",
    "set_fence_limits",
    "state_refusal",
    "upload_fence",
]

FENCE_ATTEMPTS: Final = 3  # the first try and at most two more after late telemetry
FENCE_STATE_WAIT_S: Final = 3.0  # how long to wait for fresh "disarmed on the ground"
_POLL_S: Final = 0.1


def state_refusal(exc: ForbiddenMavlinkError) -> bool:
    """The allowlist refused only because the vehicle was not known to stand disarmed on
    the ground at that moment (for example late telemetry), not for what was sent."""
    return exc.reason == FENCE_GROUND_STATE


async def fence_allowed_again(vehicle: Vehicle, timeout_s: float = FENCE_STATE_WAIT_S) -> bool:
    """Wait until fresh telemetry shows the vehicle disarmed on the ground again, that is
    until the allowlist would take a fence upload. ``False`` after ``timeout_s``: nothing
    is sent while the state is unknown."""
    probe = mavlink.MAVLink_mission_count_message(vehicle.fc_sysid, FC_COMPID, 3, FENCE)
    end = vehicle.clock.monotonic() + timeout_s
    while vehicle.would_refuse(probe, origin=SendOrigin.GROUND) is not None:
        if vehicle.clock.monotonic() >= end:
            return False
        await vehicle.clock.sleep(_POLL_S)
    return True


async def upload_fence(vehicle: Vehicle, corners: Sequence[tuple[float, float]]) -> str:
    """Upload the inclusion polygon ``corners`` (lat, lon), read it back and return its
    ``fence_hash``. ``MissionError`` when what the flight controller gives back is not
    exactly what was sent; ``MissionError`` / ``ForbiddenMavlinkError`` on failure."""
    items = fence_items(corners)
    await upload(vehicle, items, origin=SendOrigin.GROUND)
    stored = await read_fence_hash(vehicle)
    if stored != fence_hash(items):
        raise MissionError("the fence read back differs from the uploaded polygon")
    return stored


async def read_fence_hash(vehicle: Vehicle) -> str:
    stored = await download(vehicle, FENCE, origin=SendOrigin.GROUND)
    return fence_hash(stored)


async def read_params(vehicle: Vehicle, names: Iterable[str]) -> dict[str, float | None]:
    """Each parameter's value (``None`` when the flight controller did not answer)."""
    out: dict[str, float | None] = {}
    for name in names:
        value = await vehicle.read_param(name, origin=SendOrigin.GROUND)
        if value is None:  # one retry: a PARAM_VALUE can be lost on the way
            value = await vehicle.read_param(name, origin=SendOrigin.GROUND)
        out[name] = value
    return out


async def set_fence_limits(
    vehicle: Vehicle, targets: Mapping[str, float]
) -> dict[str, float | None]:
    """Move the fence's height, radius and LOITER speed to ``targets``; returns what the
    flight controller reports back for each."""
    out: dict[str, float | None] = {}
    for name, value in targets.items():
        out[name] = await vehicle.set_fence_limit(name, value, origin=SendOrigin.GROUND)
    return out
