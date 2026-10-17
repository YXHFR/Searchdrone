"""The fence in the flight controller: upload, read back, read and move its limits.

These coroutines run in the MAVLink thread's loop on the ground station's own channel
(250/190); the main loop reaches them through ``ThreadSafeMav.call``. Every message
passes the ground allowlist, which allows the fence upload and the fence limits only
while the vehicle stands disarmed on the ground. The AI has no way to start them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from sd_mav.mission import FENCE, MissionError, download, fence_hash, fence_items, upload
from sd_mav.vehicle import SendOrigin, Vehicle

__all__ = ["read_fence_hash", "read_params", "set_fence_limits", "upload_fence"]


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
