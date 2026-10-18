"""``/health``, ``/system/status`` and the live view of one vehicle."""

from __future__ import annotations

from typing import Literal

from sd_ground.context import Ground
from sd_ground.sources import PartState
from sd_schemas.api import Health, PartStatus, SystemStatus, VehicleLive
from sd_schemas.mission import OperatorLatch

__all__ = ["READY_PARTS", "health", "parts", "system_status", "vehicle_live", "vehicle_state"]

READY_PARTS = ("db", "mavlink", "sdlink")  # ready = these three are ok


def parts(g: Ground) -> dict[str, PartState]:
    brain: PartState = (
        "ok"
        if g.brain_state in ("ready", "busy")
        else ("degraded" if g.brain_state == "degraded" else "down")
    )
    return {
        "db": "ok" if g.db_ok else "down",
        "mavlink": g.vehicle_link.state(),
        "sdlink": g.sdlink.state(),
        "onboard": g.vehicle_link.onboard_state(),
        "brain": brain,
        "mediamtx": g.video_state,  # services.video.VideoMonitor
    }


def health(g: Ground) -> Health:
    p = parts(g)
    return Health(
        ready=all(p[name] == "ok" for name in READY_PARTS),
        version=g.version,
        profile=g.config.profile,
        parts=dict(p),
        test_rig=g.test_rig,
    )


def system_status(g: Ground) -> SystemStatus:
    p = parts(g)
    internet: Literal["up", "down", "unknown"] = "unknown"  # the probe is off in v0.1.0
    return SystemStatus(
        parts={name: PartStatus(state=state, since=None) for name, state in p.items()},
        internet=internet,
        operator_present=g.pilot_ws > 0,
        version=g.version,
        profile=g.config.profile,
    )


def vehicle_state(g: Ground, vehicle_id: str, latch: OperatorLatch | None) -> VehicleLive:
    """The vehicle as the panel sees it (REST ``/state`` and the live ``vehicle.state``)."""
    telemetry = g.vehicle_link.telemetry(vehicle_id)
    return VehicleLive(
        vehicle_id=vehicle_id,
        telemetry=telemetry,
        snapshot=g.sdlink.snapshot(vehicle_id),
        link=g.sdlink.link_status(vehicle_id),
        latch=latch,
        airborne=telemetry is not None and telemetry.landed_state == "in_air",
        in_flight=g.vehicle_link.in_flight(vehicle_id),
        manual=None if g.vehicle_actions is None else g.vehicle_actions.manual_state(vehicle_id),
    )


async def vehicle_live(g: Ground, vehicle_id: str) -> VehicleLive:
    latch = None
    if g.repo is not None and g.db_ok:
        latch = await g.repo.latch(vehicle_id)
    return vehicle_state(g, vehicle_id, latch)
