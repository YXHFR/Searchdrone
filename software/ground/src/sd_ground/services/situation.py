"""The AI's picture of a mission (``Situation``): built from the same parts as the
gate's, read-only. Nothing in it lets the AI change anything."""

from __future__ import annotations

import datetime as dt
from typing import Any

from sd_ground.mav.thread import MavSnapshot
from sd_ground.services.clearance import ActiveClearance
from sd_safety.constants import AIRBORNE_MIN_ALT_M
from sd_safety.targets import distance_m
from sd_schemas.api import Annotation, Approval, Charter, Command, Event, Mission
from sd_schemas.base import LatLon
from sd_schemas.commander import MapPin, Situation, VehicleSummary
from sd_schemas.link import StateSnapshot
from sd_schemas.mission import OperatorLatch

__all__ = ["build_situation"]

_MIN_SATS = 8


def _vehicle(
    mav: MavSnapshot | None, snapshot: StateSnapshot | None, sdlink_up: bool, pilot: LatLon | None
) -> VehicleSummary:
    connected = mav is not None and mav.connected
    t = None if mav is None else mav.telemetry
    lat = None if t is None or not connected else t.lat
    lon = None if t is None or not connected else t.lon
    dist = None
    if pilot is not None and lat is not None and lon is not None:
        dist = distance_m(pilot.lat, pilot.lon, lat, lon)
    return VehicleSummary(
        connected=connected,
        armed=bool(t is not None and t.armed),
        landed=bool(t is None or t.landed_state == "on_ground"),
        airborne=bool(
            t is not None
            and t.armed
            and t.landed_state == "in_air"
            and t.alt_rel_home_m > AIRBORNE_MIN_ALT_M
        ),
        flight_mode="UNKNOWN" if t is None else t.fc_mode,
        executor_state=None if snapshot is None else snapshot.executor_state,
        lat=lat,
        lon=lon,
        alt_rel_home_m=None if t is None or not connected else t.alt_rel_home_m,
        groundspeed_ms=None if t is None or not connected else t.groundspeed_ms,
        battery_pct=None if t is None else t.battery_pct,
        battery_to_reserve_s=None,
        dist_to_pilot_m=dist,
        gnss_ok=bool(t is not None and t.gps_fix_type >= 3 and t.gps_satellites >= _MIN_SATS),
        link_ok=sdlink_up,
        manual_control=bool(
            (mav is not None and mav.manual) or (snapshot is not None and snapshot.manual_control)
        ),
    )


def _pins(annotations: list[Annotation]) -> list[MapPin]:
    pins = []
    for a in annotations:
        coords = a.geometry.get("coordinates") or [None, None]
        if (
            a.kind == "pin"
            and isinstance(a.label, str)
            and a.label[:1] == "P"
            and a.label[1:].isdigit()
        ):
            pins.append(
                MapPin.model_validate(
                    {"id": a.label, "annotation_id": a.id, "lat": coords[1], "lon": coords[0]}
                )
            )
    return pins


def build_situation(
    *,
    mission: Mission,
    charter: Charter | None,
    pending_proposal: Charter | None,
    pending_approvals: list[Approval],
    clearance: ActiveClearance | None,
    now: dt.datetime,
    now_ms: int,
    latch: OperatorLatch | None,
    mav: MavSnapshot | None,
    snapshot: StateSnapshot | None,
    sdlink_up: bool,
    active_command: Command | None,
    sectors: list[dict[str, Any]],
    annotations: list[Annotation],
    recent_events: list[Event],
) -> Situation:
    body = None if charter is None else charter.body
    home = None if body is None else LatLon(lat=body.home.lat, lon=body.home.lon)
    if home is None and mav is not None and mav.telemetry.home is not None:
        home = LatLon(lat=mav.telemetry.home.lat, lon=mav.telemetry.home.lon)
    pilot = None if body is None else body.pilot_position
    valid_until = None
    if clearance is not None and str(clearance.state.mission_id) == str(mission.id):
        left_ms = clearance.state.received_mono_ms + clearance.state.valid_for_ms - now_ms
        if left_ms > 0:
            valid_until = now + dt.timedelta(milliseconds=left_ms)
    installed = charter is not None and charter.install_status == "installed"
    verified = charter is not None and charter.fence_verified_at is not None
    return Situation(
        mission_id=mission.id,
        vehicle_id=mission.vehicle_id,
        state_seq=mission.state_seq,
        snapshot_state_epoch=None if snapshot is None else snapshot.state_epoch,
        snapshot_state_seq=None if snapshot is None else snapshot.state_seq,
        mission_status=mission.status,
        paused_by=mission.paused_by,
        returning_cause=mission.returning_cause,
        autonomy_mode=None if body is None else body.autonomy_mode,
        ai_enabled=mission.ai_enabled,
        limits=mission.limits,
        charter=body,
        charter_installed=installed,
        fence_verified=verified,
        pending_proposal=pending_proposal,
        pending_approvals=pending_approvals,
        takeoff_clearance_valid_until=valid_until,
        brain_may_takeoff=installed and verified and valid_until is not None,
        override=latch,
        vehicle=_vehicle(mav, snapshot, sdlink_up, pilot),
        active_command=active_command,
        sectors=[
            {
                "sector_id": s["sector_id"],
                "name": s["name"],
                "area_m2": s["area_m2"],
                "status": s["status"],
                "progress": float(s["progress"]),
            }
            for s in sectors
        ],
        pins=_pins(annotations),
        areas=[a for a in annotations if a.kind == "area"],
        paths=[a for a in annotations if a.kind == "path"],
        home=home,
        pilot=pilot,
        alerts=[],
        recent_events=recent_events,
    )
