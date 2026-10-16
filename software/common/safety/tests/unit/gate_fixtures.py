"""Builders for gate tests: a charter around CMAC (ArduPilot's public SITL site), a vehicle
state and a context in which a typical AI order is allowed. Each test changes only what it
tests (``ctx(...)``, ``req(...)``).

The registry here marks the nine foundation skills as implemented, because the production
registry keeps every skill a stub until its executor exists.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from sd_safety.targets import Plane
from sd_schemas.base import LatLon, LatLonAlt, new_id
from sd_schemas.charter import AutonomyMode, MissionCharter, Sector
from sd_schemas.executor import ExecutorState
from sd_schemas.gate import ClearanceState, GateContext, GateRequest, StateRef
from sd_schemas.mission import MissionState, OperatorLatch
from sd_schemas.safety import SafetyLimits
from sd_schemas.skills import SKILL_REGISTRY, SkillRegistry, SkillSpec
from sd_schemas.vehicle import VehicleState

HOME_LAT, HOME_LON, HOME_AMSL = -35.363261, 149.165230, 584.0
PLANE = Plane(HOME_LAT, HOME_LON)
NOW = 1_000_000  # monotonic ms

IMPLEMENTED = (
    "takeoff",
    "goto",
    "survey_sector",
    "hold",
    "resume",
    "return_home",
    "report",
    "mark_location",
    "propose_charter_change",
)
REGISTRY: SkillRegistry = SkillRegistry(
    spec.model_copy(update={"status": "implemented"}) if spec.name in IMPLEMENTED else spec
    for spec in SKILL_REGISTRY
)
MISSION, VEHICLE, CHARTER_ID, CLEARANCE_ID = new_id(), "sim-01", new_id(), new_id()


def latlon(east: float, north: float) -> tuple[float, float]:
    """The exact inverse of the gate's plane around home: (lat, lon) for metres east/north."""
    phi = math.radians(HOME_LAT)
    a, e2 = 6_378_137.0, (1 / 298.257223563) * (2 - 1 / 298.257223563)
    w = math.sqrt(1 - e2 * math.sin(phi) ** 2)
    n_radius, m_radius = a / w, a * (1 - e2) / w**3
    lat = HOME_LAT + math.degrees(north / m_radius)
    lon = HOME_LON + math.degrees(east / (n_radius * math.cos(phi)))
    return lat, lon


def point(east: float, north: float) -> LatLon:
    lat, lon = latlon(east, north)
    return LatLon(lat=lat, lon=lon)


def polygon(*corners: tuple[float, float]) -> dict[str, Any]:
    ring = [[latlon(e, n)[1], latlon(e, n)[0]] for e, n in corners]
    return {"type": "Polygon", "coordinates": [[*ring, ring[0]]]}


# A concave "L": the notch is the square east 100..300, north 100..300.
L_AREA = polygon((-300, -300), (300, -300), (300, 100), (100, 100), (100, 300), (-300, 300))
SQUARE_A = polygon((-250, -250), (-50, -250), (-50, -50), (-250, -50))

HOME = LatLonAlt(lat=HOME_LAT, lon=HOME_LON, alt_amsl_m=HOME_AMSL)
PILOT = LatLon(lat=HOME_LAT, lon=HOME_LON)

CHARTER = MissionCharter(
    charter_id=CHARTER_ID,
    mission_id=MISSION,
    vehicle_id=VEHICLE,
    version=2,
    area=L_AREA,
    home=HOME,
    pilot_position=PILOT,
    allowed_skills=list(IMPLEMENTED),
    autonomy_mode=AutonomyMode.WITHIN_CHARTER,
    sectors=[Sector(sector_id="A", name="A", area=SQUARE_A, order_idx=0)],
    fence_hash="abc",
    approved_by="operator:demo",
    approved_at=datetime(2026, 10, 16, 12, 0, tzinfo=UTC),
)


def vehicle(**update: Any) -> VehicleState:
    """Airborne at 40 m, 50 m north of home, battery 80 %."""
    lat, lon = latlon(0, 50)
    base = VehicleState(
        vehicle_id=VEHICLE,
        mono_ms=NOW,
        fc_connected=True,
        gcs_mavlink_seen_mono_ms=NOW,
        armed=True,
        fc_mode="GUIDED",
        fc_system_status=4,
        landed_state="in_air",
        lat=lat,
        lon=lon,
        alt_rel_home_m=40.0,
        alt_amsl_m=HOME_AMSL + 40,
        groundspeed_ms=0.0,
        climb_ms=0.0,
        heading_deg=0.0,
        battery_pct=80.0,
        battery_charge_state=1,
        battery_voltage_v=24.0,
        gps_fix_type=3,
        gps_satellites=12,
        ekf_ok=True,
        home=HOME,
        mission_seq_current=None,
        pilot_position=PILOT,
        pilot_position_source="charter",
    )
    return base.model_copy(update=update)


ON_GROUND = {
    "armed": False,
    "landed_state": "on_ground",
    "alt_rel_home_m": 0.0,
    "alt_amsl_m": HOME_AMSL,
    "lat": HOME_LAT,
    "lon": HOME_LON,
}


def clearance(**update: Any) -> ClearanceState:
    base = ClearanceState(
        clearance_id=CLEARANCE_ID,
        origin="local",
        mission_id=MISSION,
        vehicle_id=VEHICLE,
        charter_id=CHARTER_ID,
        charter_version=2,
        received_mono_ms=NOW - 10_000,
        valid_for_ms=120_000,
        consumed=False,
    )
    return base.model_copy(update=update)


def latch(action: str = "stop", **update: Any) -> OperatorLatch:
    base = OperatorLatch(
        vehicle_id=VEHICLE,
        action="stop",
        action_seq=1,
        set_by="operator:demo",
        set_at=datetime(2026, 10, 16, 12, 0, tzinfo=UTC),
        set_mono_ms=NOW - 1000,
    )
    return base.model_copy(update={"action": action, **update})


def ctx(**update: Any) -> GateContext:
    """Ground, mission ACTIVE in mode B, everything fresh: a typical AI goto is allowed."""
    base = GateContext(
        side="ground",
        now_mono_ms=NOW,
        registry=REGISTRY.as_mapping(),
        limits=SafetyLimits(),
        charter=CHARTER,
        charter_installed=True,
        fence_verified=True,
        autonomy_mode=AutonomyMode.WITHIN_CHARTER,
        mission_state=MissionState.ACTIVE,
        paused_by=None,
        returning_cause=None,
        ai_enabled=True,
        vehicle=vehicle(),
        vehicle_age_ms=100,
        snapshot_age_ms=100,
        executor_state=ExecutorState.TRANSIT,
        state_epoch=7,
        last_transition_state_seq=40,
        preflight_ok=True,
        sdlink_up=True,
        stop_latch=False,
        latch=None,
        manual_control=False,
        clearance=None,
        recent_brain_actions_mono_ms=(),
        recent_brain_reports_mono_ms=(),
        brain_pending_approvals=0,
        operator_present=True,
    )
    return base.model_copy(update=update)


def onboard_ctx(**update: Any) -> GateContext:
    return ctx(
        side="onboard",
        mission_state=None,
        snapshot_age_ms=0,
        **update,
    )


def ground_ctx(**update: Any) -> GateContext:
    """On the ground, mission READY, with a valid clearance: an AI takeoff is allowed."""
    base: dict[str, Any] = {
        "mission_state": MissionState.READY,
        "vehicle": vehicle(**ON_GROUND),
        "executor_state": ExecutorState.PREFLIGHT,
        "clearance": clearance(),
    }
    return ctx(**{**base, **update})


def goto(
    east: float = 0.0, north: float = 150.0, alt_m: float = 40.0, **extra: Any
) -> dict[str, Any]:
    lat, lon = latlon(east, north)
    return {"lat": lat, "lon": lon, "alt_m": alt_m, **extra}


def req(skill: str = "goto", params: dict[str, Any] | None = None, **update: Any) -> GateRequest:
    base = GateRequest(
        command_id=new_id(),
        mission_id=MISSION,
        vehicle_id=VEHICLE,
        skill=skill,
        params=goto() if params is None and skill == "goto" else (params or {}),
        issuer_kind="brain",
        issuer_ref="brain:ground",
        charter_version=2,
        based_on=StateRef(state_epoch=7, state_seq=41),
        phase="submit",
    )
    return base.model_copy(update=update)


def spec(name: str, **update: Any) -> SkillSpec:
    return REGISTRY.get(name).model_copy(update=update)


def registry_with(*specs: SkillSpec) -> dict[str, SkillSpec]:
    reg = dict(REGISTRY.as_mapping())
    for s in specs:
        reg[s.name] = s
    return reg
