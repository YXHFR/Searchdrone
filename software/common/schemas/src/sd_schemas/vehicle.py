"""What the onboard computer knows about its flight controller."""

from __future__ import annotations

from typing import Literal

from sd_schemas.base import LatLon, LatLonAlt, SdModel, VehicleId

__all__ = ["LandedState", "VehicleState"]

LandedState = Literal["on_ground", "in_air", "takeoff", "landing", "unknown"]


class VehicleState(SdModel):
    vehicle_id: VehicleId
    mono_ms: int
    fc_connected: bool  # HEARTBEAT from the flight controller within its timeout
    gcs_mavlink_seen_mono_ms: int | None  # last HEARTBEAT from the ground station
    armed: bool
    fc_mode: str  # ArduCopter mode name, e.g. "LOITER" (sd_mav.modes.CopterMode)
    fc_system_status: int  # MAV_STATE
    landed_state: LandedState
    lat: float
    lon: float
    alt_rel_home_m: float  # height above home (= above ground in v1, flat world)
    alt_amsl_m: float
    groundspeed_ms: float
    climb_ms: float
    heading_deg: float
    battery_pct: float | None  # None when the flight controller reports -1
    battery_charge_state: int | None  # MAV_BATTERY_CHARGE_STATE
    battery_voltage_v: float | None
    gps_fix_type: int
    gps_satellites: int
    ekf_ok: bool
    home: LatLonAlt | None
    mission_seq_current: int | None
    pilot_position: LatLon  # effective pilot position
    pilot_position_source: Literal["charter", "ground_gnss", "phone"]
