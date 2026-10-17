"""The allowlist is default-deny: only its rows pass, with their number checks."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from mav_helpers import (
    FC,
    HOME,
    all_commands,
    all_message_classes,
    command,
    ctx,
    mission_item,
    offset,
    position_target,
    set_mode,
    zero_message,
)
from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_mav.allowlist import (
    GROUND_STREAMS,
    ONBOARD_STREAMS,
    Profile,
    SendEnvelope,
    check,
)
from sd_mav.denylist import denied_reason
from sd_mav.modes import CopterMode
from sd_mav.vehicle import GROUND_STREAM_US, ONBOARD_STREAM_US, SendOrigin
from sd_safety.limits import DEFAULT_LIMITS
from sd_schemas.safety import LimitCaps, SafetyLimits

ALLOWED_MESSAGES = {
    "HEARTBEAT",
    "COMMAND_LONG",
    "SET_POSITION_TARGET_GLOBAL_INT",
    "MISSION_COUNT",
    "MISSION_ITEM_INT",
    "MISSION_REQUEST_LIST",
    "MISSION_REQUEST_INT",
    "MISSION_ACK",
    "MISSION_SET_CURRENT",
    "PARAM_REQUEST_READ",
    "PARAM_SET",
    "MANUAL_CONTROL",
}
ONBOARD_COMMANDS = {176, 400, 22, 178, 224, 511, 512}
GROUND_COMMANDS = {176, 400, 511}
ALL_ORIGINS = [o.value for o in SendOrigin]
MSG_CLASSES = all_message_classes()
COMMANDS = all_commands()


def sends(msg: Any, c: Any) -> bool:
    return denied_reason(msg) is None and check(msg, c) is None


# --- default deny ------------------------------------------------------------------------


@pytest.mark.parametrize("profile", list(Profile))
def test_send_default_deny_every_message_type(profile: Profile) -> None:
    """Every message type outside the allowlist is refused for every origin."""
    for cls in MSG_CLASSES:
        msg = zero_message(cls)
        for origin in ALL_ORIGINS:
            allowed = sends(msg, ctx(profile, origin))
            if cls.msgname not in ALLOWED_MESSAGES:
                assert not allowed, (cls.msgname, origin)


@settings(max_examples=600, deadline=None)
@given(
    cmd=st.sampled_from(COMMANDS) | st.integers(0, 65535),
    params=st.lists(
        st.floats(allow_nan=True, allow_infinity=True, width=32), min_size=7, max_size=7
    ),
    profile=st.sampled_from(list(Profile)),
    origin=st.sampled_from(ALL_ORIGINS),
)
def test_send_default_deny(cmd: int, params: list[float], profile: Profile, origin: str) -> None:
    """Only the allowlisted commands can ever pass, whatever their parameters."""
    msg = command(cmd, *params)
    if sends(msg, ctx(profile, origin, on_ground=True)):
        allowed = ONBOARD_COMMANDS if profile is Profile.ONBOARD else GROUND_COMMANDS
        assert cmd in allowed


def test_send_default_deny_edges() -> None:
    c = ctx()
    takeoff = mavlink.MAV_CMD_NAV_TAKEOFF
    assert sends(command(takeoff, 0, 0, 0, 0, 0, 0, 120.0), c)
    assert not sends(command(takeoff, 0, 0, 0, 0, 0, 0, 120.1), c)
    assert not sends(command(takeoff, 0, 0, 0, 0, 0, 0, -0.1), c)
    assert not sends(command(takeoff, 0, 0, 0, 0, 0, 0, 0.0), c)
    speed = mavlink.MAV_CMD_DO_CHANGE_SPEED
    assert sends(command(speed, 1, 15.0, -1), c)
    assert not sends(command(speed, 1, 15.1, -1), c)
    assert not sends(command(speed, 1, 0.0, -1), c)
    assert not sends(command(speed, 1, -1.0, -1), c)
    assert not sends(command(speed, 0, 5.0, -1), c)  # airspeed, not ground speed
    assert not sends(command(speed, 1, 5.0, 50), c)  # throttle percent
    lat, lon = HOME
    assert sends(position_target(lat, lon, 120.0), c)
    assert not sends(position_target(lat, lon, 120.1), c)
    assert not sends(position_target(lat, lon, -0.1), c)
    assert sends(position_target(*offset(lat, lon, 529.0), 40), c)
    assert not sends(position_target(*offset(lat, lon, 531.0), 40), c)
    assert not sends(position_target(*offset(lat, lon, 0, 531.0), 40), c)


@settings(max_examples=400, deadline=None)
@given(alt=st.floats(-1000, 1000, allow_nan=False), north=st.floats(-2000, 2000, allow_nan=False))
def test_position_target_numbers(alt: float, north: float) -> None:
    lat, lon = offset(HOME[0], HOME[1], north)
    ok = sends(position_target(lat, lon, alt), ctx())
    if ok:
        assert 0.0 <= alt <= 120.0
        assert abs(north) <= 531.0
    if 0.0 <= alt <= 120.0 and abs(north) <= 528.0:
        assert ok


def test_position_target_needs_frame_mask_home_and_target() -> None:
    lat, lon = HOME
    assert not sends(position_target(lat, lon, 40, frame=0), ctx())  # absolute height
    assert not sends(position_target(lat, lon, 40, frame=11), ctx())  # terrain frame
    assert not sends(position_target(lat, lon, 40, mask=0x0DC7), ctx())  # velocity
    assert not sends(position_target(lat, lon, 40), ctx(home=None))
    msg = position_target(lat, lon, 40)
    msg.target_system = 2
    assert not sends(msg, ctx())
    assert not sends(position_target(lat, lon, 40), ctx(Profile.GROUND, "ground"))
    assert not sends(position_target(lat, lon, 40), ctx(origin="startup"))
    assert sends(position_target(lat, lon, 40), ctx(origin="limit"))


def test_raised_limits_move_the_envelope() -> None:
    limits = SafetyLimits(max_alt_agl_m=200, max_speed_ms=20, max_range_from_pilot_m=1500)
    env = SendEnvelope.from_limits(limits, LimitCaps())
    assert (env.max_alt_agl_m, env.max_speed_ms, env.fence_radius_m) == (200, 20, 1530)
    c = ctx(envelope=env)
    assert sends(command(mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 200.0), c)
    assert not sends(command(mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 200.1), c)
    assert sends(position_target(*offset(HOME[0], HOME[1], 1520.0), 40), c)
    with pytest.raises(ValueError, match="limits not allowed"):
        SendEnvelope.from_limits(SafetyLimits(max_alt_agl_m=500), LimitCaps())
    assert SendEnvelope.from_limits(DEFAULT_LIMITS) == SendEnvelope()


# --- modes -------------------------------------------------------------------------------


def test_set_mode_allowlist() -> None:
    onboard_ok = {
        CopterMode.GUIDED,
        CopterMode.AUTO,
        CopterMode.LOITER,
        CopterMode.SMART_RTL,
        CopterMode.RTL,
    }
    for mode in CopterMode:
        assert sends(set_mode(mode), ctx()) == (mode in onboard_ok), mode
    for name in ("STABILIZE", "ACRO", "FLIP", "THROW", "ALT_HOLD", "SPORT", "DRIFT", "TURTLE"):
        assert not sends(set_mode(CopterMode[name]), ctx())
        assert not sends(set_mode(CopterMode[name]), ctx(origin="override_fallback"))
    assert not sends(set_mode(99), ctx())
    assert not sends(command(mavlink.MAV_CMD_DO_SET_MODE, 0, 4), ctx())  # base mode only
    assert not sends(command(mavlink.MAV_CMD_DO_SET_MODE, 1, 4, 1), ctx())  # sub mode
    assert not sends(command(mavlink.MAV_CMD_DO_SET_MODE, 1, 4.5), ctx())


def test_brake_land_only_from_override() -> None:
    for mode in (CopterMode.BRAKE, CopterMode.LAND):
        for origin in ALL_ORIGINS:
            if origin == "override_fallback":
                assert sends(set_mode(mode), ctx(origin=origin))
            elif origin != "ground":
                assert not sends(set_mode(mode), ctx(origin=origin)), origin
    # The fallback path may only use the STOP and FLY HOME steps.
    for mode in (CopterMode.GUIDED, CopterMode.AUTO):
        assert not sends(set_mode(mode), ctx(origin="override_fallback"))
    for mode in (CopterMode.SMART_RTL, CopterMode.RTL, CopterMode.LOITER):
        assert sends(set_mode(mode), ctx(origin="override_fallback"))


def test_startup_and_manual_origins_cannot_fly() -> None:
    for origin in ("startup", "manual"):
        assert not sends(set_mode(CopterMode.GUIDED), ctx(origin=origin))
        assert not sends(command(mavlink.MAV_CMD_DO_CHANGE_SPEED, 1, 5, -1), ctx(origin=origin))
        assert not sends(command(mavlink.MAV_CMD_DO_SET_MISSION_CURRENT, 1), ctx(origin=origin))
    # The sticks are the ground station's: onboard, "manual" sends nothing at all.
    assert not sends(set_mode(CopterMode.LOITER), ctx(origin="manual"))
    # On the ground station the manual origin only sends sticks; LOITER goes as "ground".
    assert not sends(set_mode(CopterMode.LOITER), ctx(Profile.GROUND, "manual"))
    assert sends(set_mode(CopterMode.LOITER), ctx(Profile.GROUND, "ground"))
    assert not sends(set_mode(CopterMode.GUIDED), ctx(origin="ground"))  # onboard never "ground"
    assert not sends(set_mode(CopterMode.GUIDED), ctx(origin="nobody"))


# --- arming and takeoff ------------------------------------------------------------------


def test_arm_only_from_skill_and_never_disarm_onboard() -> None:
    arm = mavlink.MAV_CMD_COMPONENT_ARM_DISARM
    assert sends(command(arm, 1, 0), ctx())
    for origin in ("limit", "startup", "override_fallback", "manual"):
        assert not sends(command(arm, 1, 0), ctx(origin=origin))
    assert not sends(command(arm, 0, 0), ctx(on_ground=True))
    assert not sends(command(arm, 1, 21196), ctx())
    assert not sends(command(arm, 1, 0, 1), ctx())
    takeoff = mavlink.MAV_CMD_NAV_TAKEOFF
    assert not sends(command(takeoff, 0, 0, 0, 0, 0, 0, 40), ctx(origin="limit"))
    assert not sends(command(takeoff, 0, 0, 0, 0, -35.0, 149.0, 40), ctx())


def test_ground_station_allowlist() -> None:
    g = ctx(Profile.GROUND, "ground")
    for mode in CopterMode:
        expected = mode in {
            CopterMode.BRAKE,
            CopterMode.LOITER,
            CopterMode.LAND,
            CopterMode.SMART_RTL,
            CopterMode.RTL,
        }
        assert sends(set_mode(mode), g) == expected, mode
    arm = mavlink.MAV_CMD_COMPONENT_ARM_DISARM
    assert sends(command(arm, 0, 0), ctx(Profile.GROUND, "ground", on_ground=True))
    assert not sends(command(arm, 0, 0), g)  # not known to stand on the ground
    assert not sends(command(arm, 1, 0), ctx(Profile.GROUND, "ground", on_ground=True))
    assert not sends(command(arm, 0, 21196), ctx(Profile.GROUND, "ground", on_ground=True))
    for cmd in (22, 178, 224, 512):
        assert not sends(command(cmd, 1, 5, -1), g), cmd
    assert not sends(set_mode(CopterMode.LAND), ctx(Profile.GROUND, "skill"))
    hb_gcs = mavlink.MAVLink_heartbeat_message(6, 8, 0, 0, mavlink.MAV_STATE_ACTIVE, 3)
    hb_onb = mavlink.MAVLink_heartbeat_message(18, 8, 0, 0, mavlink.MAV_STATE_ACTIVE, 3)
    assert sends(hb_gcs, g) and not sends(hb_onb, g)
    assert sends(hb_onb, ctx(origin="startup")) and not sends(hb_gcs, ctx(origin="startup"))


def test_targets_must_be_the_autopilot() -> None:
    assert not sends(command(mavlink.MAV_CMD_DO_SET_MODE, 1, 4, target=2), ctx())
    assert not sends(command(mavlink.MAV_CMD_DO_SET_MODE, 1, 4, target=0), ctx())
    assert not sends(command(mavlink.MAV_CMD_DO_SET_MODE, 1, 4, compid=0), ctx())
    assert not sends(command(mavlink.MAV_CMD_DO_SET_MODE, 1, 4, compid=191), ctx())


# --- streams and reads -------------------------------------------------------------------


def test_stream_tables_match_the_allowlist() -> None:
    assert set(ONBOARD_STREAM_US) == ONBOARD_STREAMS
    assert set(GROUND_STREAM_US) == GROUND_STREAMS
    for table, profile in (
        (ONBOARD_STREAM_US, Profile.ONBOARD),
        (GROUND_STREAM_US, Profile.GROUND),
    ):
        origin = "ground" if profile is Profile.GROUND else "startup"
        for msg_id, interval in table.items():
            assert sends(command(511, msg_id, interval), ctx(profile, origin))


def test_message_interval_and_request_limits() -> None:
    c = ctx(origin="startup")
    assert not sends(command(511, 0, 0), c)  # default rate
    assert not sends(command(511, 0, -1), c)  # switch the heartbeat off
    assert not sends(command(511, 0, 5_000), c)  # 200 Hz
    assert not sends(command(511, mavlink.MAVLINK_MSG_ID_PARAM_VALUE, 200_000), c)
    assert sends(command(512, mavlink.MAVLINK_MSG_ID_HOME_POSITION), c)
    assert sends(command(512, mavlink.MAVLINK_MSG_ID_AUTOPILOT_VERSION), c)
    assert not sends(command(512, mavlink.MAVLINK_MSG_ID_PARAM_VALUE), c)


def test_param_read_by_name_only() -> None:
    read = mavlink.MAVLink_param_request_read_message(FC, 1, b"FENCE_RADIUS", -1)
    assert sends(read, ctx(origin="startup"))
    assert sends(read, ctx(Profile.GROUND, "ground"))
    by_index = mavlink.MAVLink_param_request_read_message(FC, 1, b"", 3)
    assert not sends(by_index, ctx(origin="startup"))


# --- missions and the fence --------------------------------------------------------------


def _wp(seq: int, lat: float, lon: float, z: float, frame: int = 3) -> Any:
    return mission_item(seq, 16, frame, lat=lat, lon=lon, z=z)


def test_mission_item_allowlist() -> None:
    c = ctx()
    lat, lon = HOME
    assert sends(mission_item(0, 16, 0, lat=lat, lon=lon, z=584.0), c)  # home slot
    assert sends(_wp(2, lat, lon, 120.0), c)
    assert sends(_wp(2, lat, lon, 120.0, frame=6), c)
    assert not sends(_wp(2, lat, lon, 120.1), c)
    assert not sends(_wp(2, lat, lon, -0.1), c)
    assert not sends(_wp(2, lat, lon, 40, frame=0), c)  # absolute height after item 0
    assert not sends(_wp(2, lat, lon, 40, frame=10), c)  # terrain
    assert sends(_wp(2, *offset(lat, lon, 529.0), 40), c)
    assert not sends(_wp(2, *offset(lat, lon, 531.0), 40), c)
    assert not sends(_wp(2, lat, lon, 40), ctx(home=None))
    assert not sends(mission_item(2, 16, 3, 5, lat=lat, lon=lon, z=40), c)  # hold time
    assert sends(mission_item(1, 178, 2, 1, 15.0, -1), c)
    assert not sends(mission_item(1, 178, 2, 1, 15.1, -1), c)
    assert not sends(mission_item(1, 178, 2, 1, 0.0, -1), c)
    assert not sends(mission_item(1, 178, 2, 0, 5.0, -1), c)
    assert not sends(mission_item(1, 178, 2, 1, 5.0, 40), c)
    assert sends(mission_item(5, 19, 3, 30, lat=lat, lon=lon, z=40), c)
    assert not sends(mission_item(5, 19, 3, 30, lat=lat, lon=lon, z=121), c)
    assert not sends(mission_item(5, 19, 3, 30, lat=-35.0, lon=149.0, z=40), c)
    assert sends(mission_item(6, 20, 2), c)
    assert not sends(mission_item(6, 20, 2, 1), c)
    assert not sends(mission_item(0, 20, 2), c)
    for cmd in (21, 22, 31, 177, 179, 181, 183, 185, 207, 208, 209, 218, 5001, 5003):
        assert not sends(mission_item(3, cmd, 3, 1, lat=lat, lon=lon, z=40), c), cmd
    assert not sends(_wp(2, lat, lon, 40), ctx(origin="startup"))
    assert not sends(_wp(2, lat, lon, 40), ctx(Profile.GROUND, "ground"))


@settings(max_examples=400, deadline=None)
@given(z=st.floats(-500, 500, allow_nan=False), speed=st.floats(-50, 50, allow_nan=False))
def test_mission_item_numbers(z: float, speed: float) -> None:
    lat, lon = HOME
    if sends(_wp(3, lat, lon, z), ctx()):
        assert 0.0 <= z <= 120.0
    if sends(mission_item(1, 178, 2, 1, speed, -1), ctx()):
        assert 0.0 < speed <= 15.0


def test_mission_count_and_read_back() -> None:
    count = mavlink.MAVLink_mission_count_message
    assert sends(count(FC, 1, 10, 0), ctx())
    assert not sends(count(FC, 1, 10, 1), ctx())  # the onboard computer never uploads a fence
    assert not sends(count(FC, 1, 10, 2), ctx())  # rally points
    assert not sends(count(FC, 1, 0, 0), ctx())  # an empty upload clears the mission
    assert not sends(count(FC, 1, 401, 0), ctx())
    req_list = mavlink.MAVLink_mission_request_list_message
    assert sends(req_list(FC, 1, 0), ctx(origin="startup"))
    assert sends(req_list(FC, 1, 1), ctx(origin="startup"))  # read the fence back
    assert not sends(req_list(FC, 1, 2), ctx(origin="startup"))
    ack = mavlink.MAVLink_mission_ack_message
    assert sends(ack(FC, 1, 0, 0), ctx())
    assert not sends(ack(FC, 1, 0, 2), ctx())
    assert sends(mavlink.MAVLink_mission_set_current_message(FC, 1, 1), ctx())
    assert not sends(mavlink.MAVLink_mission_set_current_message(FC, 1, 1), ctx(origin="startup"))


def test_ground_fence_upload() -> None:
    g = ctx(Profile.GROUND, "ground", on_ground=True, armed=False)
    lat, lon = HOME
    count = mavlink.MAVLink_mission_count_message
    assert sends(count(FC, 1, 4, 1), g)
    assert not sends(count(FC, 1, 4, 0), g)
    assert not sends(count(FC, 1, 71, 1), g)
    assert not sends(count(FC, 1, 2, 1), g)
    vertex = mission_item(0, 5001, 0, 4, lat=lat, lon=lon, mission_type=1)
    assert sends(vertex, g)
    assert not sends(
        mission_item(0, 5001, 0, 4, lat=lat, lon=lon, mission_type=1),
        ctx(Profile.GROUND, "ground", home=None),
    )
    assert not sends(mission_item(0, 5002, 0, 4, lat=lat, lon=lon, mission_type=1), g)  # exclusion
    assert not sends(mission_item(0, 5003, 0, 400, lat=lat, lon=lon, mission_type=1), g)  # circle
    assert not sends(mission_item(0, 5001, 0, 4, lat=lat, lon=lon, mission_type=0), g)
    assert not sends(mission_item(0, 5001, 0, 4, lat=lat, lon=lon, mission_type=2), g)
    far = offset(lat, lon, 600.0)
    assert not sends(mission_item(0, 5001, 0, 4, lat=far[0], lon=far[1], mission_type=1), g)
    req_list = mavlink.MAVLink_mission_request_list_message
    assert sends(req_list(FC, 1, 1), g)
    assert not sends(req_list(FC, 1, 0), g)
    # A new fence replaces the old one at once: never armed, never in the air.
    for state in (
        ctx(Profile.GROUND, "ground", on_ground=True, armed=True),
        ctx(Profile.GROUND, "ground", on_ground=True, armed=None),
        ctx(Profile.GROUND, "ground", armed=True, airborne=True),
        ctx(Profile.GROUND, "ground", armed=False),
    ):
        assert not sends(count(FC, 1, 4, 1), state)
        assert not sends(vertex, state)


# --- manual control (the pilot's sticks, LOITER only) -------------------------------------


def _sticks(x: int = 0, y: int = 0, z: int = 500, r: int = 0, buttons: int = 0) -> Any:
    return mavlink.MAVLink_manual_control_message(FC, x, y, z, r, buttons)


def test_manual_control_only_sticks_in_loiter() -> None:
    def gcs(**kw: Any) -> Any:
        return ctx(Profile.GROUND, "manual", **kw)

    loiter = gcs(fc_mode=CopterMode.LOITER, armed=True, airborne=True)
    assert sends(_sticks(), loiter)
    assert sends(_sticks(1000, -1000, 1000, 1000), loiter)
    assert sends(_sticks(z=0), loiter)
    assert not sends(_sticks(1001), loiter)
    assert not sends(_sticks(z=-1), loiter)
    assert not sends(_sticks(buttons=1), loiter)
    ext = _sticks()
    ext.enabled_extensions = 1
    assert not sends(ext, loiter)
    other = _sticks()
    other.target = 2
    assert not sends(other, loiter)
    for mode in CopterMode:
        if mode != CopterMode.LOITER:
            assert not sends(_sticks(), gcs(fc_mode=mode, armed=True, airborne=True)), mode
    assert not sends(_sticks(), gcs(fc_mode=None, armed=True, airborne=True))
    # On the ground (or disarmed) the throttle-down plus full-yaw gesture would arm or
    # disarm the motors, so no sticks there at all.
    for state in (
        gcs(fc_mode=CopterMode.LOITER, armed=True, on_ground=True),
        gcs(fc_mode=CopterMode.LOITER, armed=False),
        gcs(fc_mode=CopterMode.LOITER, armed=None),
    ):
        assert not sends(_sticks(z=0, r=1000), state)
        assert not sends(_sticks(z=0, r=-1000), state)
        assert not sends(_sticks(), state)
    for profile in Profile:
        for origin in ALL_ORIGINS:
            if (profile, origin) != (Profile.GROUND, "manual"):
                c = ctx(profile, origin, fc_mode=CopterMode.LOITER, armed=True, airborne=True)
                assert not sends(_sticks(), c), (profile, origin)
    rc = zero_message(mavlink.MAVLink_rc_channels_override_message)
    assert not sends(rc, loiter)


def test_malformed_message_is_refused() -> None:
    msg = set_mode(CopterMode.GUIDED)
    msg.param2 = "GUIDED"  # type: ignore[assignment]
    assert check(msg, ctx()) is not None


def test_origin_strings_match_send_origin() -> None:
    from sd_mav import allowlist

    assert {o.value for o in SendOrigin} == allowlist._ONBOARD_ORIGINS | allowlist._GROUND_ORIGINS
    assert not allowlist._ONBOARD_ORIGINS & allowlist._GROUND_ORIGINS


def test_context_and_envelope_sanity() -> None:
    with pytest.raises(ValueError):
        ctx(on_ground=True, airborne=True)
    for bad in (float("inf"), float("nan"), 0.0, -1.0):
        with pytest.raises(ValueError):
            SendEnvelope(max_alt_agl_m=bad)


# --- the fence's limits (owner decision: raised limits move the fence) ---------------


def fence_limit(name: str, value: float, ptype: int = mavlink.MAV_PARAM_TYPE_REAL32) -> Any:
    return mavlink.MAVLink_param_set_message(FC, 1, name.encode(), value, ptype)


GROUND_STILL = {"on_ground": True, "armed": False, "fc_mode": CopterMode.LOITER}


@pytest.mark.parametrize(
    ("name", "value"),
    [("FENCE_ALT_MAX", 125), ("FENCE_ALT_MAX", 305), ("FENCE_RADIUS", 2030), ("LOIT_SPEED_MS", 25)],
)
def test_ground_moves_the_fence_outward_on_the_ground(name: str, value: float) -> None:
    assert sends(fence_limit(name, value), ctx(Profile.GROUND, "ground", **GROUND_STILL))


@pytest.mark.parametrize(
    ("msg", "c"),
    [
        # never from the onboard computer, whatever its origin
        (
            fence_limit("FENCE_ALT_MAX", 200),
            ctx(Profile.ONBOARD, "skill", on_ground=True, armed=False),
        ),
        (
            fence_limit("FENCE_ALT_MAX", 200),
            ctx(Profile.ONBOARD, "startup", on_ground=True, armed=False),
        ),
        # never in the air or armed
        (
            fence_limit("FENCE_ALT_MAX", 200),
            ctx(Profile.GROUND, "ground", armed=True, airborne=True),
        ),
        (
            fence_limit("FENCE_ALT_MAX", 200),
            ctx(Profile.GROUND, "ground", on_ground=True, armed=True),
        ),
        # never below the standard fence or beyond what the flight controller holds
        (fence_limit("FENCE_ALT_MAX", 100), ctx(Profile.GROUND, "ground", **GROUND_STILL)),
        (fence_limit("FENCE_RADIUS", 500), ctx(Profile.GROUND, "ground", **GROUND_STILL)),
        (fence_limit("LOIT_SPEED_MS", 10), ctx(Profile.GROUND, "ground", **GROUND_STILL)),
        (fence_limit("FENCE_RADIUS", 20_000), ctx(Profile.GROUND, "ground", **GROUND_STILL)),
        (fence_limit("FENCE_ALT_MAX", float("nan")), ctx(Profile.GROUND, "ground", **GROUND_STILL)),
        (fence_limit("FENCE_ALT_MAX", float("inf")), ctx(Profile.GROUND, "ground", **GROUND_STILL)),
        # only the three, only as floats, only the ground's own origin
        (fence_limit("FENCE_ENABLE", 0), ctx(Profile.GROUND, "ground", **GROUND_STILL)),
        (fence_limit("FENCE_ACTION", 0), ctx(Profile.GROUND, "ground", **GROUND_STILL)),
        (fence_limit("FS_GCS_ENABLE", 0), ctx(Profile.GROUND, "ground", **GROUND_STILL)),
        (fence_limit("ARMING_SKIPCHK", 1), ctx(Profile.GROUND, "ground", **GROUND_STILL)),
        (
            fence_limit("FENCE_ALT_MAX", 200, mavlink.MAV_PARAM_TYPE_INT32),
            ctx(Profile.GROUND, "ground", **GROUND_STILL),
        ),
        (fence_limit("FENCE_ALT_MAX", 200), ctx(Profile.GROUND, "manual", **GROUND_STILL)),
    ],
)
def test_fence_limit_refused(msg: Any, c: Any) -> None:
    assert not sends(msg, c)


def test_every_other_parameter_stays_on_the_deny_list() -> None:
    for name in ("FENCE_ENABLE", "FENCE_TYPE", "FENCE_ACTION", "FS_GCS_ENABLE", "BATT_FS_LOW_ACT"):
        assert denied_reason(fence_limit(name, 1)) is not None
    for name in ("FENCE_ALT_MAX", "FENCE_RADIUS", "LOIT_SPEED_MS"):
        assert denied_reason(fence_limit(name, 1)) is None
