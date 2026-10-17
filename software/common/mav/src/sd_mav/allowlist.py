"""The allowlist: the only MAVLink traffic Searchdrone may send to the flight controller.

``check(msg, ctx)`` is default-deny. A message passes only when its type, its command,
the listed parameters and the number checks match a row below for the sender's
profile (onboard computer or ground station) and origin. Everything else gives a
reason string, and ``Vehicle.send`` refuses to send it. No row writes a parameter, so no
code path in Searchdrone can change a flight-controller parameter or switch off the
geofence.

Onboard (sysid 1, compid 191):

=====================================  ==================================================
What                                   Allowed
=====================================  ==================================================
``HEARTBEAT``                          onboard controller, autopilot invalid, all zero
``DO_SET_MODE`` (176)                  ``param1 = 1``; GUIDED, AUTO, LOITER, SMART_RTL,
                                       RTL; BRAKE and LAND only from the fallback path
``COMPONENT_ARM_DISARM`` (400)         arm only (``param1 = 1``, ``param2 = 0``), skills
``NAV_TAKEOFF`` (22)                   ``0 < param7 <= max height``, skills
``SET_POSITION_TARGET_GLOBAL_INT``     frame 6, type mask 0x0DF8, ``0 <= alt <= max``,
                                       within the fence radius from home
``DO_CHANGE_SPEED`` (178)              ``param1 = 1``, ``0 < param2 <= max``, ``param3 = -1``
``MISSION_COUNT``, ``MISSION_ITEM_INT``  mission type 0, commands 16, 178, 19, 20 with the
                                       same number checks
``MISSION_REQUEST_LIST``/``_INT``,     mission type 0 or 1 (1 = read the fence back)
``MISSION_ACK``
``DO_SET_MISSION_CURRENT`` (224),      a mission index
``MISSION_SET_CURRENT``
``SET_MESSAGE_INTERVAL`` (511)         the telemetry streams, 1-100 Hz
``REQUEST_MESSAGE`` (512)              ``HOME_POSITION``, ``AUTOPILOT_VERSION``
``PARAM_REQUEST_READ``                 read one parameter by name
=====================================  ==================================================

Ground station (sysid 250, compid 190): ``HEARTBEAT`` as a GCS, ``DO_SET_MODE`` to the
STOP and FLY HOME steps, disarm only when standing on the ground, ``SET_MESSAGE_INTERVAL``,
``PARAM_REQUEST_READ``, the fence protocol (mission type 1, polygon inclusion vertices
only, only while the vehicle stands disarmed on the ground) and ``MANUAL_CONTROL``: the
pilot's sticks (origin ``manual``), only while the flight controller is in LOITER and
the vehicle armed in the air, no buttons. ArduCopter takes sticks only from its GCS
system id, so they never come from the onboard computer.

Origins (``sd_mav.vehicle.SendOrigin``): ``startup`` may only read and ask for streams;
``skill`` and ``limit`` may fly; ``override_fallback`` may only change to the STOP and
FLY HOME modes; ``ground`` is the ground station's origin for everything except the
sticks, which only the ground station's ``manual`` origin sends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_mav.modes import (
    HOME_MODES,
    ONBOARD_MODES,
    STOP_MODES,
    CopterMode,
)
from sd_safety.constants import MISSION_MAX_ITEMS
from sd_safety.limits import DEFAULT_CAPS, DEFAULT_LIMITS, check_limits, fc_fence_params
from sd_safety.targets import distance_m
from sd_schemas.safety import LimitCaps, SafetyLimits

__all__ = [
    "FC_COMPID",
    "FENCE_MAX_VERTICES",
    "GROUND_STREAMS",
    "ONBOARD_STREAMS",
    "POSITION_TYPE_MASK",
    "Profile",
    "SendContext",
    "SendEnvelope",
    "check",
]

FC_COMPID: Final = mavlink.MAV_COMP_ID_AUTOPILOT1  # 1
POSITION_TYPE_MASK: Final = 0x0DF8  # position only: ignore velocity, acceleration and yaw
FENCE_MAX_VERTICES: Final = 70
_GUIDED_FRAME: Final = mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT  # 6
_REL_ALT_FRAMES: Final = frozenset(
    {mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT}
)
_FENCE_FRAMES: Final = frozenset({mavlink.MAV_FRAME_GLOBAL, mavlink.MAV_FRAME_GLOBAL_INT})
_MIN_INTERVAL_US: Final = 10_000  # 100 Hz
_MAX_INTERVAL_US: Final = 1_000_000  # 1 Hz

# The streams each side asks for with SET_MESSAGE_INTERVAL.
ONBOARD_STREAMS: Final = frozenset(
    {
        mavlink.MAVLINK_MSG_ID_HEARTBEAT,
        mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
        mavlink.MAVLINK_MSG_ID_ATTITUDE,
        mavlink.MAVLINK_MSG_ID_VFR_HUD,
        mavlink.MAVLINK_MSG_ID_SYS_STATUS,
        mavlink.MAVLINK_MSG_ID_BATTERY_STATUS,
        mavlink.MAVLINK_MSG_ID_GPS_RAW_INT,
        mavlink.MAVLINK_MSG_ID_EKF_STATUS_REPORT,
        mavlink.MAVLINK_MSG_ID_EXTENDED_SYS_STATE,
        mavlink.MAVLINK_MSG_ID_MISSION_CURRENT,
        mavlink.MAVLINK_MSG_ID_HOME_POSITION,
    }
)
GROUND_STREAMS: Final = frozenset(
    {
        mavlink.MAVLINK_MSG_ID_HEARTBEAT,
        mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
        mavlink.MAVLINK_MSG_ID_ATTITUDE,
        mavlink.MAVLINK_MSG_ID_VFR_HUD,
        mavlink.MAVLINK_MSG_ID_SYS_STATUS,
        mavlink.MAVLINK_MSG_ID_BATTERY_STATUS,
        mavlink.MAVLINK_MSG_ID_GPS_RAW_INT,
        mavlink.MAVLINK_MSG_ID_EXTENDED_SYS_STATE,
        mavlink.MAVLINK_MSG_ID_HOME_POSITION,
    }
)
_REQUESTABLE: Final = frozenset(
    {mavlink.MAVLINK_MSG_ID_HOME_POSITION, mavlink.MAVLINK_MSG_ID_AUTOPILOT_VERSION}
)
_MISSION_COMMANDS: Final = frozenset(
    {
        mavlink.MAV_CMD_NAV_WAYPOINT,  # 16
        mavlink.MAV_CMD_DO_CHANGE_SPEED,  # 178
        mavlink.MAV_CMD_NAV_LOITER_TIME,  # 19
        mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,  # 20
    }
)
_GROUND_MODES: Final = STOP_MODES | HOME_MODES
_FALLBACK_ONLY_MODES: Final = frozenset({CopterMode.BRAKE, CopterMode.LAND})
_STICK_MAX: Final = 1000

# Origins, as the values of sd_mav.vehicle.SendOrigin.
_SKILL: Final = "skill"
_LIMIT: Final = "limit"
_STARTUP: Final = "startup"
_FALLBACK: Final = "override_fallback"
_MANUAL: Final = "manual"
_GROUND: Final = "ground"
_FLYING_ORIGINS: Final = frozenset({_SKILL, _LIMIT})
_ONBOARD_ORIGINS: Final = frozenset({_SKILL, _LIMIT, _STARTUP, _FALLBACK})
_GROUND_ORIGINS: Final = frozenset({_GROUND, _MANUAL})


class Profile(StrEnum):
    ONBOARD = "onboard"
    GROUND = "ground"


@dataclass(frozen=True, slots=True)
class SendEnvelope:
    """The number limits the allowlist checks against.

    Built from the mission's limits: the height and speed limits themselves, and the
    flight controller's fence radius for them (530 m with the standard limits). The
    defaults are the standard limits.
    """

    max_alt_agl_m: float = DEFAULT_LIMITS.max_alt_agl_m
    max_speed_ms: float = DEFAULT_LIMITS.max_speed_ms
    fence_radius_m: float = fc_fence_params(DEFAULT_LIMITS)["FENCE_RADIUS"]

    def __post_init__(self) -> None:
        values = (self.max_alt_agl_m, self.max_speed_ms, self.fence_radius_m)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError(f"envelope values must be finite and positive: {values}")

    @classmethod
    def from_limits(cls, limits: SafetyLimits, caps: LimitCaps = DEFAULT_CAPS) -> SendEnvelope:
        """The envelope for ``limits``; ``ValueError`` if they break ``caps``."""
        problems = check_limits(limits, caps)
        if problems:
            raise ValueError(f"limits not allowed: {[p.field for p in problems]}")
        return cls(
            max_alt_agl_m=limits.max_alt_agl_m,
            max_speed_ms=limits.max_speed_ms,
            fence_radius_m=fc_fence_params(limits)["FENCE_RADIUS"],
        )


@dataclass(frozen=True, slots=True)
class SendContext:
    """What the sender knows at the moment a message goes out."""

    profile: Profile
    origin: str
    target_system: int
    envelope: SendEnvelope = field(default_factory=SendEnvelope)
    home: tuple[float, float] | None = None  # last HOME_POSITION (lat, lon), degrees
    fc_mode: int | None = None  # custom_mode in a fresh HEARTBEAT, None when stale
    on_ground: bool = False  # fresh telemetry: landed, armed or not, at most 0.5 m up
    armed: bool | None = None  # from a fresh HEARTBEAT, None when stale
    airborne: bool = False  # fresh: armed and in the air, taking off or landing

    def __post_init__(self) -> None:
        if self.on_ground and self.airborne:
            raise ValueError("a vehicle cannot be on the ground and in the air")


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(v)) for v in values)


def _int(value: float) -> int | None:
    v = float(value)
    if not math.isfinite(v) or v != int(v):
        return None
    return int(v)


def _within_fence(lat_int: int, lon_int: int, ctx: SendContext) -> str | None:
    if ctx.home is None:
        return "no home position known"
    lat, lon = lat_int / 1e7, lon_int / 1e7
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return "position out of range"
    d = distance_m(ctx.home[0], ctx.home[1], lat, lon)
    if not d <= ctx.envelope.fence_radius_m:
        return f"position {d:.1f} m from home, limit {ctx.envelope.fence_radius_m:.1f} m"
    return None


def _alt_ok(alt: float, ctx: SendContext) -> str | None:
    if not _finite(alt) or not 0.0 <= alt <= ctx.envelope.max_alt_agl_m:
        return f"height {alt} outside 0..{ctx.envelope.max_alt_agl_m} m"
    return None


def _speed_ok(p1: float, p2: float, p3: float, p4: float, ctx: SendContext) -> str | None:
    if not _finite(p1, p2, p3, p4):
        return "speed parameters not finite"
    if p1 != 1 or p3 != -1 or p4 != 0:
        return "speed command needs param1 = 1, param3 = -1, param4 = 0"
    if not 0.0 < p2 <= ctx.envelope.max_speed_ms:
        return f"speed {p2} outside 0..{ctx.envelope.max_speed_ms} m/s"
    return None


def _target_ok(msg: Any, ctx: SendContext) -> str | None:
    if int(msg.target_system) != ctx.target_system:
        return f"target system {int(msg.target_system)} is not the flight controller"
    if int(msg.target_component) != FC_COMPID:
        return f"target component {int(msg.target_component)} is not the autopilot"
    return None


def _origin_for(kind: str, ctx: SendContext) -> str | None:
    """``kind``: ``read`` (asks for data), ``fly`` (moves the vehicle)."""
    if ctx.profile is Profile.GROUND:
        return None if ctx.origin == _GROUND else f"origin {ctx.origin} on the ground station"
    if ctx.origin not in _ONBOARD_ORIGINS:
        return f"origin {ctx.origin} onboard"
    if kind == "fly" and ctx.origin not in _FLYING_ORIGINS:
        return f"origin {ctx.origin} may not fly the vehicle"
    return None


# --- COMMAND_LONG -------------------------------------------------------------------------


def _set_mode(p: list[float], ctx: SendContext) -> str | None:
    if p[0] != 1 or any(v != 0 for v in p[2:]):
        return "DO_SET_MODE needs param1 = 1 and param3..7 = 0"
    mode = _int(p[1])
    if mode is None:
        return "mode is not an integer"
    if ctx.profile is Profile.GROUND:
        if mode not in _GROUND_MODES:
            return f"mode {mode} is not a STOP or FLY HOME step"
        return _origin_for("fly", ctx)
    if mode not in ONBOARD_MODES:
        return f"mode {mode} is not allowed onboard"
    if ctx.origin == _FALLBACK:
        return None if mode in _GROUND_MODES else f"mode {mode} is not a fallback step"
    if mode in _FALLBACK_ONLY_MODES:
        return f"mode {mode} only from the emergency fallback"
    return _origin_for("fly", ctx)


def _arm_disarm(p: list[float], ctx: SendContext) -> str | None:
    if any(v != 0 for v in p[1:]):
        return "ARM_DISARM needs param2..7 = 0"
    if ctx.profile is Profile.GROUND:
        if p[0] != 0:
            return "the ground station may only disarm"
        if not ctx.on_ground:
            return "disarm only when standing on the ground"
        return _origin_for("fly", ctx)
    if p[0] != 1:
        return "the onboard computer never disarms"
    if ctx.origin != _SKILL:
        return "arming only from the takeoff skill"
    return None


def _takeoff(p: list[float], ctx: SendContext) -> str | None:
    if ctx.profile is Profile.GROUND:
        return "the ground station never takes off"
    if any(v != 0 for v in p[:6]):
        return "NAV_TAKEOFF needs param1..6 = 0"
    if not _finite(p[6]) or not 0.0 < p[6] <= ctx.envelope.max_alt_agl_m:
        return f"takeoff height {p[6]} outside 0..{ctx.envelope.max_alt_agl_m} m"
    return None if ctx.origin == _SKILL else "takeoff only from the takeoff skill"


def _change_speed(p: list[float], ctx: SendContext) -> str | None:
    if ctx.profile is Profile.GROUND:
        return "the ground station never sets the speed"
    if any(v != 0 for v in p[4:]):
        return "DO_CHANGE_SPEED needs param5..7 = 0"
    return _speed_ok(p[0], p[1], p[2], p[3], ctx) or _origin_for("fly", ctx)


def _mission_current(p: list[float], ctx: SendContext) -> str | None:
    if ctx.profile is Profile.GROUND:
        return "the ground station never changes the mission"
    seq = _int(p[0])
    if seq is None or not 0 <= seq < MISSION_MAX_ITEMS or any(v != 0 for v in p[1:]):
        return "DO_SET_MISSION_CURRENT needs a mission index and param2..7 = 0"
    return _origin_for("fly", ctx)


def _message_interval(p: list[float], ctx: SendContext) -> str | None:
    msg_id = _int(p[0])
    streams = GROUND_STREAMS if ctx.profile is Profile.GROUND else ONBOARD_STREAMS
    if msg_id is None or msg_id not in streams:
        return f"stream {p[0]} not allowed"
    interval = _int(p[1])
    if interval is None or not _MIN_INTERVAL_US <= interval <= _MAX_INTERVAL_US:
        return f"interval {p[1]} us outside {_MIN_INTERVAL_US}..{_MAX_INTERVAL_US}"
    if any(v != 0 for v in p[2:]):
        return "SET_MESSAGE_INTERVAL needs param3..7 = 0"
    return _origin_for("read", ctx)


def _request_message(p: list[float], ctx: SendContext) -> str | None:
    if ctx.profile is Profile.GROUND:
        return "the ground station asks for streams instead"
    msg_id = _int(p[0])
    if msg_id is None or msg_id not in _REQUESTABLE or any(v != 0 for v in p[1:]):
        return f"REQUEST_MESSAGE {p[0]} not allowed"
    return _origin_for("read", ctx)


_COMMANDS: Final = {
    mavlink.MAV_CMD_DO_SET_MODE: _set_mode,
    mavlink.MAV_CMD_COMPONENT_ARM_DISARM: _arm_disarm,
    mavlink.MAV_CMD_NAV_TAKEOFF: _takeoff,
    mavlink.MAV_CMD_DO_CHANGE_SPEED: _change_speed,
    mavlink.MAV_CMD_DO_SET_MISSION_CURRENT: _mission_current,
    mavlink.MAV_CMD_SET_MESSAGE_INTERVAL: _message_interval,
    mavlink.MAV_CMD_REQUEST_MESSAGE: _request_message,
}


def _command_long(msg: Any, ctx: SendContext) -> str | None:
    bad = _target_ok(msg, ctx)
    if bad:
        return bad
    rule = _COMMANDS.get(int(msg.command))
    if rule is None:
        return f"command {int(msg.command)} not allowed"
    p = [float(getattr(msg, f"param{i}")) for i in range(1, 8)]
    if not _finite(*p):
        return "command parameters not finite"
    return rule(p, ctx)


# --- other messages -----------------------------------------------------------------------


def _heartbeat(msg: Any, ctx: SendContext) -> str | None:
    want_type = (
        mavlink.MAV_TYPE_GCS
        if ctx.profile is Profile.GROUND
        else mavlink.MAV_TYPE_ONBOARD_CONTROLLER
    )
    if (
        int(msg.type) != want_type
        or int(msg.autopilot) != mavlink.MAV_AUTOPILOT_INVALID
        or int(msg.base_mode) != 0
        or int(msg.custom_mode) != 0
        or int(msg.system_status) != mavlink.MAV_STATE_ACTIVE
    ):
        return "HEARTBEAT fields do not match this sender"
    return _origin_for("read", ctx)


def _position_target(msg: Any, ctx: SendContext) -> str | None:
    if ctx.profile is Profile.GROUND:
        return "the ground station never steers"
    bad = _target_ok(msg, ctx)
    if bad:
        return bad
    if int(msg.coordinate_frame) != _GUIDED_FRAME:
        return f"frame {int(msg.coordinate_frame)} not allowed"
    if int(msg.type_mask) != POSITION_TYPE_MASK:
        return f"type mask {int(msg.type_mask):#06x} not allowed"
    return (
        _alt_ok(float(msg.alt), ctx)
        or _within_fence(int(msg.lat_int), int(msg.lon_int), ctx)
        or _origin_for("fly", ctx)
    )


def _fence_ground_state(ctx: SendContext) -> str | None:
    # A new fence replaces the old one at once; only on the ground and disarmed.
    if not ctx.on_ground or ctx.armed is not False:
        return "the fence is uploaded only when the vehicle stands disarmed on the ground"
    return None


def _mission_count(msg: Any, ctx: SendContext) -> str | None:
    bad = _target_ok(msg, ctx)
    if bad:
        return bad
    mtype, count = int(msg.mission_type), int(msg.count)
    if ctx.profile is Profile.GROUND:
        if mtype != mavlink.MAV_MISSION_TYPE_FENCE:
            return f"the ground station uploads only the fence, not type {mtype}"
        if not 3 <= count <= FENCE_MAX_VERTICES:
            return f"fence with {count} vertices"
        return _fence_ground_state(ctx) or _origin_for("fly", ctx)
    if mtype != mavlink.MAV_MISSION_TYPE_MISSION:
        return f"the onboard computer uploads only missions, not type {mtype}"
    if not 2 <= count <= MISSION_MAX_ITEMS:
        return f"mission with {count} items"
    return _origin_for("fly", ctx)


def _mission_item(msg: Any, ctx: SendContext) -> str | None:
    bad = _target_ok(msg, ctx)
    if bad:
        return bad
    mtype, cmd, frame, seq = (
        int(msg.mission_type),
        int(msg.command),
        int(msg.frame),
        int(msg.seq),
    )
    params = [float(msg.param1), float(msg.param2), float(msg.param3), float(msg.param4)]
    if not _finite(*params, float(msg.z)):
        return "mission item values not finite"
    if ctx.profile is Profile.GROUND:
        if mtype != mavlink.MAV_MISSION_TYPE_FENCE:
            return f"the ground station uploads only the fence, not type {mtype}"
        if cmd != mavlink.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_INCLUSION or frame not in _FENCE_FRAMES:
            return f"fence item command {cmd} frame {frame} not allowed"
        vertices = _int(params[0])
        if vertices is None or not 3 <= vertices <= FENCE_MAX_VERTICES:
            return "fence vertex count out of range"
        if any(v != 0 for v in params[1:]) or not 0 <= seq < FENCE_MAX_VERTICES:
            return "fence item needs param2..4 = 0 and a valid index"
        return (
            _fence_ground_state(ctx)
            or _within_fence(int(msg.x), int(msg.y), ctx)
            or _origin_for("fly", ctx)
        )
    if mtype != mavlink.MAV_MISSION_TYPE_MISSION:
        return f"the onboard computer uploads only missions, not type {mtype}"
    if cmd not in _MISSION_COMMANDS:
        return f"mission command {cmd} not allowed"
    if not 0 <= seq < MISSION_MAX_ITEMS:
        return f"mission index {seq} out of range"
    bad = _mission_item_numbers(cmd, frame, seq, params, msg, ctx)
    return bad or _origin_for("fly", ctx)


def _mission_item_numbers(
    cmd: int, frame: int, seq: int, params: list[float], msg: Any, ctx: SendContext
) -> str | None:
    if cmd in (mavlink.MAV_CMD_NAV_WAYPOINT, mavlink.MAV_CMD_NAV_LOITER_TIME):
        if seq == 0:  # the home slot; ArduPilot overwrites it with its home position
            if cmd != mavlink.MAV_CMD_NAV_WAYPOINT or frame != mavlink.MAV_FRAME_GLOBAL:
                return "mission item 0 must be the home waypoint in the global frame"
            return None
        if frame not in _REL_ALT_FRAMES:
            return f"frame {frame} not allowed for a waypoint"
        if cmd == mavlink.MAV_CMD_NAV_WAYPOINT and any(v != 0 for v in params):
            return "waypoint needs param1..4 = 0"
        if cmd == mavlink.MAV_CMD_NAV_LOITER_TIME:
            loiter = _int(params[0])
            if loiter is None or not 0 <= loiter <= 3600 or any(v != 0 for v in params[1:]):
                return "loiter time needs 0..3600 s and param2..4 = 0"
        return _alt_ok(float(msg.z), ctx) or _within_fence(int(msg.x), int(msg.y), ctx)
    if frame != mavlink.MAV_FRAME_MISSION or seq == 0:
        return "speed and return items use the mission frame and never index 0"
    if int(msg.x) != 0 or int(msg.y) != 0 or float(msg.z) != 0:
        return "speed and return items carry no position"
    if cmd == mavlink.MAV_CMD_DO_CHANGE_SPEED:
        return _speed_ok(params[0], params[1], params[2], params[3], ctx)
    if any(v != 0 for v in params):  # NAV_RETURN_TO_LAUNCH
        return "return item needs param1..4 = 0"
    return None


def _mission_read(msg: Any, ctx: SendContext) -> str | None:
    bad = _target_ok(msg, ctx)
    if bad:
        return bad
    mtype = int(msg.mission_type)
    allowed = (
        {mavlink.MAV_MISSION_TYPE_FENCE}
        if ctx.profile is Profile.GROUND
        else {mavlink.MAV_MISSION_TYPE_MISSION, mavlink.MAV_MISSION_TYPE_FENCE}
    )
    if mtype not in allowed:
        return f"mission type {mtype} not allowed"
    if msg.get_type() == "MISSION_REQUEST_INT" and not 0 <= int(msg.seq) < MISSION_MAX_ITEMS:
        return "mission index out of range"
    return _origin_for("read", ctx)


def _mission_set_current(msg: Any, ctx: SendContext) -> str | None:
    if ctx.profile is Profile.GROUND:
        return "the ground station never changes the mission"
    bad = _target_ok(msg, ctx)
    if bad:
        return bad
    if not 0 <= int(msg.seq) < MISSION_MAX_ITEMS:
        return "mission index out of range"
    return _origin_for("fly", ctx)


def _param_read(msg: Any, ctx: SendContext) -> str | None:
    bad = _target_ok(msg, ctx)
    if bad:
        return bad
    name = msg.param_id
    if isinstance(name, bytes):
        name = name.decode("ascii", errors="replace")
    if int(msg.param_index) != -1 or not str(name).strip("\x00"):
        return "PARAM_REQUEST_READ only by name"
    return _origin_for("read", ctx)


def _manual_control(msg: Any, ctx: SendContext) -> str | None:
    if ctx.profile is not Profile.GROUND:
        return "MANUAL_CONTROL is sent by the ground station (the GCS system id)"
    if ctx.origin != _MANUAL:
        return "MANUAL_CONTROL only for the pilot's sticks"
    if int(msg.target) != ctx.target_system:
        return "MANUAL_CONTROL to another system"
    if ctx.fc_mode != CopterMode.LOITER:
        return "MANUAL_CONTROL only while the flight controller is in LOITER"
    # Only in the air: on the ground the sticks could arm or disarm the motors (the
    # flight controller's throttle-down plus full-yaw gesture).
    if not ctx.airborne:
        return "MANUAL_CONTROL only while the vehicle is armed and in the air"
    x, y, z, r = int(msg.x), int(msg.y), int(msg.z), int(msg.r)
    if not (
        -_STICK_MAX <= x <= _STICK_MAX
        and -_STICK_MAX <= y <= _STICK_MAX
        and -_STICK_MAX <= r <= _STICK_MAX
        and 0 <= z <= _STICK_MAX
    ):
        return "stick value out of range"
    if int(msg.buttons) != 0 or int(msg.buttons2) != 0 or int(msg.enabled_extensions) != 0:
        return "MANUAL_CONTROL without buttons or extensions"
    return None


_MESSAGES: Final = {
    "HEARTBEAT": _heartbeat,
    "COMMAND_LONG": _command_long,
    "SET_POSITION_TARGET_GLOBAL_INT": _position_target,
    "MISSION_COUNT": _mission_count,
    "MISSION_ITEM_INT": _mission_item,
    "MISSION_REQUEST_LIST": _mission_read,
    "MISSION_REQUEST_INT": _mission_read,
    "MISSION_ACK": _mission_read,
    "MISSION_SET_CURRENT": _mission_set_current,
    "PARAM_REQUEST_READ": _param_read,
    "MANUAL_CONTROL": _manual_control,
}


def check(msg: Any, ctx: SendContext) -> str | None:
    """``None`` when ``msg`` may be sent in ``ctx``, otherwise why not (default deny)."""
    rule = _MESSAGES.get(msg.get_type())
    if rule is None:
        return f"{msg.get_type()} not allowed"
    try:
        return rule(msg, ctx)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        return f"malformed {msg.get_type()}: {type(exc).__name__}"
