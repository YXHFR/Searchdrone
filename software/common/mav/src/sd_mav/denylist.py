"""The most dangerous MAVLink traffic, named one by one.

The allowlist (:mod:`sd_mav.allowlist`) already refuses everything that is not on it.
This list is a second, independent check that ``Vehicle.send`` runs first, and the list
the tests and reviewers point at: writing flight-controller parameters, switching off
or moving the geofence, rebooting, terminating the flight, forcing arming, steering
around the autopilot. Nothing here can be unlocked by any origin or setting.

One named exception (owner decision on raising limits): ``PARAM_SET`` of exactly the
fence's height, radius and LOITER speed (``FENCE_LIMIT_PARAMS``) is not on this list. The
allowlist then lets only the ground station send it, only while the vehicle stands
disarmed on the ground and only to a fence at or outside the standard limits. Every
other parameter stays forbidden here.

``MANUAL_CONTROL`` is not listed: it is allowed onboard only for the pilot's own sticks
in LOITER (see :mod:`sd_mav.allowlist`). ``RC_CHANNELS_OVERRIDE`` stays forbidden.
"""

from __future__ import annotations

from typing import Any, Final

from pymavlink.dialects.v20 import ardupilotmega as mavlink

__all__ = [
    "ARM_FORCE_MAGIC",
    "DENIED_COMMANDS",
    "DENIED_MESSAGES",
    "FENCE_LIMIT_PARAMS",
    "denied_reason",
    "param_name",
]

# The only parameters the ground station may set (the fence follows raised limits).
FENCE_LIMIT_PARAMS: Final = frozenset({"FENCE_ALT_MAX", "FENCE_RADIUS", "LOIT_SPEED_MS"})

# Messages that are never sent.
DENIED_MESSAGES: Final = frozenset(
    {
        "PARAM_SET",
        "PARAM_EXT_SET",
        "PARAM_MAP_RC",
        "SET_MODE",  # modes change only through COMMAND_LONG DO_SET_MODE
        "RC_CHANNELS_OVERRIDE",
        "SET_ATTITUDE_TARGET",
        "SET_ACTUATOR_CONTROL_TARGET",
        "FILE_TRANSFER_PROTOCOL",
        "MISSION_CLEAR_ALL",
        "MISSION_WRITE_PARTIAL_LIST",
        "MISSION_ITEM",  # only MISSION_ITEM_INT
        "SET_HOME_POSITION",
        "SET_GPS_GLOBAL_ORIGIN",
        "FENCE_POINT",
        "FENCE_FETCH_POINT",
        "RALLY_POINT",
        "SERIAL_CONTROL",
        "LOG_ERASE",
        "GPS_INJECT_DATA",
        "SETUP_SIGNING",
        "CHANGE_OPERATOR_CONTROL",
    }
)

# Commands that are never sent, as COMMAND_LONG, COMMAND_INT or as a mission item.
DENIED_COMMANDS: Final = frozenset(
    {
        mavlink.MAV_CMD_DO_SET_PARAMETER,
        mavlink.MAV_CMD_DO_FENCE_ENABLE,
        mavlink.MAV_CMD_DO_SET_HOME,
        mavlink.MAV_CMD_DO_AUX_FUNCTION,
        mavlink.MAV_CMD_DO_SET_SERVO,
        mavlink.MAV_CMD_DO_REPEAT_SERVO,
        mavlink.MAV_CMD_DO_SET_RELAY,
        mavlink.MAV_CMD_DO_REPEAT_RELAY,
        mavlink.MAV_CMD_DO_PARACHUTE,
        mavlink.MAV_CMD_DO_FLIGHTTERMINATION,
        mavlink.MAV_CMD_DO_MOTOR_TEST,
        mavlink.MAV_CMD_DO_SET_ROI_LOCATION,
        mavlink.MAV_CMD_DO_GRIPPER,
        mavlink.MAV_CMD_DO_WINCH,
        mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
        mavlink.MAV_CMD_PREFLIGHT_CALIBRATION,
        mavlink.MAV_CMD_PREFLIGHT_STORAGE,
        mavlink.MAV_CMD_PREFLIGHT_SET_SENSOR_OFFSETS,
        mavlink.MAV_CMD_PREFLIGHT_UAVCAN,
        mavlink.MAV_CMD_DO_START_MAG_CAL,
        mavlink.MAV_CMD_FLASH_BOOTLOADER,
        mavlink.MAV_CMD_SCRIPTING,
        mavlink.MAV_CMD_DO_SEND_SCRIPT_MESSAGE,
    }
)

# COMPONENT_ARM_DISARM param2 with this value skips every arming and disarming check.
ARM_FORCE_MAGIC: Final = 21196


def param_name(msg: Any) -> str:
    """A parameter id as text (pymavlink gives bytes or str, NUL-padded)."""
    value = msg.param_id
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="replace")
    return str(value).rstrip("\x00")


def _command_of(msg: Any) -> int | None:
    name = msg.get_type()
    if name in ("COMMAND_LONG", "COMMAND_INT", "MISSION_ITEM_INT"):
        return int(msg.command)
    return None


def denied_reason(msg: Any) -> str | None:
    """Why ``msg`` is on the deny list, or ``None`` when it is not."""
    name = msg.get_type()
    if name == "PARAM_SET" and param_name(msg) in FENCE_LIMIT_PARAMS:
        return None  # the allowlist decides who, when and how far
    if name in DENIED_MESSAGES:
        return f"{name} is forbidden"
    command = _command_of(msg)
    if command is not None and command in DENIED_COMMANDS:
        return f"command {command} is forbidden"
    if (
        command == mavlink.MAV_CMD_COMPONENT_ARM_DISARM
        and name != "MISSION_ITEM_INT"
        and float(msg.param2) == ARM_FORCE_MAGIC  # NaN compares unequal, never raises
    ):
        return "forced arming or disarming is forbidden"
    if name in ("MISSION_COUNT", "MISSION_ITEM_INT") and int(msg.mission_type) not in (0, 1):
        return f"upload of mission_type {int(msg.mission_type)} is forbidden"
    return None
