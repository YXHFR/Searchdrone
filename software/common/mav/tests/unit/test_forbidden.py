"""The most dangerous MAVLink traffic, one test each: ``Vehicle.send`` refuses it.

Every case is tried from every origin and from both the onboard computer and the ground
station, and must raise ``ForbiddenMavlinkError`` with nothing reaching the flight
controller.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from mav_helpers import FC, HOME, command, mission_item, run_with_clock, set_mode, zero_message
from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_mav.connection import Role
from sd_mav.denylist import DENIED_COMMANDS, DENIED_MESSAGES, denied_reason
from sd_mav.fake import FakeFlightController, FakeVehicle
from sd_mav.modes import CopterMode
from sd_mav.vehicle import ForbiddenMavlinkError, SendOrigin
from sd_testing import FakeClock


async def _refused_everywhere(make: Callable[[], Any], ground_too: bool) -> None:
    clock = FakeClock()
    fc = FakeFlightController(clock)
    events: list[tuple[str, dict[str, Any]]] = []
    onboard = FakeVehicle(clock, fc=fc, on_event=lambda k, d: events.append((k, d)))
    ground = FakeVehicle(
        clock, fc=fc, role=Role.GROUND, on_event=lambda k, d: events.append((k, d))
    )
    await onboard.start()
    await ground.start()
    await clock.run_for(1.5)  # connected, home known
    try:
        before = len(fc.received)
        for vehicle in (onboard, ground) if ground_too else (onboard,):
            for origin in SendOrigin:
                with pytest.raises(ForbiddenMavlinkError):
                    await run_with_clock(clock, vehicle.send(make(), origin=origin), max_s=5)
        sent = [r.msg for r in fc.received[before:] if r.msg.get_type() != "HEARTBEAT"]
        assert not [m for m in sent if m.get_type() == make().get_type() and _same(m, make())]
        assert any(k.endswith("forbidden_mavlink_blocked") for k, _ in events)
    finally:
        await onboard.close()
        await ground.close()


def _same(a: Any, b: Any) -> bool:
    return all(getattr(a, f) == getattr(b, f) for f in b.fieldnames)


def _refused(make: Callable[[], Any], *, ground_too: bool = True) -> None:
    asyncio.run(_refused_everywhere(make, ground_too))


def _param_set() -> Any:
    return mavlink.MAVLink_param_set_message(FC, 1, b"FENCE_ENABLE", 0, mavlink.MAV_PARAM_TYPE_INT8)


def test_forbidden_param_set() -> None:
    _refused(_param_set)


def test_forbidden_param_ext_set() -> None:
    _refused(
        lambda: mavlink.MAVLink_param_ext_set_message(
            FC, 1, b"FENCE_ENABLE", b"0", mavlink.MAV_PARAM_EXT_TYPE_INT8
        )
    )


def test_forbidden_do_set_parameter() -> None:
    _refused(lambda: command(mavlink.MAV_CMD_DO_SET_PARAMETER, 1, 0))


@pytest.mark.parametrize(
    ("name", "cmd"),
    [
        ("fence_enable", mavlink.MAV_CMD_DO_FENCE_ENABLE),
        ("set_home", mavlink.MAV_CMD_DO_SET_HOME),
        ("aux_function", mavlink.MAV_CMD_DO_AUX_FUNCTION),
        ("set_servo", mavlink.MAV_CMD_DO_SET_SERVO),
        ("set_relay", mavlink.MAV_CMD_DO_SET_RELAY),
        ("parachute", mavlink.MAV_CMD_DO_PARACHUTE),
    ],
)
def test_forbidden_do_commands_as_command_and_mission_item(name: str, cmd: int) -> None:
    _refused(lambda: command(cmd, 0, 0))
    _refused(lambda: command(cmd, 1, 1))
    _refused(lambda: mission_item(3, cmd, 2, 0, mission_type=0))
    _refused(lambda: mission_item(3, cmd, 0, 1, lat=HOME[0], lon=HOME[1], mission_type=0))


def test_forbidden_fence_enable() -> None:
    _refused(lambda: command(mavlink.MAV_CMD_DO_FENCE_ENABLE, 0))
    _refused(lambda: mission_item(2, mavlink.MAV_CMD_DO_FENCE_ENABLE, 2, 0))


def test_forbidden_set_home() -> None:
    _refused(lambda: command(mavlink.MAV_CMD_DO_SET_HOME, 1))
    _refused(lambda: zero_message(mavlink.MAVLink_set_home_position_message))


def test_forbidden_upload_other_mission_types() -> None:
    _refused(
        lambda: mavlink.MAVLink_mission_count_message(FC, 1, 4, mavlink.MAV_MISSION_TYPE_RALLY)
    )
    _refused(lambda: mission_item(0, 5100, 0, mission_type=mavlink.MAV_MISSION_TYPE_RALLY))
    _refused(lambda: mavlink.MAVLink_mission_count_message(FC, 1, 4, mavlink.MAV_MISSION_TYPE_ALL))


def test_forbidden_onboard_fence_upload() -> None:
    async def run() -> None:
        clock = FakeClock()
        vehicle = FakeVehicle(clock)
        await vehicle.start()
        await clock.run_for(1.5)
        for origin in SendOrigin:
            with pytest.raises(ForbiddenMavlinkError):
                await vehicle.send(
                    mavlink.MAVLink_mission_count_message(FC, 1, 4, 1), origin=origin
                )
        await vehicle.close()

    asyncio.run(run())


def test_forbidden_mission_clear_all() -> None:
    _refused(lambda: mavlink.MAVLink_mission_clear_all_message(FC, 1, 0))


def test_forbidden_mission_write_partial_list() -> None:
    _refused(lambda: mavlink.MAVLink_mission_write_partial_list_message(FC, 1, 0, 3, 0))


def test_forbidden_disarm_from_onboard() -> None:
    # The ground station may disarm a vehicle standing on the ground (STOP); the
    # onboard computer never disarms.
    _refused(lambda: command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0), ground_too=False)


def test_forbidden_force_arm_or_disarm() -> None:
    _refused(lambda: command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1, 21196))
    _refused(lambda: command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 21196))


@pytest.mark.parametrize(
    "cmd",
    [
        mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
        mavlink.MAV_CMD_PREFLIGHT_CALIBRATION,
        mavlink.MAV_CMD_PREFLIGHT_STORAGE,
        mavlink.MAV_CMD_DO_FLIGHTTERMINATION,
        mavlink.MAV_CMD_DO_MOTOR_TEST,
    ],
)
def test_forbidden_preflight_and_termination(cmd: int) -> None:
    _refused(lambda: command(cmd, 1))
    _refused(lambda: command(cmd, 0))


def test_forbidden_flight_termination() -> None:
    _refused(lambda: command(mavlink.MAV_CMD_DO_FLIGHTTERMINATION, 1))


def test_forbidden_set_mode_message() -> None:
    _refused(lambda: mavlink.MAVLink_set_mode_message(FC, 1, int(CopterMode.STABILIZE)))


def test_forbidden_rc_channels_override() -> None:
    _refused(lambda: zero_message(mavlink.MAVLink_rc_channels_override_message))


def test_forbidden_manual_control_outside_loiter() -> None:
    # Allowed only for the pilot's sticks while the flight controller is in LOITER; the
    # fake starts in STABILIZE, so it is refused for every origin here.
    _refused(lambda: mavlink.MAVLink_manual_control_message(FC, 0, 0, 500, 0, 0))


def test_forbidden_set_attitude_target() -> None:
    _refused(lambda: zero_message(mavlink.MAVLink_set_attitude_target_message))


def test_forbidden_file_transfer_protocol() -> None:
    _refused(lambda: zero_message(mavlink.MAVLink_file_transfer_protocol_message))


@pytest.mark.parametrize(
    "mode",
    [
        m
        for m in CopterMode
        if m
        not in {
            CopterMode.GUIDED,
            CopterMode.AUTO,
            CopterMode.LOITER,
            CopterMode.BRAKE,
            CopterMode.SMART_RTL,
            CopterMode.RTL,
            CopterMode.LAND,
        }
    ],
)
def test_forbidden_modes(mode: CopterMode) -> None:
    _refused(lambda: set_mode(mode))


def test_deny_list_contents() -> None:
    for name in (
        "PARAM_SET",
        "PARAM_EXT_SET",
        "SET_MODE",
        "RC_CHANNELS_OVERRIDE",
        "MISSION_CLEAR_ALL",
    ):
        assert name in DENIED_MESSAGES
    for cmd in (180, 207, 179, 218, 183, 181, 208, 246, 241, 245, 185, 209):
        assert cmd in DENIED_COMMANDS
    assert "MANUAL_CONTROL" not in DENIED_MESSAGES  # allowed only for sticks in LOITER
    assert denied_reason(command(mavlink.MAV_CMD_DO_SET_MODE, 1, 4)) is None
