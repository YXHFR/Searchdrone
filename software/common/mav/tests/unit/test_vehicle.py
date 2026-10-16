"""``Vehicle`` against the fake flight controller (simulated time)."""

from __future__ import annotations

import asyncio
import inspect
import re
from pathlib import Path
from typing import Any

import pytest
from mav_helpers import FC, HOME, command, run_with_clock
from pymavlink.dialects.v20 import ardupilotmega as mavlink

import sd_mav
from sd_mav.connection import Role
from sd_mav.fake import FakeFlightController, FakeVehicle
from sd_mav.modes import CopterMode
from sd_mav.vehicle import (
    ONBOARD_STREAM_US,
    FcTelemetry,
    ForbiddenMavlinkError,
    MavResult,
    SendOrigin,
    Vehicle,
)
from sd_obs.clock import FakeClock
from sd_schemas.base import LatLon
from sd_schemas.link import StickAxes

SRC = Path(sd_mav.__file__).parent


class Events:
    def __init__(self) -> None:
        self.items: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, kind: str, details: dict[str, Any]) -> None:
        self.items.append((kind, details))

    def kinds(self) -> list[str]:
        return [k for k, _ in self.items]


async def _started(**kwargs: Any) -> tuple[FakeClock, FakeVehicle, Events]:
    clock = FakeClock()
    events = Events()
    vehicle = FakeVehicle(clock, on_event=events, **kwargs)
    await vehicle.start()
    return clock, vehicle, events


def test_heartbeat_within_5s_and_telemetry_at_4hz() -> None:
    async def run() -> None:
        clock, vehicle, events = await _started()
        updates: list[int] = []
        vehicle.subscribe(lambda t: updates.append(clock.monotonic_ms()))
        assert await run_with_clock(clock, vehicle.wait_connected(5.0), max_s=5.0)
        start = clock.monotonic_ms()
        await clock.run_for(10.0)
        in_window = [u for u in updates if u >= start]
        assert len(in_window) / 10.0 >= 4.0
        assert "mav.fc_connected" in events.kinds()
        t = vehicle.telemetry
        assert t.fc_connected and t.fc_mode == "STABILIZE" and not t.armed
        assert t.home is not None and abs(t.home.lat - HOME[0]) < 1e-7
        assert t.landed_state == "on_ground" and t.ekf_ok and t.gps_satellites == 12
        assert t.battery_pct == 100.0 and t.battery_voltage_v == 16.0
        assert t.fc_firmware == "4.7.1"
        await vehicle.close()

    asyncio.run(run())


def test_streams_requested_after_connect_and_reconnect() -> None:
    async def run() -> None:
        clock, vehicle, events = await _started()
        await clock.run_for(3.0)
        assert vehicle.fc.intervals_us == dict(ONBOARD_STREAM_US)
        assert vehicle.fc.intervals_us[mavlink.MAVLINK_MSG_ID_HEARTBEAT] == 200_000
        vehicle.fc.intervals_us.clear()
        vehicle.fc.silent = True
        await clock.run_for(4.0)
        assert not vehicle.telemetry.fc_connected
        vehicle.fc.silent = False
        await clock.run_for(4.0)
        assert vehicle.telemetry.fc_connected
        assert vehicle.fc.intervals_us == dict(ONBOARD_STREAM_US)
        assert events.kinds().count("mav.fc_connected") == 2
        await vehicle.close()

    asyncio.run(run())


def test_lost_connection_detected_within_3s() -> None:
    async def run() -> None:
        clock, vehicle, events = await _started()
        await clock.run_for(2.0)
        lost_at: list[int] = []
        vehicle.subscribe(
            lambda t: None if t.fc_connected else lost_at.append(clock.monotonic_ms())
        )
        cut = clock.monotonic_ms()
        vehicle.link.up = False
        await clock.run_for(4.0)
        assert lost_at, "the lost connection was never noticed"
        assert lost_at[0] - cut <= 3000
        assert "mav.fc_lost" in events.kinds()
        await vehicle.close()

    asyncio.run(run())


def test_command_ack_retries_with_rising_confirmation() -> None:
    async def run() -> None:
        clock, vehicle, _ = await _started()
        await clock.run_for(3.0)
        vehicle.fc.drop_acks = 2
        before = len(vehicle.sent)
        result = await run_with_clock(
            clock, vehicle.set_mode(CopterMode.GUIDED, origin=SendOrigin.SKILL)
        )
        assert result is MavResult.ACCEPTED
        tries = [m for m in vehicle.sent[before:] if m.get_type() == "COMMAND_LONG"]
        assert [int(m.confirmation) for m in tries] == [0, 1, 2]
        vehicle.fc.drop_acks = 5
        result = await run_with_clock(
            clock, vehicle.set_mode(CopterMode.LOITER, origin=SendOrigin.SKILL)
        )
        assert result is MavResult.TIMEOUT
        vehicle.fc.refuse_modes = {int(CopterMode.AUTO)}
        result = await run_with_clock(
            clock, vehicle.set_mode(CopterMode.AUTO, origin=SendOrigin.SKILL)
        )
        assert result is MavResult.DENIED
        await vehicle.close()

    asyncio.run(run())


def test_forbidden_is_blocked_reported_and_never_sent() -> None:
    async def run() -> None:
        clock, vehicle, events = await _started()
        await clock.run_for(2.0)
        before = len(vehicle.sent)
        with pytest.raises(ForbiddenMavlinkError) as exc:
            await vehicle.takeoff(121.0, origin=SendOrigin.SKILL)
        assert "height" in exc.value.reason
        assert len(vehicle.sent) == before
        assert vehicle.blocked_count == 1
        kind, details = events.items[-1]
        assert kind == "executor.forbidden_mavlink_blocked"
        assert details["msg_type"] == "COMMAND_LONG" and details["origin"] == "skill"
        with pytest.raises(TypeError):
            await vehicle.send(command(176, 1, 4), origin="skill")  # type: ignore[arg-type]
        await vehicle.close()

    asyncio.run(run())


def test_flight_helpers_work_end_to_end() -> None:
    async def run() -> None:
        clock, vehicle, _ = await _started()
        await clock.run_for(2.0)

        async def fly() -> None:
            assert (
                await vehicle.set_mode(CopterMode.GUIDED, origin=SendOrigin.SKILL)
                is MavResult.ACCEPTED
            )
            assert await vehicle.arm(origin=SendOrigin.SKILL) is MavResult.ACCEPTED
            assert await vehicle.takeoff(20.0, origin=SendOrigin.SKILL) is MavResult.ACCEPTED
            assert await vehicle.change_speed(7.0, origin=SendOrigin.SKILL) is MavResult.ACCEPTED
            await vehicle.goto(HOME[0] + 0.0005, HOME[1], 30.0, origin=SendOrigin.SKILL)
            assert (
                await vehicle.set_mission_current(1, origin=SendOrigin.SKILL) is MavResult.ACCEPTED
            )
            assert await vehicle.read_param("FENCE_RADIUS", origin=SendOrigin.STARTUP) == 530.0
            assert (
                await vehicle.read_param("NO_SUCH", origin=SendOrigin.STARTUP, within_s=0.5) is None
            )

        await run_with_clock(clock, fly())
        await clock.run_for(20.0)
        assert vehicle.telemetry.armed and vehicle.telemetry.alt_rel_home_m > 20
        assert vehicle.telemetry.landed_state == "in_air"
        assert vehicle.fc.speed_ms == 7.0
        assert vehicle.fc.mission_current == 1
        await vehicle.close()

    asyncio.run(run())


def test_manual_control_scales_sticks_in_loiter_only() -> None:
    async def run() -> None:
        clock, vehicle, _ = await _started()
        await clock.run_for(2.0)
        axes = StickAxes.model_validate(
            {
                "takeover_id": "0192f4a3-0000-7000-8000-0000000000c0",
                "input_seq": 1,
                "pitch": 0.4,
                "roll": -1.0,
                "throttle": 0.5,
                "yaw": 0.25,
            }
        )
        with pytest.raises(ForbiddenMavlinkError):
            await vehicle.manual_control(axes, origin=SendOrigin.MANUAL)  # STABILIZE
        await run_with_clock(clock, vehicle.set_mode(CopterMode.LOITER, origin=SendOrigin.MANUAL))
        await clock.run_for(0.5)
        with pytest.raises(ForbiddenMavlinkError):
            await vehicle.manual_control(axes, origin=SendOrigin.SKILL)
        await vehicle.manual_control(axes, origin=SendOrigin.MANUAL)
        await clock.run_for(0.1)
        sticks = vehicle.fc.sticks
        assert sticks is not None
        assert (sticks.x, sticks.y, sticks.z, sticks.r, sticks.buttons) == (400, -1000, 500, 250, 0)
        assert sticks.get_srcSystem() == FC and sticks.get_srcComponent() == 191
        # A HEARTBEAT older than 1.5 s does not tell the mode: sticks are refused.
        vehicle.fc.silent = True
        await clock.run_for(1.6)
        with pytest.raises(ForbiddenMavlinkError):
            await vehicle.manual_control(axes, origin=SendOrigin.MANUAL)
        await vehicle.close()

    asyncio.run(run())


def test_onboard_heartbeat_and_identity() -> None:
    async def run() -> None:
        clock, vehicle, _ = await _started()
        await clock.run_for(3.0)
        beats = [m for m in vehicle.sent if m.get_type() == "HEARTBEAT"]
        assert 2 <= len(beats) <= 4  # 1 Hz
        hb = beats[0]
        assert (int(hb.type), int(hb.autopilot), int(hb.base_mode), int(hb.custom_mode)) == (
            18,
            8,
            0,
            0,
        )
        assert int(hb.system_status) == mavlink.MAV_STATE_ACTIVE
        assert {(m.get_srcSystem(), m.get_srcComponent()) for m in vehicle.sent} == {(FC, 191)}
        await vehicle.close()

    asyncio.run(run())


def test_ground_vehicle_heartbeat_disarm_and_gate() -> None:
    async def run() -> None:
        clock = FakeClock()
        fc = FakeFlightController(clock)
        alive = [True]
        ground = FakeVehicle(clock, fc=fc, role=Role.GROUND, heartbeat_allowed=lambda: alive[0])
        onboard = FakeVehicle(clock, fc=fc)
        await ground.start()
        await onboard.start()
        await clock.run_for(3.0)
        hb = next(m for m in ground.sent if m.get_type() == "HEARTBEAT")
        assert (int(hb.type), int(hb.autopilot)) == (
            mavlink.MAV_TYPE_GCS,
            mavlink.MAV_AUTOPILOT_INVALID,
        )
        assert {(m.get_srcSystem(), m.get_srcComponent()) for m in ground.sent} == {(250, 190)}
        assert onboard.telemetry.gcs_mavlink_seen_mono_ms is not None
        assert ground.telemetry.onboard_seen_mono_ms is not None
        # STOP on the ground: disarm only while standing still on the ground.
        fc.armed = True
        await clock.run_for(1.0)
        assert (
            await run_with_clock(clock, ground.disarm(origin=SendOrigin.GROUND))
            is MavResult.ACCEPTED
        )
        fc.armed, fc.alt_rel, fc.target = True, 10.0, (HOME[0], HOME[1], 10.0)
        await clock.run_for(1.0)
        with pytest.raises(ForbiddenMavlinkError):
            await ground.disarm(origin=SendOrigin.GROUND)
        with pytest.raises(ForbiddenMavlinkError):
            await ground.set_mode(CopterMode.GUIDED, origin=SendOrigin.GROUND)
        assert await run_with_clock(
            clock, ground.set_mode(CopterMode.BRAKE, origin=SendOrigin.GROUND)
        ) is (MavResult.ACCEPTED)
        # A stalled ground station stops its heartbeat (the flight controller's GCS failsafe).
        alive[0] = False
        count = len([m for m in ground.sent if m.get_type() == "HEARTBEAT"])
        await clock.run_for(3.0)
        assert len([m for m in ground.sent if m.get_type() == "HEARTBEAT"]) == count
        await ground.close()
        await onboard.close()

    asyncio.run(run())


def test_inbound_command_ignored() -> None:
    async def run() -> None:
        clock, vehicle, events = await _started()
        await clock.run_for(1.0)
        for sysid, compid in ((255, 190), (253, 190)):
            intruder = vehicle.fc.link(sysid, compid)
            intruder.write_message(
                mavlink.MAVLink_command_long_message(
                    FC, 191, mavlink.MAV_CMD_USER_1, 0, 1, 0, 0, 0, 0, 0, 0
                )
            )
        # Delivered by the fake as a forwarded message to the onboard link.
        vehicle.link.receive(
            bytes(
                mavlink.MAVLink_command_long_message(
                    FC, 191, mavlink.MAV_CMD_USER_1, 0, 0, 0, 0, 0, 0, 0, 0
                ).pack(mavlink.MAVLink(None, 255, 190))
            )
        )
        await clock.run_for(0.5)
        ignored = [d for k, d in events.items if k == "executor.inbound_command_ignored"]
        assert (
            ignored
            and ignored[0]["sysid"] == 255
            and ignored[0]["command"] == mavlink.MAV_CMD_USER_1
        )
        assert vehicle.telemetry.fc_mode == "STABILIZE"
        await vehicle.close()

    asyncio.run(run())


def test_failsafe_signal_and_statustext_events() -> None:
    async def run() -> None:
        clock, vehicle, events = await _started()
        await clock.run_for(1.0)
        vehicle.fc.statustext("Battery 1 is low 14.0V used 3000 mAh")
        vehicle.fc.statustext("Mode changed")
        await clock.run_for(0.2)
        signal = vehicle.telemetry.last_failsafe_signal
        assert signal is not None and signal.name == "battery"
        assert events.kinds().count("mav.statustext") == 2
        assert events.kinds().count("mav.failsafe_signal") == 1
        await vehicle.close()

    asyncio.run(run())


def test_to_vehicle_state() -> None:
    t = FcTelemetry(
        fc_connected=True, fc_custom_mode=5, armed=True, landed_state="in_air", lat=1.0, lon=2.0
    )
    state = t.to_vehicle_state("sim-01", 1234, LatLon(lat=1.0, lon=2.0))
    assert state.fc_mode == "LOITER" and state.armed and state.vehicle_id == "sim-01"
    assert FcTelemetry().fc_mode == "UNKNOWN"


def test_no_parameter_write_in_the_api() -> None:
    """No public name, and no line of the package outside the deny list, writes a parameter."""
    for cls in (Vehicle, FakeVehicle):
        for name, _ in inspect.getmembers(cls):
            assert not re.search(
                r"param_set|set_param|write_param|fence_enable", name, re.IGNORECASE
            ), name
    pattern = re.compile(
        r"\b(PARAM_SET|PARAM_EXT_SET|param_set_send|MAV_CMD_DO_SET_PARAMETER|DO_FENCE_ENABLE|MAV_CMD_DO_FENCE_ENABLE|21196|MAV_CMD_DO_FLIGHTTERMINATION)\b"
    )
    for path in SRC.glob("*.py"):
        if path.name == "denylist.py":
            continue
        hits = [
            n for n, line in enumerate(path.read_text().splitlines(), 1) if pattern.search(line)
        ]
        assert not hits, (path.name, hits)


def test_origin_override_fallback_only_in_definition() -> None:
    for path in SRC.glob("*.py"):
        text = path.read_text()
        if path.name != "vehicle.py":
            assert "SendOrigin.OVERRIDE_FALLBACK" not in text, path.name
