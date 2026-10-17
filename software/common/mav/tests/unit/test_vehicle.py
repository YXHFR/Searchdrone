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
    ControlNotAllowedError,
    FcTelemetry,
    ForbiddenMavlinkError,
    MavResult,
    SendOrigin,
    Vehicle,
)
from sd_schemas.base import LatLon
from sd_schemas.link import StickAxes
from sd_testing import FakeClock

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
        vehicle.fc.drop_commands = 2
        before = len(vehicle.sent)
        result = await run_with_clock(
            clock, vehicle.set_mode(CopterMode.GUIDED, origin=SendOrigin.SKILL)
        )
        assert result is MavResult.ACCEPTED
        tries = [m for m in vehicle.sent[before:] if m.get_type() == "COMMAND_LONG"]
        assert [int(m.confirmation) for m in tries] == [0, 1, 2]
        await clock.run_for(0.5)  # GUIDED shows in the HEARTBEAT before the next change
        vehicle.fc.drop_commands = 5
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
    """The ground station (250/190) sends the sticks; the onboard computer never can."""

    async def run() -> None:
        clock = FakeClock()
        fc = FakeFlightController(clock)
        ground = FakeVehicle(clock, fc=fc, role=Role.GROUND)
        onboard = FakeVehicle(clock, fc=fc)
        await ground.start()
        await onboard.start()
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
            await ground.manual_control(axes, origin=SendOrigin.MANUAL)  # STABILIZE
        await run_with_clock(clock, ground.set_mode(CopterMode.LOITER, origin=SendOrigin.GROUND))
        await clock.run_for(0.5)
        with pytest.raises(ForbiddenMavlinkError):  # in LOITER, but on the ground
            await ground.manual_control(axes, origin=SendOrigin.MANUAL)

        async def climb() -> None:
            await onboard.set_mode(CopterMode.GUIDED, origin=SendOrigin.SKILL)
            await onboard.arm(origin=SendOrigin.SKILL)
            await onboard.takeoff(10.0, origin=SendOrigin.SKILL)

        await run_with_clock(clock, climb())
        await clock.run_for(6.0)
        await run_with_clock(clock, ground.set_mode(CopterMode.LOITER, origin=SendOrigin.GROUND))
        await clock.run_for(0.6)
        with pytest.raises(ForbiddenMavlinkError):
            await ground.manual_control(axes, origin=SendOrigin.GROUND)
        for origin in SendOrigin:  # the onboard computer: refused for every origin
            with pytest.raises(ForbiddenMavlinkError):
                await onboard.manual_control(axes, origin=origin)
        await ground.manual_control(axes, origin=SendOrigin.MANUAL)
        await clock.run_for(0.1)
        sticks = fc.sticks
        assert sticks is not None
        assert (sticks.x, sticks.y, sticks.z, sticks.r, sticks.buttons) == (400, -1000, 500, 250, 0)
        assert (sticks.get_srcSystem(), sticks.get_srcComponent()) == (250, 190)
        # A HEARTBEAT older than 1.5 s does not tell the mode: sticks are refused.
        fc.silent = True
        await clock.run_for(1.6)
        with pytest.raises(ForbiddenMavlinkError):
            await ground.manual_control(axes, origin=SendOrigin.MANUAL)
        await ground.close()
        await onboard.close()

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


def test_a_failing_subscriber_never_stops_lost_link_detection() -> None:
    async def run() -> None:
        clock, vehicle, events = await _started()

        def bad(t: FcTelemetry) -> None:
            if not t.fc_connected:
                raise RuntimeError("subscriber bug")

        vehicle.subscribe(bad)
        await clock.run_for(2.0)
        for _ in range(2):  # lost, back, lost again: both losses are noticed
            vehicle.fc.silent = True
            await clock.run_for(3.2)
            assert not vehicle.telemetry.fc_connected
            vehicle.fc.silent = False
            await clock.run_for(1.0)
            assert vehicle.telemetry.fc_connected
        assert "mav.subscriber_error" in events.kinds()
        await vehicle.close()

    asyncio.run(run())


def test_ack_for_another_channel_is_not_ours() -> None:
    async def run() -> None:
        clock, vehicle, _ = await _started()
        await clock.run_for(2.0)
        vehicle.fc.drop_commands = 3  # our own commands never arrive
        task = asyncio.ensure_future(vehicle.set_mode(CopterMode.GUIDED, origin=SendOrigin.SKILL))
        await clock.run_for(0.1)
        # The flight controller's answer to the ground station reaches this channel.
        vehicle.link.receive(
            bytes(
                mavlink.MAVLink_command_ack_message(
                    mavlink.MAV_CMD_DO_SET_MODE, 0, 0, 0, 250, 190
                ).pack(mavlink.MAVLink(None, FC, 1))
            )
        )
        result = await run_with_clock(clock, task)
        assert result is MavResult.TIMEOUT
        await vehicle.close()

    asyncio.run(run())


def test_retry_never_overrides_a_failsafe() -> None:
    async def run() -> None:
        clock, vehicle, events = await _started()
        await clock.run_for(2.0)
        fc = vehicle.fc
        fc.drop_acks = 1  # the first answer is lost ...
        task = asyncio.ensure_future(vehicle.set_mode(CopterMode.GUIDED, origin=SendOrigin.SKILL))
        await clock.run_for(0.3)
        fc.set_mode_external(CopterMode.RTL)  # ... and a failsafe takes over meanwhile
        fc.statustext("Battery 1 is critical")
        result = await run_with_clock(clock, task)
        assert result is MavResult.CANCELLED
        tries = [
            m for m in vehicle.sent if m.get_type() == "COMMAND_LONG" and int(m.command) == 176
        ]
        assert [int(m.param2) for m in tries] == [4]  # no second GUIDED
        assert "mav.retry_cancelled" in events.kinds()
        # Without a failsafe or a foreign mode, the retry still goes out.
        fc.mode = CopterMode.LOITER
        await clock.run_for(0.5)
        fc.drop_acks = 1
        result = await run_with_clock(
            clock, vehicle.set_mode(CopterMode.GUIDED, origin=SendOrigin.SKILL)
        )
        assert result is MavResult.ACCEPTED
        with pytest.raises(ValueError):
            await vehicle.send(command(176, 1, 4), origin=SendOrigin.SKILL, attempts=0)
        fc.drop_acks = 5
        once = await run_with_clock(
            clock,
            vehicle.send(
                command(176, 1, 5), origin=SendOrigin.SKILL, attempts=1, ack_timeout_s=0.2
            ),
        )
        assert once is MavResult.TIMEOUT
        await vehicle.close()

    asyncio.run(run())


def test_stream_requests_restart_after_a_loss_mid_request() -> None:
    async def run() -> None:
        clock, vehicle, events = await _started()
        vehicle.fc.drop_acks = 100  # the first round of requests hangs on missing answers
        await clock.run_for(1.0)
        vehicle.fc.silent = True
        await clock.run_for(3.5)
        assert not vehicle.telemetry.fc_connected
        vehicle.fc.drop_acks = 0
        vehicle.fc.intervals_us.clear()
        vehicle.fc.silent = False
        await clock.run_for(4.0)
        assert vehicle.fc.intervals_us == dict(ONBOARD_STREAM_US)
        assert "mav.streams_requested" in events.kinds()
        await vehicle.close()

    asyncio.run(run())


def test_active_failsafe_does_not_stop_retries_of_stop_steps() -> None:
    """A failsafe that was already active (CRITICAL heartbeat, LOW battery repeated by
    the flight controller) never cancels the retry of a STOP step; only a new one does."""

    async def run() -> None:
        clock = FakeClock()
        fc = FakeFlightController(clock)
        ground = FakeVehicle(clock, fc=fc, role=Role.GROUND)
        await ground.start()
        fc.armed, fc.alt_rel, fc.target = True, 20.0, (HOME[0], HOME[1], 20.0)
        fc.mode = CopterMode.GUIDED
        fc.charge_state = mavlink.MAV_BATTERY_CHARGE_STATE_LOW
        fc.system_status = mavlink.MAV_STATE_CRITICAL
        await clock.run_for(3.0)
        assert ground.telemetry.last_failsafe_signal is not None
        fc.drop_acks = 1
        fc.refuse_modes = {int(CopterMode.BRAKE)}  # the first try must not change the mode
        task = asyncio.ensure_future(ground.set_mode(CopterMode.BRAKE, origin=SendOrigin.GROUND))
        await clock.run_for(0.5)
        fc.refuse_modes = set()
        result = await run_with_clock(clock, task)
        assert result is MavResult.ACCEPTED
        tries = [m for m in ground.sent if m.get_type() == "COMMAND_LONG" and int(m.command) == 176]
        assert len(tries) == 2
        await ground.close()

    asyncio.run(run())


def test_lost_ack_with_mode_reached_is_accepted() -> None:
    async def run() -> None:
        clock, vehicle, events = await _started()
        await clock.run_for(2.0)
        vehicle.fc.drop_acks = 1  # the change happens, its answer is lost
        result = await run_with_clock(
            clock, vehicle.set_mode(CopterMode.GUIDED, origin=SendOrigin.SKILL)
        )
        assert result is MavResult.ACCEPTED
        assert "mav.ack_lost_mode_reached" in events.kinds()
        tries = [
            m for m in vehicle.sent if m.get_type() == "COMMAND_LONG" and int(m.command) == 176
        ]
        assert len(tries) == 1
        await vehicle.close()

    asyncio.run(run())


def test_a_failing_event_sink_never_stops_lost_link_detection() -> None:
    async def run() -> None:
        clock = FakeClock()

        def sink(kind: str, details: dict[str, Any]) -> None:
            raise RuntimeError("sink bug")

        vehicle = FakeVehicle(clock, on_event=sink)
        await vehicle.start()
        await clock.run_for(2.0)
        for _ in range(2):
            vehicle.fc.silent = True
            await clock.run_for(3.2)
            assert not vehicle.telemetry.fc_connected
            vehicle.fc.silent = False
            await clock.run_for(1.0)
            assert vehicle.telemetry.fc_connected
        assert vehicle.event_errors > 0
        await vehicle.close()

    asyncio.run(run())


def test_stale_heartbeat_never_counts_as_mode_reached() -> None:
    async def run() -> None:
        clock, vehicle, _ = await _started()
        await clock.run_for(2.0)
        vehicle.fc.mode = CopterMode.LOITER
        await clock.run_for(0.5)  # LOITER is the last mode the vehicle saw ...
        vehicle.fc.silent = True  # ... and then the link goes quiet
        await clock.run_for(1.6)
        result = await run_with_clock(
            clock, vehicle.set_mode(CopterMode.LOITER, origin=SendOrigin.SKILL)
        )
        assert result is MavResult.TIMEOUT  # never ACCEPTED from a stale HEARTBEAT
        await vehicle.close()

    asyncio.run(run())


def test_send_gate_is_asked_before_every_attempt() -> None:
    """The owner's gate closes between two retries: the retry is never written."""

    async def run() -> None:
        closed: list[str] = []

        def gate(origin: SendOrigin) -> str | None:
            return closed[0] if closed and origin is SendOrigin.SKILL else None

        clock, vehicle, events = await _started(send_gate=gate)
        assert await run_with_clock(clock, vehicle.wait_connected(5.0), max_s=5.0)
        vehicle.fc.drop_acks = 1  # the first answer is lost, so a retry would follow
        before = len(vehicle.sent)
        task = asyncio.ensure_future(vehicle.change_speed(5.0, origin=SendOrigin.SKILL))
        await clock.run_for(0.2)
        closed.append("held")
        await clock.run_for(2.0)
        with pytest.raises(ControlNotAllowedError, match="held"):
            await task
        speed = [
            m
            for m in vehicle.sent[before:]
            if m.get_type() == "COMMAND_LONG"
            and int(m.command) == mavlink.MAV_CMD_DO_CHANGE_SPEED  # not the stream requests
        ]
        assert len(speed) == 1  # the first attempt only
        assert "mav.forbidden_mavlink_blocked" not in events.kinds()
        # Startup traffic (heartbeat, streams) still passes the same gate.
        await vehicle.request_message(
            mavlink.MAVLINK_MSG_ID_HOME_POSITION, origin=SendOrigin.STARTUP
        )
        await vehicle.close()

    asyncio.run(run())


def test_rc_mode_channel_parsing_and_switch_rule() -> None:
    from sd_mav.chain import rc_switch_moved

    async def run() -> None:
        clock, vehicle, _ = await _started()
        await clock.run_for(1.0)
        assert vehicle.telemetry.rc_mode_pwm is None  # no RC receiver
        assert not rc_switch_moved(vehicle, None)

        async def feed(count: int, ch5: int) -> None:
            c = 1500
            msg = mavlink.MAVLink_rc_channels_message(
                0, count, c, c, c, c, ch5, c, c, c, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255
            )
            vehicle.fc._send(msg)  # pyright: ignore[reportPrivateUsage]
            await clock.run_for(0.05)

        await feed(4, 1500)  # fewer than 5 channels: no mode switch
        assert vehicle.telemetry.rc_mode_pwm is None
        await feed(8, 0)  # channel not in use
        assert vehicle.telemetry.rc_mode_pwm is None
        await feed(8, 1100)
        assert vehicle.telemetry.rc_mode_pwm == 1100
        assert rc_switch_moved(vehicle, None)  # an RC receiver came on: a person
        assert not rc_switch_moved(vehicle, 1150)  # noise
        assert rc_switch_moved(vehicle, 1500)
        await vehicle.close()

    asyncio.run(run())
