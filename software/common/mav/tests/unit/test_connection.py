"""Identities, and a real UDP connection on loopback."""

from __future__ import annotations

import asyncio
import socket

import pytest
from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_mav.connection import MavConnection, Role, check_identity, connect
from sd_mav.fake import FakeFlightController
from sd_mav.vehicle import SendOrigin, Vehicle
from sd_obs.clock import FakeClock, MonotonicClock


@pytest.mark.parametrize(
    ("sysid", "compid"),
    [(250, 191), (255, 191), (1, 190), (250, 190), (0, 191), (1, 1), (251, 191)],
)
def test_onboard_never_uses_gcs_sysid(sysid: int, compid: int) -> None:
    with pytest.raises(ValueError):
        check_identity(Role.ONBOARD, sysid, compid)
    with pytest.raises(ValueError):
        connect(
            "udpin:127.0.0.1:0", role=Role.ONBOARD, source_system=sysid, source_component=compid
        )


def test_identities_that_are_allowed() -> None:
    check_identity(Role.ONBOARD, 1, 191)
    check_identity(Role.GROUND, 250, 190)
    for sysid, compid in ((1, 191), (250, 191), (255, 190)):
        with pytest.raises(ValueError):
            check_identity(Role.GROUND, sysid, compid)


def test_vehicle_refuses_a_transport_with_another_identity() -> None:
    clock = FakeClock()
    fc = FakeFlightController(clock)
    with pytest.raises(ValueError):
        Vehicle(fc.link(250, 190), role=Role.ONBOARD, clock=clock)
    with pytest.raises(ValueError):
        Vehicle(fc.link(2, 191), role=Role.ONBOARD, clock=clock, fc_sysid=1)
    with pytest.raises(ValueError):
        Vehicle(fc.link(1, 191), role=Role.GROUND, clock=clock)


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_udp_connection_receives_and_sends(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A Vehicle on a real UDP socket: sees a heartbeat and answers with its own."""

    async def run() -> None:
        port = _free_udp_port()
        conn = connect(
            f"udpin:127.0.0.1:{port}", role=Role.ONBOARD, source_system=1, source_component=191
        )
        assert isinstance(conn, MavConnection)
        conn.start_tlog(tmp_path / "flight.tlog", MonotonicClock())
        vehicle = Vehicle(conn, role=Role.ONBOARD, clock=MonotonicClock())
        await vehicle.start()
        fc_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        fc_sock.bind(("127.0.0.1", 0))
        fc_sock.setblocking(False)
        encoder = mavlink.MAVLink(None, srcSystem=1, srcComponent=1)
        hb = mavlink.MAVLink_heartbeat_message(
            mavlink.MAV_TYPE_QUADROTOR, mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA, 1, 5, 3, 3
        )
        try:
            for _ in range(20):
                fc_sock.sendto(bytes(hb.pack(encoder)), ("127.0.0.1", port))
                if vehicle.telemetry.fc_connected:
                    break
                await asyncio.sleep(0.05)
            assert vehicle.telemetry.fc_connected and vehicle.telemetry.fc_mode == "LOITER"
            got = b""
            for _ in range(60):
                try:
                    got += fc_sock.recv(4096)
                except BlockingIOError:
                    await asyncio.sleep(0.05)
                    continue
                if got:
                    break
            parser = mavlink.MAVLink(None)
            msgs = parser.parse_buffer(got) or []
            assert msgs and msgs[0].get_srcSystem() == 1 and msgs[0].get_srcComponent() == 191
            assert msgs[0].get_type() in ("HEARTBEAT", "COMMAND_LONG")
            await vehicle.send(
                mavlink.MAVLink_param_request_read_message(1, 1, b"FENCE_RADIUS", -1),
                origin=SendOrigin.STARTUP,
            )
        finally:
            fc_sock.close()
            await vehicle.close()
        log = mavutil.mavlink_connection(str(tmp_path / "flight.tlog"))
        first = log.recv_match(type="HEARTBEAT")
        assert first is not None and first.get_srcSystem() == 1 and first.get_srcComponent() == 1

    asyncio.run(run())


def test_read_errors_do_not_stop_the_reader() -> None:
    """A failing read (an unplugged serial port) is retried and the port reopened."""

    async def run() -> None:
        port = _free_udp_port()
        conn = connect(
            f"udpin:127.0.0.1:{port}", role=Role.ONBOARD, source_system=1, source_component=191
        )
        conn.READ_RETRY_S = 0.05  # type: ignore[misc]
        real = conn._conn.recv_match
        calls = {"n": 0, "reset": 0}

        def flaky(**kwargs):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if calls["n"] <= 3:
                raise OSError("device disconnected")
            return real(**kwargs)

        conn._conn.recv_match = flaky
        conn._conn.reset = lambda: calls.__setitem__("reset", calls["reset"] + 1)
        got: list[object] = []
        conn.attach(got.append)
        encoder = mavlink.MAVLink(None, srcSystem=1, srcComponent=1)
        hb = mavlink.MAVLink_heartbeat_message(2, 3, 1, 5, 3, 3)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as fc_sock:
            for _ in range(40):
                fc_sock.sendto(bytes(hb.pack(encoder)), ("127.0.0.1", port))
                await asyncio.sleep(0.05)
                if got:
                    break
        conn.close()
        assert got, "no message after the read errors"
        assert conn.read_errors == 3 and calls["reset"] == 3

    asyncio.run(run())
