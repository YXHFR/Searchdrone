"""STOP while the main loop hangs (04 T14c, T33, T32) with the real MAVLink thread.

The panel listener runs as in production (guard, ingress, bridge to a main loop in
another thread); the MAVLink thread runs ``GroundMav`` on a fake flight controller in
its own event loop with real time.
"""

from __future__ import annotations

import asyncio
import itertools
import socket
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from stop_helpers import VID, TimedFc

from sd_ground.app import create_panel_app, panel_asgi
from sd_ground.context import Ground
from sd_ground.control_socket import READ_TIMEOUT_S, ControlSocket, call
from sd_ground.liveness import LoopLiveness
from sd_ground.mav.thread import GroundMav, HeartbeatGate, MavThread
from sd_ground.services.latches import Latches
from sd_ground.services.vehicle_actions import VehicleActions
from sd_mav.connection import Role
from sd_mav.fake import FakeVehicle
from sd_mav.modes import CopterMode
from sd_obs.clock import MonotonicClock

pytestmark = pytest.mark.component

H = {"X-SD-Client": "panel", "Content-Type": "application/json"}


@dataclass
class Live:
    thread: MavThread
    actions: VehicleActions
    fc: TimedFc
    liveness: LoopLiveness

    def gcs_heartbeats(self, since: float) -> list[float]:
        return [
            t
            for t, sysid, compid, msg in list(self.fc.arrivals)
            if t >= since and (sysid, compid) == (250, 190) and msg.get_type() == "HEARTBEAT"
        ]

    def first_set_mode(self, since: float) -> float | None:
        for t, sysid, compid, msg in list(self.fc.arrivals):
            if (
                t >= since
                and (sysid, compid) == (250, 190)
                and msg.get_type() == "COMMAND_LONG"
                and int(msg.command) == 176
            ):
                return t
        return None


@pytest.fixture
def live() -> Iterator[Live]:
    clock = MonotonicClock()
    latches = Latches(clock)
    actions = VehicleActions(clock, latches, vehicle_ids=[VID])
    liveness = LoopLiveness()
    holder: dict[str, TimedFc] = {}

    def make_core() -> GroundMav:
        fc = TimedFc(clock)
        fc.armed, fc.mode, fc.alt_rel = True, CopterMode.GUIDED, 20.0
        holder["fc"] = fc
        vehicle = FakeVehicle(
            clock, fc=fc, role=Role.GROUND, heartbeat_allowed=HeartbeatGate(liveness)
        )
        return GroundMav(
            vehicle,
            clock,
            vehicle_id=VID,
            latches=latches,
            on_report=actions.on_report,
            on_event=actions.on_event,
            system_action=actions.system,
        )

    thread = MavThread(make_core)
    thread.start()
    assert thread.ready.wait(5) and thread.mav is not None
    actions.mav = thread.mav
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        snap = thread.mav.snapshot()
        if snap is not None and snap.connected and snap.in_flight:
            break
        time.sleep(0.05)
    yield Live(thread, actions, holder["fc"], liveness)
    thread.stop()


@pytest.fixture
def main_loop(live: Live) -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()

    async def beat() -> None:
        while True:
            live.liveness.beat("main")
            await asyncio.sleep(0.2)

    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    asyncio.run_coroutine_threadsafe(beat(), loop)
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def _ground(make_ground: Callable[..., Ground], live: Live) -> Ground:
    g = make_ground()
    assert live.thread.mav is not None
    g.vehicle_link = live.thread.mav
    g.emergency = live.actions
    g.vehicle_actions = live.actions
    return g


def test_stop_reaches_the_fc_within_100ms_while_the_main_loop_hangs(
    make_ground: Callable[..., Ground], live: Live, main_loop: asyncio.AbstractEventLoop
) -> None:
    g = _ground(make_ground, live)
    app = panel_asgi(g, create_panel_app(g), main_loop=main_loop)
    with TestClient(app, base_url="http://127.0.0.1:8080") as client:  # pyright: ignore[reportArgumentType]
        code, _ = g.login_link()
        res = client.post("/api/v1/session", json={"login_code": code}, headers=H)
        assert res.status_code == 200, res.text
        client.cookies.set("sd_session", res.cookies["sd_session"])
        client.post(  # warm up the path once (imports, first request)
            "/api/v1/vehicles/sim-01/actions/stop",
            json={"client_action_id": "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a00"},
            headers=H,
        )
        time.sleep(0.5)
        res = client.post(  # Resume, so the first STOP's watch ends
            "/api/v1/vehicles/sim-01/actions/resume",
            json={"client_action_id": "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a01"},
            headers=H,
        )
        assert res.status_code == 202, res.text
        mav = live.thread.mav
        assert mav is not None
        mav.loop.call_soon_threadsafe(live.fc.set_mode_external, CopterMode.GUIDED)
        block_start = time.monotonic()
        main_loop.call_soon_threadsafe(time.sleep, 2.0)  # the main loop hangs for 2 s
        time.sleep(0.1)
        sent = time.monotonic()
        res = client.post(
            "/api/v1/vehicles/sim-01/actions/stop",
            json={"client_action_id": "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a11"},
            headers=H,
        )
        assert res.status_code == 202, res.text
        arrived = live.first_set_mode(sent)
        for _ in range(50):
            if arrived is not None:
                break
            time.sleep(0.01)
            arrived = live.first_set_mode(sent)
        assert arrived is not None and arrived - sent < 0.1, arrived and arrived - sent
        time.sleep(2.2 - (time.monotonic() - block_start))
        beats = live.gcs_heartbeats(block_start)
        gaps = [b - a for a, b in itertools.pairwise(beats)]
        assert len(beats) >= 2 and all(0.9 <= gap <= 1.1 for gap in gaps), gaps
        assert live.fc.mode == CopterMode.BRAKE


def test_heartbeat_gate_stops_on_a_stalled_loop() -> None:
    now = [0.0]
    events: list[str] = []
    liveness = LoopLiveness(now=lambda: now[0])
    gate = HeartbeatGate(liveness, lambda t, _s, _p: events.append(t))
    liveness.beat("main")
    liveness.beat("ingress")
    assert gate()
    now[0] = 7.9
    assert gate()
    now[0] = 8.5  # no beat for more than LOOP_STALL_S
    assert not gate() and not gate()
    assert events == ["system.loop_stall"]
    liveness.beat("main")
    liveness.beat("ingress")
    assert gate()
    assert events == ["system.loop_stall", "system.loop_recovered"]


def test_cli_stop_through_the_socket_with_a_silent_client(
    tmp_path: Path, make_ground: Callable[..., Ground], live: Live
) -> None:
    """A client that connects and says nothing never holds up the next STOP."""
    g = _ground(make_ground, live)
    path = tmp_path / "run" / "sd-ground.sock"
    control = ControlSocket(path, login_link=g.login_link, pair=g.pair, emergency=live.actions)
    control.start()
    try:
        silent = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        silent.connect(str(path))
        drip = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        drip.connect(str(path))
        drip.sendall(b'{"op"')  # never finishes its line
        time.sleep(0.2)
        started = time.monotonic()
        reply = call(path, {"op": "action", "vehicle_id": VID, "action": "stop"}, timeout=2.0)
        assert time.monotonic() - started < 0.5
        assert reply is not None and reply["ok"] is True
        assert isinstance(reply["action_seq"], int) and reply["action_id"]
        time.sleep(0.3)
        assert live.fc.mode == CopterMode.BRAKE
        # The slow client is cut off after the read timeout.
        drip.settimeout(READ_TIMEOUT_S + 1.0)
        assert drip.recv(100) == b""
        silent.close()
        drip.close()
        assert call(path, {"op": "action", "vehicle_id": VID, "action": "resume"}) == {
            "ok": False,
            "error": "UNSUPPORTED_ACTION",
        }
    finally:
        control.stop()
