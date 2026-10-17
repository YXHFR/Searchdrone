"""``/ws/v1/live``: hello, subscribe, resume and telemetry (04 T17, D2; B1-05 point 3)."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from sd_ground.bus import RING_SIZE, EventBus
from sd_ground.context import Ground
from sd_ground.services.telemetry import TelemetryService
from sd_testing import FakeClock

URL = "ws://127.0.0.1:8080/ws/v1/live"


def _sub(
    ws: object, g: Ground, topics: list[str], since: int | None, epoch: str | None = None
) -> None:
    ws.send_json(  # type: ignore[attr-defined]
        {
            "op": "subscribe",
            "topics": topics,
            "vehicle_id": "sim-01",
            "mission_id": None,
            "server_epoch": epoch if epoch is not None else g.bus.server_epoch,
            "since_seq": since,
        }
    )


def test_hello_then_resume_exactly_the_missed_events(logged_in: TestClient, ground: Ground) -> None:
    ground.bus.publish("mission", "mission.updated", {"n": 1}, ts=ground.now(), vehicle_id="sim-01")
    since = ground.bus.last_seq
    ground.bus.publish("chat", "chat.message", {"n": 2}, ts=ground.now(), vehicle_id="sim-01")
    ground.bus.publish("mission", "mission.updated", {"n": 3}, ts=ground.now(), vehicle_id="sim-01")
    ground.bus.publish("mission", "mission.updated", {"n": 4}, ts=ground.now(), vehicle_id="sim-99")
    with logged_in.websocket_connect(URL) as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert hello["data"]["server_epoch"] == ground.bus.server_epoch
        _sub(ws, ground, ["mission"], since)
        replay = ws.receive_json()
        assert (replay["data"], replay["seq"]) == ({"n": 3}, since + 2)
        ws.send_json({"op": "ping", "client_ns": 1})
        pong = ws.receive_json()
        assert pong["type"] == "pong"
        assert isinstance(pong["server_ns"], int)


def test_other_epoch_or_old_seq_needs_resync(logged_in: TestClient, ground: Ground) -> None:
    for epoch, since in (("other-epoch", 0), (None, None)):
        with logged_in.websocket_connect(URL) as ws:
            ws.receive_json()
            _sub(ws, ground, ["mission"], since, epoch)
            assert ws.receive_json()["type"] == "resync_required"


def test_ring_keeps_2000() -> None:
    from sd_testing import FakeClock

    bus, clock = EventBus(), FakeClock()
    for i in range(RING_SIZE + 10):
        bus.publish("mission", "x", {"i": i}, ts=clock.utc_now())
    assert bus.replay(bus.server_epoch, 5) is None  # fell out of the ring
    events = bus.replay(bus.server_epoch, bus.last_seq - 3)
    assert events is not None
    assert [e.data["i"] for e in events] == [RING_SIZE + 7, RING_SIZE + 8, RING_SIZE + 9]
    assert bus.replay(bus.server_epoch, bus.last_seq) == []


def test_telemetry_has_no_seq() -> None:
    from sd_testing import FakeClock

    bus = EventBus()
    before = bus.last_seq
    event = bus.publish("telemetry", "telemetry", {}, ts=FakeClock().utc_now())
    assert event.seq is None
    assert bus.last_seq == before


def test_unauthenticated_ws_is_closed(panel: TestClient) -> None:
    import pytest
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as closed, panel.websocket_connect(URL) as ws:
        ws.receive_json()
    assert closed.value.code == 4401


def test_pilot_ws_counts_as_present(logged_in: TestClient, ground: Ground) -> None:
    assert ground.pilot_ws == 0
    with logged_in.websocket_connect(URL) as ws:
        ws.receive_json()
        _sub(ws, ground, ["system"], None)
        ws.receive_json()
        assert ground.pilot_ws == 1
        status = logged_in.get("/api/v1/system/status").json()
        assert status["operator_present"] is True
    assert ground.pilot_ws == 0


async def test_telemetry_rates(ground: Ground, clock: FakeClock, vehicle_link: object) -> None:
    """5 Hz to the panel (seq null) over 10 simulated seconds."""
    from sd_schemas.api import Telemetry

    vehicle_link.values["sim-01"] = Telemetry(  # type: ignore[attr-defined]
        t=clock.utc_now(),
        time_boot_ms=1,
        lat=-35.363261,
        lon=149.16523,
        alt_amsl_m=584,
        alt_rel_home_m=0,
        heading_deg=0,
        groundspeed_ms=0,
        climb_ms=0,
        battery_pct=90,
        battery_v=16,
        flight_mode="LOITER",
        armed=False,
        landed_state="on_ground",
        gps_fix=3,
        gps_sats=12,
        home=None,
        dist_to_pilot_m=0,
    )
    sub = ground.bus.subscribe(["telemetry"])
    service = TelemetryService(ground)
    task = asyncio.create_task(service.run())
    await clock.run_for(10.0)
    task.cancel()
    got = []
    while not sub.queue.empty():
        event = sub.queue.get_nowait()
        assert event is not None
        got.append(event)
    assert 49 <= len(got) <= 52
    assert all(e.seq is None for e in got)
