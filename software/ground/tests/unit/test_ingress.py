"""The ingress thread: STOP / RETURN HOME never wait for the main loop (04 §12.2, T17)."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from sd_ground.app import create_panel_app, panel_asgi
from sd_ground.context import Ground

H = {"X-SD-Client": "panel", "Content-Type": "application/json"}
STOP = {"client_action_id": "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a11"}


@pytest.fixture
def main_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


@pytest.fixture
def bridged(ground: Ground, main_loop: asyncio.AbstractEventLoop) -> Iterator[TestClient]:
    """The panel listener as in production: the routes run in another thread's loop."""
    app = panel_asgi(ground, create_panel_app(ground), main_loop=main_loop)
    with TestClient(app, base_url="http://127.0.0.1:8080") as client:  # pyright: ignore[reportArgumentType]
        yield client


def _login(client: TestClient, ground: Ground) -> None:
    code, _ = ground.login_link()
    res = client.post("/api/v1/session", json={"login_code": code}, headers=H)
    assert res.status_code == 200, res.text
    client.cookies.set("sd_session", res.cookies["sd_session"])


def test_requests_cross_the_bridge(bridged: TestClient, ground: Ground) -> None:
    _login(bridged, ground)  # POST /session runs in the main loop
    assert bridged.get("/api/v1/session").json()["issuer_ref"] == "operator:demo"


def test_ws_through_the_bridge(bridged: TestClient, ground: Ground) -> None:
    _login(bridged, ground)
    with bridged.websocket_connect("ws://127.0.0.1:8080/ws/v1/live") as ws:
        assert ws.receive_json()["type"] == "hello"
        ws.send_json(
            {"op": "subscribe", "topics": ["mission"], "server_epoch": None, "since_seq": None}
        )
        assert ws.receive_json()["type"] == "resync_required"


def test_stop_while_main_loop_is_blocked(
    bridged: TestClient, ground: Ground, main_loop: asyncio.AbstractEventLoop, emergency: object
) -> None:
    _login(bridged, ground)
    main_loop.call_soon_threadsafe(time.sleep, 2.0)  # the main loop hangs for 2 s
    time.sleep(0.1)
    started = time.monotonic()
    res = bridged.post("/api/v1/vehicles/sim-01/actions/stop", json=STOP, headers=H)
    elapsed = time.monotonic() - started
    assert res.status_code == 202, res.text
    assert elapsed < 0.5, elapsed
    assert emergency.calls == [("stop", "sim-01", "operator:demo")]  # type: ignore[attr-defined]
    home = bridged.post("/api/v1/actions/return_home_all", json=STOP, headers=H)
    assert home.status_code == 202
    assert home.json()["actions"][0]["vehicle_id"] == "sim-01"


def test_emergency_route_keeps_the_guard(bridged: TestClient, ground: Ground) -> None:
    _login(bridged, ground)
    no_csrf = bridged.post("/api/v1/vehicles/sim-01/actions/stop", json=STOP)
    assert no_csrf.status_code == 403
    assert no_csrf.json()["code"] == "CSRF"
    bridged.cookies.clear()
    assert (
        bridged.post("/api/v1/vehicles/sim-01/actions/stop", json=STOP, headers=H).status_code
        == 401
    )
    assert "content-security-policy" in no_csrf.headers
