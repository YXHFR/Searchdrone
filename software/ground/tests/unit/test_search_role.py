"""B2-01: the search service's role. It reads its own health route and nothing that steers.

The search token is refused on the commander API, on approvals and on the emergency
actions (403), and on every other panel route too.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from sd_ground.app import commander_asgi, create_commander_app, openapi_documents
from sd_ground.auth import BrainToken
from sd_ground.context import Ground

SEARCH = "search-token-for-tests-0123456789abcdef"
BRAIN = "brain-token-for-tests-0123456789abcdef"  # same as conftest.BRAIN_TOKEN
AS_SEARCH = {"Authorization": f"Bearer {SEARCH}"}
PANEL_H = {"X-SD-Client": "panel", "Content-Type": "application/json"}
SAMPLE_ID = "0192f49e-0b3a-7c55-8e2f-1a2b3c4d5e6f"


@pytest.fixture
def searching(ground: Ground) -> Ground:
    ground.search_token = BrainToken(SEARCH)
    return ground


@pytest.fixture
def loopback(searching: Ground) -> Iterator[TestClient]:
    app = commander_asgi(searching, create_commander_app(searching))
    with TestClient(app, base_url="http://127.0.0.1:7700") as client:  # pyright: ignore[reportArgumentType]
        yield client


def _routes(doc: dict[str, Any]) -> list[tuple[str, str]]:
    return [(m.upper(), p) for p, item in doc["paths"].items() for m in item]


def _fill(path: str) -> str:
    out = path
    for key in ("{vehicle_id}", "{vehicle}"):
        out = out.replace(key, "sim-01")
    while "{" in out:
        start = out.index("{")
        out = out[:start] + SAMPLE_ID + out[out.index("}", start) + 1 :]
    return out


def test_search_role(searching: Ground, loopback: TestClient, panel: TestClient) -> None:
    res = loopback.get("/search/v1/health", headers=AS_SEARCH)
    assert res.status_code == 200, res.text
    assert res.json()["role"] == "search"
    for path in (
        "/commander/v1/skills",
        f"/commander/v1/missions/{SAMPLE_ID}/situation",
    ):
        res = loopback.get(path, headers=AS_SEARCH)
        assert res.status_code == 403 and res.json()["code"] == "SEARCH_FORBIDDEN", path
    res = loopback.post(f"/commander/v1/missions/{SAMPLE_ID}/commands", json={}, headers=AS_SEARCH)
    assert res.status_code == 403
    for path in (
        f"/api/v1/approvals/{SAMPLE_ID}/decision",
        "/api/v1/vehicles/sim-01/actions/stop",
        "/api/v1/vehicles/sim-01/actions/return_home",
        "/api/v1/vehicles/sim-01/actions/resume",
    ):
        res = panel.post(path, json={}, headers={**PANEL_H, **AS_SEARCH})
        assert res.status_code == 403 and res.json()["code"] == "SEARCH_FORBIDDEN", path


def test_search_token_reaches_no_commander_route(loopback: TestClient) -> None:
    for method, path in _routes(openapi_documents()["commander"]):
        res = loopback.request(method, _fill(path), json={}, headers=AS_SEARCH)
        assert res.status_code == 403, (method, path, res.status_code)


def test_search_token_reaches_no_panel_route(searching: Ground, panel: TestClient) -> None:
    routes = _routes(openapi_documents()["panel"])
    assert len(routes) > 30
    for method, path in routes:
        res = panel.request(method, _fill(path), json={}, headers={**PANEL_H, **AS_SEARCH})
        assert res.status_code == 403, (method, path, res.status_code)
        assert res.json()["code"] == "SEARCH_FORBIDDEN"


def test_search_token_cannot_open_the_commander_socket(loopback: TestClient) -> None:
    with (
        pytest.raises(WebSocketDisconnect) as closed,
        loopback.websocket_connect("ws://127.0.0.1:7700/commander/v1/ws", headers=AS_SEARCH) as ws,
    ):
        ws.receive_json()
    assert closed.value.code == 4403


def test_search_health_needs_the_search_token(loopback: TestClient) -> None:
    assert loopback.get("/search/v1/health").status_code == 401
    assert loopback.get("/search/v1/health", headers={"Authorization": "Bearer x"}).status_code == (
        401
    )
    res = loopback.get("/search/v1/health", headers={"Authorization": f"Bearer {BRAIN}"})
    assert res.status_code == 403 and res.json()["code"] == "COMMANDER_FORBIDDEN"


def test_search_health_is_off_without_a_token_file(ground: Ground) -> None:
    """No token configured: nobody gets in, not even with an empty bearer."""
    app = commander_asgi(ground, create_commander_app(ground))
    with TestClient(app, base_url="http://127.0.0.1:7700") as client:  # pyright: ignore[reportArgumentType]
        assert client.get("/search/v1/health", headers=AS_SEARCH).status_code == 401
        assert client.get(
            "/search/v1/health", headers={"Authorization": "Bearer "}
        ).status_code == (401)


def test_search_api_has_only_its_own_routes() -> None:
    """E7: the search service has no route that gives orders, approves or decides."""
    assert _routes(openapi_documents()["search"]) == [("GET", "/search/v1/health")]
    assert "/search/v1/health" not in openapi_documents()["commander"]["paths"]
