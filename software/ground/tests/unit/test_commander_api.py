"""What the AI commander can and cannot reach (B1-05 point 7, 04 T13)."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from sd_ground.api.commander import clean_text
from sd_ground.app import openapi_documents
from sd_ground.context import Ground

BRAIN = "brain-token-for-tests-0123456789abcdef"  # same as conftest.BRAIN_TOKEN
AUTH = {"Authorization": f"Bearer {BRAIN}"}
H = {"X-SD-Client": "panel", "Content-Type": "application/json"}
SAMPLE_ID = "0192f49e-0b3a-7c55-8e2f-1a2b3c4d5e6f"

# Routes on the panel listener the commander token may reach: none. Every one is 403.
PANEL_WHITELIST: frozenset[tuple[str, str]] = frozenset()
# Routes on :7700 the commander may use (05's list, 04 §7.2).
COMMANDER_WHITELIST = frozenset(
    {
        ("GET", "/commander/v1/skills"),
        ("GET", "/commander/v1/missions/active"),
        ("GET", "/commander/v1/missions/{mission_id}/situation"),
        ("GET", "/commander/v1/missions/{mission_id}/chat"),
        ("POST", "/commander/v1/missions/{mission_id}/messages"),
        ("POST", "/commander/v1/missions/{mission_id}/commands"),
        ("POST", "/commander/v1/missions/{mission_id}/estimate"),
        ("POST", "/commander/v1/missions/{mission_id}/commands/{command_id}/cancel"),
        ("POST", "/commander/v1/missions/{mission_id}/charter-proposals"),
        # The AI's stop check: a pilot's own recent line only, never a release.
        ("POST", "/commander/v1/missions/{mission_id}/chat-intents"),
        ("GET", "/commander/v1/missions/{mission_id}/memory"),
        ("PUT", "/commander/v1/missions/{mission_id}/memory"),
        ("POST", "/commander/v1/missions/{mission_id}/decisions"),
    }
)
# Operator actions: must not exist on :7700 at all.
OPERATOR_ONLY = [
    ("POST", "/commander/v1/approvals/{id}/decision"),
    ("POST", "/commander/v1/missions/{id}/clearance"),
    ("PUT", "/commander/v1/missions/{id}/autonomy"),
    ("PUT", "/commander/v1/missions/{id}/limits"),
    ("PUT", "/commander/v1/missions/{id}/ai"),
    ("POST", "/commander/v1/vehicles/sim-01/actions/stop"),
    ("POST", "/commander/v1/vehicles/sim-01/actions/return_home"),
    ("POST", "/commander/v1/vehicles/sim-01/actions/resume"),
    ("POST", "/commander/v1/missions/{id}/annotations"),
    ("POST", "/api/v1/vehicles/sim-01/actions/stop"),
]


def _routes(doc: dict[str, Any]) -> list[tuple[str, str]]:
    return [(m.upper(), p) for p, item in doc["paths"].items() for m in item]


def _fill(path: str) -> str:
    return re.sub(
        r"\{[a-z_]+(:path)?\}", lambda m: "sim-01" if "vehicle" in m.group(0) else SAMPLE_ID, path
    )


def test_panel_routes_refuse_commander_token(panel: TestClient) -> None:
    routes = _routes(openapi_documents()["panel"])
    assert len(routes) > 30
    for method, path in routes:
        if (method, path) in PANEL_WHITELIST:
            continue
        res = panel.request(method, _fill(path), json={}, headers={**H, **AUTH})
        assert res.status_code == 403, (method, path, res.status_code)
        assert res.json()["code"] == "COMMANDER_FORBIDDEN"


def test_commander_routes_are_the_whitelist(commander: TestClient) -> None:
    routes = set(_routes(openapi_documents()["commander"]))
    assert routes == COMMANDER_WHITELIST
    for method, path in routes:
        res = commander.request(method, _fill(path), json={})
        assert res.status_code == 401, (method, path)  # no token, no access


@pytest.mark.parametrize(("method", "path"), OPERATOR_ONLY)
def test_operator_actions_do_not_exist_on_commander_port(
    commander: TestClient, method: str, path: str
) -> None:
    res = commander.request(method, path.replace("{id}", SAMPLE_ID), json={}, headers=AUTH)
    assert res.status_code == 404


@pytest.mark.parametrize("path", ["/api/v1/brain/turn", "/api/v1/brain/x/y", "/api/v1/brain"])
def test_brain_paths_on_panel_are_404(logged_in: TestClient, path: str) -> None:
    assert logged_in.post(path, json={}, headers=H).status_code == 404
    assert logged_in.get(path).status_code == 404


def test_panel_without_any_token_is_401(panel: TestClient) -> None:
    assert panel.get("/api/v1/vehicles").status_code == 401
    assert panel.post("/api/v1/vehicles/sim-01/actions/stop", json={}, headers=H).status_code == 401


def test_commander_rest_with_origin_is_403(commander: TestClient) -> None:
    res = commander.get("/commander/v1/skills", headers={**AUTH, "Origin": "http://127.0.0.1:7700"})
    assert res.status_code == 403
    assert res.json()["code"] == "ORIGIN_NOT_ALLOWED"


def test_commander_wrong_host_is_421(commander: TestClient) -> None:
    res = commander.get("/commander/v1/skills", headers={**AUTH, "Host": "192.168.1.5:7700"})
    assert res.status_code == 421


def test_skills_lists_registry(commander: TestClient) -> None:
    res = commander.get("/commander/v1/skills", headers=AUTH)
    assert res.status_code == 200
    names = {s["name"] for s in res.json()}
    assert len(names) == 19
    assert "takeoff" in names


def test_commander_body_over_2_mib_is_413(commander: TestClient) -> None:
    big = json.dumps({"text": "a" * (2 * 1024 * 1024 + 10)})
    res = commander.post(
        f"/commander/v1/missions/{SAMPLE_ID}/messages",
        content=big,
        headers={**AUTH, "Content-Type": "application/json"},
    )
    assert res.status_code == 413


def test_prompt_json_over_1_mib_is_413(commander: TestClient) -> None:
    body = {"id": SAMPLE_ID, "prompt_json": {"x": "a" * (1024 * 1024 + 10)}}
    res = commander.post(f"/commander/v1/missions/{SAMPLE_ID}/decisions", json=body, headers=AUTH)
    assert res.status_code == 413
    assert res.json()["code"] == "PROMPT_TOO_LARGE"


def test_decision_log_with_prompt_file_is_422(commander: TestClient) -> None:
    body = {"id": SAMPLE_ID, "prompt_file": "../../etc/passwd"}
    res = commander.post(f"/commander/v1/missions/{SAMPLE_ID}/decisions", json=body, headers=AUTH)
    assert res.status_code == 422


def test_rationale_over_500_is_refused_not_cut(commander: TestClient) -> None:
    body = {"client_msg_id": SAMPLE_ID, "text": "a" * 4001}
    res = commander.post(f"/commander/v1/missions/{SAMPLE_ID}/messages", json=body, headers=AUTH)
    assert res.status_code == 422


def test_clean_text() -> None:
    assert clean_text("a\u0007b") == "ab"
    assert clean_text("a\nb") == "a\nb"
    assert clean_text("x\u0000\u001by") == "xy"


def test_ws_without_token_is_4401(commander: TestClient) -> None:
    with (
        pytest.raises(WebSocketDisconnect) as closed,
        commander.websocket_connect("ws://127.0.0.1:7700/commander/v1/ws") as ws,
    ):
        ws.receive_json()
    assert closed.value.code == 4401


def test_ws_with_origin_is_4403(commander: TestClient) -> None:
    with (
        pytest.raises(WebSocketDisconnect) as closed,
        commander.websocket_connect(
            "ws://127.0.0.1:7700/commander/v1/ws", headers={**AUTH, "Origin": "http://x"}
        ) as ws,
    ):
        ws.receive_json()
    assert closed.value.code == 4403


def test_only_one_commander_and_intruders_cannot_take_the_place(
    commander: TestClient, ground: Ground
) -> None:
    url = "ws://127.0.0.1:7700/commander/v1/ws"
    with commander.websocket_connect(url, headers=AUTH) as first:
        assert first.receive_json()["type"] == "hello"
        first.send_json({"type": "brain.hello", "body": {"version": "t", "server_epoch": None}})
        assert first.receive_json()["type"] == "server.hello"
        assert first.receive_json()["type"] == "resync_required"
        for headers in ({"Authorization": "Bearer wrong"}, {**AUTH, "Origin": "http://x"}):
            with (
                pytest.raises(WebSocketDisconnect) as bad,
                commander.websocket_connect(url, headers=headers) as ws,
            ):
                ws.receive_json()
            assert bad.value.code in (4401, 4403)
        with (
            pytest.raises(WebSocketDisconnect) as second,
            commander.websocket_connect(url, headers=AUTH) as ws,
        ):
            ws.receive_json()
        assert second.value.code == 4409
        assert ground.brain_state == "ready"  # the first commander is still the one connected


def test_commander_resume_stream(commander: TestClient, ground: Ground) -> None:
    url = "ws://127.0.0.1:7700/commander/v1/ws"
    pin = {
        "id": SAMPLE_ID,
        "kind": "pin",
        "label": "P1",
        "geometry": {"type": "Point", "coordinates": [149.16523, -35.36146]},
    }
    ground.bus.publish("mission", "annotation.created", pin, ts=ground.now())
    since = ground.bus.last_seq
    ground.bus.publish("mission", "annotation.created", {**pin, "label": "P2"}, ts=ground.now())
    with commander.websocket_connect(url, headers=AUTH) as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "brain.hello",
                "body": {"server_epoch": ground.bus.server_epoch, "since_seq": since},
            }
        )
        assert ws.receive_json()["type"] == "server.hello"
        missed = ws.receive_json()
        assert missed["type"] == "event.pin_set"
        assert missed["body"]["pin_id"] == "P2"


def test_cli_pilot_token_cannot_use_commander_port(commander: TestClient, pilot_token: str) -> None:
    res = commander.get("/commander/v1/skills", headers={"Authorization": f"Bearer {pilot_token}"})
    assert res.status_code == 401


def test_a_new_mission_reaches_the_ai_as_created() -> None:
    """sd-brain switches to a mission the panel opened after it started only on created."""
    import datetime as dt

    from sd_ground.api.commander import to_commander
    from sd_ground.bus import BusEvent

    def event(data: dict[str, object]) -> BusEvent:
        ts = dt.datetime(2026, 10, 16, tzinfo=dt.UTC)
        return BusEvent(5, ts, "mission", "mission.state_changed", "sim-01", "M2", data)

    new = to_commander(event({"from": None, "to": "DRAFT", "created": True, "state_seq": 1}))
    assert new is not None
    assert new["type"] == "event.mission"
    assert new["mission_id"] == "M2"
    assert new["body"]["created"] is True
    later = to_commander(event({"from": "DRAFT", "to": "READY", "state_seq": 2}))
    assert later is not None and later["body"]["created"] is False


def test_a_drawn_path_reaches_the_ai_by_its_name_only() -> None:
    import datetime as dt

    from sd_ground.api.commander import to_commander
    from sd_ground.bus import BusEvent

    ts = dt.datetime(2026, 10, 18, tzinfo=dt.UTC)
    data = {
        "id": "a1",
        "kind": "path",
        "label": "pa-1",
        "length_m": 420.0,
        "geometry": {
            "type": "LineString",
            "coordinates": [[149.16523, -35.36326], [149.16583, -35.36226]],
        },
    }
    out = to_commander(BusEvent(6, ts, "mission", "annotation.created", "sim-01", "M2", data))
    assert out is not None and out["type"] == "event.path_drawn"
    assert out["body"] == {"path_id": "pa-1", "length_m": 420.0}
