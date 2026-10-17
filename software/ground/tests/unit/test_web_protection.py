"""Browser protection on the panel listeners (04 T26, B1-05 points 5 and 6)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from sd_ground.app import create_panel_app, panel_asgi
from sd_ground.context import Ground
from sd_ground.web import SECURITY_HEADERS

H = {"X-SD-Client": "panel", "Content-Type": "application/json"}
ROOT = Path(__file__).resolve().parents[4]


def _has_security_headers(res: object) -> None:
    headers = res.headers  # type: ignore[attr-defined]
    for name, value in SECURITY_HEADERS.items():
        assert headers.get(name) == value, name
    assert not any(k.lower().startswith("access-control-") for k in headers)


def test_headers_match_shared_file() -> None:
    shared = json.loads((ROOT / "software/common/schemas/security_headers.json").read_text())
    assert shared == SECURITY_HEADERS


def test_changing_request_without_client_header_is_csrf(logged_in: TestClient) -> None:
    res = logged_in.post(
        "/api/v1/session/logout", json={}, headers={"Content-Type": "application/json"}
    )
    assert res.status_code == 403
    assert res.json()["code"] == "CSRF"


def test_text_plain_is_csrf(logged_in: TestClient) -> None:
    res = logged_in.post(
        "/api/v1/session/logout",
        content="{}",
        headers={"X-SD-Client": "panel", "Content-Type": "text/plain"},
    )
    assert res.status_code == 403
    assert res.json()["code"] == "CSRF"


def test_foreign_host_is_421(panel: TestClient) -> None:
    res = panel.get("/api/v1/health", headers={"Host": "evil.example"})
    assert res.status_code == 421
    assert res.json()["code"] == "HOST_NOT_ALLOWED"
    _has_security_headers(res)


def test_tunnel_host_rejected(logged_in: TestClient) -> None:
    res = logged_in.get(
        "/api/v1/vehicles",
        headers={"Host": "searchdrone.example.com", "Cf-Connecting-Ip": "203.0.113.9"},
    )
    assert res.status_code == 421


def test_foreign_origin_is_403(logged_in: TestClient) -> None:
    res = logged_in.post(
        "/api/v1/session/logout", json={}, headers={**H, "Origin": "http://evil.example"}
    )
    assert res.status_code == 403
    assert res.json()["code"] == "ORIGIN_NOT_ALLOWED"
    same = logged_in.get("/api/v1/session", headers={"Origin": "http://127.0.0.1:8080"})
    assert same.status_code == 200


def test_ws_with_foreign_origin_closes(logged_in: TestClient) -> None:
    with (
        pytest.raises(WebSocketDisconnect) as closed,
        logged_in.websocket_connect(
            "ws://127.0.0.1:8080/ws/v1/live", headers={"Origin": "http://evil.example"}
        ) as ws,
    ):
        ws.receive_json()
    assert closed.value.code == 4403


def test_bearer_ws_with_origin_closes(panel: TestClient, pilot_token: str) -> None:
    with (
        pytest.raises(WebSocketDisconnect) as closed,
        panel.websocket_connect(
            "ws://127.0.0.1:8080/ws/v1/live",
            headers={"Authorization": f"Bearer {pilot_token}", "Origin": "http://127.0.0.1:8080"},
        ) as ws,
    ):
        ws.receive_json()
    assert closed.value.code == 4403


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/v1/health"),
        ("GET", "/api/v1/nope"),
        ("GET", "/api/v1/session"),
        ("POST", "/api/v1/session"),
    ],
)
def test_every_response_has_security_headers(panel: TestClient, method: str, path: str) -> None:
    res = panel.request(method, path, json={}, headers=H)
    _has_security_headers(res)
    assert res.headers["cache-control"] == "no-store"


def test_static_files_have_security_headers(
    make_ground: Callable[..., Ground], tmp_path: Path
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>p</title>")
    g = make_ground(http={"static_dir": str(dist)})
    with TestClient(panel_asgi(g, create_panel_app(g)), base_url="http://127.0.0.1:8080") as c:  # pyright: ignore[reportArgumentType]
        res = c.get("/")
        assert res.status_code == 200
        _has_security_headers(res)
        _has_security_headers(c.get("/missing.js"))


def test_dev_proxy_header(make_ground: Callable[..., Ground]) -> None:
    for profile, expected in (("bench", 403), ("sim", 200)):
        g = make_ground(profile=profile)
        with TestClient(panel_asgi(g, create_panel_app(g)), base_url="http://127.0.0.1:8080") as c:  # pyright: ignore[reportArgumentType]
            res = c.get("/api/v1/health", headers={"X-SD-Dev-Proxy": "1"})
            assert res.status_code == expected
            if expected == 403:
                assert res.json()["code"] == "DEV_PROXY_FORBIDDEN"


def test_whep_content_types(logged_in: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from sd_ground.api import video_proxy

    session = "0d0a7c48-1b8b-4c4e-9a7c-6d4b0f1f2e3a"
    answer = httpx.Response(201, headers={"Location": f"/drone/sim-01/main/whep/{session}"})
    monkeypatch.setattr(
        video_proxy,
        "_transport",
        httpx.MockTransport(lambda r: answer if r.method == "POST" else httpx.Response(204)),
    )
    base = "/api/v1/video/sim-01/main/whep"
    no_header = logged_in.post(base, content="v=0", headers={"Content-Type": "application/sdp"})
    assert no_header.status_code == 403
    sdp = logged_in.post(
        base, content="v=0", headers={"X-SD-Client": "panel", "Content-Type": "application/sdp"}
    )
    assert sdp.status_code == 201  # passes the guard to the video proxy
    frag = logged_in.patch(
        f"{base}/{session}",
        content="a=x",
        headers={"X-SD-Client": "panel", "Content-Type": "application/trickle-ice-sdpfrag"},
    )
    assert frag.status_code == 204
    elsewhere = logged_in.post(
        "/api/v1/session/logout",
        content="v=0",
        headers={"X-SD-Client": "panel", "Content-Type": "application/sdp"},
    )
    assert elsewhere.status_code == 403  # SDP is only accepted on the video paths


def test_body_over_256_kib_is_413(logged_in: TestClient) -> None:
    big = json.dumps({"login_code": "x" * (300 * 1024)})
    res = logged_in.post("/api/v1/session", content=big, headers=H)
    assert res.status_code == 413
    assert res.json()["code"] == "BODY_TOO_LARGE"


def test_six_wrong_codes_from_one_address_is_429(panel: TestClient) -> None:
    codes = [
        panel.post("/api/v1/session", json={"login_code": "wrong"}, headers=H).status_code
        for _ in range(6)
    ]
    assert codes == [401] * 5 + [429]
    assert (
        panel.post("/api/v1/session", json={"login_code": "wrong"}, headers=H).json()["code"]
        == "TOO_MANY_ATTEMPTS"
    )


def test_no_session_is_401(panel: TestClient) -> None:
    res = panel.get("/api/v1/vehicles")
    assert res.status_code == 401
    assert res.json()["code"] == "UNAUTHENTICATED"


def test_cookie_flags(panel: TestClient, ground: Ground) -> None:
    code, _ = ground.login_link()
    res = panel.post("/api/v1/session", json={"login_code": code}, headers=H)
    cookie = res.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=strict" in cookie
    assert "path=/" in cookie
    assert "secure" not in cookie  # plain http on loopback; LAN mode adds Secure


def test_login_code_works_once(panel: TestClient, ground: Ground) -> None:
    code, _ = ground.login_link()
    assert panel.post("/api/v1/session", json={"login_code": code}, headers=H).status_code == 200
    assert panel.post("/api/v1/session", json={"login_code": code}, headers=H).status_code == 401
