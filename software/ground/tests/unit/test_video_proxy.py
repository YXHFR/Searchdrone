"""The WHEP proxy in front of MediaMTX (B1-09: Location rewrite, PATCH and DELETE routing).

MediaMTX is faked with an ``httpx.MockTransport`` that records every upstream call.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from sd_ground.api import video_proxy
from sd_ground.app import create_panel_app, panel_asgi
from sd_ground.context import Ground

BASE = "/api/v1/video/sim-01/main/whep"
SESSION = "0d0a7c48-1b8b-4c4e-9a7c-6d4b0f1f2e3a"
SDP_H = {"X-SD-Client": "panel", "Content-Type": "application/sdp"}
FRAG_H = {"X-SD-Client": "panel", "Content-Type": "application/trickle-ice-sdpfrag"}
JSON_H = {"X-SD-Client": "panel", "Content-Type": "application/json"}
OFFER = "v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\n"
ANSWER = "v=0\r\no=- 2 2 IN IP4 127.0.0.1\r\n"


class FakeMtx:
    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []
        self.status = 201
        self.location: str | None = f"/drone/sim-01/main/whep/{SESSION}"
        self.fail = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self.fail:
            raise httpx.ConnectError("refused", request=request)
        if request.method == "POST":
            headers = {"Content-Type": "application/sdp", "ETag": '"e1"', "Set-Cookie": "x=1"}
            if self.location is not None:
                headers["Location"] = self.location
            return httpx.Response(self.status, content=ANSWER.encode(), headers=headers)
        return httpx.Response(204 if self.status == 201 else self.status)


@pytest.fixture
def mtx() -> Iterator[FakeMtx]:
    fake = FakeMtx()
    video_proxy.set_transport(httpx.MockTransport(fake.handler))
    video_proxy._sessions.clear()  # pyright: ignore[reportPrivateUsage]
    yield fake
    video_proxy.set_transport(None)
    video_proxy._sessions.clear()  # pyright: ignore[reportPrivateUsage]


def test_offer_is_forwarded_and_location_rewritten(logged_in: TestClient, mtx: FakeMtx) -> None:
    res = logged_in.post(BASE, content=OFFER, headers=SDP_H)
    assert res.status_code == 201, res.text
    assert res.text == ANSWER
    assert res.headers["content-type"].startswith("application/sdp")
    assert res.headers["location"] == f"{BASE}/{SESSION}"
    assert res.headers["etag"] == '"e1"'
    assert "set-cookie" not in res.headers
    assert res.headers["cache-control"] == "no-store"
    (call,) = mtx.calls
    assert str(call.url) == "http://127.0.0.1:8889/drone/sim-01/main/whep"
    assert call.content == OFFER.encode()
    assert "cookie" not in call.headers  # the panel's session never reaches MediaMTX
    assert "x-sd-client" not in call.headers


def test_patch_and_delete_go_to_the_session(logged_in: TestClient, mtx: FakeMtx) -> None:
    assert logged_in.post(BASE, content=OFFER, headers=SDP_H).status_code == 201
    mtx.calls.clear()
    patch = logged_in.patch(
        f"{BASE}/{SESSION}", content="a=candidate:1", headers={**FRAG_H, "If-Match": "*"}
    )
    assert patch.status_code == 204
    delete = logged_in.request("DELETE", f"{BASE}/{SESSION}", content="{}", headers=JSON_H)
    assert delete.status_code == 204
    assert [(c.method, c.url.path) for c in mtx.calls] == [
        ("PATCH", f"/drone/sim-01/main/whep/{SESSION}"),
        ("DELETE", f"/drone/sim-01/main/whep/{SESSION}"),
    ]
    assert mtx.calls[0].headers["if-match"] == "*"


def test_session_id_must_be_a_uuid(logged_in: TestClient, mtx: FakeMtx) -> None:
    res = logged_in.request("DELETE", f"{BASE}/..%2F..%2Fv3", content="{}", headers=JSON_H)
    assert res.status_code in (404, 422)
    assert mtx.calls == []


def test_needs_a_session(panel: TestClient, mtx: FakeMtx) -> None:
    res = panel.post(BASE, content=OFFER, headers=SDP_H)
    assert res.status_code == 401
    assert mtx.calls == []


def test_offer_needs_sdp(logged_in: TestClient, mtx: FakeMtx) -> None:
    res = logged_in.post(BASE, json={}, headers=JSON_H)
    assert res.status_code == 415
    assert mtx.calls == []


def test_no_stream_is_404(logged_in: TestClient, mtx: FakeMtx) -> None:
    mtx.status = 404
    res = logged_in.post(BASE, content=OFFER, headers=SDP_H)
    assert res.status_code == 404
    assert res.json()["code"] == "VIDEO_NOT_FOUND"


def test_unexpected_location_is_refused(logged_in: TestClient, mtx: FakeMtx) -> None:
    for location in ("http://evil.example/x", "/drone/sim-02/main/whep/" + SESSION, None):
        mtx.location = location
        res = logged_in.post(BASE, content=OFFER, headers=SDP_H)
        assert res.status_code == 502, location
        assert "location" not in res.headers


def test_mediamtx_down_is_503(logged_in: TestClient, mtx: FakeMtx) -> None:
    mtx.fail = True
    res = logged_in.post(BASE, content=OFFER, headers=SDP_H)
    assert res.status_code == 503
    assert res.json()["code"] == "VIDEO_UNAVAILABLE"


def test_video_disabled(make_ground: Callable[..., Ground], mtx: FakeMtx) -> None:
    g = make_ground(video={"enabled": False})
    with TestClient(panel_asgi(g, create_panel_app(g)), base_url="http://127.0.0.1:8080") as c:  # pyright: ignore[reportArgumentType]
        code, _ = g.login_link()
        c.post("/api/v1/session", json={"login_code": code}, headers=JSON_H)
        res = c.post(BASE, content=OFFER, headers=SDP_H)
    assert res.status_code == 503
    assert res.json()["code"] == "VIDEO_DISABLED"
    assert mtx.calls == []


def test_hls_is_not_built(logged_in: TestClient) -> None:
    res = logged_in.get("/api/v1/video/sim-01/main/hls/index.m3u8")
    assert res.status_code == 501


@pytest.mark.parametrize(
    ("upstream", "expected"),
    [
        (f"/drone/sim-01/main/whep/{SESSION}", f"{BASE}/{SESSION}"),
        (f"http://127.0.0.1:8889/drone/sim-01/main/whep/{SESSION}", f"{BASE}/{SESSION}"),
        (f"/drone/sim-01/main/whep/{SESSION}/x", None),
        ("/drone/sim-01/main/whep/not-a-session", None),
        (f"/drone/sim-01/main/whep/{SESSION}\n", f"{BASE}/{SESSION}"),  # urlsplit drops it
        ("", None),
    ],
)
def test_panel_location(upstream: str, expected: str | None) -> None:
    found = video_proxy.panel_location("sim-01", upstream)
    assert (None if found is None else found[0]) == expected


def test_only_sessions_this_proxy_opened(logged_in: TestClient, mtx: FakeMtx) -> None:
    """A made-up or already ended session never reaches MediaMTX."""
    other = "11111111-2222-4333-8444-555555555555"
    res = logged_in.request("DELETE", f"{BASE}/{other}", content="{}", headers=JSON_H)
    assert res.status_code == 404
    assert res.json()["code"] == "VIDEO_SESSION_UNKNOWN"
    assert logged_in.post(BASE, content=OFFER, headers=SDP_H).status_code == 201
    first = logged_in.request("DELETE", f"{BASE}/{SESSION}", content="{}", headers=JSON_H)
    again = logged_in.request("DELETE", f"{BASE}/{SESSION}", content="{}", headers=JSON_H)
    assert (first.status_code, again.status_code) == (204, 404)
    assert [c.method for c in mtx.calls] == ["POST", "DELETE"]


def test_other_vehicle_is_404(logged_in: TestClient, mtx: FakeMtx) -> None:
    res = logged_in.post("/api/v1/video/sim-02/main/whep", content=OFFER, headers=SDP_H)
    assert res.status_code == 404
    assert mtx.calls == []


def test_session_memory_is_bounded(logged_in: TestClient, mtx: FakeMtx) -> None:
    for i in range(video_proxy.MAX_SESSIONS + 5):
        mtx.location = f"/drone/sim-01/main/whep/{i:08x}-0000-4000-8000-000000000000"
        assert logged_in.post(BASE, content=OFFER, headers=SDP_H).status_code == 201
    assert len(video_proxy._sessions) == video_proxy.MAX_SESSIONS  # pyright: ignore[reportPrivateUsage]
