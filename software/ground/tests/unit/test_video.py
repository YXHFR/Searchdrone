"""Video: the WHEP proxy, MediaMTX's state and the recording per mission (B1-09).

A small HTTP server on loopback plays MediaMTX (WHEP on ``/drone/<vid>/main/whep`` and the
API on ``/v3/...``), so the proxy is tested over real HTTP without the video server.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sd_ground.api.video_proxy import rewrite_location
from sd_ground.app import create_panel_app, panel_asgi
from sd_ground.context import Ground
from sd_ground.services import health
from sd_ground.services.video import (
    NAME_FORMAT,
    VideoMonitor,
    client,
    close_mission_video,
    plan_segments,
    segment_file,
)

SESSION = "0b0e5a9c-1d2e-4f30-8a41-5c6d7e8f9a0b"
WHEP = "/api/v1/video/sim-01/main/whep"
SDP_HEADERS = {"X-SD-Client": "panel", "Content-Type": "application/sdp"}
T0 = dt.datetime(2026, 10, 17, 12, 0, 0, tzinfo=dt.UTC)


@dataclass
class FakeMtx:
    url: str = ""
    requests: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    whep_status: int = 201
    ready: bool = True
    segments: list[str] = field(default_factory=list[str])


@pytest.fixture
def mtx() -> Iterator[FakeMtx]:
    state = FakeMtx()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _reply(self, status: int, body: bytes = b"", headers: dict[str, str] | None = None):
            self.send_response(status)
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _record(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            state.requests.append(
                {
                    "method": self.command,
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": self.rfile.read(length) if length else b"",
                }
            )

        def do_POST(self) -> None:
            self._record()
            if self.path != "/drone/sim-01/main/whep" or state.whep_status != 201:
                self._reply(404 if state.whep_status == 201 else state.whep_status)
                return
            self._reply(
                201,
                b"v=0 answer",
                {
                    "Content-Type": "application/sdp",
                    "Location": f"/drone/sim-01/main/whep/{SESSION}",
                    "ETag": '"abc"',
                    "Accept-Patch": "application/trickle-ice-sdpfrag",
                    "Set-Cookie": "upstream=1",
                },
            )

        def do_PATCH(self) -> None:
            self._record()
            self._reply(204 if self.path.endswith(SESSION) else 404)

        def do_DELETE(self) -> None:
            self._record()
            self._reply(200 if self.path.endswith(SESSION) else 404)

        def do_GET(self) -> None:
            self._record()
            if self.path == "/v3/paths/get/drone/sim-01/main":
                body = json.dumps({"ready": state.ready}).encode()
                self._reply(200, body, {"Content-Type": "application/json"})
            elif self.path == "/v3/recordings/get/drone/sim-01/main" and state.segments:
                body = json.dumps({"segments": [{"start": s} for s in state.segments]}).encode()
                self._reply(200, body, {"Content-Type": "application/json"})
            else:
                self._reply(404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield state
    server.shutdown()
    server.server_close()


@pytest.fixture
def video_ground(make_ground: Callable[..., Ground], mtx: FakeMtx) -> Ground:
    return make_ground(video={"mediamtx_whep": mtx.url, "mediamtx_api": mtx.url})


@pytest.fixture
def client_for(video_ground: Ground) -> Iterator[TestClient]:
    app = panel_asgi(video_ground, create_panel_app(video_ground))
    with TestClient(app, base_url="http://127.0.0.1:8080") as c:  # pyright: ignore[reportArgumentType]
        code, _ = video_ground.login_link()
        res = c.post(
            "/api/v1/session",
            json={"login_code": code},
            headers={"X-SD-Client": "panel", "Content-Type": "application/json"},
        )
        assert res.status_code == 200
        yield c


# --- the proxy ------------------------------------------------------------------------


def test_whep_needs_a_session(video_ground: Ground, mtx: FakeMtx) -> None:
    app = panel_asgi(video_ground, create_panel_app(video_ground))
    with TestClient(app, base_url="http://127.0.0.1:8080") as c:  # pyright: ignore[reportArgumentType]
        res = c.post(WHEP, content="v=0", headers=SDP_HEADERS)
        assert res.status_code == 401
        delete = c.request(
            "DELETE",
            f"{WHEP}/{SESSION}",
            content="{}",
            headers={"X-SD-Client": "panel", "Content-Type": "application/json"},
        )
        assert delete.status_code == 401
    assert mtx.requests == []  # nothing reached the video server


def test_offer_is_forwarded_and_location_rewritten(client_for: TestClient, mtx: FakeMtx) -> None:
    res = client_for.post(WHEP, content="v=0 offer", headers=SDP_HEADERS)
    assert res.status_code == 201
    assert res.content == b"v=0 answer"
    assert res.headers["location"] == f"{WHEP}/{SESSION}"
    assert res.headers["content-type"] == "application/sdp"
    assert res.headers["etag"] == '"abc"'
    assert "upstream" not in res.headers.get("set-cookie", "")  # only WHEP headers come back
    sent = mtx.requests[-1]
    assert sent["path"] == "/drone/sim-01/main/whep"
    assert sent["body"] == b"v=0 offer"
    assert sent["headers"]["content-type"] == "application/sdp"
    assert "cookie" not in sent["headers"]  # the panel's session never leaves sd-ground
    assert "x-sd-client" not in sent["headers"]


def test_patch_and_delete_reach_the_session(client_for: TestClient, mtx: FakeMtx) -> None:
    patch = client_for.patch(
        f"{WHEP}/{SESSION}",
        content="a=candidate",
        headers={
            "X-SD-Client": "panel",
            "Content-Type": "application/trickle-ice-sdpfrag",
            "If-Match": '"abc"',
        },
    )
    assert patch.status_code == 204
    assert mtx.requests[-1]["path"] == f"/drone/sim-01/main/whep/{SESSION}"
    assert mtx.requests[-1]["headers"]["if-match"] == '"abc"'
    delete = client_for.request(
        "DELETE",
        f"{WHEP}/{SESSION}",
        content="{}",
        headers={"X-SD-Client": "panel", "Content-Type": "application/json"},
    )
    assert delete.status_code == 200
    assert mtx.requests[-1]["method"] == "DELETE"
    assert mtx.requests[-1]["body"] == b""


@pytest.mark.parametrize("session", ["../x", "a" * 65, "a.b"])
def test_bad_session_ids_never_reach_mediamtx(
    client_for: TestClient, mtx: FakeMtx, session: str
) -> None:
    res = client_for.request(
        "DELETE",
        f"{WHEP}/{session}",
        content="{}",
        headers={"X-SD-Client": "panel", "Content-Type": "application/json"},
    )
    assert res.status_code in (404, 422)
    assert mtx.requests == []


def test_no_stream_is_404_and_errors_are_502(client_for: TestClient, mtx: FakeMtx) -> None:
    other = client_for.post("/api/v1/video/sim-02/main/whep", content="v=0", headers=SDP_HEADERS)
    assert other.status_code == 404
    assert other.json()["code"] == "VIDEO_NOT_FOUND"
    mtx.whep_status = 400
    bad = client_for.post(WHEP, content="v=0", headers=SDP_HEADERS)
    assert bad.status_code == 502


def test_video_server_down_is_503(make_ground: Callable[..., Ground]) -> None:
    g = make_ground(video={"mediamtx_whep": "http://127.0.0.1:9"})
    app = panel_asgi(g, create_panel_app(g))
    with TestClient(app, base_url="http://127.0.0.1:8080") as c:  # pyright: ignore[reportArgumentType]
        code, _ = g.login_link()
        c.post(
            "/api/v1/session",
            json={"login_code": code},
            headers={"X-SD-Client": "panel", "Content-Type": "application/json"},
        )
        res = c.post(WHEP, content="v=0", headers=SDP_HEADERS)
    assert res.status_code == 503
    assert res.json()["code"] == "VIDEO_UNAVAILABLE"


def test_disabled_video(make_ground: Callable[..., Ground]) -> None:
    g = make_ground(video={"enabled": False})
    app = panel_asgi(g, create_panel_app(g))
    with TestClient(app, base_url="http://127.0.0.1:8080") as c:  # pyright: ignore[reportArgumentType]
        code, _ = g.login_link()
        c.post(
            "/api/v1/session",
            json={"login_code": code},
            headers={"X-SD-Client": "panel", "Content-Type": "application/json"},
        )
        assert c.get("/api/v1/panel/config").json()["video"] is None
        assert c.post(WHEP, content="v=0", headers=SDP_HEADERS).status_code == 503
    g.video_state = "ok"
    assert health.parts(g)["mediamtx"] == "down"


def test_panel_config_points_at_the_proxy(client_for: TestClient) -> None:
    video = client_for.get("/api/v1/panel/config").json()["video"]
    assert video == {"whep_url": WHEP, "hls_url": None}


def test_hls_on_the_server_is_not_built(client_for: TestClient) -> None:
    assert client_for.get("/api/v1/video/sim-01/main/hls/index.m3u8").status_code == 501


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        (f"/drone/sim-01/main/whep/{SESSION}", f"{WHEP}/{SESSION}"),
        (f"http://127.0.0.1:8889/drone/sim-01/main/whep/{SESSION}?x=1", f"{WHEP}/{SESSION}"),
        ("/somewhere/else", None),
        (None, None),
        ("/drone/sim-01/main/whep/../../x", None),
    ],
)
def test_rewrite_location(location: str | None, expected: str | None) -> None:
    assert rewrite_location(location, "sim-01") == expected


# --- MediaMTX's state -----------------------------------------------------------------


def test_monitor_states(video_ground: Ground, mtx: FakeMtx) -> None:
    async def state() -> str:
        async with client() as http:
            return await VideoMonitor(video_ground).state(http)

    assert asyncio.run(state()) == "ok"
    mtx.ready = False
    assert asyncio.run(state()) == "degraded"


def test_monitor_down(make_ground: Callable[..., Ground]) -> None:
    g = make_ground(video={"mediamtx_api": "http://127.0.0.1:9"})

    async def state() -> str:
        async with client() as http:
            return await VideoMonitor(g).state(http)

    assert asyncio.run(state()) == "down"
    assert health.parts(g)["mediamtx"] == "down"
    g.video_state = "ok"
    assert health.parts(g)["mediamtx"] == "ok"


# --- the recording per mission --------------------------------------------------------


def _segment(folder: Path, start: dt.datetime, size: int = 100, age_s: float = 60.0) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{start.strftime(NAME_FORMAT)}.mp4"
    path.write_bytes(b"x" * size)
    mtime = (dt.datetime.now(dt.UTC) - dt.timedelta(seconds=age_s)).timestamp()
    os.utime(path, (mtime, mtime))
    return path


def test_segment_file_exact_and_near(tmp_path: Path) -> None:
    exact = _segment(tmp_path, T0)
    assert segment_file(tmp_path, T0) == exact
    near = _segment(tmp_path, T0 + dt.timedelta(minutes=1, microseconds=400_000))
    assert segment_file(tmp_path, T0 + dt.timedelta(minutes=1)) == near
    assert segment_file(tmp_path, T0 + dt.timedelta(minutes=5)) is None


def test_plan_picks_overlapping_segments(tmp_path: Path) -> None:
    starts = [T0 + dt.timedelta(minutes=i) for i in range(4)]
    for s in starts[:3]:
        _segment(tmp_path, s)
    _segment(tmp_path, starts[3], age_s=0.0)  # still being written
    now = dt.datetime.now(dt.UTC)
    chosen, missing = plan_segments(
        starts, tmp_path, T0 + dt.timedelta(seconds=90), T0 + dt.timedelta(minutes=10), now
    )
    assert [c.start for c in chosen] == starts[1:]
    assert [c.finished for c in chosen] == [True, True, False]
    assert missing == []
    # A start whose file is gone is reported.
    gone = T0 + dt.timedelta(minutes=4)
    _, missing = plan_segments([*starts, gone], tmp_path, T0, gone + dt.timedelta(1), now)
    assert missing == [gone]


@dataclass
class FakeRepo:
    rows: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    async def add_file(self, **row: Any) -> None:
        self.rows.append(row)


@dataclass
class FakeFlow:
    g: Ground
    repo: FakeRepo = field(default_factory=FakeRepo)

    def now(self) -> dt.datetime:
        return dt.datetime.now(dt.UTC)

    def mission_dir(self, mission_id: str) -> Path:
        return self.g.loaded.data_dir / "missions" / mission_id


@dataclass
class FakeMission:
    id: str = "m1"
    vehicle_id: str = "sim-01"
    activated_at: dt.datetime | None = None
    landed_at: dt.datetime | None = None


def test_close_mission_video_links_and_copies(video_ground: Ground, mtx: FakeMtx) -> None:
    source = video_ground.loaded.data_dir / "video" / "drone" / "sim-01" / "main"
    done = _segment(source, T0, size=1000)
    _segment(source, T0 + dt.timedelta(minutes=1), size=500, age_s=0.0)
    mtx.segments = ["2026-10-17T12:00:00Z", "2026-10-17T12:01:00Z"]
    flow = FakeFlow(video_ground)
    mission = FakeMission(activated_at=T0 + dt.timedelta(seconds=30))
    asyncio.run(close_mission_video(flow, mission))  # pyright: ignore[reportArgumentType]
    folder = flow.mission_dir("m1") / "video"
    linked = folder / done.name
    assert linked.stat().st_ino == done.stat().st_ino  # a hard link, not a copy
    assert [r["notes"] for r in flow.repo.rows] == [None, "partial"]
    assert {r["kind"] for r in flow.repo.rows} == {"video"}
    assert all(r["rel_path"].startswith("video/") for r in flow.repo.rows)
    assert all(len(r["sha256"]) == 64 for r in flow.repo.rows)
    index = json.loads((folder / "index.json").read_text())
    assert len(index["segments"]) == 2


def test_close_mission_video_without_flight_or_server(
    make_ground: Callable[..., Ground],
) -> None:
    g = make_ground(video={"mediamtx_api": "http://127.0.0.1:9"})
    flow = FakeFlow(g)
    asyncio.run(close_mission_video(flow, FakeMission()))  # pyright: ignore[reportArgumentType]
    assert flow.repo.rows == []
    asyncio.run(close_mission_video(flow, FakeMission(activated_at=T0)))  # pyright: ignore[reportArgumentType]
    assert flow.repo.rows == []
    assert any(e["type"] == "video.unavailable" for e in g._events)  # pyright: ignore[reportPrivateUsage]
