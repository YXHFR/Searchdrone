"""Live video and recordings against a running MediaMTX.

Needs ``just video-up`` (MediaMTX from compose.yaml) and GStreamer for Python; the
browser test also needs the database (``just db-up migrate``), the built panel and Google
Chrome. Run with ``just test-video``. CI runs them on every pull request with
``SD_SIM__REQUIRE_VIDEO=1`` (a missing MediaMTX fails instead of skipping). The source
is the synthetic camera with the time code strip (``video=test``, the builtin-physics
picture); the Gazebo camera test is marked ``gz`` and runs on the owner's computer.

* the running configuration and ``ss``: signalling, HLS, RTSP and the API
  only on 127.0.0.1, no RTP/RTCP UDP ports, Media over QUIC off.
* publishing with ``CHANGE_ME`` (and without a password) is refused, with the
  real password it works.
* the stream reaches MediaMTX; the delay measured with the strip; in the
  browser through ``/api/v1/video/sim-01/main/whep`` within 5 s, median < 1 000 ms.
* the mission's video segments in its folder, H.264, long enough.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from video_rig import (  # pyright: ignore[reportMissingImports]
    REPO,
    VID,
    GroundServer,
    VideoChain,
    api_get,
    api_request,
    delay_summary,
    ffprobe,
    listeners,
    need_mediamtx,
    playwright,
    rtsp_frame_count,
    rtsp_samples,
    stream_ready,
    try_publish,
    wait_for,
)

from sd_ground.auth import ServiceToken
from sd_ground.config import load
from sd_ground.context import Ground
from sd_ground.services import video as video_service
from sd_obs.clock import MonotonicClock
from sd_schemas.api import Mission
from sd_schemas.base import new_id
from sd_schemas.mission import MissionState
from sd_schemas.safety import SafetyLimits
from sd_sim import config
from sd_sim.launch import Stack, read_env_value
from sd_sim.report import measure
from sd_sim.util import LOOPBACK_OPENER, new_run_dir

pytestmark = [pytest.mark.video]

LOOPBACK_TCP = {8889: "WHEP", 8888: "HLS", 8554: "RTSP", 9997: "API"}
MAX_MEDIAN_MS = 1000
SAMPLES = 50
MAX_INVALID = 5


@pytest.fixture(scope="module")
def mtx() -> dict[str, Any]:
    """The running configuration, without the users' passwords (never in a traceback)."""
    cfg = need_mediamtx()
    cfg["authInternalUsers"] = [
        {k: v for k, v in u.items() if k != "pass"} for u in cfg["authInternalUsers"]
    ]
    return cfg


@pytest.fixture(scope="module")
def chain(mtx: dict[str, Any]) -> Iterator[VideoChain]:
    with VideoChain.running() as running:
        wait_for(stream_ready, 15, "videon når MediaMTX")
        yield running


def test_running_config_is_loopback_only(mtx: dict[str, Any]) -> None:
    assert mtx["webrtcAddress"] == "127.0.0.1:8889"
    assert mtx["hlsAddress"] == "127.0.0.1:8888"
    assert mtx["rtspAddress"] == "127.0.0.1:8554"
    assert mtx["apiAddress"] == "127.0.0.1:9997"
    assert mtx["rtspTransports"] == ["tcp"]
    assert mtx["rtmp"] is False
    assert mtx["srt"] is False
    assert mtx["moq"] is False
    assert mtx["playback"] is False
    assert mtx["webrtcLocalUDPAddress"].endswith(":8189")
    users = {u["user"]: u for u in mtx["authInternalUsers"]}
    assert users["publisher"]["ips"] == ["127.0.0.1/32"]
    assert [p["action"] for p in users["publisher"]["permissions"]] == ["publish"]
    assert users["any"]["ips"] == ["127.0.0.1/32"]
    assert [p["action"] for p in users["any"]["permissions"]] == ["read"]
    assert [p["action"] for p in users["ground"]["permissions"]] == ["api"]


def test_ports_bind_only_loopback(mtx: dict[str, Any]) -> None:
    tcp = listeners("t")
    for port, name in LOOPBACK_TCP.items():
        hosts = sorted({h for h, p in tcp if p == port})
        assert hosts == ["127.0.0.1"], f"{name} (:{port}) lyssnar på {hosts}"
    udp = listeners("u")
    for port in (8000, 8001, 8892, 8893):
        assert all(p != port for _, p in udp), f"UDP {port} ska inte lyssna"
    for port in (8892, 8890, 1935):
        assert all(p != port for _, p in tcp), f"TCP {port} ska inte lyssna"


@pytest.mark.parametrize(("user", "password"), [("publisher", "CHANGE_ME"), ("any", "")])
def test_publish_without_the_real_password_is_refused(
    mtx: dict[str, Any], user: str, password: str
) -> None:
    result = try_publish(user, password, "sim-99")
    assert result != "published", result
    assert not stream_ready("sim-99")


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/v3/config/global/get"),
        ("PATCH", "/v3/config/global/patch"),
        ("POST", "/v3/config/paths/add/drone/sim-97/main"),
    ],
)
def test_control_api_needs_its_password(mtx: dict[str, Any], method: str, path: str) -> None:
    """No password-less local client (or web page) may read or change the configuration."""
    body = b'{"logLevel": "debug"}' if method != "GET" else None
    status, _ = api_request(path, method=method, body=body, auth=False)
    assert status == 401
    assert api_get("/v3/config/global/get")["logLevel"] == "info"  # pyright: ignore[reportOptionalSubscript]


def test_publish_with_the_real_password_works(mtx: dict[str, Any]) -> None:
    password = read_env_value(REPO / ".env", "SD_GROUND__VIDEO_PUBLISH_PASSWORD")
    assert try_publish("publisher", password, "sim-98") == "published"


def test_stream_reaches_mediamtx_with_low_delay(chain: VideoChain) -> None:
    state = api_get(f"/v3/paths/get/drone/{VID}/main")
    assert state is not None
    assert state["tracks"] == ["H264"]
    summary = delay_summary(rtsp_samples(SAMPLES))
    (chain.run_dir / "rtsp_delay.json").write_text(json.dumps(summary), encoding="utf-8")
    assert summary["valid"] >= SAMPLES - MAX_INVALID, summary
    median_ok = summary["median_ms"] < MAX_MEDIAN_MS
    measure("fördröjning, median", summary["median_ms"], MAX_MEDIAN_MS, unit="ms", passed=median_ok)
    assert summary["median_ms"] < MAX_MEDIAN_MS, summary


def test_radio_off_stops_the_picture(chain: VideoChain) -> None:
    """The video goes through the simulated radio: no radio, no new frames."""
    ports = chain.stack.ports

    def control(body: dict[str, Any]) -> None:
        req = urllib.request.Request(
            f"http://127.0.0.1:{ports['linksim_control']}/radio",
            data=json.dumps(body).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {chain.stack.linksim_token}",
                "Content-Type": "application/json",
            },
        )
        with LOOPBACK_OPENER.open(req, timeout=3):
            pass

    def inbound() -> int:
        state = api_get(f"/v3/paths/get/drone/{VID}/main") or {}
        return int(state.get("inboundBytes", 0))

    control({"state": "down"})
    try:
        time.sleep(1.0)
        before = inbound()
        time.sleep(2.0)
        assert inbound() - before < 2000  # at most a few stray packets
    finally:
        control({"state": "up"})
    before = inbound()
    wait_for(lambda: inbound() > before + 20_000, 10, "videon kommer tillbaka")


def _mission(start: dt.datetime, end: dt.datetime) -> Mission:
    return Mission(
        id=new_id(),
        vehicle_id=VID,
        title="video",
        status=MissionState.LANDED,
        paused_by=None,
        returning_cause=None,
        state_seq=1,
        limits=SafetyLimits(),
        active_charter_id=None,
        autonomy_mode=None,
        ai_enabled=True,
        created_by="operator:demo",
        created_at=start,
        activated_at=start,
        landed_at=end,
        closed_at=None,
        closed_reason=None,
        notes=None,
    )


class _Repo:
    def __init__(self) -> None:
        self.files: list[dict[str, Any]] = []

    async def add_file(self, **row: Any) -> None:
        self.files.append(row)


class _Flow:
    def __init__(self, g: Ground) -> None:
        self.g = g
        self.repo = _Repo()

    def now(self) -> dt.datetime:
        return self.g.clock.utc_now()

    def mission_dir(self, mission_id: str) -> Path:
        return self.g.loaded.data_dir / "missions" / mission_id


@pytest.mark.slow
async def test_mission_recording_is_h264_and_long_enough(chain: VideoChain) -> None:
    loaded = load(REPO)
    g = Ground(loaded=loaded, clock=MonotonicClock(), brain_token=ServiceToken(None))
    start = g.now()
    await asyncio.sleep(25)
    end = g.now()
    await asyncio.sleep(1.5)
    flow: Any = _Flow(g)
    m = _mission(start, end)
    await video_service.close_video(flow, m)
    folder = flow.mission_dir(str(m.id)) / "video"
    mp4s = sorted(folder.glob("*.mp4"))
    assert mp4s, [r["rel_path"] for r in flow.repo.files]
    total = 0.0
    for path in mp4s:
        info = ffprobe(path)
        codecs = [s["codec_name"] for s in info["streams"] if s["codec_type"] == "video"]
        assert codecs == ["h264"], (path.name, codecs)
        total += float(info["format"].get("duration") or 0.0)
    length = (end - start).total_seconds()
    assert total >= length - 5, (total, length)
    rows = {r["rel_path"] for r in flow.repo.files}
    assert {f"video/{p.name}" for p in mp4s} | {"video/index.json"} == rows


@pytest.mark.slow
def test_browser_plays_through_the_proxy(chain: VideoChain) -> None:
    """In a real browser: picture within 5 s, median delay < 1 000 ms."""
    result = chain.run_dir / "browser_video.json"
    with GroundServer.running(chain.run_dir) as ground:
        # The ground station sees the stream through MediaMTX's API (its own user).
        wait_for(lambda: ground.health()["parts"]["mediamtx"] == "ok", 10, "mediamtx ok i /health")
        code = ground.login_code()
        assert playwright("http://localhost:8080", code, result) == 0
    summary = json.loads(result.read_text(encoding="utf-8"))
    first_ok = summary["first_frame_ms"] < 5000
    measure(
        "första bilden i webbläsaren", summary["first_frame_ms"], 5000, unit="ms", passed=first_ok
    )
    assert summary["first_frame_ms"] < 5000, summary
    assert summary["valid"] >= SAMPLES and summary["invalid"] <= MAX_INVALID, summary
    assert summary["median"] < MAX_MEDIAN_MS, summary


@pytest.mark.slow
def test_panel_shows_live_video(chain: VideoChain) -> None:
    """PS-1 in the panel's test build: LIVE · WebRTC within 10 s, the picture moves, >= 12
    frames a second (counted by the panel itself)."""
    with GroundServer.running(
        chain.run_dir, SD_GROUND__HTTP__STATIC_DIR="software/panel/dist-test"
    ) as ground:
        wait_for(lambda: ground.health()["parts"]["mediamtx"] == "ok", 10, "mediamtx ok i /health")
        code = ground.login_code()
        result = chain.run_dir / "panel_video.json"
        assert (
            playwright("http://localhost:8080", code, result, spec="e2e/video/panel.spec.ts") == 0
        )


@pytest.mark.gz
@pytest.mark.slow
def test_gazebo_camera_reaches_mediamtx(mtx: dict[str, Any]) -> None:
    """The Gazebo camera through the radio and the relay: >= 135 frames in 10 s."""
    cfg = config.load({"physics": "gz", "gui": "false", "video": "camera", "instance": "0"})
    stack = Stack(cfg, new_run_dir(0, "video-gz"), observer=True)
    try:
        stack.up()
        wait_for(stream_ready, 30, "kamerans video når MediaMTX")
        assert rtsp_frame_count(10.0) >= 135
    finally:
        stack.down()
