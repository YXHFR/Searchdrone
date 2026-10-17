"""Live video end to end: MediaMTX from compose.yaml, the synthetic camera with
the time strip, the RTSP relay, sd-ground's WHEP proxy and Google Chrome.

Run with ``just test-video`` (it starts MediaMTX and builds the panel first). Needs
Docker, GStreamer with an H.264 encoder, ffprobe and Google Chrome. Not part of the SITL
selection (no ``sitl`` marker).

* point 1 and 2: Chrome shows the picture within 5 s and the median latency of 50 valid
  samples of the time strip is under 1 000 ms (``e2e/video/live.spec.ts``);
* point 3: the mission's recording, taken from MediaMTX's segments, is H.264 and at least
  as long as the mission minus 5 s (``ffprobe``);
* point 5: every MediaMTX TCP port on 127.0.0.1 only, also in LAN mode, where only UDP
  8189 moves to the LAN address; WHEP straight to the LAN address fails, through
  sd-ground's LAN listener it works with a session and gives 401 without;
* point 6: publishing with ``CHANGE_ME`` or a wrong password is refused.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from sd_ground.config import load, read_env_file
from sd_ground.context import lan_addresses
from sd_ground.services import video as video_service
from sd_sim.video_relay import rtsp

pytestmark = pytest.mark.video

ROOT = Path(__file__).resolve().parents[3]
API = "http://127.0.0.1:9997"
PATH = "drone/sim-01/main"
RELAY_UDP = 5602
PANEL_DIST = ROOT / "software" / "panel" / "dist" / "index.html"
BRAIN = "b" * 43
REQUIRE_LAN = os.environ.get("SD_VIDEO_REQUIRE_LAN") == "1"
MISSION_S = 30.0


def _env() -> dict[str, str]:
    return read_env_file(ROOT / ".env")


def _password() -> str:
    value = _env().get("SD_GROUND__VIDEO_PUBLISH_PASSWORD", "")
    assert value and value.lower() != "change_me", "kör just setup-config"
    return value


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _compose_up(lan_ip: str | None) -> None:
    env = {k: v for k, v in os.environ.items() if k not in ("SD_VIDEO_BIND", "SD_LAN_IP")}
    if lan_ip:
        env |= {"SD_VIDEO_BIND": lan_ip, "SD_LAN_IP": lan_ip}
    subprocess.run(
        ["docker", "compose", "--env-file", ".env", "up", "-d", "--force-recreate", "mediamtx"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{API}/v3/config/global/get", timeout=1).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.3)
    raise AssertionError("MediaMTX startade inte")


def _wait_ready(seconds: float = 30.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            res = httpx.get(f"{API}/v3/paths/get/{PATH}", timeout=1)
            if res.status_code == 200 and res.json().get("ready"):
                return
        except httpx.TransportError:
            pass
        time.sleep(0.3)
    raise AssertionError("Videon kom aldrig fram till MediaMTX")


def _listeners() -> list[tuple[str, str]]:
    """(protocol, local address) of every socket MediaMTX listens on."""
    out = subprocess.run(["ss", "-ltnupH"], capture_output=True, text=True, check=True).stdout
    found = []
    for line in out.splitlines():
        if '"mediamtx"' in line:
            cols = line.split()
            found.append((cols[0], cols[4]))
    return found


@dataclass
class Chain:
    procs: list[subprocess.Popen[bytes]] = field(default_factory=list[subprocess.Popen[bytes]])
    started: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))

    def stop(self) -> None:
        for p in reversed(self.procs):
            p.send_signal(signal.SIGINT)
        for p in self.procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


@pytest.fixture(scope="module")
def chain(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Chain]:
    """MediaMTX on loopback, the relay and the synthetic camera, for the whole module."""
    if shutil.which("docker") is None or shutil.which("gst-launch-1.0") is None:
        pytest.fail("Docker och GStreamer krävs (just test-video)")
    _compose_up(None)
    secret = tmp_path_factory.mktemp("secret") / "mediamtx_publish"
    secret.write_text(_password())
    secret.chmod(0o600)
    py = [sys.executable, "-m", "sd_sim.video_relay"]
    c = Chain()
    c.procs.append(
        subprocess.Popen(
            [*py, "relay", "--udp-port", str(RELAY_UDP), "--password-file", str(secret)],
            cwd=ROOT,
        )
    )
    c.procs.append(subprocess.Popen([*py, "source", "--port", str(RELAY_UDP)], cwd=ROOT))
    try:
        _wait_ready()
        c.started = dt.datetime.now(dt.UTC)
        yield c
    finally:
        c.stop()
        _compose_up(None)  # leave MediaMTX in its normal loopback mode


@dataclass
class Ground:
    proc: subprocess.Popen[str]
    base_url: str
    env: dict[str, str]

    def login_code(self) -> str:
        res = subprocess.run(
            [sys.executable, "-m", "sd_ground", "login-link", "--print-code"],
            cwd=ROOT,
            env={**os.environ, **self.env},
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()


def _start_ground(tmp: Path, lan_ip: str | None) -> Ground:
    data = Path(_env().get("SD_DATA_DIR") or tmp / "data")
    token = tmp / "secrets" / "brain_token"
    token.parent.mkdir(mode=0o700, exist_ok=True)
    token.write_text(BRAIN)
    token.chmod(0o600)
    panel_port = _free_port()
    env = {
        "SD_GROUND__HTTP__PANEL_BIND": f"127.0.0.1:{panel_port}",
        "SD_GROUND__HTTP__COMMANDER_BIND": f"127.0.0.1:{_free_port()}",
        "SD_GROUND__SDLINK__BIND": f"127.0.0.1:{_free_port()}",
        "SD_GROUND__MAVLINK__LISTEN": f"udpin:127.0.0.1:{_free_port()}",
        "SD_GROUND__DATA_DIR": str(data),
        "SD_DATA_DIR": str(data),
        "SD_GROUND__AUTH__BRAIN_TOKEN_FILE": str(token),
        # No database: the panel and the video work in the safe start mode.
        "SD_GROUND__DB__URL": f"postgresql+asyncpg://sd_app@127.0.0.1:{_free_port()}/none",
        "SD_GROUND__DB__CONNECT_TIMEOUT_S": "1",
        "XDG_RUNTIME_DIR": str(tmp),
    }
    base = f"http://127.0.0.1:{panel_port}"
    if lan_ip:
        from sd_ground.__main__ import _make_cert  # pyright: ignore[reportPrivateUsage]

        tls = tmp / "tls"
        _make_cert(tls)
        lan_port = _free_port()
        env |= {
            "SD_GROUND__LAN__ENABLED": "true",
            "SD_GROUND__LAN__BIND": f"0.0.0.0:{lan_port}",
            "SD_GROUND__LAN__TLS_DIR": str(tls),
        }
        base = f"https://{lan_ip}:{lan_port}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "sd_ground", "serve"],
        cwd=ROOT,
        env={**os.environ, **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(
                f"sd-ground avslutades: {proc.stdout.read() if proc.stdout else ''}"
            )
        try:
            res = httpx.get(f"http://127.0.0.1:{panel_port}/api/v1/health", timeout=1)
            if res.status_code == 200 and res.json()["parts"]["mediamtx"] == "ok":
                return Ground(proc, base, env)
        except httpx.TransportError:
            pass
        time.sleep(0.3)
    proc.kill()
    raise AssertionError("sd-ground blev inte redo med mediamtx = ok")


def _stop_ground(g: Ground) -> None:
    g.proc.send_signal(signal.SIGTERM)
    try:
        g.proc.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        g.proc.kill()


def _run_chrome(g: Ground, tmp: Path) -> dict[str, Any]:
    result = tmp / "video-result.json"
    pnpm = shutil.which("pnpm")
    assert pnpm, "pnpm saknas"
    res = subprocess.run(
        [
            pnpm,
            "--filter",
            "@searchdrone/panel",
            "exec",
            "playwright",
            "test",
            "-c",
            "playwright.video.config.ts",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "SD_VIDEO_BASE_URL": g.base_url,
            "SD_VIDEO_LOGIN_CODE": g.login_code(),
            "SD_VIDEO_RESULT": str(result),
        },
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    sys.stdout.write(res.stdout[-4000:])
    assert res.returncode == 0, f"Chrome-testet föll:\n{res.stdout[-4000:]}\n{res.stderr[-2000:]}"
    return json.loads(result.read_text())


def _lan_ip() -> str | None:
    ips = [a for a in lan_addresses() if not a.endswith(".local")]
    return ips[0] if ips else None


# --- point 6: the publish password ------------------------------------------------------


@pytest.mark.parametrize("password", ["CHANGE_ME", "wrong-password-123"])
def test_publish_with_a_wrong_password_is_refused(chain: Chain, password: str) -> None:
    async def attempt() -> None:
        pub = rtsp.RtspPublisher(
            rtsp.stream_url(8554, "sim-99"), "publisher", password, _free_port()
        )
        reader, writer = await asyncio.open_connection("127.0.0.1", 8554)
        try:
            await pub.handshake(reader, writer)
        finally:
            writer.close()

    with pytest.raises(rtsp.PublishRefusedError):
        asyncio.run(attempt())


def test_running_config_has_no_placeholder(chain: Chain) -> None:
    cfg = httpx.get(f"{API}/v3/config/global/get").json()
    assert cfg["webrtcAddress"] == "127.0.0.1:8889"
    assert cfg["rtspTransports"] == ["tcp"]
    publisher = next(u for u in cfg["authInternalUsers"] if u["user"] == "publisher")
    assert publisher["pass"] != "CHANGE_ME"
    assert publisher["ips"] == ["127.0.0.1/32"]


# --- point 5: ports ---------------------------------------------------------------------


def test_every_port_on_loopback(chain: Chain) -> None:
    found = _listeners()
    assert found, "ss hittar inte mediamtx"
    assert all(addr.startswith("127.0.0.1:") for _, addr in found), found
    ports = {addr.rsplit(":", 1)[1] for _, addr in found}
    assert {"8554", "8888", "8889", "9997", "8189"} <= ports
    assert not ports & {
        "8000",
        "8001",
        "8890",
        "8892",
        "8893",
        "1935",
    }  # no RTP/UDP, SRT, MoQ, RTMP


# --- points 1 and 2: Chrome on loopback ---------------------------------------------------


def test_panel_shows_live_video_with_low_latency(chain: Chain, tmp_path: Path) -> None:
    assert PANEL_DIST.is_file(), "bygg panelen först (pnpm --filter @searchdrone/panel build)"
    g = _start_ground(tmp_path, None)
    try:
        result = _run_chrome(g, tmp_path)
    finally:
        _stop_ground(g)
    assert result["shown_ms"] <= 5000
    assert result["samples"] == 50 and result["invalid"] <= 5
    assert result["median_ms"] < 1000


# --- point 3: the recording per mission ---------------------------------------------------


@dataclass
class _Repo:
    rows: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    async def add_file(self, **row: Any) -> None:
        self.rows.append(row)


@dataclass
class _Flow:
    g: Any
    folder: Path
    repo: _Repo = field(default_factory=_Repo)

    def now(self) -> dt.datetime:
        return dt.datetime.now(dt.UTC)

    def mission_dir(self, mission_id: str) -> Path:
        return self.folder / mission_id


@dataclass
class _Mission:
    activated_at: dt.datetime
    landed_at: dt.datetime
    id: str = "m-video"
    vehicle_id: str = "sim-01"


@dataclass
class _G:
    loaded: Any
    events: list[str] = field(default_factory=list[str])

    @property
    def config(self) -> Any:
        return self.loaded.config

    def add_event(self, type_: str, **_: Any) -> None:
        self.events.append(type_)


def _ffprobe(path: Path) -> tuple[str, float]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    info = json.loads(out)
    return info["streams"][0]["codec_name"], float(info["format"]["duration"])


def test_recording_per_mission(chain: Chain, tmp_path: Path) -> None:
    assert shutil.which("ffprobe"), "ffprobe saknas (apt install ffmpeg)"
    activated = chain.started + dt.timedelta(seconds=2)
    landed = activated + dt.timedelta(seconds=MISSION_S)
    wait = (landed - dt.datetime.now(dt.UTC)).total_seconds() + 1.5
    if wait > 0:
        time.sleep(wait)
    data = Path(_env()["SD_DATA_DIR"]).expanduser()
    loaded = load(ROOT, {"SD_GROUND__DATA_DIR": str(data)})
    g = _G(loaded)
    flow = _Flow(g, tmp_path / "missions")
    mission = _Mission(activated_at=activated, landed_at=landed)
    asyncio.run(video_service.close_mission_video(flow, mission))  # pyright: ignore[reportArgumentType]
    assert flow.repo.rows, f"inga segment togs med (händelser: {g.events})"
    total = 0.0
    for row in flow.repo.rows:
        codec, duration = _ffprobe(flow.mission_dir(mission.id) / row["rel_path"])
        assert codec == "h264"
        total += duration
    assert total >= MISSION_S - 5, f"{total:.1f} s video för ett uppdrag på {MISSION_S} s"
    assert (flow.mission_dir(mission.id) / "video" / "index.json").is_file()


# --- point 5: LAN mode --------------------------------------------------------------------


def test_lan_mode(chain: Chain, tmp_path: Path) -> None:
    lan = _lan_ip()
    if lan is None:
        if REQUIRE_LAN:
            pytest.fail("ingen LAN-adress på maskinen")
        pytest.skip("ingen privat LAN-adress (körs i CI med SD_VIDEO_REQUIRE_LAN=1)")
    _compose_up(lan)
    _wait_ready()
    found = _listeners()
    tcp = [addr for proto, addr in found if proto == "tcp"]
    udp = [addr for proto, addr in found if proto == "udp"]
    assert tcp and all(addr.startswith("127.0.0.1:") for addr in tcp), tcp
    assert f"{lan}:8189" in udp, udp
    assert not any(addr.endswith(":8889") and not addr.startswith("127.") for addr in tcp)
    direct = subprocess.run(
        [
            "curl",
            "-sS",
            "-m",
            "3",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/sdp",
            "--data",
            "v=0",
            f"http://{lan}:8889/{PATH}/whep",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert direct.returncode != 0, "WHEP svarar direkt på LAN-adressen"
    g = _start_ground(tmp_path, lan)
    try:
        result = _run_chrome(g, tmp_path)  # also checks 401 without a session
    finally:
        _stop_ground(g)
    assert result["median_ms"] < 1000
