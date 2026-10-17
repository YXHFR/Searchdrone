"""LAN mode: only the WebRTC media port leaves loopback.

MediaMTX is restarted with ``SD_VIDEO_BIND=0.0.0.0`` and the LAN address, the ground
station runs with its LAN listener (:8443, TLS). Then:

* ``ss``: WHEP signalling (8889), HLS, RTSP and the API stay on 127.0.0.1; only UDP 8189
  is on every address;
* ``curl -X POST http://<LAN-IP>:8889/drone/sim-01/main/whep`` fails;
* ``/api/v1/video/sim-01/main/whep`` on :8443 gives 401 without a session, and with a
  session the browser plays the video through it.

MediaMTX is put back to loopback-only afterwards, also when the test fails.
"""

from __future__ import annotations

import ipaddress
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from video_rig import (  # pyright: ignore[reportMissingImports]
    REPO,
    GroundServer,
    VideoChain,
    listeners,
    need_mediamtx,
    playwright,
    stream_ready,
    wait_for,
)

from sd_ground.context import lan_addresses

pytestmark = [pytest.mark.video, pytest.mark.slow]
LAN_VARS = ("SD_VIDEO_BIND", "SD_LAN_IP")


def lan_ip() -> str:
    """The machine's first IPv4 address that is not loopback or a Docker bridge."""
    out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, check=True).stdout
    for word in out.split():
        try:
            addr = ipaddress.ip_address(word)
        except ValueError:
            continue
        if addr.version == 4 and not addr.is_loopback and not word.startswith("172.17."):
            return word
    pytest.skip("ingen LAN-adress på den här datorn")


def compose_mediamtx(**env: str) -> None:
    docker = shutil.which("docker")
    assert docker is not None
    res = subprocess.run(
        [
            docker,
            "compose",
            "--env-file",
            ".env",
            "up",
            "-d",
            "--force-recreate",
            "--wait",
            "mediamtx",
        ],
        cwd=REPO,
        env={**{k: v for k, v in os.environ.items() if k not in LAN_VARS}, **env},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert res.returncode == 0, res.stderr


@pytest.fixture
def lan_mediamtx() -> Iterator[str]:
    need_mediamtx()
    ip = lan_ip()
    compose_mediamtx(SD_VIDEO_BIND="0.0.0.0", SD_LAN_IP=ip)  # noqa: S104 - the LAN test
    try:
        wait_for(lambda: need_mediamtx() is not None, 20, "MediaMTX i LAN-läge")
        yield ip
    finally:
        compose_mediamtx()


def curl(*args: str) -> subprocess.CompletedProcess[str]:
    exe = shutil.which("curl")
    assert exe is not None
    return subprocess.run(
        [exe, "-sS", "--noproxy", "*", "--max-time", "5", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_lan_mode_keeps_signalling_on_loopback(lan_mediamtx: str) -> None:
    ip = lan_mediamtx
    tcp, udp = listeners("t"), listeners("u")
    for port in (8889, 8888, 8554, 9997):
        assert sorted({h for h, p in tcp if p == port}) == ["127.0.0.1"], port
    assert {h for h, p in udp if p == 8189} == {"0.0.0.0"}  # noqa: S104 - the media port
    direct = curl(
        "-X",
        "POST",
        "-H",
        "Content-Type: application/sdp",
        "--data",
        "v=0",
        f"http://{ip}:8889/drone/sim-01/main/whep",
    )
    assert direct.returncode != 0, "WHEP direkt mot MediaMTX på LAN-adressen ska misslyckas"

    with VideoChain.running() as chain:
        wait_for(stream_ready, 15, "videon når MediaMTX")
        hosts = [h for h in lan_addresses() if not h.endswith(".local")] or lan_addresses()
        host = hosts[0]
        base = f"https://{host}:8443"
        resolve = f"{host}:8443:{ip}"
        with GroundServer.running(chain.run_dir, SD_GROUND__LAN__ENABLED="true") as ground:
            no_session = curl(
                "-k",
                "--resolve",
                resolve,
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "-X",
                "POST",
                "-H",
                "X-SD-Client: panel",
                "-H",
                "Content-Type: application/sdp",
                "--data",
                "v=0",
                f"{base}/api/v1/video/sim-01/main/whep",
            )
            assert no_session.stdout == "401", no_session.stdout + no_session.stderr
            result = chain.run_dir / "browser_video_lan.json"
            os.environ["SD_VIDEO_HOST_RULES"] = f"MAP {host} {ip}"
            try:
                code = playwright(base, ground.login_code(), result, ignore_https=True)
            finally:
                os.environ.pop("SD_VIDEO_HOST_RULES", None)
            assert code == 0
    summary = json.loads(result.read_text(encoding="utf-8"))
    assert summary["median"] < 1000, summary
