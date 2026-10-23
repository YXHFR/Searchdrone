"""Small helpers shared by the launcher, the smoke test and the doctor."""

from __future__ import annotations

import socket
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from sd_sim import paths

# Ports that carry UDP; every other simulator port is TCP.
UDP_PORTS = frozenset(
    {
        "ground_mavlink",
        "radio_mavlink_in",
        "mavtap_in",
        "onboard_mavlink",
        "harness_mavlink",
        "qgc_mavlink",
        "sitl_json_fdm",
        "sitl_rc_in",
        "video_raw_udp",
        "video_out_udp",
        "mediamtx_webrtc_udp",
        "gz_discovery_msg",
        "gz_discovery_srv",
    }
)

# Calls to our own loopback services never go through an HTTP proxy (the bearer
# token must not reach a proxy, and a proxy cannot reach 127.0.0.1 anyway).
LOOPBACK_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def say(message: str) -> None:
    print(message, flush=True)  # noqa: T201 - command-line output


def new_run_dir(instance: int, label: str = "sim", repo: Path = paths.REPO) -> Path:
    """A new, empty run folder, never one an earlier run made: a test that fails at once
    and the next one on the same instance can start in the same second."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    runs = paths.runs_dir(repo)
    runs.mkdir(parents=True, exist_ok=True)
    run, n = runs / f"{stamp}-{label}-i{instance}", 1
    while True:
        try:
            run.mkdir(exist_ok=False)
            return run
        except FileExistsError:
            n += 1
            run = runs / f"{stamp}-{label}-i{instance}-{n}"


def udp_port_taken(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


def tcp_port_taken(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


def port_taken(name: str, port: int) -> bool:
    """Is the named simulator port in use, for the protocol it uses?"""
    return udp_port_taken(port) if name in UDP_PORTS else tcp_port_taken(port)


def busy_ports(ports: dict[str, int], names: list[str] | tuple[str, ...]) -> list[int]:
    return [ports[name] for name in names if name in ports and port_taken(name, ports[name])]
