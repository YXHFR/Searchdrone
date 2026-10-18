"""Small helpers shared by the launcher, the smoke test and the doctor."""

from __future__ import annotations

import os
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
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run = paths.runs_dir(repo) / f"{stamp}-{label}-i{instance}"
    run.mkdir(parents=True, exist_ok=False)
    return run


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


def read_env_value(path: Path, key: str) -> str:
    """One value from ``.env`` (the process environment wins), without reading the rest."""
    if os.environ.get(key):
        return os.environ[key]
    if not path.is_file():
        return ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        name, sep, value = raw.strip().partition("=")
        if sep and name.strip() == key:
            return value.strip().strip('"').strip("'")
    return ""


def write_secret(directory: Path, name: str, value: str) -> Path:
    """Write a secret file with mode 0600 inside a 0700 folder."""
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    path = directory / name
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        out.write(value)
    path.chmod(0o600)
    return path
