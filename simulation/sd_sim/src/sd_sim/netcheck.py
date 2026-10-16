"""Check that simulator ports listen on loopback only (``ss -ltnupH``)."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from sd_sim import paths

LOOPBACK = {"127.0.0.1", "::1", "[::1]", "::ffff:127.0.0.1"}
_ADDR = re.compile(r"^(?P<addr>\[[^\]]*\]|[^%\s]*?)(?:%\S+)?:(?P<port>\d+|\*)$")


@dataclass(frozen=True)
class Listener:
    proto: str
    addr: str
    port: int
    process: str

    def __str__(self) -> str:
        return f"{self.proto} {self.addr}:{self.port} ({self.process or 'okänd process'})"


def parse_ss(output: str) -> list[Listener]:
    """Parse ``ss -ltnupH``: netid, state, recv-q, send-q, local, peer[, process]."""
    out: list[Listener] = []
    for line in output.splitlines():
        cols = line.split()
        if len(cols) < 5 or cols[0] not in ("tcp", "udp"):
            continue
        match = _ADDR.match(cols[4])
        if match is None or match["port"] == "*":
            continue
        process = ""
        if len(cols) > 6:
            found = re.search(r'\("([^"]+)"', " ".join(cols[6:]))
            process = found.group(1) if found else ""
        out.append(Listener(cols[0], match["addr"], int(match["port"]), process))
    return out


def load_allowlist(path: Path = paths.LISTEN_ALLOWLIST) -> set[tuple[str, str]]:
    """(port name, protocol) pairs that may listen outside loopback."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    allowed: set[tuple[str, str]] = set()
    for row in raw:
        if not str(row.get("reason", "")).strip():
            raise ValueError(f"{path}: raden {row.get('port')} saknar skäl")
        allowed.add((str(row["port"]), str(row["proto"])))
    return allowed


def violations(
    listeners: list[Listener],
    ports_by_name: dict[str, int],
    allowlist: set[tuple[str, str]],
) -> list[Listener]:
    """Listeners on a simulator port that are not on loopback and not allow-listed."""
    names_by_port: dict[int, set[str]] = {}
    for name, port in ports_by_name.items():
        names_by_port.setdefault(port, set()).add(name)
    bad: list[Listener] = []
    for item in listeners:
        names = names_by_port.get(item.port)
        if not names or item.addr in LOOPBACK:
            continue
        if any((name, item.proto) in allowlist for name in names):
            continue
        bad.append(item)
    return bad


def run_ss() -> str:
    exe = shutil.which("ss")
    if exe is None:
        raise FileNotFoundError("ss saknas (installera iproute2)")
    return subprocess.run(  # noqa: S603 - fixed arguments
        [exe, "-ltnupH"], capture_output=True, text=True, check=True
    ).stdout
