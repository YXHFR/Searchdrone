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
    pid: int | None = None

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
        process, pid = "", None
        if len(cols) > 6:
            found = re.search(r'\("([^"]+)",pid=(\d+)', " ".join(cols[6:]))
            if found:
                process, pid = found.group(1), int(found.group(2))
        out.append(Listener(cols[0], match["addr"], int(match["port"]), process, pid))
    return out


def load_allowlist(path: Path = paths.LISTEN_ALLOWLIST) -> set[tuple[str, str]]:
    """(port name, protocol) pairs that may listen outside loopback. A random port of a
    named process is the pair ("ephemeral:<process>", protocol)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    allowed: set[tuple[str, str]] = set()
    for row in raw:
        if not str(row.get("reason", "")).strip():
            raise ValueError(f"{path}: raden {row.get('port')} saknar skäl")
        name = str(row["port"])
        if name == "ephemeral":
            if not row.get("process"):
                raise ValueError(f"{path}: en ephemeral-rad måste ange process")
            name = f"ephemeral:{row['process']}"
        allowed.add((name, str(row["proto"])))
    return allowed


def violations(
    listeners: list[Listener],
    ports_by_name: dict[str, int],
    allowlist: set[tuple[str, str]],
    owned_pids: set[int] | frozenset[int] = frozenset(),
) -> list[Listener]:
    """Non-loopback listeners that are not allow-listed, on a simulator port or owned by
    one of our processes (so an unexpected port of ours is caught too)."""
    names_by_port: dict[int, set[str]] = {}
    for name, port in ports_by_name.items():
        names_by_port.setdefault(port, set()).add(name)
    bad: list[Listener] = []
    for item in listeners:
        if item.addr in LOOPBACK:
            continue
        names = names_by_port.get(item.port, set())
        if not names and item.pid not in owned_pids:
            continue
        if not names and (f"ephemeral:{item.process}", item.proto) in allowlist:
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
