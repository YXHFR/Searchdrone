"""Where things live: the repository, the version lock and the per-instance build folders."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
SIMULATION = REPO / "simulation"
VERSIONS_LOCK = SIMULATION / "versions.lock"
PORTS_YAML = SIMULATION / "ports.yaml"
LISTEN_ALLOWLIST = SIMULATION / "listen_allowlist.yaml"


def versions() -> dict[str, Any]:
    return tomllib.loads(VERSIONS_LOCK.read_text(encoding="utf-8"))


def instance_dir(instance: int, repo: Path = REPO) -> Path:
    """``build/sim/i<N>/``: pids.json, ports.json and secrets/ (gitignored)."""
    return repo / "build" / "sim" / f"i{instance}"


def runs_dir(repo: Path = REPO) -> Path:
    return repo / "runs"


def ardupilot_home() -> Path:
    return Path(os.environ.get("ARDUPILOT_HOME", str(Path.home() / "ardupilot")))


def arducopter_bin() -> Path:
    default = ardupilot_home() / "build" / "sitl" / "bin" / "arducopter"
    return Path(os.environ.get("SD_ARDUCOPTER_BIN", str(default)))
