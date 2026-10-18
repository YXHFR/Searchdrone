"""The Searchdrone services the launcher starts around SITL: database, ground, onboard, AI.

The environment of each process is built by a pure function from the instance's ports
and folders, so a test can compare it without starting anything. Secrets never go into
argv or these environments: ``sd-ground`` and ``sd-brain`` read ``.env`` themselves, and
the tokens are files in ``build/sim/i<N>/secrets/`` (folder 0700, files 0600).

Each instance gets its own database ``searchdrone_i<N>``, made anew at every start, so
two instances (or two runs) never see each other's missions.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sd_sim.errors import InfraError
from sd_sim.util import LOOPBACK_OPENER

__all__ = [
    "DB_HOST",
    "DB_PORT",
    "VEHICLE_ID",
    "Health",
    "StackPaths",
    "brain_env",
    "database_name",
    "ground_cli",
    "ground_env",
    "onboard_env",
    "read_health",
    "recreate_database",
    "tcp_open",
]

DB_HOST = "127.0.0.1"
DB_PORT = 5433  # ports.yaml postgres_local
VEHICLE_ID = "sim-01"
CLI_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class StackPaths:
    """Where one launched stack keeps its files."""

    repo: Path
    run_dir: Path  # runs/<run>/: logs, SITL work folder, onboard state
    secrets_dir: Path  # build/sim/i<N>/secrets/
    data_dir: Path  # the ground station's data folder (missions, logs, control socket)
    onboard_dir: Path  # sd-onboard's link token and state, outside the repository

    @property
    def brain_token(self) -> Path:
        return self.secrets_dir / "brain_token"

    @property
    def link_token(self) -> Path:
        # sd-onboard refuses a token file or state folder inside the repository.
        return self.onboard_dir / "link_token"

    @property
    def link_tokens(self) -> Path:
        return self.secrets_dir / "link_tokens.toml"


def database_name(instance: int) -> str:
    return f"searchdrone_i{instance}"


def ground_env(
    ports: dict[str, int],
    paths: StackPaths,
    instance: int,
    *,
    video: bool,
    lan: bool,
    test_rig: bool,
) -> dict[str, str]:
    """``sd-ground``'s settings for one instance (no secrets: it reads ``.env`` itself)."""
    db = f"{DB_HOST}:{DB_PORT}/{database_name(instance)}"
    env = {
        "SD_GROUND__PROFILE": "sim",
        "SD_GROUND__DB__URL": f"postgresql+asyncpg://sd_app@{db}",
        "SD_GROUND__DB__OWNER_URL": f"postgresql+asyncpg://sd_owner@{db}",
        "SD_GROUND__AUTH__BRAIN_TOKEN_FILE": str(paths.brain_token),
        "SD_GROUND__AUTH__CLI_MAY_CONFIRM": "true" if test_rig else "false",
        "SD_GROUND__HTTP__PANEL_BIND": f"127.0.0.1:{ports['ground_panel']}",
        "SD_GROUND__HTTP__COMMANDER_BIND": f"127.0.0.1:{ports['ground_commander']}",
        "SD_GROUND__SDLINK__BIND": f"127.0.0.1:{ports['ground_sdlink']}",
        "SD_GROUND__SDLINK__LINK_TOKENS_FILE": str(paths.link_tokens),
        "SD_GROUND__MAVLINK__LISTEN": f"udpin:127.0.0.1:{ports['ground_mavlink']}",
        "SD_GROUND__DATA_DIR": str(paths.data_dir),
        # The ground station checks that .env's SD_DATA_DIR and its data_dir agree; the
        # process environment wins over .env, so both name the instance's folder.
        "SD_DATA_DIR": str(paths.data_dir),
        "SD_GROUND__SIM__DATAFLASH_DIR": str(paths.run_dir / "sitl" / "logs"),
        "SD_GROUND__VIDEO__ENABLED": "true" if video else "false",
        "SD_GROUND__REMOTE__ENABLED": "false",
        "SD_GROUND__LAN__ENABLED": "true" if lan else "false",
    }
    if "ground_lan" in ports:
        env["SD_GROUND__LAN__BIND"] = f"0.0.0.0:{ports['ground_lan']}"
    return env


def onboard_env(ports: dict[str, int], paths: StackPaths) -> dict[str, str]:
    """``sd-onboard`` in the simulator: MAVLink through mavtap, SD-link through linksim."""
    return {
        "SD_ONBOARD__VEHICLE_ID": VEHICLE_ID,
        "SD_ONBOARD__PROFILE": "sim",
        "SD_ONBOARD__MAV_URL": f"udpin:127.0.0.1:{ports['onboard_mavlink']}",
        "SD_ONBOARD__SDLINK_URL": f"ws://127.0.0.1:{ports['linksim_sdlink']}/sdlink/v1/control",
        "SD_ONBOARD__LINK_TOKEN_FILE": str(paths.link_token),
        "SD_ONBOARD__STATE_DIR": str(paths.onboard_dir / "state"),
        # SITL has no RC transmitter; the onboard computer refuses this outside "sim".
        "SD_ONBOARD__REQUIRE_RC": "false",
        "SD_DATA_DIR": str(paths.data_dir),
    }


def brain_env(
    ports: dict[str, int],
    paths: StackPaths,
    kind: str,
    script: Path | None,
    *,
    cloud: bool = True,
) -> dict[str, str]:
    """``sd-brain`` against this instance's commander API (keys: it reads ``.env``).

    ``cloud=False`` turns the cloud model off whatever ``.env`` and ``brain.toml`` say
    (an empty model disables it; the process environment wins), so only the local model
    answers (``just perf``)."""
    env = {
        "SD_BRAIN__PROFILE": "sim",
        "SD_BRAIN__TOKEN_FILE": str(paths.brain_token),
        "SD_BRAIN__GROUND__COMMANDER_URL": (
            f"http://127.0.0.1:{ports['ground_commander']}/commander/v1"
        ),
        "SD_BRAIN__GROUND__VEHICLE_ID": VEHICLE_ID,
        "SD_BRAIN__BACKEND__KIND": kind,
        "SD_BRAIN__LLM__ROOT_URL": f"http://127.0.0.1:{ports['llm_http']}",
    }
    if kind == "scripted":
        if script is None:
            raise InfraError("brain=scripted kräver ett manus (script=...).")
        env["SD_BRAIN__BACKEND__SCRIPTED_FILE"] = str(script)
    if not cloud:
        env["SD_BRAIN__CLOUD__MODEL"] = ""
    return env


def tcp_open(port: int, host: str = DB_HOST, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


async def _recreate(password: str, name: str, port: int) -> None:
    import asyncpg  # the ground station's driver, in the same workspace

    admin = await asyncpg.connect(
        host=DB_HOST, port=port, user="sd_owner", password=password, database="postgres"
    )
    try:
        ident = '"' + name.replace('"', '""') + '"'
        await admin.execute(f"DROP DATABASE IF EXISTS {ident} WITH (FORCE)")
        await admin.execute(f"CREATE DATABASE {ident} OWNER sd_owner")
    finally:
        await admin.close()
    db = await asyncpg.connect(
        host=DB_HOST, port=port, user="sd_owner", password=password, database=name
    )
    try:
        # PostGIS is not a trusted extension: the owner (superuser in the container).
        await db.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    finally:
        await db.close()


def recreate_database(password: str, name: str, port: int = DB_PORT) -> None:
    """Drop and create the instance's database with PostGIS (as ``sd_owner``)."""
    try:
        asyncio.run(_recreate(password, name, port))
    except OSError as exc:
        raise InfraError(
            f"Databasen svarar inte på {DB_HOST}:{port} ({type(exc).__name__}). Kör just db-up."
        ) from None
    except Exception as exc:  # asyncpg's own errors: a wrong password, a missing role
        raise InfraError(f"Databasen {name} gick inte att skapa: {type(exc).__name__}") from None


def ground_cli(
    repo: Path, env: dict[str, str], *args: str, stdout_file: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """``sd-ground <args>`` with the service's environment; stdout optionally to a 0600 file
    (a token is never in this process's output)."""
    cmd = [sys.executable, "-m", "sd_ground", *args]
    if stdout_file is None:
        res = subprocess.run(  # noqa: S603 - fixed argv, no shell
            cmd,
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=CLI_TIMEOUT_S,
        )
    else:
        fd = os.open(stdout_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            res = subprocess.run(  # noqa: S603 - fixed argv, no shell
                cmd,
                cwd=repo,
                env=env,
                stdout=out,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=CLI_TIMEOUT_S,
            )
    if res.returncode != 0:
        raise InfraError(f"sd-ground {args[0]} misslyckades: {(res.stderr or '').strip()[-500:]}")
    return res


@dataclass(frozen=True)
class Health:
    ready: bool
    parts: dict[str, str]

    def ok(self, *names: str) -> bool:
        return all(self.parts.get(name) == "ok" for name in names)


def read_health(port: int) -> Health | None:
    """``GET /api/v1/health`` on the panel port; ``None`` while the server does not answer."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/v1/health")
    try:
        with LOOPBACK_OPENER.open(req, timeout=2) as resp:
            body: Any = json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    parts = body.get("parts") or {}
    return Health(bool(body.get("ready")), {str(k): str(v) for k, v in dict(parts).items()})
