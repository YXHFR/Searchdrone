"""The services the launcher starts around SITL: database, ground station, onboard
computer, AI commander, video server and panel (``just demo``, ``just sim``, ``just e2e``).

Each service is the real program, started the way an operator would start it
(``sd-ground serve``, ``sd-onboard``, ``sd-brain run``), with its settings for this
instance passed as ``SD_<PART>__<KEY>`` variables. Secrets stay in files (mode 0600) or
in ``.env``, which the services read themselves; nothing secret is put in argv.

* Database: Postgres from ``compose.yaml`` on 127.0.0.1:5433. When it already answers
  (CI starts it before the simulator container) it is used as it is. Every instance has
  its own database; ``fresh_db`` drops it first (the end-to-end test).
* Ground station: one data folder per instance (instance 0 uses ``SD_DATA_DIR`` from
  ``.env``, where MediaMTX writes its recordings), its own commander token and SD-link
  token, SITL's dataflash folder for the mission folder.
* Onboard computer: the MAVLink port behind mavtap and SD-link through linksim (the
  simulated radio). SITL has no RC transmitter, so ``require_rc`` is off (``sim`` only).
* AI commander: ``brain=llm_local`` is the language model chain in ``config/brain.toml``
  (the cloud model when its key is in ``.env``); when no model answers, the commander's
  own command mode takes over. The local model server is never started.
  ``brain=scripted`` plays a script, ``brain=grammar`` is the command mode only,
  ``brain=none`` starts no commander.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sd_sim import procs
from sd_sim.errors import InfraError
from sd_sim.util import LOOPBACK_OPENER, say, tcp_port_taken

if TYPE_CHECKING:
    from sd_sim.launch import Stack

__all__ = [
    "BRAIN_KINDS",
    "DB_OWNER",
    "HEALTH_GROUND_S",
    "HEALTH_SERVICES_S",
    "ServiceLayout",
    "brain_env",
    "ground_env",
    "layout_for",
    "onboard_env",
]

DB_OWNER = "sd_owner"
DB_APP = "sd_app"
DB_HOST = "127.0.0.1"
HEALTH_DB_S = 60.0
HEALTH_GROUND_S = 30.0
HEALTH_SERVICES_S = 60.0
VEHICLE_ID = "sim-01"
# sim brain= value -> sd-brain [backend] kind
BRAIN_KINDS = {"llm_local": "llm_local", "scripted": "scripted", "grammar": "grammar"}
PANEL_DIST = Path("software/panel/dist")
# The AI commander in its command mode (no model answers) is ready too: without a cloud
# model that is how the demo runs (only buttons, menus and fixed commands).
READY_STATES = {"brain": ("ok", "degraded")}
DB_PASSWORD_KEY = "SD_GROUND__DB_PASSWORD"  # noqa: S105 - a variable name
DATA_DIR_KEY = "SD_DATA_DIR"


@dataclass(frozen=True)
class ServiceLayout:
    """Where one instance's services keep their things."""

    db_name: str
    data_dir: Path  # outside the repository (the onboard token must be)
    brain_token: Path
    link_tokens: Path
    onboard_token: Path
    dataflash_dir: Path
    static_dir: Path
    brain_config: Path


def default_data_dir(repo: Path) -> Path:
    from sd_sim.launch import read_env_value

    raw = read_env_value(repo / ".env", DATA_DIR_KEY)
    return Path(raw).expanduser() if raw else Path.home() / ".local/share/searchdrone"


def layout_for(stack: Stack) -> ServiceLayout:
    inst = stack.cfg.instance
    base = default_data_dir(stack.repo)
    # Instance 0 is the operator's own folder (MediaMTX records into <SD_DATA_DIR>/video);
    # the other instances get their own, so their control sockets never meet.
    data_dir = base if inst == 0 else base / "sim" / f"i{inst}"
    brain_config = stack.repo / "config" / "brain.toml"
    if not brain_config.is_file():
        brain_config = stack.repo / "config" / "brain.example.toml"
    return ServiceLayout(
        db_name="searchdrone" if inst == 0 and not stack.fresh_db else f"searchdrone_i{inst}",
        data_dir=data_dir,
        brain_token=stack.secrets_dir / "brain_token",
        link_tokens=stack.secrets_dir / "link_tokens.toml",
        onboard_token=data_dir / "secrets" / f"onboard_link_token_i{inst}",
        dataflash_dir=stack.run_dir / "sitl" / "logs",
        static_dir=stack.repo / PANEL_DIST,
        brain_config=brain_config,
    )


def _db_url(driver: str, user: str, port: int, name: str) -> str:
    return f"postgresql+{driver}://{user}@{DB_HOST}:{port}/{name}"


def ground_env(stack: Stack, lay: ServiceLayout, *, video: bool) -> dict[str, str]:
    p = stack.ports
    pg = p["postgres_local"]
    env = {
        DATA_DIR_KEY: str(lay.data_dir),
        "SD_GROUND__DATA_DIR": str(lay.data_dir),
        "SD_GROUND__PROFILE": "sim",
        "SD_GROUND__DEFAULT_VEHICLE_ID": VEHICLE_ID,
        "SD_GROUND__DB__URL": _db_url("asyncpg", DB_APP, pg, lay.db_name),
        "SD_GROUND__DB__OWNER_URL": _db_url("asyncpg", DB_OWNER, pg, lay.db_name),
        "SD_GROUND__HTTP__PANEL_BIND": f"127.0.0.1:{p['ground_panel']}",
        "SD_GROUND__HTTP__COMMANDER_BIND": f"127.0.0.1:{p['ground_commander']}",
        "SD_GROUND__HTTP__STATIC_DIR": str(lay.static_dir),
        "SD_GROUND__SDLINK__BIND": f"127.0.0.1:{p['ground_sdlink']}",
        "SD_GROUND__SDLINK__LINK_TOKENS_FILE": str(lay.link_tokens),
        "SD_GROUND__MAVLINK__LISTEN": f"udpin:127.0.0.1:{p['ground_mavlink']}",
        "SD_GROUND__AUTH__BRAIN_TOKEN_FILE": str(lay.brain_token),
        "SD_GROUND__SIM__DATAFLASH_DIR": str(lay.dataflash_dir),
        "SD_GROUND__VIDEO__ENABLED": "true" if video else "false",
        "SD_GROUND__LAN__ENABLED": "true" if stack.cfg.lan else "false",
    }
    if stack.cfg.lan and "ground_lan" in p:
        env["SD_GROUND__LAN__BIND"] = f"0.0.0.0:{p['ground_lan']}"
    return env


def onboard_env(stack: Stack, lay: ServiceLayout) -> dict[str, str]:
    p = stack.ports
    return {
        DATA_DIR_KEY: str(lay.data_dir),
        "SD_ONBOARD__VEHICLE_ID": VEHICLE_ID,
        "SD_ONBOARD__PROFILE": "sim",
        "SD_ONBOARD__MAV_URL": f"udpin:127.0.0.1:{p['onboard_mavlink']}",
        "SD_ONBOARD__SDLINK_URL": f"ws://127.0.0.1:{p['linksim_sdlink']}/sdlink/v1/control",
        "SD_ONBOARD__LINK_TOKEN_FILE": str(lay.onboard_token),
        "SD_ONBOARD__STATE_DIR": str(
            lay.data_dir / "onboard" / f"{VEHICLE_ID}-i{stack.cfg.instance}"
        ),
        # SITL has no RC transmitter; only the simulator profile may turn the check off.
        "SD_ONBOARD__REQUIRE_RC": "false",
    }


def brain_env(stack: Stack, lay: ServiceLayout) -> dict[str, str]:
    env = {
        "SD_BRAIN__PROFILE": "sim",
        "SD_BRAIN__TOKEN_FILE": str(lay.brain_token),
        "SD_BRAIN__GROUND__COMMANDER_URL": (
            f"http://127.0.0.1:{stack.ports['ground_commander']}/commander/v1"
        ),
        "SD_BRAIN__GROUND__VEHICLE_ID": VEHICLE_ID,
        "SD_BRAIN__BACKEND__KIND": BRAIN_KINDS[stack.cfg.brain],
    }
    if stack.cfg.brain == "scripted":
        script = Path(stack.cfg.script)
        if not script.is_absolute():
            script = stack.repo / script
        env["SD_BRAIN__BACKEND__SCRIPTED_FILE"] = str(script)
    return env


# --- steps --------------------------------------------------------------------------


def _python_argv(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def _run(stack: Stack, step: str, argv: list[str], env: dict[str, str]) -> str:
    """A one-shot command of a service (set-up), its output masked into the run's log."""
    res = subprocess.run(  # noqa: S603 - argv is built here, never a shell
        argv,
        cwd=stack.repo,
        env=procs.clean_env(env),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    log = stack.run_dir / "logs" / f"{step}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as out:
        out.write(stack.mask(res.stdout + res.stderr))
    if res.returncode != 0:
        tail = "\n".join(stack.mask(res.stderr or res.stdout).splitlines()[-15:])
        raise InfraError(f"Steget '{step}' misslyckades (kod {res.returncode}).\n{tail}")
    return res.stdout


def _docker_compose_up(stack: Stack, service: str) -> None:
    if shutil.which("docker") is None:
        raise InfraError(
            f"{service} kör inte och docker saknas här. Starta den med just "
            f"{'db-up' if service == 'db' else 'video-up'}."
        )
    env_file = stack.repo / ".env"
    if not env_file.is_file():
        raise InfraError("Filen .env saknas. Kör: just setup-config")
    if service == "mediamtx":
        data = default_data_dir(stack.repo)
        (data / "video").mkdir(parents=True, exist_ok=True)
    res = subprocess.run(  # noqa: S603 - fixed argv
        ["docker", "compose", "--env-file", str(env_file), "up", "-d", "--wait", service],  # noqa: S607
        cwd=stack.repo,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if res.returncode != 0:
        raise InfraError(f"docker compose up {service} misslyckades:\n{res.stderr[-1500:]}")


async def _provision(port: int, password: str, name: str, *, fresh: bool) -> None:
    import asyncpg

    deadline = asyncio.get_running_loop().time() + HEALTH_DB_S
    while True:
        try:
            conn = await asyncpg.connect(
                host=DB_HOST, port=port, user=DB_OWNER, password=password, database="postgres"
            )
            break
        except (OSError, asyncpg.PostgresError, TimeoutError) as exc:
            if asyncio.get_running_loop().time() > deadline:
                raise InfraError("Databasen svarar inte. Kör just db-up.") from exc
            await asyncio.sleep(1.0)
    try:
        quoted = '"' + name.replace('"', '""') + '"'
        if fresh:
            await conn.execute(f"DROP DATABASE IF EXISTS {quoted} WITH (FORCE)")
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name)
        if not exists:
            await conn.execute(f"CREATE DATABASE {quoted}")
    finally:
        await conn.close()


def start_database(stack: Stack, lay: ServiceLayout) -> None:
    """Postgres answering on 127.0.0.1:5433, this instance's database at the schema head."""
    from sd_sim.launch import read_env_value

    port = stack.ports["postgres_local"]
    if not tcp_port_taken(port):
        _docker_compose_up(stack, "db")
    password = read_env_value(stack.repo / ".env", DB_PASSWORD_KEY)
    if not password:
        raise InfraError(f"{DB_PASSWORD_KEY} saknas i .env. Kör just setup-config.")
    asyncio.run(_provision(port, password, lay.db_name, fresh=stack.fresh_db))
    _run(
        stack,
        "db-setup",
        _python_argv("sd_ground", "db", "setup"),
        ground_env(stack, lay, video=False),
    )


def _write_token(path: Path) -> None:
    import secrets as _secrets

    from sd_sim.launch import write_secret

    write_secret(path.parent, path.name, _secrets.token_urlsafe(32))


def make_service_secrets(stack: Stack, lay: ServiceLayout) -> None:
    """A new commander token and SD-link token per run (the ground reads both at start)."""
    _write_token(lay.brain_token)
    lay.onboard_token.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lay.link_tokens.unlink(missing_ok=True)
    lay.onboard_token.unlink(missing_ok=True)
    _run(
        stack,
        "link-token",
        _python_argv(
            "sd_ground",
            "link-token",
            "create",
            "--vehicle",
            VEHICLE_ID,
            "--write-onboard",
            str(lay.onboard_token),
        ),
        ground_env(stack, lay, video=False),
    )


def health(stack: Stack) -> dict[str, Any] | None:
    """The ground station's ``GET /api/v1/health``, or ``None`` when it does not answer."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{stack.ports['ground_panel']}/api/v1/health",
        headers={"Host": f"127.0.0.1:{stack.ports['ground_panel']}"},
    )
    try:
        with LOOPBACK_OPENER.open(req, timeout=2) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def mediamtx_up(stack: Stack) -> bool:
    """MediaMTX for instance 0's video: started from compose.yaml when it is not running."""
    if stack.cfg.instance != 0 or stack.cfg.video == "off":
        return False
    if stack.mediamtx_get("/v3/config/global/get") is not None:
        return True
    try:
        _docker_compose_up(stack, "mediamtx")
    except InfraError as exc:
        say(f"Varning: ingen video i panelen ({str(exc).splitlines()[0]})")
        return False
    return stack.mediamtx_get("/v3/config/global/get") is not None


def panel_ready(stack: Stack, lay: ServiceLayout) -> None:
    """The panel's built files, which the ground station serves; built once if missing."""
    if (lay.static_dir / "index.html").is_file():
        return
    if shutil.which("pnpm") is None:
        raise InfraError("Panelen är inte byggd och pnpm saknas. Kör just build först.")
    say("Bygger panelen (första gången)")
    res = subprocess.run(
        ["pnpm", "--filter", "@searchdrone/panel", "build"],  # noqa: S607
        cwd=stack.repo,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if res.returncode != 0 or not (lay.static_dir / "index.html").is_file():
        raise InfraError(f"Panelen gick inte att bygga:\n{res.stderr[-1500:]}")


def start_ground(stack: Stack, lay: ServiceLayout, *, video: bool) -> procs.Proc:
    proc = stack.group.start(
        "sd-ground",
        _python_argv("sd_ground", "serve"),
        env=procs.clean_env(ground_env(stack, lay, video=video)),
        cwd=stack.repo,
    )
    stack.wait_for(
        "sd-ground",
        proc,
        HEALTH_GROUND_S,
        lambda: health(stack) is not None,
        "Markstationen svarar inte på /api/v1/health.",
    )
    return proc


def start_onboard(stack: Stack, lay: ServiceLayout) -> procs.Proc:
    return stack.group.start(
        "sd-onboard",
        _python_argv("sd_onboard"),
        env=procs.clean_env(onboard_env(stack, lay)),
        cwd=stack.repo,
    )


def start_brain(stack: Stack, lay: ServiceLayout) -> procs.Proc | None:
    if stack.cfg.brain == "none":
        say("hoppar över sd-brain (brain=none: bara knapparna och stopporden)")
        return None
    if stack.cfg.brain == "scripted" and not stack.cfg.script:
        raise InfraError("brain=scripted kräver script=<manus>, t.ex. SC01.yaml.")
    return stack.group.start(
        "sd-brain",
        _python_argv("sd_brain", "run", "--config", str(lay.brain_config)),
        env=procs.clean_env(brain_env(stack, lay)),
        cwd=stack.repo,
    )


def wanted_parts(stack: Stack) -> tuple[str, ...]:
    parts = ("db", "mavlink", "sdlink", "onboard")
    return (*parts, "brain") if stack.cfg.brain != "none" else parts


def all_ready(stack: Stack, want_video: bool) -> bool:
    h = health(stack)
    if h is None or h.get("ready") is not True:
        return False
    parts = h.get("parts") or {}
    names = wanted_parts(stack) + (("mediamtx",) if want_video else ())
    return all(parts.get(name) in READY_STATES.get(name, ("ok",)) for name in names)


def brain_mode(stack: Stack) -> str:
    """How the AI commander answers, in words for the start-up summary."""
    if stack.cfg.brain == "none":
        return "ingen AI (bara knapparna och stopporden)"
    part = ((health(stack) or {}).get("parts") or {}).get("brain")
    if part == "degraded":
        return "kommandoläget (ingen språkmodell svarar; fasta kommandon och knappar)"
    return {"scripted": "skriptad (test)", "grammar": "kommandoläget"}.get(
        stack.cfg.brain, "språkmodellen (molnet när nyckeln finns i .env)"
    )


def not_ready_parts(stack: Stack) -> str:
    h = health(stack) or {}
    parts = h.get("parts") or {}
    return ", ".join(f"{k}={v}" for k, v in sorted(parts.items()) if v != "ok") or "okänt"


def login_link(stack: Stack, lay: ServiceLayout) -> str:
    """A one-time login link from the running ground station (its control socket)."""
    out = _run(
        stack,
        "login-link",
        _python_argv("sd_ground", "login-link"),
        ground_env(stack, lay, video=False),
    )
    return out.strip().splitlines()[-1] if out.strip() else ""


def open_browser(url: str) -> None:
    """The login link in the operator's browser: Windows' browser from WSL, else xdg-open."""
    for opener in ("wslview", "xdg-open"):
        path = shutil.which(opener)
        if path is not None:
            subprocess.Popen(  # noqa: S603 - the link from our own server
                [path, url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env={k: v for k, v in os.environ.items() if k in procs.BASE_ENV_KEYS}
                | {k: v for k, v in os.environ.items() if k in procs.GRAPHICS_ENV_KEYS},
            )
            return
