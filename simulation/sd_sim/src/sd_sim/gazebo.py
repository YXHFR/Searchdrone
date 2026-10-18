"""Gazebo Harmonic helpers: which world to run, its command line and environment.

Gazebo Transport carries its topics on loopback (``GZ_IP=127.0.0.1``; its discovery
sockets bind all addresses by design, see listen_allowlist.yaml) and every instance gets
its own partition, so two worlds never mix their topics. The launcher renders the
Searchdrone world ``sd_lake_forest`` with ``sd_quad14`` before Gazebo starts
(``sd_sim.world``); ``vehicle=stock`` uses ArduPilot's ``iris_runway`` world instead.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from sd_sim import procs
from sd_sim.config import SimConfig
from sd_sim.errors import InfraError

STOCK_WORLD = "iris_runway"
# Camera frame rates of the models, used for the frame-count check.
CAMERA_FPS = {"sd_quad14": 15, "stock": 10}
CAMERA_WAIT_S = 30.0
GZ_PASSTHROUGH = ("GZ_VERSION", "GZ_SIM_SYSTEM_PLUGIN_PATH", "GZ_SIM_RESOURCE_PATH")
# Gazebo Harmonic's Python bindings are built for Ubuntu 24.04's system Python
# (sd_sim.gz_stills runs with it).
GZ_PYTHON = "/usr/bin/python3.12"


@dataclass(frozen=True)
class World:
    name: str
    path: Path
    vehicle: str  # "sd_quad14" | "stock"
    note: str = ""


def ardupilot_gazebo_home() -> Path:
    default = Path.home() / "gz_ws" / "src" / "ardupilot_gazebo"
    return Path(os.environ.get("SD_ARDUPILOT_GAZEBO_HOME", str(default)))


def resolve_world(cfg: SimConfig, repo: Path) -> World:
    generated = repo / "build" / "sim" / "worlds" / f"{cfg.world}_i{cfg.instance}.sdf"
    if cfg.vehicle == "sd_quad14" and generated.is_file():
        return World(cfg.world, generated, "sd_quad14")
    if cfg.instance != 0:
        raise InfraError(
            f"Standardvärlden {STOCK_WORLD} fungerar bara för instans 0 (fast FDM-port)."
        )
    stock = ardupilot_gazebo_home() / "worlds" / f"{STOCK_WORLD}.sdf"
    if not stock.is_file():
        raise InfraError(f"{stock} finns inte. Kör just sim-setup först.")
    note = (
        ""
        if cfg.vehicle == "stock"
        else (
            f"Världen {cfg.world} finns inte än; använder ArduPilots standardvärld {STOCK_WORLD}."
        )
    )
    return World(STOCK_WORLD, stock, "stock", note)


def gz_env(instance: int, *, gui: bool) -> dict[str, str]:
    passthrough = GZ_PASSTHROUGH + (procs.GRAPHICS_ENV_KEYS if gui else ())
    return procs.clean_env(
        {"GZ_IP": "127.0.0.1", "GZ_PARTITION": f"sd_i{instance}"}, passthrough=passthrough
    )


def gz_sim_argv(world: World, *, gui: bool) -> list[str]:
    argv = ["gz", "sim", "-v3", "-r"]
    if not gui:
        argv += ["-s", "--headless-rendering"]
    return [*argv, str(world.path)]


def _gz(args: list[str], instance: int, timeout: float) -> subprocess.CompletedProcess[str]:
    exe = shutil.which("gz")
    if exe is None:
        raise InfraError("gz saknas. Kör just sim-setup först.")
    return subprocess.run(  # noqa: S603 - fixed program, arguments built here
        [exe, *args],
        env=gz_env(instance, gui=False),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def world_running(world: World, instance: int) -> bool:
    try:
        res = _gz(["topic", "-e", "-n", "1", "-t", f"/world/{world.name}/stats"], instance, 5)
    except subprocess.TimeoutExpired:
        return False
    return res.returncode == 0 and bool(res.stdout.strip())


def enable_camera(instance: int, timeout: float = CAMERA_WAIT_S) -> str:
    """Turn on the camera's UDP stream; returns the topic that was used.

    The camera sensor appears a few seconds after the world, so this waits for it.
    """
    end = time.monotonic() + timeout
    while True:
        listing = _gz(["topic", "-l"], instance, 10).stdout.split()
        topics = [t for t in listing if t.endswith("/enable_streaming")]
        if topics:
            break
        if time.monotonic() > end:
            raise InfraError("Hittade ingen kamera med enable_streaming i Gazebo.")
        time.sleep(1.0)
    res = _gz(["topic", "-t", topics[0], "-m", "gz.msgs.Boolean", "-p", "data: 1"], instance, 10)
    if res.returncode != 0:
        raise InfraError(f"Kunde inte slå på kameran: {res.stderr.strip()}")
    return topics[0]
