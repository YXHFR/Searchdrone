"""Gazebo Harmonic helpers: which world to run, its command line and environment.

Gazebo Transport carries its topics on loopback (``GZ_IP=127.0.0.1``; its discovery
sockets bind all addresses by design, see listen_allowlist.yaml) and every instance gets
its own partition, so two worlds never mix their topics. The launcher renders the
Searchdrone world ``sd_lake_forest`` with ``sd_quad14`` (or ``vehicle=sd_fpv7``) before
Gazebo starts (``sd_sim.world``); ``vehicle=stock`` uses ArduPilot's ``iris_runway`` world
instead.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sd_sim import procs
from sd_sim.config import SimConfig
from sd_sim.errors import InfraError

STOCK_WORLD = "iris_runway"
# Camera frame rates of the models, used for the frame-count check.
CAMERA_FPS = {"sd_quad14": 15, "sd_fpv7": 15, "stock": 10}
CAMERA_WAIT_S = 30.0
# How long a request to start the stream may go unanswered before it is sent again. The
# plugin starts streaming on the camera's next image (10-15 per simulated second), so
# frames arrive well within this once the request got through.
CAMERA_CONFIRM_S = 5.0
GZ_PASSTHROUGH = ("GZ_VERSION", "GZ_SIM_SYSTEM_PLUGIN_PATH", "GZ_SIM_RESOURCE_PATH")
# Gazebo Harmonic's Python bindings are built for Ubuntu 24.04's system Python
# (sd_sim.gz_stills runs with it).
GZ_PYTHON = "/usr/bin/python3.12"


@dataclass(frozen=True)
class World:
    name: str
    path: Path
    vehicle: str  # "sd_quad14" | "sd_fpv7" | "stock"
    note: str = ""


def ardupilot_gazebo_home() -> Path:
    default = Path.home() / "gz_ws" / "src" / "ardupilot_gazebo"
    return Path(os.environ.get("SD_ARDUPILOT_GAZEBO_HOME", str(default)))


def resolve_world(cfg: SimConfig, repo: Path) -> World:
    generated = repo / "build" / "sim" / "worlds" / f"{cfg.world}_i{cfg.instance}.sdf"
    if cfg.vehicle != "stock" and generated.is_file():
        return World(cfg.world, generated, cfg.vehicle)
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
    passthrough = GZ_PASSTHROUGH + procs.GPU_ENV_KEYS + (procs.DISPLAY_ENV_KEYS if gui else ())
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


_TIME = re.compile(r"(sim_time|real_time)\s*\{\s*(?:sec:\s*(\d+))?\s*(?:nsec:\s*(\d+))?\s*\}")


def world_times(world: World, instance: int) -> tuple[float, float] | None:
    """(simulated, wall-clock) seconds from one ``/world/<name>/stats`` message."""
    try:
        res = _gz(["topic", "-e", "-n", "1", "-t", f"/world/{world.name}/stats"], instance, 10)
    except subprocess.TimeoutExpired:
        return None
    found = {
        m.group(1): int(m.group(2) or 0) + int(m.group(3) or 0) / 1e9
        for m in _TIME.finditer(res.stdout)
    }
    if "sim_time" not in found or "real_time" not in found:
        return None
    return found["sim_time"], found["real_time"]


def real_time_factor(
    start: tuple[float, float] | None, end: tuple[float, float] | None
) -> float | None:
    """Simulated time over wall-clock time between two ``world_times`` samples."""
    if start is None or end is None or end[1] <= start[1]:
        return None
    return round((end[0] - start[0]) / (end[1] - start[1]), 3)


def enable_camera(
    instance: int,
    timeout: float = CAMERA_WAIT_S,
    streaming: Callable[[], bool] | None = None,
) -> str:
    """Turn on the camera's UDP stream; returns the topic that was used.

    The camera sensor appears a few seconds after the world, so this waits for it.
    ``gz topic -p`` publishes once after a fixed 0.8 s and exits 0 whether or not the
    camera plugin had been found by then (gz-transport 13, its issue #47): on a busy
    computer the request is lost and no frame ever comes. With ``streaming`` (true once
    frames arrive) the request is sent again until the stream runs, within ``timeout``.
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
    while True:
        res = _gz(
            ["topic", "-t", topics[0], "-m", "gz.msgs.Boolean", "-p", "data: 1"], instance, 10
        )
        if res.returncode != 0:
            raise InfraError(f"Kunde inte slå på kameran: {res.stderr.strip()}")
        if streaming is None:
            return topics[0]
        confirm = time.monotonic() + CAMERA_CONFIRM_S
        while time.monotonic() < confirm:
            if streaming():
                return topics[0]
            time.sleep(0.25)
        if time.monotonic() > end:
            raise InfraError("Kameran skickar inga bilder fast den har slagits på.")
