"""The smoke flight in a terrain world: real-time factor, settings and the hover measurement."""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from sd_sim import config, gazebo, site, smoke
from sd_sim.sitl import CMAC_HOME

STATS = """header {
  stamp { sec: 61 nsec: 500000000 }
}
sim_time {
  sec: 30
  nsec: 250000000
}
real_time {
  sec: 60
  nsec: 500000000
}
real_time_factor: 0.5
iterations: 30250
"""


def _world() -> gazebo.World:
    return gazebo.World("sd_hellasgarden", Path("w.sdf"), "sd_quad14")


def test_world_times_reads_simulated_and_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake(args: list[str], instance: int, timeout: float) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, STATS, "")

    monkeypatch.setattr(gazebo, "_gz", fake)
    assert gazebo.world_times(_world(), 0) == (30.25, 60.5)
    assert calls[0][-1] == "/world/sd_hellasgarden/stats"


def test_world_times_without_a_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gazebo, "_gz", lambda a, i, t: subprocess.CompletedProcess(a, 1, "", ""))
    assert gazebo.world_times(_world(), 0) is None

    def hang(args: list[str], instance: int, timeout: float) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(args, timeout)

    monkeypatch.setattr(gazebo, "_gz", hang)
    assert gazebo.world_times(_world(), 0) is None


def test_real_time_factor() -> None:
    assert gazebo.real_time_factor((10.0, 20.0), (15.0, 30.0)) == 0.5
    assert gazebo.real_time_factor(None, (15.0, 30.0)) is None
    assert gazebo.real_time_factor((10.0, 30.0), (15.0, 30.0)) is None


def _cfg(world: str) -> config.SimConfig:
    cfg = config.SimConfig()
    cfg.world, cfg.physics, cfg.gui, cfg.objects = world, "gz", False, "x.yaml"
    cfg.home, cfg.personal_home = "1,2,3,4", True
    return cfg


def test_harden_flies_our_vehicle_at_the_public_site_home() -> None:
    cfg = smoke.harden(_cfg("sd_hellasgarden"))
    assert cfg.vehicle == "sd_quad14" and cfg.home == site.load("hellasgarden").home_arg
    assert not cfg.personal_home and not cfg.qgc and cfg.objects == ""
    assert cfg.video == "camera"


def test_harden_keeps_cmac_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = smoke.harden(_cfg("sd_lake_forest"))
    assert (cfg.vehicle, cfg.home, cfg.personal_home) == ("stock", CMAC_HOME, False)
    # A local site (the owner's own area) is never flown by the smoke flight.
    monkeypatch.setattr(site, "world_sites", lambda: {"sd_mine": "mine"})
    monkeypatch.setattr(site, "folder_for", lambda name: (site.PUBLIC_DIR, False))
    cfg = smoke.harden(_cfg("sd_mine"))
    assert (cfg.vehicle, cfg.home) == ("stock", CMAC_HOME)


class _Harness:
    def __init__(self) -> None:
        self.pumped = 0.0

    def pump(self, seconds: float) -> None:
        self.pumped += seconds
        time.sleep(0.01)


def test_hover_keeps_the_heartbeat_while_measuring() -> None:
    harness = _Harness()
    done = threading.Event()

    def measure() -> None:
        time.sleep(0.1)
        done.set()

    result: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    smoke.hover_until(harness, measure, 5.0, result, checks)  # type: ignore[arg-type]
    assert done.is_set() and harness.pumped > 0
    assert "measure_error" not in result and checks == {}


def test_hover_gives_up_on_a_hanging_measurement() -> None:
    release = threading.Event()
    result: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    start = time.monotonic()
    smoke.hover_until(_Harness(), release.wait, 0.3, result, checks)  # type: ignore[arg-type]
    release.set()
    assert time.monotonic() - start < 2.0
    assert checks == {"camera": False} and "0.3" in result["measure_error"]


def test_camera_frames_follow_simulated_time() -> None:
    # At a real-time factor of 0.75, 10 wall-clock seconds are 7.5 simulated seconds.
    assert smoke.min_frames("stock") == 90
    assert smoke.min_frames("stock", 7.5) == 67
    assert smoke.min_frames("sd_quad14", 5.0) == 67


def test_launch_default_worlds() -> None:
    from sd_sim import launch

    assert launch.default_world_for(demo=False) == "sd_hellasgarden"
    assert launch.default_world_for(demo=True) == "sd_lake_forest"
