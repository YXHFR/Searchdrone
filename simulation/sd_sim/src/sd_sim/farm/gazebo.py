"""The farm round in Gazebo: fly a seeded search with the chosen commander and keep the stills.

The flight is ``simulation/tests/scenarios/farm_flight.py``, on the scenario tests' rig
(SITL in Gazebo, the ground station with its database, the onboard side with perception,
sd-brain): the runner starts it with pytest in a child process, with the round's spec and
an empty work folder, and reads back what it left there (the stills with their camera
poses, the world's facit, the round's draw, the flight record). ``finalize()`` makes the
dataset from it.

Needs the simulator (``just sim-setup``), the person models (``just sim-farm-setup``) and
the database (``just db-up``); a missing piece stops the round with what to run.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from sd_sim import paths, world
from sd_sim.farm import people
from sd_sim.farm.camera import Intrinsics
from sd_sim.farm.round import RawRound, RoundError, RoundSpec

FLIGHT = paths.SIMULATION / "tests" / "scenarios" / "farm_flight.py"
SPEC_ENV, WORK_ENV = "SD_FARM_SPEC", "SD_FARM_WORK"
FLIGHT_TIMEOUT_S = 2 * 3600  # below the loop's cap for a whole round
TAIL_LINES = 25

Run = Callable[..., subprocess.CompletedProcess[str]]


def camera() -> Intrinsics:
    cam = world.STILLS
    return Intrinsics(int(cam["width"]), int(cam["height"]), math.radians(float(cam["hfov_deg"])))


def flight_argv(spec: RoundSpec, flight: Path = FLIGHT) -> list[str]:
    argv = [sys.executable, "-m", "pytest", "-p", "sd_sim.pytest_plugin", "-p", "no:cacheprovider"]
    argv += ["-q", "-x", "--physics", "gz", "--brain", "scripted"]
    if spec.commander == "sd-brain":
        # The cloud model is the only thing that leaves the machine; everything else stays
        # on loopback as in every simulator test.
        argv.append("--force-enable-socket")
    return [*argv, str(flight)]


def flight_env(
    spec_path: Path, work: Path, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    parts = [str(people.assets_dir()), env.get("GZ_SIM_RESOURCE_PATH", "")]
    env["GZ_SIM_RESOURCE_PATH"] = ":".join(p for p in dict.fromkeys(parts) if p)
    env[SPEC_ENV], env[WORK_ENV] = str(spec_path), str(work)
    return env


def _read(path: Path, what: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RoundError(f"flygningen lämnade ingen läsbar {what}: {type(exc).__name__}") from None
    if not isinstance(data, dict):
        raise RoundError(f"{what} är inget JSON-objekt")
    return data


def _tail(text: str) -> str:
    return "\n".join(text.strip().splitlines()[-TAIL_LINES:])


class GazeboRunner:
    def __init__(
        self,
        run: Run = subprocess.run,
        repo: Path = paths.REPO,
        timeout_s: float = FLIGHT_TIMEOUT_S,
    ) -> None:
        self._run = run
        self.repo = repo
        self.timeout_s = timeout_s

    def check(self) -> None:
        """What a round needs before anything starts."""
        if not paths.arducopter_bin().is_file():
            raise RoundError("ArduPilot SITL saknas. Kör: just sim-setup")
        missing = people.missing(self.repo)
        if missing:
            raise RoundError(
                f"personmodellerna saknas ({', '.join(missing)}). Kör: just sim-farm-setup"
            )

    def run(self, spec: RoundSpec, work: Path) -> RawRound:
        self.check()
        work.mkdir(parents=True, exist_ok=True)
        spec_path = work / "spec.json"
        spec.write(spec_path)
        try:
            done = self._run(
                flight_argv(spec),
                cwd=self.repo,
                env=flight_env(spec_path, work),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise RoundError(f"flygningen tog mer än {self.timeout_s / 60:g} min") from None
        if done.returncode != 0:
            raise RoundError(
                f"flygningen misslyckades (kod {done.returncode}):\n{_tail(done.stdout)}"
            )
        return self.collect(work)

    def collect(self, work: Path) -> RawRound:
        """What the flight left in ``work``, as a raw round."""
        info = _read(work / "flight.json", "flight.json")
        runs = work / "runs"
        extra: dict[str, Any] = {
            "flight": {k: info[k] for k in sorted(info) if k != "note"},
        }
        if info.get("note"):
            extra["commander_note"] = info["note"]
        return RawRound(
            stills_dir=work / "stills",
            truth=_read(work / "truth.json", "truth.json"),
            scenario=_read(work / "scenario.json", "scenario.json"),
            camera=camera(),
            max_height_m=float(info.get("max_height_m") or 0.0),
            extra=extra,
            runs_dir=runs if runs.is_dir() else None,
        )
