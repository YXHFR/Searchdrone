"""One simulator-farm round flown in Gazebo: ``python -m sd_sim.farm round`` runs this file.

Not a test of the suite (the name keeps it out of every collection): the farm's Gazebo
runner (``sd_sim.farm.gazebo``) starts it with pytest, since it flies on the scenario
tests' rig, with ``SD_FARM_SPEC`` (the round's ``spec.json``) and ``SD_FARM_WORK`` (an
empty folder for what the round leaves behind).

The round's world comes from the seed (``sd_sim.farm.height.draw``): people, decoys,
sun, weather and the flight height, inside the demo's search box. The height is raised to
clear the tallest tree under the flight (``sd_sim.farm.height``). The vehicle carries the
stills camera; every still is kept with the camera's pose in the world. The onboard side
runs the perception task as in SK-RED; until there is a person model for the simulator,
people are reported from the simulator's labels (the oracle source), as cars are.

The decision maker (``commander`` in the spec):

* ``script``: the farm's script (``simulation/scenarios/scripts/SIMFARM.yaml``) with the
  round's height. It takes off, searches the box, asks the pilot to confirm each person it
  is told about and flies home.
* ``sd-brain``: sd-brain's decision maker for simulator exams
  (``sd_brain.simeval.decider``): the cloud model while the key and the day's cost cap
  allow, the fixed search script after.

The pilot's part is the same in both: the charter, "clear around the drone", the start;
FLYG HEM if the round runs out of time. What is written into ``SD_FARM_WORK``::

    stills/         every still with its pose (``gz_stills capture --archive``)
    truth.json      the world's facit
    scenario.json   the round's draw
    flight.json     what the farm knows of the flight (height, landing, time)
    runs/flight/    ``flight.json`` (and ``brain.json`` for sd-brain) for the grade
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario_rig import (  # pyright: ignore[reportMissingImports]
    ROOT,
    SCRIPTS,
    Scenario,
    fresh_db,
    migrated_db,
    passwords,
    scenario,
)
from test_mission_flow_sitl import SitlOnboard  # pyright: ignore[reportMissingImports]
from test_onboard_sitl import Sim, sim_stack  # pyright: ignore[reportMissingImports]

from sd_brain.simeval import budget
from sd_brain.simeval import farm as brain_farm
from sd_brain.simeval.decider import make_decision_maker
from sd_geo import PinholeCamera
from sd_onboard.perception.loop import PerceptionLoop
from sd_onboard.perception.sources import DirStillSource
from sd_sim import world
from sd_sim.farm import flight
from sd_sim.farm import height as farm_height
from sd_sim.farm import scenario as farm_scenario
from sd_sim.farm.round import RoundSpec
from sd_sim.invariants import check_track

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.brain, pytest.mark.gz]
__all__ = ["fresh_db", "migrated_db", "passwords"]

SPEC_ENV, WORK_ENV = "SD_FARM_SPEC", "SD_FARM_WORK"
SCRIPT = "SIMFARM"
# The day's cost cap is shared with ``sd-brain eval``: the farm's loop names the owner's
# ledger (its rounds run in a worktree of their own).
LEDGER_ENV = "SD_FARM_COST_LEDGER"
LEDGER = Path("data/brain-eval/cost-ledger.json")
SPACING_M = 30.0
FLIGHT_LIMIT_S = 900.0  # the search; then the pilot presses FLYG HEM
HOME_LIMIT_S = 300.0
PERIOD_S = 0.2
SEARCH_ORDER = (
    "Sök igenom sektor A efter en försvunnen person. Markera det du ser och be mig "
    "bekräfta. Flyg hem och landa när sektorn är sökt."
)


def _spec() -> tuple[RoundSpec, Path]:
    spec, work = os.environ.get(SPEC_ENV), os.environ.get(WORK_ENV)
    if not spec or not work:
        pytest.skip(f"bara i simfarmens varv ({SPEC_ENV} och {WORK_ENV})")
    return RoundSpec.read(Path(spec)), Path(work)


def draw(spec: RoundSpec) -> farm_scenario.FarmScenario:
    """The round's world: the seed's draw inside the demo's search box, at a height clear
    of the trees under the flight."""
    return farm_height.draw(spec.seed, spec.worlds)


class FarmOnboard(SitlOnboard):
    """The onboard executor plus the perception task on the stills folder."""

    def __init__(self, sim: Sim, url: str, token: str, keep_dir: Path) -> None:
        super().__init__(sim, url, token)
        cam = world.STILLS
        self.perception = PerceptionLoop(
            DirStillSource(sim.stack.stills_dir, synthetic=True),
            lambda: self.vehicle.telemetry,
            self.link.perception,
            self.clock,
            PinholeCamera(int(cam["width"]), int(cam["height"]), float(cam["hfov_deg"])),
            registry=flight.farm_registry(),
            period_s=PERIOD_S,
            keep_dir=keep_dir,
            env={"SD_SIM": "1"},
        )

    async def start(self, home: tuple[float, float]) -> None:
        await super().start(home)
        self._tasks.append(asyncio.ensure_future(self.perception.run(self._stop)))


@pytest.fixture
def farm_sim() -> Iterator[tuple[Sim, RoundSpec, Path, farm_scenario.FarmScenario]]:
    spec, work = _spec()
    if spec.commander not in ("script", "sd-brain"):
        raise ValueError(f"okänd beslutsfattare {spec.commander!r}")
    sc = draw(spec)
    if sc.world != world.WORLD_NAME:
        raise ValueError(f"Gazebo-varvet flyger bara {world.WORLD_NAME} än, inte {sc.world}")
    stills = work / "stills"
    with sim_stack("simfarm", physics="gz", scene=sc, light=sc.light, stills_archive=stills) as sim:
        yield sim, spec, work, sc


@dataclass
class Exam:
    """sd-brain's exam decision maker for the flight, with its meter and model."""

    decider: Any
    meter: Any
    model: str
    client: Any  # the cloud model's HTTP client, closed after the flight


async def _start_with_decider(sc: Scenario, height_m: float) -> Exam:
    """The pilot readies the mission; sd-brain flies with the exam decision maker."""
    import httpx

    from sd_brain.config import load_config
    from sd_brain.runtime.cloud import CloudModelRuntime
    from sd_brain.secrets import brain_env, optional_secret

    # The AI picks its own heights: the charter keeps them at the round's height or above.
    await sc.ready("charter_test", **farm_height.charter_floor(height_m))
    brain = await sc.start_brain(SCRIPT)
    local = ROOT / "config" / "brain.toml"
    cfg = load_config(local if local.is_file() else None)  # the owner's cloud model
    cfg.cloud.provider.follow_ground("sim")  # the exam flies against the SITL ground
    key = optional_secret("SD_CLOUD_API_KEY", brain_env().get("SD_CLOUD_API_KEY"))
    meter = client = None
    if key is not None and cfg.cloud.model:
        client = httpx.AsyncClient()
        ledger = Path(os.environ.get(LEDGER_ENV) or ROOT / LEDGER)
        cloud = CloudModelRuntime(cfg.cloud, key, client)
        meter = budget.MeteredRuntime(cloud, budget.CostLedger(ledger))
    decider = make_decision_maker(meter, cfg.behaviour)
    brain.loop.backend = decider
    await sc.clearance()
    res = await sc.order("takeoff", {"alt_m": height_m})
    assert res.get("decision") == "ALLOW", res
    await sc.wait_command("takeoff", "done", 120)
    await asyncio.sleep(3.1)  # the gate's rate limit
    await sc.say(SEARCH_ORDER)
    return Exam(decider, meter, cfg.cloud.model, client)


def brain_record(decider: Any, meter: Any, model: str) -> dict[str, Any]:
    return brain_farm.brain_record(decider, meter, model=model, ai_config_hash="simfarm")


@pytest.mark.budget_s(2400)
async def test_farm_round(
    farm_sim: tuple[Sim, RoundSpec, Path, farm_scenario.FarmScenario],
    migrated_db: Any,
    tmp_path: Path,
) -> None:
    sim, spec, work, draw_ = farm_sim
    height = draw_.height_m
    truth_path = sim.stack.ground_truth
    assert truth_path is not None
    shutil.copyfile(truth_path, work / "truth.json")
    (work / "scenario.json").write_text(
        json.dumps(draw_.as_dict(), indent=1) + "\n", encoding="utf-8"
    )
    onboard: list[FarmOnboard] = []

    def factory(s: Sim, url: str, token: str) -> FarmOnboard:
        onboard.append(FarmOnboard(s, url, token, tmp_path / "kept"))
        return onboard[-1]

    exam: Exam | None = None
    async with scenario(sim, migrated_db, tmp_path, onboard_factory=factory) as sc:

        def ekf_on_gps() -> bool:
            t = onboard[0].vehicle.telemetry
            return t.ekf_ok and t.gps_fix_type >= 3

        await sc.until(ekf_on_gps, 180, "EKF using GPS")
        await asyncio.sleep(5.0)
        started = time.monotonic()
        if spec.commander == "sd-brain":
            exam = await _start_with_decider(sc, height)
        else:
            script = work / "script.yaml"
            script.write_text(flight.script_at(SCRIPTS / f"{SCRIPT}.yaml", height), "utf-8")
            await sc.airborne_with_ai(
                SCRIPT, backend={"kind": "scripted", "scripted_file": str(script)}
            )
        search_from = time.monotonic()
        homed = False
        try:
            await sc.wait_command("return_home", "done", FLIGHT_LIMIT_S, issuer_kind="brain")
            homed = True
        except AssertionError:
            homed = False  # out of time, or the AI's return home ended otherwise
        if not homed:
            res = await sc.press("return_home")  # FLYG HEM, past the AI
            assert res.status_code == 202, res.text
        await sc.until(lambda: not sc.tel.armed, HOME_LIMIT_S, "disarmed after landing")
        ended = time.monotonic()
        home_m = sc.from_home_m()
        finds = (await sc.r.panel.get(f"/api/v1/missions/{sc.mid}/finds")).json()
        rows = (await sc.r.panel.get(f"/api/v1/missions/{sc.mid}/detections")).json()
        queue = (await sc.r.panel.get(f"/api/v1/missions/{sc.mid}/candidates?limit=200")).json()
        commands = await sc.commands()
        gates = {c["id"]: await sc.gate(c["id"]) for c in commands}
        chat = await sc.chat()
        charters = (await sc.r.panel.get(f"/api/v1/missions/{sc.mid}/charters")).json()
        approved = [c["body"] for c in charters if c.get("status") in ("approved", "superseded")]
        breaches = len(check_track(sc.track, approved)) if approved and sc.track else 0
        coverage = sc.coverage(search_from, alt_m=height, spacing_m=SPACING_M)
        turns = len(sc.brain_turns())
        track = [s for s in sc.track if s.armed]
        utc_offset = time.time() - time.monotonic()  # the track's clock to the orders' UTC
        stats = onboard[0].perception.stats
    if exam is not None and exam.client is not None:
        await exam.client.aclose()

    run = work / "runs" / "flight"
    run.mkdir(parents=True, exist_ok=True)
    record_ = flight.flight_record(
        started=started,
        ended=ended,
        commands=commands,
        gates=gates,
        finds=finds,
        detections=rows,
        chat=chat,
        candidates=queue["items"],
        track=[(s.t + utc_offset, s.lat, s.lon) for s in track],
        breaches=breaches,
        coverage=coverage,
        landed_home=home_m <= flight.HOME_RADIUS_M,
        turns=turns,
        max_time_s=FLIGHT_LIMIT_S + HOME_LIMIT_S,
    )
    (run / "flight.json").write_text(json.dumps(record_, indent=1) + "\n", encoding="utf-8")
    if exam is not None:
        record = brain_record(exam.decider, exam.meter, exam.model)
        (run / "brain.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    info = {
        "max_height_m": round(max((s.alt_m for s in track), default=0.0), 2),
        "took_off": bool(track) and max(s.alt_m for s in track) > 2.0,
        "landed_from_home_m": round(home_m, 2),
        "flew_home_by": "ai" if homed else "pilot",
        "finds": len(finds),
        "track_points": len(track),
        "perception": {"stills": stats.stills, "detections": stats.detections},
    }
    (work / "flight.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    assert home_m <= 10.0, f"landade {home_m:.1f} m från hem"
