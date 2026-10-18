"""SK-RED: search for red blocks with the camera in Gazebo, and find every one of them.

Gazebo Harmonic (headless rendering) with SITL; the world has the SK-RED objects of one
seed: one to three red 0.5 m blocks, a red car and a red house as decoys, a blue car and a
road, each with its label. The vehicle carries the stills camera (1280 x 720, 66 degrees,
1 Hz) and a bounding-box camera with the same picture; a capture process writes both to
a folder. The onboard side runs the executor and the perception task, which reads that
folder: the colour detector on the real pixels for red blocks, the simulator's labels
(oracle) for cars, buildings and the road, georeferencing with the flight controller's
position and attitude, reports to the ground over SD-link.

The scripted commander searches the box at P1, flies over each new red block and asks
the pilot to confirm, searches on, then flies home and lands. Passes when:

* every red block has a find within 3 m of the ground truth, and there is no other red
  block find (the red car and the red house are no blocks);
* the oracle reported the cars as cars;
* the AI asked the pilot to confirm, and never confirmed anything itself;
* the drone landed at home and the invariants held.

Time to the first find and every error are written to ``runs/<run>/search/result.json``
with a map from above and the stills with the detector's boxes. Gazebo only:
``just test-sitl sc=SK-RED physics=gz`` (and the nightly ``sitl-nightly-gz``).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario_rig import (  # pyright: ignore[reportMissingImports]
    fresh_db,
    migrated_db,
    passwords,
    scenario,
)
from test_mission_flow_sitl import SitlOnboard  # pyright: ignore[reportMissingImports]
from test_onboard_sitl import Sim, sim_stack  # pyright: ignore[reportMissingImports]

from sd_geo import PinholeCamera
from sd_onboard.perception.loop import PerceptionLoop
from sd_onboard.perception.sources import DirStillSource
from sd_sim import search_report, world
from sd_sim.targets import Scene

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.brain, pytest.mark.gz]
__all__ = ["fresh_db", "migrated_db", "passwords"]

# Seed 11 has three blocks; SD_SK_RED_SEED picks another layout.
SEED = int(os.environ.get("SD_SK_RED_SEED", "11"))
MAX_ERROR_M = 3.0
SEARCH_LIMIT_S = 1200.0
ASK = "Rött block hittat, bekräfta?"


@pytest.fixture
def gz_sim() -> Iterator[Sim]:
    with sim_stack("sk_red", physics="gz", scene=Scene("SK-RED", SEED)) as s:
        yield s


class PerceivingOnboard(SitlOnboard):
    """The onboard executor plus the perception task on the Gazebo stills folder."""

    def __init__(self, sim: Sim, url: str, token: str, keep_dir: Path) -> None:
        super().__init__(sim, url, token)
        cam = world.STILLS
        self.perception = PerceptionLoop(
            DirStillSource(sim.stack.stills_dir, synthetic=True),
            lambda: self.vehicle.telemetry,
            self.link.perception,
            self.clock,
            PinholeCamera(int(cam["width"]), int(cam["height"]), float(cam["hfov_deg"])),
            keep_dir=keep_dir,
            env={"SD_SIM": "1"},
        )

    async def start(self, home: tuple[float, float]) -> None:
        await super().start(home)
        self._tasks.append(asyncio.ensure_future(self.perception.run(self._stop)))


def _utc(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


@pytest.mark.parametrize("scenario_id", ["SK-RED"])
@pytest.mark.budget_s(1800)
async def test_sk_red(gz_sim: Sim, migrated_db: Any, tmp_path: Path, scenario_id: str) -> None:
    run_dir = gz_sim.stack.run_dir
    truth_path = gz_sim.stack.ground_truth
    assert truth_path is not None
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    keep_dir = run_dir / "kept"
    onboard: list[PerceivingOnboard] = []

    def factory(sim: Sim, url: str, token: str) -> PerceivingOnboard:
        onboard.append(PerceivingOnboard(sim, url, token, keep_dir))
        return onboard[-1]

    started = time.monotonic()
    async with scenario(gz_sim, migrated_db, tmp_path, onboard_factory=factory) as sc:
        # Under Gazebo the EKF takes a while to use GPS (and the IMUs to settle); an arm
        # before that is refused ("Gyros inconsistent") and the take-off skill does not
        # retry, so the pilot waits for the EKF's position as one would in the field.
        def ekf_on_gps() -> bool:
            t = onboard[0].vehicle.telemetry
            return t.ekf_ok and t.gps_fix_type >= 3

        await sc.until(ekf_on_gps, 180, "EKF using GPS")
        await asyncio.sleep(5.0)
        await sc.airborne_with_ai(scenario_id)
        await sc.wait_command("survey_sector", "running", 180, issuer_kind="brain")
        # The search ends with the AI's own return home once the box is searched.
        await sc.wait_command("return_home", "done", SEARCH_LIMIT_S, issuer_kind="brain")
        await sc.until(lambda: not sc.tel.armed, 90, "disarmed after landing")
        home_m = sc.from_home_m()

        finds = (await sc.r.panel.get(f"/api/v1/missions/{sc.mid}/finds")).json()
        rows = (await sc.r.panel.get(f"/api/v1/missions/{sc.mid}/detections")).json()
        commands = await sc.commands()
        chat = await sc.chat()
        stats = onboard[0].perception.stats
        await sc.check_invariants()

    result = search_report.match_finds(truth, finds, "red_block", MAX_ERROR_M)
    takeoff = min(_utc(c["created_at"]) for c in commands if c["skill"] == "takeoff")
    surveys = [_utc(c["created_at"]) for c in commands if c["skill"] == "survey_sector"]
    blocks = sorted(_utc(r["captured_at"]) for r in rows if r["label"] == "red_block")
    first_s = (blocks[0] - takeoff).total_seconds() if blocks else None
    first_after_search_s = (
        (blocks[0] - min(surveys)).total_seconds() if blocks and surveys else None
    )
    asked = [m for m in chat if m.get("role") == "brain" and m.get("text") == ASK]
    gotos_to_finds = [c for c in commands if c["skill"] == "goto" and c["issuer_kind"] == "brain"]

    # The record for people: map, annotated stills, numbers.
    track = [(s.lat, s.lon) for s in sc.track if s.armed]
    search_report.draw_map(truth, finds, result, track, run_dir / "search" / "map.png")
    kept = sorted(keep_dir.glob("*.json")) if keep_dir.is_dir() else []
    shown = 0
    for meta in kept:
        report = json.loads(meta.read_text(encoding="utf-8"))
        if any(d["label"] == "red_block" for d in report["detections"]) and shown < 8:
            search_report.annotate_still(
                meta.with_suffix(".jpg"), report, run_dir / "search" / f"still_{shown}.jpg"
            )
            shown += 1
    summary = {
        "scenario": scenario_id,
        "seed": SEED,
        "blocks": [o["name"] for o in truth["objects"] if o["cls"] == "red_block"],
        "result": result.as_dict(),
        "time_to_first_find_s": {
            "from_takeoff": first_s,
            "from_search_start": first_after_search_s,
        },
        "finds": finds,
        "perception": {
            "stills": stats.stills,
            "reports": stats.reports,
            "detections": stats.detections,
            "by_label": stats.by_label,
            "skipped_no_pose": stats.skipped_no_pose,
            "failed": stats.failed,
        },
        "ai_asked_to_confirm": len(asked),
        "ai_gotos": len(gotos_to_finds),
        "landed_from_home_m": round(home_m, 2),
        "wall_s": round(time.monotonic() - started, 1),
    }
    search_report.write(run_dir, summary)

    assert result.ok, json.dumps(result.as_dict(), indent=1)
    assert len(result.matches) == len(summary["blocks"]) >= 1
    assert any(f["label"] == "car" and f["source"] == "oracle" for f in finds), finds
    assert not any(f["label"] == "red_block" and f["source"] == "oracle" for f in finds)
    assert asked, "the AI never asked the pilot to confirm a find"
    assert not any(
        "bekräftat" in str(m.get("text", "")).lower() for m in chat if m.get("role") == "brain"
    )
    assert stats.failed == 0
    assert first_s is not None
    assert home_m <= 3.0
