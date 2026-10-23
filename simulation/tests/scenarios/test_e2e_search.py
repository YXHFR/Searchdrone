"""The search demo end to end (``just e2e-search``; ``just demo-search`` with Gazebo).

SK01 for seed 1 on the chain rig (``chain_rig``): the scripted AI searches sector A and
flies over each high candidate, the scripted operator works the queue in the panel and
confirms the person in the red jacket after viewing the picture, the drone holds, flies
home and lands. ``search_report.md`` in the run's folder holds the time to the find, POS,
the queue at the end and the engines that judged the candidates; CI keeps it with the
run's ``search_metrics.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chain_rig import ChainOnboard, search_run  # pyright: ignore[reportMissingImports]
from scenario_rig import (  # pyright: ignore[reportMissingImports]
    fresh_db,
    migrated_db,
    passwords,
    scenario,
    sim,
)
from test_onboard_sitl import Sim  # pyright: ignore[reportMissingImports]

from sd_sim import search_world

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.brain, pytest.mark.e2e_search]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]

SEED = 1
LIMIT_S = 480.0


@pytest.mark.budget_s(720)
async def test_e2e_search_sk01(sim: Sim, migrated_db: Any, tmp_path: Path) -> None:
    world = search_world.load("SK01", SEED)
    made: list[ChainOnboard] = []

    def factory(s: Sim, url: str, token: str) -> ChainOnboard:
        made.append(ChainOnboard(s, url, token, world, None))
        return made[-1]

    async with scenario(sim, migrated_db, tmp_path, onboard_factory=factory) as sc:
        m = await search_run(sc, world, made[0], sim.stack.run_dir, confirm_limit_s=LIMIT_S)

    assert m.found and m.confirmed_thing == "target", m.decisions
    assert m.false_confirms == 0 and m.invariants_broken == []
    report = sim.stack.run_dir / "search_report.md"
    metrics = json.loads((sim.stack.run_dir / "search_metrics.json").read_text("utf-8"))
    assert report.is_file() and metrics["found"]
    text = report.read_text("utf-8")
    for part in ("Tid lyft", "POS", "Kön vid slutet", "Motorer"):
        assert part in text, part
