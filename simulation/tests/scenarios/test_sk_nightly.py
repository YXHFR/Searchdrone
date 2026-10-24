"""The nightly search campaign: SK01, SK02, SK03, SK04 and SK06 over many seeds.

The seeds come from ``SD_SEARCH_SEEDS`` (``just test-sitl mode=search-nightly seeds=1-50``;
three when unset). Each run is the chain rig's (``chain_rig.search_run``) with the
scenario's own time limit; a run that does not find the person is a measure, not a
failure, as long as no invariant breaks and nothing wrong is confirmed. Every run writes
``search_metrics.json`` and ``search_report.md``; ``python -m sd_sim.eval_search
aggregate`` checks the campaign's requirements over them. The detector and the image
model are oracles with modelled profiles: the results are "modellerat, ej uppmätt".

SK06 flies each seed twice, with and without the probability map: the sectors in the
map's order (highest PSR first, read again before each sector) or in the scenario's
fixed order. The sectors are ordered on the commander API, the AI's path.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chain_rig import ChainOnboard, SearchRun, search_run  # pyright: ignore[reportMissingImports]
from scenario_rig import (  # pyright: ignore[reportMissingImports]
    TERMINAL,
    fresh_db,
    migrated_db,
    passwords,
    scenario,
    sim,
)
from test_onboard_sitl import Sim  # pyright: ignore[reportMissingImports]

from sd_sim import search_world

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.brain, pytest.mark.search_nightly]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]

SEEDS = search_world.seeds(os.environ.get("SD_SEARCH_SEEDS") or "1-3")
SECTOR_LIMIT_S = 240.0


async def _run(sim: Sim, db: Any, tmp_path: Path, name: str, seed: int, **kw: Any) -> Any:
    world = search_world.load(name, seed)
    made: list[ChainOnboard] = []

    def factory(s: Sim, url: str, token: str) -> ChainOnboard:
        made.append(ChainOnboard(s, url, token, world, None))
        return made[-1]

    async with scenario(sim, db, tmp_path, onboard_factory=factory) as sc:
        return await search_run(sc, world, made[0], sim.stack.run_dir, require_found=False, **kw)


@pytest.mark.budget_s(720)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name", ["SK01", "SK02", "SK03", "SK04"])
async def test_search_campaign(
    sim: Sim, migrated_db: Any, tmp_path: Path, name: str, seed: int
) -> None:
    m = await _run(sim, migrated_db, tmp_path, name, seed)
    assert m.false_confirms == 0


async def _sector_order(run: SearchRun, with_map: bool, done: set[str]) -> str | None:
    left = [s for s in run.world.sector_order if s not in done]
    if not left:
        return None
    if with_map:
        res = await run.sc.r.panel.get(f"/api/v1/missions/{run.sc.mid}/sectors/poa")
        ranked = [s["sector_id"] for s in res.json() if s["sector_id"] in left]
        if ranked:
            return str(ranked[0])
    return left[0]


def _fly(with_map: bool) -> Any:
    """The sectors one by one, as the AI orders them. A search the keep-out reflex stops
    (``KEEPOUT_VIOLATION``: a hover or no way around) is ordered once more, planned
    around the cylinders from where the drone is, as the SK04 script does; stopped again,
    the next sector. Any other end (the operator's STOP, a replaced order) ends the run."""

    async def fly(run: SearchRun) -> None:
        sc, survey = run.sc, dict(run.world.survey)
        done: set[str] = set()
        held: set[str] = set()
        while (sector := await _sector_order(run, with_map, done)) is not None:
            await asyncio.sleep(3.1)  # the gate's rate limit: one AI action per 3 s
            res = await sc.order("survey_sector", {"sector_id": sector, **survey})
            assert res.get("decision") == "ALLOW", res
            cid = str(res["command_id"])

            async def ended(cid: str = cid) -> bool:
                return (await sc.command(cid))["status"] in TERMINAL

            await sc.until(ended, SECTOR_LIMIT_S, f"sector {sector} searched")
            after = await sc.command(cid)
            if after["status"] == "done":
                done.add(sector)
            elif after.get("status_reason") == "KEEPOUT_VIOLATION":
                if sector in held:
                    done.add(sector)  # no way around twice: on to the next sector
                held.add(sector)
            else:
                return  # stopped or replaced: the operator's or the AI's call
        await asyncio.sleep(3.1)
        await sc.order("return_home", {"reason": "Sektorerna är sökta."})

    return fly


@pytest.mark.budget_s(720)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("with_map", [True, False], ids=["med-karta", "utan-karta"])
async def test_sk06_map_against_fixed_order(
    sim: Sim, migrated_db: Any, tmp_path: Path, with_map: bool, seed: int
) -> None:
    m = await _run(sim, migrated_db, tmp_path, "SK06", seed, fly=_fly(with_map), with_map=with_map)
    assert m.false_confirms == 0
