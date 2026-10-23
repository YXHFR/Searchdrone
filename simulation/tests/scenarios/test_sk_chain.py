"""SK01, SK03, SK04 and SK09: the search chain end to end on ArduCopter SITL (PR CI).

The rig is ``chain_rig``: synthetic stills of the scenario's world, the detector and
image-model oracles, the product's perception, keep-out cylinders, ``sd-search``,
``sd-brain`` with the scenario's search script, and the scripted operator in the panel.
Each scenario flies three fixed seeds (the world moves and turns with the seed) and must
break no invariant (the flight invariants and E1, E2, E8):

* SK01: the person in the red jacket is confirmed by the operator and the drone lands.
* SK03: no person: nothing is confirmed, every card the operator reaches is rejected, and
  the drone searches both sectors and lands.
* SK04: the right person is confirmed, never the hiker in the green jacket, whose card
  shows the top in conflict when the code scored it.
* SK09: the operator rejects the person in red without a backpack in sector A; the right
  person in sector B is confirmed after that.
"""

from __future__ import annotations

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

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.brain, pytest.mark.search]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]

SEEDS = (1, 2, 3)
# PR CI: the confirmed find within 8 minutes of take-off (the nightly limits are the
# scenarios' own, 15-20 min).
PR_LIMIT_S = 480.0


async def _run(sim: Sim, db: Any, tmp_path: Path, name: str, seed: int, **kw: Any) -> Any:
    world = search_world.load(name, seed)
    made: list[ChainOnboard] = []

    def factory(s: Sim, url: str, token: str) -> ChainOnboard:
        made.append(ChainOnboard(s, url, token, world, None))
        return made[-1]

    async with scenario(sim, db, tmp_path, onboard_factory=factory) as sc:
        return await search_run(
            sc, world, made[0], sim.stack.run_dir, confirm_limit_s=PR_LIMIT_S, **kw
        )


@pytest.mark.budget_s(600)
@pytest.mark.parametrize("seed", SEEDS)
async def test_sk01_chain(sim: Sim, migrated_db: Any, tmp_path: Path, seed: int) -> None:
    m = await _run(sim, migrated_db, tmp_path, "SK01", seed)
    assert m.found or m.time_to_confirmed_s is not None


@pytest.mark.budget_s(600)
@pytest.mark.parametrize("seed", SEEDS)
async def test_sk03_chain_no_person(sim: Sim, migrated_db: Any, tmp_path: Path, seed: int) -> None:
    m = await _run(sim, migrated_db, tmp_path, "SK03", seed)
    assert m.confirmed_thing is None
    assert all(d["decision"] == "reject" for d in m.decisions), m.decisions


@pytest.mark.budget_s(600)
@pytest.mark.parametrize("seed", SEEDS)
async def test_sk04_chain_right_person(
    sim: Sim, migrated_db: Any, tmp_path: Path, seed: int
) -> None:
    m = await _run(sim, migrated_db, tmp_path, "SK04", seed)
    assert m.confirmed_thing == "target"
    hiker = m.fields_by_thing.get("green-jacket")
    if hiker is not None and hiker.get("garment:0") not in (None, "not_visible"):
        assert hiker["garment:0"] == "conflict", hiker


@pytest.mark.budget_s(600)
@pytest.mark.parametrize("seed", SEEDS)
async def test_sk09_chain_wrong_person_first(
    sim: Sim, migrated_db: Any, tmp_path: Path, seed: int
) -> None:
    m = await _run(sim, migrated_db, tmp_path, "SK09", seed)
    assert m.confirmed_thing == "target"
    wrong = [d for d in m.decisions if d["truth"] == "wrong-person"]
    assert wrong and all(d["decision"] == "reject" for d in wrong), m.decisions
    confirm = next(d for d in m.decisions if d["decision"] == "confirm")
    assert wrong[0]["t_s"] <= confirm["t_s"], m.decisions
