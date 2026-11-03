"""The closer look's scenarios on the onboard rig (spec "Titta närmare" §16.4, §17).

Each scenario's world, map errors and laser are in ``simulation/scenarios/TNxx_*.yaml``
(``sd_sim.inspect_world``) and flies with its three fixed seeds (the person's place, the
candidate's error, the detections' noise and the map's random errors move with the seed).
``inspect_rig.TnLook`` flies the look with the executor's own ``InspectHandler`` against
ArduCopter SITL (builtin physics) and checks acceptance 7 (clearance 5 m, at most 1.5 m
under the floor in force) and 20 (every step a still per camera, with the order's id and
its step) in every run, and writes the §16.3 numbers to the run's ``search_report.md``.

* TN01: meadow, exercise, no laser: down to 30 m by itself, "Gå lägre?", the pilot's yes,
  down to the map floor without the laser (about 21 m), never below 30 m before the yes.
* TN02: 24 m spruce, no laser: stops at the trees plus 15 m ("golvet nått").
* TN03: spruce with a GRF-250: the pilot's yes takes it to the trees plus 5 m, never
  nearer than 12 m to the surface the laser sees.
* TN04: meadow with a TF02-Pro: no confirmation over dark ground, a confirmation over the
  light meadow.
* TN05: live without the permit: the side look, never nearer than the cylinder's radius.
* TN06: SD-link lost on the lowest step: straight up to the start height, then the usual
  rule's hold at 23 s and home at 53 s.
* TN07: the laser lost below the trees plus 15 m: the climb to that floor within 1 s.
* TN08: a hole in the map under the person: counted as 30 m forest.
* TN09: a power line 20 m away: no descent below the start height within 30 m of it.
* TN13: a tree 8 m taller than the map: clear without the laser, a higher floor with it.
* TN14: the battery's reserve during the look: home.
* TN19: another person 25 m away: its own cylinder, none new around the target.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
import sys
import time
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_rig import TnLook, tn_stack  # pyright: ignore[reportMissingImports]
from test_onboard_sitl import flight  # pyright: ignore[reportMissingImports]

from sd_safety.constants import (
    LINK_LOSS_DETECT_S,
    PERSON_KEEPOUT_M,
    SDLINK_GRACE_S,
    SDLINK_HOLD_S,
)
from sd_schemas.executor import ExecutorState as S
from sd_schemas.inspect import AUTO_FLOOR_ABOVE_GROUND_M, CLIMB_MIN_MS, match_radius_m
from sd_sim import inspect_world, paths

pytestmark = [
    pytest.mark.sitl,
    pytest.mark.slow,
    pytest.mark.inspect,
    pytest.mark.skipif(not paths.arducopter_bin().is_file(), reason="ArduCopter SITL is not built"),
]

TOL_M = 1.5  # GOTO_ALT_STEP_TOL_M: how near a height counts as reached
HOME_MODES = frozenset({"RTL", "SMART_RTL"})


def seeds(scenario: str) -> list[int]:
    return list(inspect_world.load(scenario, 1).seeds)


def world(scenario: str, seed: int, **changes: Any) -> inspect_world.InspectWorld:
    w = inspect_world.load(scenario, seed)
    return dataclasses.replace(w, **changes) if changes else w


def no_floor_m(w: inspect_world.InspectWorld, tree_m: float = 0.0) -> float:
    """The map's floor without the laser over ``tree_m`` of trees (spec §6.3): the trees
    within 10 m, plus 2 m, plus 0.5 m a year since the scan (at most 5), plus 15 m."""
    year = w.map_errors.scan_year
    growth = 0.0 if year is None else min(0.5 * (time.gmtime().tm_year - year), 5.0)
    return tree_m + 2.0 + growth + 15.0


def run(coro: Any) -> None:
    asyncio.run(coro)


# --- TN01: meadow, exercise, no laser ---------------------------------------------------------


@pytest.mark.budget_s(330)
@pytest.mark.parametrize("seed", seeds("TN01"))
def test_tn01_meadow_down_to_30_m_then_the_pilots_yes_to_the_map_floor(seed: int) -> None:
    w = world("TN01", seed)

    async def flown() -> None:
        with tn_stack(w) as sim:
            async with flight(sim) as f:
                look = TnLook(f, w)
                await look.start()
                await look.order()
                offer = await look.wait_event("inspect.lower_offer", 180)
                # By itself never below 30 m: the question comes at the automatic floor.
                assert abs(look.alt() - AUTO_FLOOR_ABOVE_GROUND_M) <= TOL_M, offer
                assert min(a for *_, a in look.track(look.t_order)) >= 30 - TOL_M
                assert offer["lowest_above_ground_m"] == pytest.approx(no_floor_m(w), abs=0.5)
                await look.yes()
                done = await look.waiting(120)
                assert done["end"] in ("clear", "floor"), done
                m = look.report()
                # Acceptance 8: the lowest height 20.9 ± 1.5 m over the meadow.
                assert m.lowest_above_ground_m == pytest.approx(no_floor_m(w), abs=TOL_M), m
                assert 20.9 - TOL_M <= m.lowest_above_ground_m <= 20.9 + TOL_M, m
                look.check_steps()
                assert not look.of("inspect.aborted")

    run(flown())
