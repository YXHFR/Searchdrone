"""SC01 (main scenario, search part) and the pilot's manual takeover, on SITL.

* SC01: the AI proposes the charter, takes off on "Starta.", flies to P1 and searches the
  100 x 100 m box there in a lawnmower pattern; at least 95 % of the box is covered, then
  it flies home and lands within 3 m.
* SC01_takeover: mid-search the pilot takes over, moves the drone 20 m with the sticks,
  lets go (the ground station stops sending after 1 s: the drone stands in LOITER) and
  hands back; the AI continues the search from where it stopped.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario_rig import (  # pyright: ignore[reportMissingImports]
    fresh_db,
    migrated_db,
    passwords,
    scenario,
    sim,
)

from sd_safety.targets import distance_m

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.brain]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]

ALT_M = 40.0
SPACING_M = 30.0
MIN_COVERAGE = 0.95
SURVEY_LIMIT_S = 240.0


@pytest.mark.budget_s(600)
async def test_sc01_survey_on_sitl(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne_with_ai("SC01")
        assert abs(sc.tel.alt_rel_home_m - ALT_M) <= 2.0
        await sc.wait_command("goto", "done", 90, issuer_kind="brain")
        survey = await sc.wait_command("survey_sector", "running", 30, issuer_kind="brain")
        t0 = time.monotonic()
        await sc.wait_command("survey_sector", "done", SURVEY_LIMIT_S, issuer_kind="brain")
        took = time.monotonic() - t0
        assert took <= SURVEY_LIMIT_S
        got = sc.coverage(t0, ALT_M, SPACING_M)
        assert got >= MIN_COVERAGE, f"coverage {got:.3f}"
        gates = await sc.gate(survey["id"])
        assert [g["decision"] for g in gates if g["side"] == "ground"] == ["ALLOW"]

        await sc.wait_command("return_home", "done", 240, issuer_kind="brain")
        await sc.until(lambda: not sc.tel.armed, 60, "disarmed after landing")
        assert sc.from_home_m() <= 3.0
        await sc.check_invariants()


@pytest.mark.budget_s(660)
async def test_sc01_takeover_on_sitl(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne_with_ai("SC01_takeover", behaviour={"idle_airborne_tick_s": 5})
        await sc.wait_command("survey_sector", "running", 120, issuer_kind="brain")
        await asyncio.sleep(20.0)  # some of the search flown

        res = await sc.press("takeover")
        assert res.status_code == 202, res.text
        takeover_id = res.json()["action_id"]
        await sc.until(lambda: sc.mode() == "LOITER", 3, "LOITER after the takeover")
        await sc.wait_command("survey_sector", "superseded", 10, issuer_kind="brain")

        # The pilot pushes the stick forward at 10 Hz until the drone has moved 20 m.
        start = (sc.tel.lat, sc.tel.lon)
        moved = 0.0
        t0 = time.monotonic()
        while moved < 20.0:
            assert time.monotonic() - t0 < 30, f"the sticks moved the drone only {moved:.1f} m"
            assert await sc.sticks(pitch=0.6, roll=0.0, takeover_id=takeover_id)
            await asyncio.sleep(0.1)
            moved = distance_m(*start, sc.tel.lat, sc.tel.lon)
        # Let go: no more values; after 1 s the ground station stops sending and the
        # drone stands still in LOITER.
        await sc.until(lambda: sc.tel.groundspeed_ms < 0.5, 10, "standing after letting go")
        here = (sc.tel.lat, sc.tel.lon)
        await sc.hold_for(
            5.0,
            lambda: sc.mode() == "LOITER" and distance_m(*here, sc.tel.lat, sc.tel.lon) < 2.0,
            "stands in LOITER without stick values",
        )
        alt = sc.tel.alt_rel_home_m
        assert abs(alt - ALT_M) <= 2.0  # the flight controller held the height
        # No AI order while the pilot steers.
        assert not [
            c for c in await sc.commands() if c["issuer_kind"] == "brain" and c["skill"] == "resume"
        ]

        res = await sc.press("handback")
        assert res.status_code == 202, res.text
        # The AI's resume carries the search on from where it stopped, to its end.
        await sc.wait_command("resume", "done", 30 + SURVEY_LIMIT_S, issuer_kind="brain")
        await sc.wait_command("return_home", "done", 240, issuer_kind="brain")
        await sc.until(lambda: not sc.tel.armed, 60, "disarmed after landing")
        assert sc.from_home_m() <= 3.0
        await sc.check_invariants()
