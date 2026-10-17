"""The AI commander on ArduCopter SITL: when it dies, and in mode A.

* SC08a: the AI dies 30 s into the search: the ground station notices within 6 s and says
  so in the chat; the search is flown to its end and covers the box; then the drone holds
  where it is (no flight home on its own) until the pilot decides.
* SC11: mode A: every order from the AI waits for the pilot's click; nothing moves before
  it; after four clicks the flight is the same as SC01.
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
    Scenario,
    fresh_db,
    migrated_db,
    passwords,
    scenario,
    sim,
    survey_box_enu,
)

from sd_geo.coverage import coverage_fraction
from sd_safety.constants import BRAIN_HEARTBEAT_TIMEOUT_S
from sd_safety.targets import Plane, distance_m

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.brain]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]

AI_SILENT_S = 150.0  # the drone waits at least this long without a flight home


def coverage(sc: Scenario, since: float) -> float:
    plane = Plane(*sc.r.home)
    track = [(*plane.xy(s.lat, s.lon), s.alt_m) for s in sc.track if s.t >= since and s.armed]
    return coverage_fraction(survey_box_enu(), track, 40.0, 30.0)


async def still_for(sc: Scenario, seconds: float, what: str) -> None:
    here = (sc.tel.lat, sc.tel.lon, sc.tel.alt_rel_home_m)
    await sc.hold_for(
        seconds,
        lambda: (
            distance_m(here[0], here[1], sc.tel.lat, sc.tel.lon) <= 3.0
            and abs(sc.tel.alt_rel_home_m - here[2]) <= 1.0
        ),
        what,
    )


@pytest.mark.budget_s(660)
async def test_sc08a_ai_dies_mid_search(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne_with_ai("SC08")
        await sc.wait_command("survey_sector", "running", 120, issuer_kind="brain")
        t_survey = time.monotonic()
        await asyncio.sleep(30.0)

        await sc.kill_brain()
        t0 = time.monotonic()
        await sc.until(
            lambda: sc.r.g.brain_state == "down", BRAIN_HEARTBEAT_TIMEOUT_S + 1.0, "brain down"
        )

        alerts = [a for a in sc.published("alert") if a.get("code") == "BRAIN_DISCONNECTED"]
        assert alerts and "svarar inte" in alerts[0]["message_sv"], alerts
        assert time.monotonic() - t0 <= BRAIN_HEARTBEAT_TIMEOUT_S + 1.0

        await sc.wait_command("survey_sector", "done", 300, issuer_kind="brain")
        done = time.monotonic()
        assert coverage(sc, t_survey) >= 0.95
        await sc.until(lambda: sc.tel.groundspeed_ms < 0.5, 10, "stopped after the search")

        async def holding() -> bool:
            live = await sc.onboard_state()
            return live["executor_state"] == "HOLD" and live["hold_reason"] in (
                "ai_silent",
                "skill_done",
            )

        await sc.until(holding, 5, "executor HOLD")
        here = (sc.tel.lat, sc.tel.lon)
        await sc.hold_for(
            AI_SILENT_S - (time.monotonic() - done),
            lambda: (
                sc.mode() in ("LOITER", "BRAKE", "GUIDED")
                and distance_m(*here, sc.tel.lat, sc.tel.lon) <= 3.0
            ),
            "waits where the search ended, no flight home",
        )
        assert await holding()
        skills = [c["skill"] for c in await sc.commands()]
        assert skills.count("survey_sector") == 1 and "return_home" not in skills

        assert (await sc.press("return_home")).status_code == 202
        await sc.until(lambda: not sc.tel.armed, 240, "landed")
        await sc.check_invariants()


@pytest.mark.budget_s(660)
async def test_sc11_mode_a_every_step_waits_for_a_click(
    sim: Any, migrated_db: Any, tmp_path: Path
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.open()
        await sc.start_brain("SC11")
        await sc.say("Använd ramen charter_test.")
        await sc.approve_proposed(autonomy_mode="APPROVE_EACH")
        assert (await sc.r.mission(sc.mid))["charter"]["body"]["autonomy_mode"] == "APPROVE_EACH"
        await sc.clearance()
        await sc.say("Starta.")
        clicked: list[str] = []
        for skill in ("takeoff", "goto", "survey_sector", "return_home"):
            cmd = await sc.wait_command(skill, "pending_approval", 300, issuer_kind="brain")
            cards = [
                a
                for a in (
                    await sc.r.panel.get(
                        f"/api/v1/missions/{sc.mid}/approvals", params={"status": "pending"}
                    )
                ).json()
                if a["subject_id"] == cmd["id"]
            ]
            assert len(cards) == 1, cards
            await still_for(sc, 5.0, f"nothing moves before the click on {skill}")
            res = await sc.r.panel.post(
                f"/api/v1/approvals/{cards[0]['id']}/decision",
                json={
                    "decision": "approve",
                    "subject_version": cards[0]["subject_version"],
                    "confirmations": cards[0]["required_confirmations"],
                },
                headers={"X-SD-Client": "panel", "Content-Type": "application/json"},
            )
            assert res.status_code == 200, res.text
            clicked.append(skill)
            await sc.wait_command(skill, "done", 300, issuer_kind="brain")
        assert clicked == ["takeoff", "goto", "survey_sector", "return_home"]
        await sc.until(lambda: not sc.tel.armed, 60, "disarmed")
        assert sc.from_home_m() <= 3.0
        await sc.check_invariants()
