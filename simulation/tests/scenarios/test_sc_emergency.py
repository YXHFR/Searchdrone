"""STOP, FLY HOME, the chat's stop words and the absent operator, on ArduCopter SITL.

* SC02: STOP mid-flight: BRAKE within a second, standing still within 5 s and for 20 s;
  the AI cannot resume (only the pilot's click releases a stop), a question or "Fortsätt."
  in the chat releases nothing; after the pilot's Resume the AI flies on to P1.
* SC03: FLY HOME mid-search: SMART_RTL within a second, the AI's goto is refused while
  the drone flies home, the search is not taken up again, landing within 3 m of home.
* The chat's stop words: "stopp" and "flyg hem" from the pilot act like the buttons
  without the AI process; from an observer they do nothing (and the chat still works);
  one chat line acts once, also when the AI reports the same line.
* SC19: the operator closes the panel: after 30 s an alarm and a hold, never a flight
  home; the AI's orders are refused; the drone waits for the pilot's Resume.
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
    VID,
    H,
    Scenario,
    fresh_db,
    migrated_db,
    passwords,
    pin,
    scenario,
    sim,
)

from sd_safety.constants import OPERATOR_ABSENT_S
from sd_safety.targets import distance_m
from sd_schemas.base import new_id

pytestmark = [pytest.mark.sitl, pytest.mark.slow]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]

STOP_ACK_S = 1.0  # the button's request -> the mode in the ground's telemetry (+1 s slack)
STILL_S = 5.0
HOLD_S = 20.0
HOME_MODES = ("SMART_RTL", "RTL")


async def stopped_and_holding(sc: Scenario, t0: float) -> None:
    """BRAKE within L-STOP-HB, standing still within 5 s, then within 3 m / 1 m for 20 s."""
    await sc.until(lambda: sc.mode() == "BRAKE", 2.0 * STOP_ACK_S, "BRAKE")
    await sc.until(lambda: sc.tel.groundspeed_ms < 0.5, STILL_S - (time.monotonic() - t0), "still")
    here = (sc.tel.lat, sc.tel.lon, sc.tel.alt_rel_home_m)
    await sc.hold_for(
        HOLD_S,
        lambda: (
            distance_m(here[0], here[1], sc.tel.lat, sc.tel.lon) <= 3.0
            and abs(sc.tel.alt_rel_home_m - here[2]) <= 1.0
        ),
        "holds where it stopped",
    )


@pytest.mark.brain
@pytest.mark.budget_s(480)
async def test_sc02_stop_and_resume(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne_with_ai("SC02")
        await sc.wait_command("goto", "running", 30, issuer_kind="brain")
        await sc.until(lambda: sc.from_home_m() >= 80.0, 90, "80 m out")

        t0 = time.monotonic()
        res = await sc.press("stop")
        assert res.status_code == 202, res.text
        await stopped_and_holding(sc, t0)
        live = await sc.onboard_state()
        assert live["executor_state"] == "HELD", live
        assert live["held_by"] == "operator_stop", live
        mission = (await sc.r.mission(sc.mid))["mission"]
        assert mission["status"] == "PAUSED" and mission["paused_by"] == "operator", mission
        # The AI's resume is refused on the ground: only the pilot releases a stop.
        tried = await sc.wait_command("resume", "rejected_by_gate", 20, issuer_kind="brain")
        assert "RESUME_NEEDS_OPERATOR" in str(tried["status_reason"]), tried
        assert [a["via"] for a in sc.actions("stop")][:1] == ["button"]  # on the live stream

        # A question and "Fortsätt." in the chat release nothing.
        line = await sc.say("Vad hände?")
        assert line["handled_by"] is None
        line = await sc.say("Fortsätt.")
        assert line["handled_by"] is None and line["parsed_intent"] is None
        await asyncio.sleep(10.0)
        assert sc.mode() == "BRAKE"
        assert sc.r.actions.latches.get(VID) is not None
        assert not sc.actions("resume")

        # The pilot's Resume: the AI flies on to P1.
        before = [c["id"] for c in await sc.commands()]
        res = await sc.press("resume")
        assert res.status_code == 202, res.text
        resumed = await sc.wait_command("resume", "done", 120, issuer_kind="brain", besides=before)
        assert resumed["status"] == "done"
        p1 = pin("P1")
        assert distance_m(*p1, sc.tel.lat, sc.tel.lon) <= 5.0
        res = await sc.press("return_home")
        assert res.status_code == 202
        await sc.until(lambda: not sc.tel.armed, 180, "landed")
        assert sc.from_home_m() <= 3.0
        await sc.check_invariants()


@pytest.mark.brain
@pytest.mark.budget_s(480)
async def test_sc03_fly_home_mid_search(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne_with_ai("SC03")
        survey = await sc.wait_command("survey_sector", "running", 120, issuer_kind="brain")
        await asyncio.sleep(30.0)
        before = [c["id"] for c in await sc.commands()]

        res = await sc.press("return_home")
        assert res.status_code == 202, res.text
        await sc.until(lambda: sc.mode() in HOME_MODES, 2.0 * STOP_ACK_S, "flying home")
        assert sc.mode() == "SMART_RTL"  # RTL only when SmartRTL is refused
        # The AI's goto to P1 is refused while the drone flies home.
        tried = await sc.wait_command(
            "goto", "rejected_by_gate", 20, issuer_kind="brain", besides=before
        )
        gates = await sc.gate(tried["id"])
        assert "OPERATOR_OVERRIDE_ACTIVE" in gates[0]["reason_codes"]
        await sc.until(lambda: not sc.tel.armed, 180, "landed")
        assert sc.from_home_m() <= 3.0
        after = await sc.command(survey["id"])
        assert after["status"] in ("cancelled", "superseded"), after
        assert [c for c in await sc.commands() if c["skill"] == "survey_sector"] == [after]
        await sc.check_invariants()


@pytest.mark.budget_s(420)
async def test_chat_stop_words_without_the_ai(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne()  # no AI process at all
        assert sc.r.g.brain_state != "ready"
        lat, lon = pin("P1")
        res = await sc.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 5})
        assert res.get("decision") == "ALLOW", res
        await sc.until(lambda: sc.from_home_m() >= 40.0, 60, "on the way")

        # An observer cannot write in the chat, so its stop word does nothing; the chat
        # still works for the pilot.
        res = await sc.r.panel.post(
            f"/api/v1/missions/{sc.mid}/chat",
            json={"client_msg_id": str(new_id()), "text": "stopp"},
            headers={**H, "Authorization": f"Bearer {sc.observer_token}"},
        )
        assert res.status_code == 403, res.text
        await asyncio.sleep(3.0)
        assert sc.r.actions.latches.get(VID) is None
        assert sc.mode() == "GUIDED"

        # The pilot's "stopp": the button's path, at once.
        t0 = time.monotonic()
        line = await sc.say("stopp")
        assert line["handled_by"] == "ground" and line["parsed_intent"] == "stop", line
        await stopped_and_holding(sc, t0)
        stops = [a for a in sc.actions("stop") if a["result"] == "dispatched"]
        assert len(stops) == 1, sorted({t for t, _ in sc.bus_log})
        assert stops[0]["via"] == "chat", stops
        assert stops[0]["by"] == "operator:ci-harness"
        # The AI reports the same line as well: it still acts only once.
        res = await sc.r.commander.post(
            f"/commander/v1/missions/{sc.mid}/chat-intents",
            json={
                "request_id": str(new_id()),
                "message_id": line["id"],
                "intent": "stop",
                "probability": 0.99,
                "engine": "jev",
                "location": "cloud",
                "latency_ms": 120,
            },
        )
        assert res.status_code in (202, 409), res.text
        await asyncio.sleep(2.0)
        assert len([a for a in sc.actions("stop") if a["result"] == "dispatched"]) == 1

        # Resume, then the pilot's "flyg hem": home and land, still without the AI.
        assert (await sc.press("resume")).status_code == 202
        await sc.until(lambda: sc.r.actions.latches.get(VID) is None, 5, "released")
        line = await sc.say("flyg hem")
        assert line["handled_by"] == "ground" and line["parsed_intent"] == "return_home"
        await sc.until(lambda: sc.mode() in HOME_MODES, 2.0 * STOP_ACK_S, "flying home")
        homes = [a for a in sc.actions("return_home") if a["result"] == "dispatched"]
        assert len(homes) == 1 and homes[0]["via"] == "chat", homes
        await sc.until(lambda: not sc.tel.armed, 180, "landed")
        assert sc.from_home_m() <= 3.0
        await sc.check_invariants()


@pytest.mark.budget_s(420)
async def test_sc19_operator_absent(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne()
        await sc.r.wait_status(sc.mid, "ACTIVE", 30)
        lat, lon = pin("P1")
        res = await sc.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 3})
        assert res.get("decision") == "ALLOW", res

        sc.r.operator_away.set()  # the pilot closes the panel
        gone = time.monotonic()
        await sc.until(
            lambda: sc.mode() == "BRAKE", OPERATOR_ABSENT_S + 2.0 + 2.0, "a hold for no operator"
        )
        assert time.monotonic() - gone >= OPERATOR_ABSENT_S - 1.0
        assert sc.published("alarm.OPERATOR_ABSENT")
        await sc.until(lambda: sc.tel.groundspeed_ms < 0.5, STILL_S, "still")
        await asyncio.sleep(10.0)
        res = await sc.order("goto", {"lat": lat, "lon": lon, "alt_m": 40})
        assert res.get("decision") == "REJECT", res
        # No flight home on its own: it waits.
        await sc.hold_for(45.0, lambda: sc.mode() not in (*HOME_MODES, "LAND"), "no flight home")

        sc.r.operator_away.clear()  # the pilot is back; the drone still waits for the click
        await asyncio.sleep(3.0)
        assert sc.mode() == "BRAKE"
        assert (await sc.press("resume")).status_code == 202
        assert (await sc.press("return_home")).status_code == 202
        await sc.until(lambda: not sc.tel.armed, 180, "landed")
        await sc.check_invariants()
