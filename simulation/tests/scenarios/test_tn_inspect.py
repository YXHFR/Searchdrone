"""The closer look's scenarios with the ground (spec "Titta närmare" §16.4, §17): the panel's
buttons, the gate and the pilot's decisions against ArduCopter SITL (builtin physics), the
whole chain from the ground station to the onboard executor.

The charter is approved in ``exercise`` (the look flies over its candidate) and the
executor has the map pack of the simulator's own world, so a look goes down to 30 m by
itself and asks "Gå lägre?". The camera is ``test_sk_inspect.Camera`` (an empty report per
still). The candidate's place moves with the seed.

* TN10: STOP, FLY HOME and the pilot's takeover while the drone goes down: the same
  steps within the same times as SK08 and SC01, the look aborted and never resumed.
* TN11: no decision: the return starts 120 s after ``inspect.done`` and the candidate is
  ``ready`` again in the queue.
* TN12: "Inte hen" while the drone waits: the return starts within 2 s, and the search
  goes on from where it broke off.
* TN15: a second look while one is open is refused (``INSPECT_BUSY``); so is every AI
  order that moves the drone, and the AI's cancel of the look (owner decision: during a
  look only the pilot moves the drone); an observer's decision is a note for the pilot.
* TN16: the AI proposes a look, the pilot clicks yes on the card; the AI's next proposal
  is refused until the first look is over.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario_rig import (  # pyright: ignore[reportMissingImports]
    FIXTURE,
    H,
    Scenario,
    fresh_db,
    migrated_db,
    passwords,
    pin,
    scenario,
    sim,
)
from test_sc_emergency import (  # pyright: ignore[reportMissingImports]
    HOME_MODES,
    STOP_ACK_S,
    stopped_and_holding,
)
from test_sc_linkloss import survey_box  # pyright: ignore[reportMissingImports]
from test_sk_inspect import (  # pyright: ignore[reportMissingImports]
    INSPECT_SKILLS,
    Camera,
    a_candidate,
)

from sd_ground.app import create_panel_app, panel_asgi
from sd_safety.targets import Plane, distance_m
from sd_schemas.inspect import AWAIT_DECISION_S
from sd_schemas.skills import SKILL_REGISTRY
from sd_sim.search_world import seed_offset

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.inspect]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]

SEEDS = [1, 2, 3]
TARGET_ENU = (40.0, 60.0)  # a person on the way to P1, inside the charter's area
JITTER_M = 10.0
DESCENDING_M = 37.0  # below the start height (40 m): the drone is going down
RETURN_SLACK_S = 2.0


def target_of(seed: int, salt: str, base: tuple[float, float] = TARGET_ENU) -> tuple[float, float]:
    de, dn = seed_offset(seed, JITTER_M, salt)
    return base[0] + de, base[1] + dn


class Events:
    """The executor's own events with the time they happened onboard (the ground's log
    has its own clock): the look's ``inspect.*`` and ``executor.*``."""

    def __init__(self, sc: Scenario) -> None:
        self.seen: list[tuple[float, str, dict[str, Any]]] = []
        ex = sc.r.onboard.ex
        send_inspect, send = ex.emit_inspect, ex.emit

        def inspect(kind: str, data: dict[str, Any]) -> None:
            self.seen.append((time.monotonic(), kind, data))
            send_inspect(kind, data)

        def executor(kind: str, data: dict[str, Any] | None = None) -> None:
            self.seen.append((time.monotonic(), f"executor.{kind}", dict(data or {})))
            send(kind, data)

        ex.emit_inspect = inspect
        ex.emit = executor

    def first(self, kind: str, since: float = 0.0) -> tuple[float, dict[str, Any]] | None:
        return next(((t, d) for t, k, d in self.seen if k == kind and t >= since), None)


async def exercise_airborne(sc: Scenario, sim: Any, **overrides: Any) -> Events:
    """In the air in ``exercise`` with the closer look allowed, the map pack onboard."""
    charter = FIXTURE["charters"]["charter_test"]
    skills = [*charter["allowed_skills"], *INSPECT_SKILLS, "survey_sector", "resume"]
    assert all(SKILL_REGISTRY.get(s) is not None for s in skills)
    await sc.airborne(allowed_skills=sorted(set(skills)), mode="exercise", **overrides)
    pack = await asyncio.to_thread(sim.stack.write_floor_pack, "tn")
    sc.r.onboard.ex.floor_pack = pack
    Camera(sc)
    return Events(sc)


async def looks(sc: Scenario) -> list[dict[str, Any]]:
    res = await sc.r.panel.get(f"/api/v1/missions/{sc.mid}/inspections")
    assert res.status_code == 200, res.text
    return list(res.json())


async def the_look(sc: Scenario) -> dict[str, Any]:
    return (await looks(sc))[-1]


async def look_at(sc: Scenario, cid: str) -> dict[str, Any]:
    """The pilot's "Titta närmare" through the gate: the order, running."""
    res = await sc.operator_order("inspect_detection", {"candidate_id": cid})
    assert res["gate"]["decision"] == "ALLOW", res
    return await sc.wait_command("inspect_detection", "running", 30)


async def asked(sc: Scenario, limit_s: float = 90.0) -> dict[str, Any]:
    """The drone's "Gå lägre?" in the panel."""

    async def offered() -> bool:
        return (await the_look(sc))["phase"] == "lower_offer"

    await sc.until(offered, limit_s, "Gå lägre?")
    return await the_look(sc)


async def answer(sc: Scenario, look: dict[str, Any], yes: bool) -> None:
    res = await sc.r.panel.post(
        f"/api/v1/inspections/{look['command_id']}/{'lower' if yes else 'stay'}", headers=H
    )
    assert res.status_code == 200, res.text


async def waiting(sc: Scenario, limit_s: float = 120.0) -> dict[str, Any]:
    """The steps are over: the drone hovers for the decision."""

    async def awaiting() -> bool:
        return (await the_look(sc))["phase"] == "await"

    await sc.until(awaiting, limit_s, "the wait for the decision")
    return await the_look(sc)


async def ended(sc: Scenario, limit_s: float = 15.0) -> dict[str, Any]:
    async def over() -> bool:
        return (await the_look(sc))["ended_at"] is not None

    await sc.until(over, limit_s, "the look ended")
    return await the_look(sc)


async def card_state(sc: Scenario, cid: str) -> str:
    res = await sc.r.panel.get(f"/api/v1/candidates/{cid}")
    assert res.status_code == 200, res.text
    return str(res.json()["card"]["state"])


async def queued(sc: Scenario, cid: str) -> bool:
    res = await sc.r.panel.get(f"/api/v1/missions/{sc.mid}/candidates")
    assert res.status_code == 200, res.text
    return any(c["candidate_id"] == cid for c in res.json()["items"])


@contextlib.asynccontextmanager
async def person(sc: Scenario, role: str, name: str) -> AsyncIterator[httpx.AsyncClient]:
    """A logged-in person's panel session (the cookie, as the browser sends it)."""
    g = sc.r.g
    token, _ = g.sessions.create_session(
        role=role, issuer_ref=f"operator:{name}", auth_mode="local"
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=panel_asgi(g, create_panel_app(g))),  # pyright: ignore[reportArgumentType]
        base_url="http://127.0.0.1:8080",
        cookies={"sd_session": token},
        headers=H,
    ) as client:
        yield client


async def decide(c: httpx.AsyncClient, cid: str, decision: str) -> httpx.Response:
    rev = (await c.get(f"/api/v1/candidates/{cid}")).json()["card"]["rev"]
    return await c.post(
        f"/api/v1/candidates/{cid}/decision",
        json={"decision": decision, "note": "", "subject_rev": rev},
    )


# --- TN10: STOP, FLY HOME and the takeover while going down -----------------------------------


@pytest.mark.budget_s(720)
@pytest.mark.parametrize("seed", SEEDS)
async def test_tn10_stop_fly_home_and_takeover_during_the_descent(
    sim: Any, migrated_db: Any, tmp_path: Path, seed: int
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await exercise_airborne(sc, sim)
        cid = await a_candidate(sc, sc.at(*target_of(seed, "TN10")))

        async def going_down() -> None:
            await look_at(sc, cid)
            await answer(sc, await asked(sc), yes=True)
            await sc.until(lambda: sc.tel.alt_rel_home_m < 30.0 - 2.0, 60, "below 30 m")

        # STOP: BRAKE within its time, still, holding; the look is over, never resumed.
        await going_down()
        t0 = time.monotonic()
        assert (await sc.press("stop")).status_code == 202
        await stopped_and_holding(sc, t0)
        look = await ended(sc)
        assert (look["outcome"], look["detail"]) == ("aborted", "stop"), look
        assert await card_state(sc, cid) == "ready"
        assert (await sc.press("resume")).status_code == 202

        async def let_go() -> bool:
            return (await sc.onboard_state())["executor_state"] != "HELD"

        await sc.until(let_go, 30, "released onboard")

        # The pilot's takeover: LOITER at once, the look over.
        await going_down()
        res = await sc.press("takeover")
        assert res.status_code == 202, res.text
        await sc.until(lambda: sc.mode() == "LOITER", 3, "LOITER after the takeover")
        look = await ended(sc)
        assert (look["outcome"], look["detail"]) == ("aborted", "override"), look
        await sc.until(lambda: sc.tel.groundspeed_ms < 0.5, 10, "standing")
        assert (await sc.press("resume")).status_code == 202
        await sc.until(let_go, 30, "released onboard")

        # FLY HOME: SMART_RTL within its time, the look over, landed at home.
        await going_down()
        assert (await sc.press("return_home")).status_code == 202
        await sc.until(lambda: sc.mode() in HOME_MODES, 2.0 * STOP_ACK_S, "flying home")
        assert sc.mode() == "SMART_RTL"
        look = await ended(sc)
        assert (look["outcome"], look["detail"]) == ("aborted", "home"), look
        await sc.until(lambda: not sc.tel.armed, 240, "landed")
        assert sc.from_home_m() <= 3.0
        await sc.check_invariants()


# --- TN11 / TN12: the wait for the pilot's decision ------------------------------------------


@pytest.mark.budget_s(600)
@pytest.mark.parametrize("seed", SEEDS)
async def test_tn11_no_decision_returns_after_2_min_and_the_candidate_is_queued(
    sim: Any, migrated_db: Any, tmp_path: Path, seed: int
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        ev = await exercise_airborne(sc, sim)
        cid = await a_candidate(sc, sc.at(*target_of(seed, "TN11")))
        here = (sc.tel.lat, sc.tel.lon)  # where the look breaks off
        await look_at(sc, cid)
        await answer(sc, await asked(sc), yes=False)  # "Nej, stanna här"
        look = await waiting(sc)
        assert look["done"]["end"] == "lower_declined", look
        done = ev.first("inspect.done")
        assert done is not None
        # Acceptance 16: the return 120 +- 2 s after inspect.done ...
        await sc.until(
            lambda: ev.first("executor.inspect_return") is not None,
            AWAIT_DECISION_S + 10.0,
            "the return",
        )
        back = ev.first("executor.inspect_return")
        assert back is not None and back[1]["cause"] == "timeout", back
        assert abs(back[0] - done[0] - AWAIT_DECISION_S) <= RETURN_SLACK_S, back[0] - done[0]
        # ... the candidate ready in the queue ...
        look = await ended(sc)
        assert look["outcome"] == "timeout", look
        assert await card_state(sc, cid) == "ready"
        assert await queued(sc, cid)
        # ... and the drone back where the look broke off.
        await sc.until(
            lambda: distance_m(*here, sc.tel.lat, sc.tel.lon) <= 5.0, 90, "back at the break"
        )
        await sc.check_invariants()


@pytest.mark.budget_s(600)
@pytest.mark.parametrize("seed", SEEDS)
async def test_tn12_not_them_returns_at_once_and_the_search_goes_on(
    sim: Any, migrated_db: Any, tmp_path: Path, seed: int
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        ev = await exercise_airborne(sc, sim)
        # The person inside the search box at P1, seen while the survey flies its lines.
        p1 = Plane(*sc.r.home).xy(*pin("P1"))
        cid = await a_candidate(sc, sc.at(*target_of(seed, "TN12", (p1[0] + 20.0, p1[1]))))
        res = await sc.order("survey_sector", {"polygon": survey_box(sc)})
        assert res.get("decision") == "ALLOW", res
        survey = await sc.wait_command("survey_sector", "running", 60)
        await sc.until(lambda: sc.mode() == "GUIDED" and sc.tel.groundspeed_ms > 2.0, 90, "lines")
        brk = (sc.tel.lat, sc.tel.lon)
        await look_at(sc, cid)
        await answer(sc, await asked(sc), yes=False)
        await waiting(sc)
        async with person(sc, "pilot", "kalle") as pilot:
            t0 = time.monotonic()
            res = await decide(pilot, cid, "reject")
            assert res.status_code == 200, res.text
        # Acceptance 17: the return within 2 s of the decision ...
        await sc.until(
            lambda: ev.first("executor.inspect_return", t0) is not None, 5.0, "the return"
        )
        back = ev.first("executor.inspect_return", t0)
        assert back is not None and back[0] - t0 <= 2.0, back
        look = await ended(sc)
        assert look["outcome"] == "rejected", look
        assert await card_state(sc, cid) == "rejected"
        # ... and the survey goes on from where it broke off: its lines again, first
        # within 30 m of the break.
        ex = sc.r.onboard.ex
        await sc.until(lambda: ex.state == "SURVEY", 90, "the survey again")
        assert distance_m(*brk, sc.tel.lat, sc.tel.lon) <= 30.0
        assert (await sc.command(survey["id"]))["status"] in ("running", "superseded")
        await sc.check_invariants()


# --- TN15 / TN16: one look at a time, the pilot's ----------------------------------------------


@pytest.mark.budget_s(600)
@pytest.mark.parametrize("seed", SEEDS)
async def test_tn15_one_look_at_a_time_and_only_the_pilot_moves_the_drone(
    sim: Any, migrated_db: Any, tmp_path: Path, seed: int
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await exercise_airborne(sc, sim)
        first = await a_candidate(sc, sc.at(*target_of(seed, "TN15")))
        other = await a_candidate(sc, sc.at(*target_of(seed, "TN15b", (-40.0, 60.0))))
        look = await look_at(sc, first)

        # TN15: a second look while the first is open.
        res = await sc.operator_order("inspect_detection", {"candidate_id": other})
        assert res["gate"]["decision"] == "REJECT", res
        assert "INSPECT_BUSY" in res["gate"]["reason_codes"], res
        # Owner decision: during a look only the pilot moves the drone, the AI included.
        await asyncio.sleep(3.1)  # the gate's rate limit for the AI
        lat, lon = sc.at(0.0, 0.0)
        res = await sc.order("goto", {"lat": lat, "lon": lon, "alt_m": 40})
        assert res["decision"] == "REJECT" and "INSPECT_BUSY" in res["reason_codes"], res
        cancel = await sc.r.commander.post(
            f"/commander/v1/missions/{sc.mid}/commands/{look['id']}/cancel"
        )
        assert cancel.status_code == 409 and cancel.json()["code"] == "INSPECT_BUSY", cancel.text
        # An observer's decision is a note for the pilot: the look goes on.
        async with person(sc, "observer", "olle") as observer:
            res = await decide(observer, first, "reject")
            assert res.status_code == 200, res.text
        assert (await the_look(sc))["ended_at"] is None
        assert await card_state(sc, first) == "inspecting"
        assert (await sc.command(look["id"]))["status"] == "running"
        # The pilot's own decision ends it.
        await answer(sc, await asked(sc), yes=False)
        await waiting(sc)
        async with person(sc, "pilot", "kalle") as pilot:
            res = await decide(pilot, first, "unsure")
            assert res.status_code == 200, res.text
        assert (await ended(sc))["outcome"] == "unsure"
        await sc.check_invariants()


@pytest.mark.budget_s(600)
@pytest.mark.parametrize("seed", SEEDS)
async def test_tn16_the_ai_proposes_and_the_pilot_clicks_yes(
    sim: Any, migrated_db: Any, tmp_path: Path, seed: int
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await exercise_airborne(sc, sim)
        first = await a_candidate(sc, sc.at(*target_of(seed, "TN16")))
        other = await a_candidate(sc, sc.at(*target_of(seed, "TN16b", (-40.0, 60.0))))
        # The AI's proposal is a card; nothing flies until the pilot's click.
        res = await sc.order("inspect_detection", {"candidate_id": first})
        assert res["decision"] == "PENDING_APPROVAL", res
        await asyncio.sleep(2.0)
        assert not await looks(sc)
        assert await card_state(sc, first) == "ready"
        ok = await sc.r.approve(str(res["approval_id"]))
        assert ok.status_code == 200, ok.text
        await sc.wait_command("inspect_detection", "running", 30)
        look = await the_look(sc)
        assert look["started_by"].startswith("operator:"), look
        # Not the next one until the first is over.
        await asyncio.sleep(3.1)
        res = await sc.order("inspect_detection", {"candidate_id": other})
        assert res["decision"] == "REJECT" and "INSPECT_BUSY" in res["reason_codes"], res
        await answer(sc, await asked(sc), yes=False)
        await waiting(sc)
        async with person(sc, "pilot", "kalle") as pilot:
            res = await decide(pilot, first, "reject")
            assert res.status_code == 200, res.text
        await ended(sc)
        await asyncio.sleep(3.1)
        res = await sc.order("inspect_detection", {"candidate_id": other})
        assert res["decision"] == "PENDING_APPROVAL", res
        await sc.check_invariants()
