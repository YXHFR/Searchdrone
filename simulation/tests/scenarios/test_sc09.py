"""SC09: a take-off needs an approved charter and its own "clear around the drone".

Base case: the charter is approved (READY) but nobody confirmed "clear around the drone".
The AI's take-off on "Starta." is refused with ``CLEARANCE_REQUIRED`` and the AI asks for
the confirmation; for 30 s nothing arms the motors. Then the pilot confirms and the AI's
rule start takes off.

Variants, each its own run, each refused at the gate within L-GATE with no arming for
30 s: (a) no approved charter, (c) the confirmation is older than 120 s, (d) the
mission has landed, (e) the confirmation was for the previous mission, (f) a new charter
version was approved after it, (g) the flight controller's fence was changed behind the
ground's back: only the onboard computer can see it and refuses (``FENCE_NOT_VERIFIED``).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from rig_gate import (  # pyright: ignore[reportMissingImports]
    FENCE_TAMPER_TYPES,
    arms,
    clearance_status,
    commander_charter,
    end_mission,
    fence_tamper,
    takeoffs,
)
from scenario_rig import AI_RATE_S, Scenario, scenario  # pyright: ignore[reportMissingImports]

from sd_safety.constants import CLEARANCE_VALID_S
from sd_sim import expect

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.script("SC09")]

NO_ARM_S = 30.0
ARMED_WITHIN_S = 10.0


async def _no_arming(sc: Scenario, since: int, id_: str) -> None:
    await sc.still(lambda: not sc.tel.armed, NO_ARM_S, "FC armerades inte")
    sc.criterion(
        id_,
        arms(sc, since) == 0 and takeoffs(sc, since) == 0,
        "ingen ARM och ingen NAV_TAKEOFF",
        arms=arms(sc, since),
    )


async def _refused(
    sc: Scenario, code: str | tuple[str, ...], id_: str, *, side: str = "ground", after: int = 0
) -> None:
    wanted = (code,) if isinstance(code, str) else code
    t = time.monotonic()
    cmd = await sc.wait_command(
        "takeoff",
        ("rejected_by_gate", "rejected_onboard"),
        expect.L_GATE_S + 8,
        issuer_kind="brain",
        after=after,
    )
    gates = await sc.gate(cmd["id"])
    found = [c for g in gates if g.side == side for c in g.reason_codes]
    sc.criterion(
        id_,
        any(c in found for c in wanted),
        f"take-off avvisas med {code} ({side})",
        codes=found,
        measured=round(time.monotonic() - t, 2),
    )


@pytest.mark.budget_s(135)
async def test_sc09(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario("test_sc09", sim, migrated_db, tmp_path, "SC09") as sc:
        await sc.ready()
        start = len(sc.tl.sent)
        await sc.say("Starta.")
        await _refused(sc, "CLEARANCE_REQUIRED", "SC09.2")
        await sc.wait_chat(
            lambda m: m.get("role") == "brain" and "fritt runt" in m.get("text", ""),
            expect.L_CHAT_S,
        )
        await _no_arming(sc, start, "SC09.1")

        await sc.confirm_clear()
        t = time.monotonic()
        await sc.until(lambda: sc.tel.armed, ARMED_WITHIN_S, "armerad efter bekräftelsen")
        sc.criterion("SC09.3a", True, "armerad inom 10 s", measured=time.monotonic() - t)
        cmd = await sc.wait_command("takeoff", ("accepted", "running", "done"), 5, after=1)
        await sc.until(
            lambda: abs(sc.tel.alt_rel_home_m - 40) <= 2, expect.L_TAKEOFF_S, "L-TAKEOFF"
        )
        sc.criterion("SC09.3b", cmd["issuer_kind"] == "brain", "regelstarten lyfte (L-TAKEOFF)")
        sc.criterion("SC09.4", arms(sc, start) == 1, "exakt en armering", arms=arms(sc, start))
        await sc.finish()


BUDGET_S = {"a": 100, "c": 270, "d": 200, "e": 120, "f": 115, "g": 140}


@pytest.mark.parametrize(
    "variant", [pytest.param(v, marks=pytest.mark.budget_s(b)) for v, b in BUDGET_S.items()]
)
async def test_sc09_variant(sim: Any, migrated_db: Any, tmp_path: Path, variant: str) -> None:
    async with scenario(f"test_sc09[{variant}]", sim, migrated_db, tmp_path, "SC09_v") as sc:
        await VARIANTS[variant](sc)


async def _a(sc: Scenario) -> None:
    """No approved charter: the confirmation is refused, so is the take-off."""
    await sc.open_mission()
    start = len(sc.tl.sent)
    res = await sc.r.clearance(sc.mission_id)
    sc.criterion(
        "SC09a.0",
        res.status_code == 409 and res.json().get("code") == "CHARTER_NOT_APPROVED",
        "bekräftelsen ger 409 utan ram",
        status=res.status_code,
    )
    await sc.say("Starta.")
    await _refused(sc, "CHARTER_NOT_APPROVED", "SC09a.1")
    await _no_arming(sc, start, "SC09a.2")
    await sc.finish()


async def _c(sc: Scenario) -> None:
    """The confirmation is too old."""
    await sc.ready()
    start = len(sc.tl.sent)
    await sc.confirm_clear()
    await asyncio.sleep(CLEARANCE_VALID_S + 5)
    status = await clearance_status(sc)
    sc.criterion("SC09c.0", status == "expired", "bekräftelsen har gått ut", status=status)
    await sc.say("Starta.")
    # The ground drops an expired confirmation at once, so its gate then sees none:
    # CLEARANCE_REQUIRED, where 07 names CLEARANCE_EXPIRED (both refuse the take-off).
    await _refused(sc, ("CLEARANCE_EXPIRED", "CLEARANCE_REQUIRED"), "SC09c.1")
    await _no_arming(sc, start, "SC09c.2")
    await sc.finish()


async def _d(sc: Scenario) -> None:
    """After a landing (mission LANDED) no new take-off."""
    await sc.ready()
    await sc.confirm_clear()
    await sc.say("Starta.")
    await sc.wait_done("takeoff", 90, issuer_kind="brain")
    await asyncio.sleep(10)
    await sc.press("return_home")
    await sc.until(
        lambda: not sc.tel.armed and sc.tel.landed_state == "on_ground",
        expect.L_LAND_S,
        "landad och desarmerad",
    )
    start = len(sc.tl.sent)
    await sc.r.wait_status(sc.mission_id, "LANDED", 5)
    await sc.say("Starta.")
    await _refused(sc, "MISSION_STATE_INVALID", "SC09d.1", after=1)
    await _no_arming(sc, start, "SC09d.2")
    await sc.finish()


async def _e(sc: Scenario) -> None:
    """The confirmation of the previous mission does not carry over."""
    await sc.ready()
    await sc.confirm_clear()
    res = await end_mission(sc)
    assert res.status_code == 200, res.text
    await sc.r.wait_status(sc.mission_id, "CLOSED", 15)
    await sc.ready()
    start = len(sc.tl.sent)
    await asyncio.sleep(AI_RATE_S)
    await sc.say("Starta.")
    await _refused(sc, "CLEARANCE_REQUIRED", "SC09e.1")
    await _no_arming(sc, start, "SC09e.2")
    await sc.finish()


async def _f(sc: Scenario) -> None:
    """A new charter version makes the confirmation invalid."""
    await sc.ready()
    await sc.confirm_clear()
    res = await commander_charter(sc, "charter_test")
    assert res.status_code == 201, res.text
    await sc.approve_charter()
    start = len(sc.tl.sent)
    await asyncio.sleep(AI_RATE_S)
    await sc.say("Starta.")
    await _refused(sc, "CLEARANCE_REQUIRED", "SC09f.1")
    await _no_arming(sc, start, "SC09f.2")
    await sc.finish()


async def _g(sc: Scenario) -> None:
    """The fence in the flight controller is not the charter's: the onboard computer
    sees it (the ground verified the fence at approval and lets it through)."""
    await sc.ready()
    await fence_tamper(sc)
    start = len(sc.tl.sent)
    await sc.confirm_clear()
    await sc.say("Starta.")
    await _refused(sc, "FENCE_NOT_VERIFIED", "SC09g.1", side="onboard")
    await _no_arming(sc, start, "SC09g.2")
    await sc.finish(harness_extra=FENCE_TAMPER_TYPES, allow_mismatch=True)


VARIANTS = {"a": _a, "c": _c, "d": _d, "e": _e, "f": _f, "g": _g}
