"""SC12: what needs a click in mode B.

In the air in mode B: (a) the AI's ``land_at`` (a stub) is refused at once with
``SKILL_NOT_IMPLEMENTED``, no card, nothing moves. A smaller charter (``charter_test``
cut to 300 x 300 m, inside the old one) proposed on the AI's path is only a proposal
(c): the charter in force does not change and the vehicle keeps to it; (e) the pilot
rejects it and nothing changes; (d) proposed again and approved: the new version is in
force, the old one superseded, installed onboard, and no fence goes up to the flight
controller in the air (the smaller area is held by the gates): a goto inside the old but
outside the new area is refused with ``OUTSIDE_CHARTER``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from rig_gate import commander_charter  # pyright: ignore[reportMissingImports]
from scenario_rig import AI_RATE_S, Scenario, scenario  # pyright: ignore[reportMissingImports]

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.script("SC12")]

# charter_test cut to its southern 300 m, 1 m inside its edges (it may only shrink in the air)
SMALLER = [(-149.0, -99.0), (149.0, -99.0), (149.0, 200.0), (-149.0, 200.0)]
OLD_ONLY = (0.0, 250.0)  # inside charter_test, outside the smaller charter


@pytest.mark.budget_s(190)
async def test_sc12(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario("test_sc12", sim, migrated_db, tmp_path, "SC12") as sc:
        await _run(sc)


async def _charters(sc: Scenario) -> list[dict[str, Any]]:
    res = await sc.r.panel.get(f"/api/v1/missions/{sc.mission_id}/charters")
    assert res.status_code == 200, res.text
    return sorted(res.json(), key=lambda c: c["version"])


async def _in_force(sc: Scenario) -> int:
    return next(c["version"] for c in await _charters(sc) if c["status"] == "approved")


async def _run(sc: Scenario) -> None:
    await sc.arrange_airborne()
    first = await _in_force(sc)
    after_takeoff = len(sc.tl.sent)

    # a. A stub: refused at once on the ground, no card, nothing moves.
    await asyncio.sleep(AI_RATE_S)
    await sc.say("Landa här.")
    cmd = await sc.wait_command("land_at", "rejected_by_gate", 10, issuer_kind="brain")
    codes = await sc.codes(cmd["id"])
    cards = await sc.approvals("command")
    sc.criterion(
        "SC12a.1",
        "SKILL_NOT_IMPLEMENTED" in codes and not cards,
        "stubben avvisas utan kort",
        codes=codes,
    )
    await sc.assert_drift("SC12a.2", 30.0)

    # c. A smaller charter proposed: only a proposal.
    res = await commander_charter(sc, "charter_test", area_enu=SMALLER)
    assert res.status_code == 201, res.text
    card = await sc.pending("charter")
    proposed = [c for c in await _charters(sc) if c["status"] == "proposed"]
    sc.criterion(
        "SC12c.1",
        len(proposed) == 1 and await _in_force(sc) == first,
        "ett förslag; gällande ram oförändrad",
        proposed=len(proposed),
    )
    await sc.assert_drift("SC12c.2")

    # e. The pilot rejects it: nothing changes.
    res = await sc.decide(card, "reject")
    assert res.status_code == 200, res.text
    rows = await _charters(sc)
    rejected = [c for c in rows if c["status"] == "rejected"]
    sc.criterion(
        "SC12e.1",
        len(rejected) == 1 and await _in_force(sc) == first,
        "ramen avvisad; gällande ram oförändrad",
        statuses=[c["status"] for c in rows],
    )

    # d. Proposed again and approved: in force, installed, no fence upload in the air.
    res = await commander_charter(sc, "charter_test", area_enu=SMALLER)
    assert res.status_code == 201, res.text
    card = await sc.pending("charter")
    res = await sc.decide(card, "approve", autonomy_mode="WITHIN_CHARTER")
    assert res.status_code == 200, res.text

    async def installed() -> bool:
        rows = await _charters(sc)
        new = rows[-1]
        return new["status"] == "approved" and new["install_status"] == "installed"

    await sc.until(installed, 20, "den nya ramen installerad ombord")
    rows = await _charters(sc)
    old = next(c for c in rows if c["version"] == first)
    sc.criterion(
        "SC12d.1",
        old["status"] == "superseded"
        and rows[-1]["body"]["fence_hash"] == old["body"]["fence_hash"],
        "ny version gäller, den gamla ersatt, staketet fört över",
        statuses=[c["status"] for c in rows],
    )
    uploads = [
        m
        for m in sc.tl.sent[after_takeoff:]
        if m.type in ("MISSION_COUNT", "MISSION_ITEM_INT") and m.mission_type == 1
    ]
    sc.criterion("SC12d.2", not uploads, "inget staket laddas upp i luften", found=len(uploads))
    lat, lon = sc.at(*OLD_ONLY)
    res = await sc.commander_order("goto", lat=lat, lon=lon, alt_m=40)
    assert res.status_code == 201, res.text
    body = res.json()
    sc.criterion(
        "SC12d.3",
        body["decision"] == "REJECT" and "OUTSIDE_CHARTER" in body["reason_codes"],
        "mål i den gamla men utanför den nya ramen avvisas",
        codes=body["reason_codes"],
    )
    await sc.assert_drift("SC12d.4")
    await sc.finish()
