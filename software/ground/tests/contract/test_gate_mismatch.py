"""What the ground station does when the onboard computer acts on its own.

* Its gate refuses an order the ground let through: the order ends ``rejected_onboard``,
  the ground logs ``gate.mismatch`` with the onboard reasons and stores the decision.
* Its battery rule flies home: the mission returns (cause battery), the pilot gets the
  ``BATTERY_RTL`` alert and the AI's orders meet the battery failsafe.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))

from flow_rig import Rig, rig  # pyright: ignore[reportMissingImports]
from script_driver import ScriptDriver  # pyright: ignore[reportMissingImports]

from sd_schemas.executor import ExecutorState
from sd_schemas.gate import GateDecision

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]


@pytest.fixture
async def r(migrated_db: Db, tmp_path: Path) -> AsyncIterator[Rig]:
    async with rig(migrated_db.name, migrated_db.app_password, tmp_path) as built:
        yield built


async def test_onboard_refusal_is_logged_as_a_mismatch(r: Rig) -> None:
    mid = await r.ready_mission()
    d = ScriptDriver(r.commander, mid, {"p1": r.at(60.0, 40.0)})
    res = await r.clearance(mid)
    assert res.status_code == 200, res.text
    await r.until(lambda: r.onboard.clearances, 5, "clearance onboard")
    assert (await d.order("takeoff", {"alt_m": 40})).json()["decision"] == "ALLOW"
    r.fly(40.0)
    await r.until(lambda: r.flow.seen_flying("sim-01"), 10, "flight seen by the flow")

    await r.wait_status(mid, "ACTIVE")
    r.onboard.set_state(ExecutorState.HOLD, hold_reason="skill_done")
    await r.wait_snapshot(ExecutorState.HOLD)
    await asyncio.sleep(3.1)  # the gate lets the AI act once per 3 s
    r.onboard.order_outcome = "rejected"
    r.onboard.reject_gate = GateDecision(
        decision="REJECT",
        reason_codes=["OUTSIDE_CHARTER"],
        rule_ids=["R-AREA"],
        message_sv="Utanför ramen.",
        details={"outside_by_m": 12.0},
        side="onboard",
    )
    out = (await d.order("goto", {"at": "p1", "alt_m": 40})).json()
    assert out["decision"] == "ALLOW", out  # the ground let it through

    async def ended() -> bool:
        cmds = (await r.panel.get(f"/api/v1/missions/{mid}/commands")).json()
        return any(c["id"] == out["command_id"] and c["status"] == "rejected_onboard" for c in cmds)

    await r.until(ended, 10, "rejected onboard")
    await r.g.flush_events()
    events = (await r.panel.get(f"/api/v1/missions/{mid}/events")).json()
    mismatch = [e for e in events if e["type"] == "gate.mismatch"]
    assert mismatch and mismatch[0]["payload"]["reason_codes"] == ["OUTSIDE_CHARTER"], events
    gates = await r.flow.repo.gate_decisions(mid, 10, command_id=out["command_id"])
    assert {g.side for g in gates} == {"ground", "onboard"}
    assert any(g.side == "onboard" and g.decision == "REJECT" for g in gates)


async def test_onboard_battery_rule_is_seen_by_the_ground(r: Rig) -> None:
    mid = await r.ready_mission()
    d = ScriptDriver(r.commander, mid, {"p1": r.at(60.0, 40.0)})
    assert (await r.clearance(mid)).status_code == 200
    await r.until(lambda: r.onboard.clearances, 5, "clearance onboard")
    assert (await d.order("takeoff", {"alt_m": 40})).json()["decision"] == "ALLOW"
    r.fly(40.0)
    await r.until(lambda: r.flow.seen_flying("sim-01"), 10, "flight seen by the flow")
    await r.wait_status(mid, "ACTIVE")
    r.onboard.set_state(ExecutorState.RTL)
    r.onboard.battery_rtl_active = True
    await r.wait_status(mid, "RETURNING")
    assert (await r.mission(mid))["mission"]["returning_cause"] == "battery"
    await r.g.flush_events()
    events = (await r.panel.get(f"/api/v1/missions/{mid}/events")).json()
    assert [e for e in events if e["type"] == "alarm.BATTERY_RTL"], events
    await asyncio.sleep(3.1)
    out = (await d.order("goto", {"at": "p1", "alt_m": 40})).json()
    assert out["decision"] == "REJECT" and "FAILSAFE_ACTIVE" in out["reason_codes"], out
