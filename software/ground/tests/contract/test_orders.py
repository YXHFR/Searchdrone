"""Orders through the gate (``sd_ground.services.commands``) against a real database,
the fake flight controller and the fake onboard computer.

The whole mission flow is in ``test_mission_flow.py``; these tests go through the
order paths one by one: the pilot's order buttons, repeated ids, estimates, order
cards, cancelling, what the onboard computer answers and sector progress.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))

from flow_rig import VID, H, Rig, rig  # pyright: ignore[reportMissingImports]
from script_driver import ScriptDriver  # pyright: ignore[reportMissingImports]

from sd_ground.services import approvals as approvals_service
from sd_link.session import Expired
from sd_safety.targets import distance_m
from sd_schemas.base import new_id
from sd_schemas.executor import ExecutorState
from sd_schemas.gate import GateDecision
from sd_schemas.link import CommandCancel, Envelope, SkillCommand
from sd_schemas.skills import SKILL_REGISTRY

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]


@pytest.fixture
async def r(migrated_db: Db, tmp_path: Path) -> AsyncIterator[Rig]:
    async with rig(migrated_db.name, migrated_db.app_password, tmp_path) as built:
        yield built


def driver(r: Rig, mid: str) -> ScriptDriver:
    return ScriptDriver(r.commander, mid, {"p1": r.at(60.0, 40.0), "box": r.square(150.0)})


def no_rate_limit(r: Rig) -> None:
    """Forget the AI's earlier orders: the gate's 3 s rate limit has its own tests."""
    r.flow.rates.clear()


async def order(
    r: Rig, d: ScriptDriver, skill: str, params: dict[str, Any] | None = None, want: str = "ALLOW"
) -> dict[str, Any]:
    """One AI order (rate limit cleared); the gate's answer must be ``want``."""
    no_rate_limit(r)
    res = await d.order(skill, params)
    assert res.status_code == 201, res.text
    out = res.json()
    assert out["decision"] == want, out
    return out


async def settle(r: Rig, state: ExecutorState) -> None:
    """Put the fake executor in ``state`` and wait until the ground holds the snapshot
    that carries that transition (an AI order's ``based_on`` is then current)."""
    r.onboard.set_state(state)

    def seen() -> bool:
        snap, _ = r.flow.snapshot(VID)
        return (
            snap is not None
            and snap.executor_state is state
            and snap.state_seq >= r.onboard.last_transition_seq
        )

    await r.until(seen, 5, f"snapshot {state} seen")


async def take_off(r: Rig, mid: str, d: ScriptDriver) -> None:
    res = await r.clearance(mid)
    assert res.status_code == 200, res.text
    no_rate_limit(r)
    res = await d.order("takeoff", {"alt_m": 40})
    assert res.status_code == 201 and res.json()["decision"] == "ALLOW", res.text
    await r.wait_status(mid, "ACTIVE")
    r.fly(40.0)
    await r.until(lambda: r.flow.in_flight(VID), 5, "in the air for the ground")
    await r.until(lambda: r.flow.seen_flying(VID), 10, "flight seen by the flow")
    await r.until(
        lambda: r.core.vehicle.telemetry.landed_state == "in_air", 5, "in the air for the gate"
    )
    await settle(r, ExecutorState.TRANSIT)


async def commands(r: Rig, mid: str) -> dict[str, dict[str, Any]]:
    res = await r.panel.get(f"/api/v1/missions/{mid}/commands")
    assert res.status_code == 200, res.text
    return {c["id"]: c for c in res.json()}


async def status_is(r: Rig, mid: str, command_id: str, want: str) -> bool:
    return (await commands(r, mid))[command_id]["status"] == want


async def wait_order_status(r: Rig, mid: str, command_id: str, want: str, s: float = 5) -> None:
    await r.until(lambda: status_is(r, mid, command_id, want), s, f"order {want}")


async def approve_each(r: Rig, mid: str) -> None:
    """Mode A. The new charter version is installed onboard (the fake goes to PREFLIGHT
    on every install); in the air the executor is then put back in TRANSIT."""
    installed = len(r.onboard.charters)
    res = await r.panel.put(
        f"/api/v1/missions/{mid}/autonomy", json={"mode": "APPROVE_EACH"}, headers=H
    )
    assert res.status_code == 200, res.text
    await r.until(lambda: len(r.onboard.charters) > installed, 10, "charter installed")
    if r.flow.in_flight(VID):
        await settle(r, ExecutorState.TRANSIT)


async def pending_cards(r: Rig, mid: str) -> list[dict[str, Any]]:
    res = await r.panel.get(f"/api/v1/missions/{mid}/approvals?status=pending")
    assert res.status_code == 200, res.text
    return res.json()


def silent_onboard(r: Rig, *kinds: type) -> list[Any]:
    """The fake onboard computer takes messages of ``kinds`` but answers nothing; the
    test sends the Acks itself. Returns the list the messages land in."""
    got: list[Any] = []
    handle = r.onboard._handle  # pyright: ignore[reportPrivateUsage]

    async def quiet(session: Any, inbound: Any) -> None:
        if isinstance(inbound.body, kinds):
            got.append(inbound.body)
            return
        await handle(session, inbound)

    r.onboard._handle = quiet  # pyright: ignore[reportPrivateUsage]
    return got


async def onboard_ack(r: Rig, command_id: str, stage: str, **kw: Any) -> None:
    session = r.onboard.session
    assert session is not None
    await session.ack(uuid.UUID(command_id), stage, **kw)


def record_link(r: Rig, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every message the ground sends over SD-link, in order."""
    sent: list[Any] = []
    original = r.link.send_message

    async def send(vehicle_id: str, message: Any, **kw: Any) -> Any:
        sent.append(message)
        return await original(vehicle_id, message, **kw)

    monkeypatch.setattr(r.link, "send_message", send)
    return sent


# --- the pilot's order buttons and repeated ids ----------------------------------------


async def test_pilot_takeoff_button_uses_the_active_charter_version(r: Rig) -> None:
    mid = await r.ready_mission()
    assert (await r.clearance(mid)).status_code == 200
    res = await r.panel.post(
        f"/api/v1/missions/{mid}/commands",
        json={"client_command_id": str(new_id()), "skill": "takeoff", "params": {"alt_m": 30}},
        headers=H,
    )
    assert res.status_code == 201, res.text
    out = res.json()
    assert out["gate"]["decision"] == "ALLOW", out
    cmd = out["command"]
    assert cmd["issuer_kind"] == "operator" and cmd["issuer_ref"] == "operator:ci-harness"
    assert cmd["charter_version"] == 1 and cmd["based_on"] is None
    await r.until(lambda: r.onboard.orders, 5, "order onboard")
    sent = r.onboard.orders[-1]
    assert sent.skill == "takeoff" and sent.charter_version == 1
    assert (await r.mission(mid))["clearance"] is None  # the take-off used it up
    await r.wait_status(mid, "ACTIVE")


async def test_same_command_id_gets_the_first_answer(r: Rig) -> None:
    mid = await r.ready_mission()
    d = driver(r, mid)
    cid = str(new_id())
    first = await d.order("report", {"text_sv": "En gång."}, command_id=cid)
    assert first.status_code == 201, first.text
    no_rate_limit(r)
    again = await d.order("report", {"text_sv": "En gång."}, command_id=cid)
    assert again.status_code == 200, again.text
    keys = ("command_id", "decision", "reason_codes", "status")
    assert {k: again.json()[k] for k in keys} == {k: first.json()[k] for k in keys}
    chat = (await r.panel.get(f"/api/v1/missions/{mid}/chat")).json()["items"]
    assert [c["text"] for c in chat if c["role"] == "brain"] == ["En gång."]  # said once
    lat, lon = r.at(10, 10)
    other = await d.order("mark_location", {"lat": lat, "lon": lon}, command_id=cid)
    assert other.status_code == 409 and other.json()["code"] == "ID_CONFLICT", other.text


async def test_estimate_creates_no_order(r: Rig) -> None:
    mid = await r.ready_mission()
    d = driver(r, mid)
    situation = await d.situation()
    lat, lon = r.at(0.0, 120.0)
    body = {
        "command_id": str(new_id()),
        "skill": "goto",
        "params": {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 6},
        "rationale_sv": "Hur långt?",
        "in_reply_to": None,
        "backend": "scripted",
        "ai_config_hash": "script-driver",
        "based_on": {
            "state_epoch": situation["snapshot_state_epoch"],
            "state_seq": situation["snapshot_state_seq"],
        },
        "charter_version": situation["charter"]["version"],
    }
    res = await r.commander.post(f"/commander/v1/missions/{mid}/estimate", json=body)
    assert res.status_code == 200, res.text
    out = res.json()
    v = situation["vehicle"]
    want = distance_m(v["lat"], v["lon"], lat, lon) / 6
    assert out["duration_s"] == pytest.approx(want, rel=0.05)
    assert out["gate"]["decision"] == "REJECT"  # a goto on the ground; still estimated
    body = {**body, "command_id": str(new_id()), "skill": "report", "params": {"text_sv": "x"}}
    res = await r.commander.post(f"/commander/v1/missions/{mid}/estimate", json=body)
    assert res.status_code == 200 and res.json()["duration_s"] is None, res.text
    assert await commands(r, mid) == {}


# --- order cards -----------------------------------------------------------------------


async def test_cards_name_the_step_and_the_order_goes_after_the_click(r: Rig) -> None:
    mid = await r.ready_mission()
    d = driver(r, mid)
    await take_off(r, mid, d)
    await approve_each(r, mid)
    pending = "PENDING_APPROVAL"
    hold = await order(r, d, "hold", {}, pending)
    (card,) = await pending_cards(r, mid)
    assert card["subject_id"] == hold["command_id"] and card["summary_sv"] == "Stå still"
    rej = await r.panel.post(
        f"/api/v1/approvals/{hold['approval_id']}/decision",
        json={"decision": "reject", "subject_version": 1},
        headers=H,
    )
    assert rej.status_code == 200, rej.text

    lat, lon = r.at(60.0, 40.0)
    goto = await order(r, d, "goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 8}, pending)
    (card,) = await pending_cards(r, mid)
    assert card["summary_sv"] == "Flyg till punkten på 40 m, 8 m/s"
    ok = await r.approve(goto["approval_id"])
    assert ok.status_code == 200 and ok.json()["status"] == "approved", ok.text
    await r.until(lambda: [o for o in r.onboard.orders if o.skill == "goto"], 5, "goto")
    sent = next(o for o in r.onboard.orders if o.skill == "goto")
    assert sent.approval is not None and str(sent.approval.approval_id) == goto["approval_id"]
    assert [o.skill for o in r.onboard.orders].count("hold") == 0  # the rejected one


async def test_takeoff_card_uses_a_clearance_given_after_the_order(r: Rig) -> None:
    mid = await r.ready_mission()
    await approve_each(r, mid)
    d = driver(r, mid)
    out = (await d.order("takeoff", {"alt_m": 40})).json()
    assert out["decision"] == "PENDING_APPROVAL", out
    card = next(c for c in await pending_cards(r, mid) if c["subject_id"] == out["command_id"])
    assert "clear_around_drone" in card["required_confirmations"]
    refused = await r.approve(out["approval_id"])  # no clearance, box not ticked
    assert refused.status_code == 409 and refused.json()["code"] == "CLEARANCE_REQUIRED"
    assert r.onboard.orders == []
    assert (await r.clearance(mid)).status_code == 200  # confirmed on its own
    ok = await r.approve(out["approval_id"])  # box still not ticked: the clearance counts
    assert ok.status_code == 200 and ok.json()["status"] == "approved", ok.text
    async with r.flow.repo.engine.connect() as conn:
        ticked = (
            await conn.execute(
                text("SELECT confirmations FROM approvals WHERE id = :id"),
                {"id": out["approval_id"]},
            )
        ).scalar_one()
    assert "clear_around_drone" in ticked  # recorded as given by the clearance
    await r.until(lambda: r.onboard.orders, 5, "take-off onboard")
    assert (await r.mission(mid))["clearance"] is None  # consumed by the take-off


async def test_gate_says_no_at_the_click_after_the_state_changed(r: Rig) -> None:
    mid = await r.ready_mission()
    d = driver(r, mid)
    await take_off(r, mid, d)
    await approve_each(r, mid)
    lat, lon = r.at(30.0, 30.0)
    out = await order(r, d, "goto", {"lat": lat, "lon": lon, "alt_m": 40}, "PENDING_APPROVAL")
    r.onboard.set_state(ExecutorState.HOLD, hold_reason="ai_hold")  # after the AI looked
    await r.wait_snapshot(ExecutorState.HOLD)
    res = await r.approve(out["approval_id"])
    assert res.status_code == 200, res.text
    assert res.json()["status"] == "rejected" and res.json()["reason"] == "GATE_ON_APPROVE"
    cmd = (await commands(r, mid))[out["command_id"]]
    assert (cmd["status"], cmd["status_reason"]) == ("rejected_by_gate", "GATE_ON_APPROVE")
    assert [o for o in r.onboard.orders if o.skill == "goto"] == []


async def test_card_that_runs_out_while_the_clearance_is_written(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The click ticks "clear around the drone"; the card's time runs out while that
    clearance is recorded. The second check, under the lock, stops the order."""

    async def no_expiry(_flow: Any) -> None:
        return None

    monkeypatch.setattr(approvals_service, "expire", no_expiry)
    mid = await r.ready_mission()
    await approve_each(r, mid)
    d = driver(r, mid)
    out = (await d.order("takeoff", {"alt_m": 40})).json()
    assert out["decision"] == "PENDING_APPROVAL", out
    original = r.link.send_message

    async def slow_send(vehicle_id: str, message: Any, **kw: Any) -> Any:
        r.clock.skew_s = 61  # the clearance goes out a minute "later"
        return await original(vehicle_id, message, **kw)

    monkeypatch.setattr(r.link, "send_message", slow_send)
    res = await r.approve(out["approval_id"], confirmations=["clear_around_drone"])
    r.clock.skew_s = 0
    assert res.status_code == 409 and res.json()["code"] == "APPROVAL_TIMEOUT", res.text
    await asyncio.sleep(0.3)
    assert r.onboard.orders == []
    assert await status_is(r, mid, out["command_id"], "pending_approval")


async def test_order_no_longer_waiting_cannot_be_approved(r: Rig) -> None:
    """The card is still open but its order has left ``pending_approval`` (written by
    another process): the click sends nothing."""
    mid = await r.ready_mission()
    await approve_each(r, mid)
    d = driver(r, mid)
    out = await order(r, d, "takeoff", {"alt_m": 40}, "PENDING_APPROVAL")
    async with r.flow.repo.engine.begin() as conn:
        await conn.execute(
            text("UPDATE commands SET status = 'cancelled' WHERE id = :id"),
            {"id": out["command_id"]},
        )
    res = await r.approve(out["approval_id"])
    assert res.status_code == 409 and res.json()["code"] == "SUPERSEDED", res.text
    assert [c for c in await pending_cards(r, mid) if c["id"] == out["approval_id"]]


# --- surveys: the sector's polygon and its progress ------------------------------------


async def ready_with_sector(r: Rig) -> tuple[str, dict[str, Any]]:
    """READY with a charter that has one sector well inside its area."""
    mid = await r.open_mission()
    sector = {"sector_id": "A", "name": "Ängen", "area": r.square(40.0, east=50.0), "order_idx": 0}
    res = await r.panel.post(
        f"/api/v1/missions/{mid}/charters",
        json={"area": r.square(150.0), "reason_sv": "Test", "sectors": [sector]},
        headers=H,
    )
    assert res.status_code == 201, res.text
    res = await r.approve(str(res.json()["approval_id"]), res.json()["version"])
    assert res.status_code == 200, res.text
    await r.wait_status(mid, "READY", 30)
    await r.wait_snapshot(ExecutorState.PREFLIGHT)
    return mid, sector


async def test_sector_survey_gets_the_polygon_and_moves_the_sector(r: Rig) -> None:
    mid, sector = await ready_with_sector(r)
    d = driver(r, mid)
    await take_off(r, mid, d)
    r.onboard.order_outcome = "running"
    out = await order(r, d, "survey_sector", {"sector_id": "A", "alt_m": 40})
    await r.until(lambda: [o for o in r.onboard.orders if o.skill == "survey_sector"], 5, "survey")
    sent = next(o for o in r.onboard.orders if o.skill == "survey_sector")
    assert "sector_id" not in sent.params
    assert sent.params["polygon"] == sector["area"]
    await wait_order_status(r, mid, out["command_id"], "running")

    async def progress() -> dict[str, Any]:
        rows = (await d.situation())["sectors"]
        return next(s for s in rows if s["sector_id"] == sector["sector_id"])

    assert (await progress())["status"] == "active"
    r.onboard.set_state(ExecutorState.SURVEY, progress_pct=40.0)

    async def at_40() -> bool:
        return (await progress())["progress"] == pytest.approx(0.4)

    await r.until(at_40, 5, "sector at 40 %")
    active = (await r.mission(mid))["active_command"]
    assert active is not None and active["id"] == out["command_id"]
    session = r.onboard.session
    assert session is not None
    await session.ack(sent.command_id, "done")
    await wait_order_status(r, mid, out["command_id"], "done")
    done = await progress()
    assert (done["status"], done["progress"]) == ("done", 1.0)


async def test_progress_of_other_orders_moves_no_sector(r: Rig) -> None:
    mid = await r.ready_mission()
    d = driver(r, mid)
    await take_off(r, mid, d)
    r.onboard.order_outcome = "running"
    no_rate_limit(r)
    area = r.square(40.0, east=40.0)
    poly = await order(r, d, "survey_sector", {"polygon": area, "alt_m": 40})
    async with r.flow.repo.engine.connect() as conn:
        target = (
            await conn.execute(
                text("SELECT ST_AsGeoJSON(target) FROM commands WHERE id = :id"),
                {"id": poly["command_id"]},
            )
        ).scalar_one()
    assert target is not None and '"Polygon"' in target  # the map shows the area
    await wait_order_status(r, mid, poly["command_id"], "running")
    r.onboard.set_state(ExecutorState.SURVEY, progress_pct=50.0)
    await asyncio.sleep(0.6)
    no_rate_limit(r)
    lat, lon = r.at(20.0, 20.0)
    goto = await order(r, d, "goto", {"lat": lat, "lon": lon, "alt_m": 40})
    assert goto["decision"] == "ALLOW", goto
    await wait_order_status(r, mid, goto["command_id"], "running")
    r.onboard.set_state(ExecutorState.TRANSIT, progress_pct=70.0)
    await asyncio.sleep(0.6)
    sectors = (await d.situation())["sectors"]
    assert all(s["status"] == "pending" and not s["progress"] for s in sectors), sectors


# --- cancelling ------------------------------------------------------------------------


async def test_cancel_running_waiting_and_finished_orders(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    mid = await r.ready_mission()
    d = driver(r, mid)
    await take_off(r, mid, d)
    sent = record_link(r, monkeypatch)
    r.onboard.order_outcome = "running"
    no_rate_limit(r)
    lat, lon = r.at(50.0, 0.0)
    out = await order(r, d, "goto", {"lat": lat, "lon": lon, "alt_m": 40})
    await wait_order_status(r, mid, out["command_id"], "running")
    path = f"/commander/v1/missions/{mid}/commands/{out['command_id']}/cancel"
    res = await r.commander.post(path)
    assert res.status_code == 202 and res.json()["status"] == "cancelled", res.text
    cancels = [m for m in sent if isinstance(m, CommandCancel)]
    assert [str(m.command_id) for m in cancels] == [out["command_id"]]
    again = await r.commander.post(path)
    assert again.status_code == 409 and again.json()["code"] == "NOT_CANCELLABLE"

    await approve_each(r, mid)
    no_rate_limit(r)
    waiting = (await d.order("hold")).json()
    assert waiting["decision"] == "PENDING_APPROVAL", waiting
    res = await r.panel.post(f"/api/v1/commands/{waiting['command_id']}/cancel", headers=H)
    assert res.status_code == 202 and res.json()["status"] == "cancelled", res.text
    card = (await r.panel.get(f"/api/v1/missions/{mid}/approvals")).json()
    mine = next(c for c in card if c["id"] == waiting["approval_id"])
    assert (mine["status"], mine["reason"]) == ("superseded", "CANCELLED")
    assert len([m for m in sent if isinstance(m, CommandCancel)]) == 1  # never sent onboard


async def test_ai_cannot_cancel_the_pilots_order(r: Rig) -> None:
    mid = await r.ready_mission()
    assert (await r.clearance(mid)).status_code == 200
    r.onboard.order_outcome = "running"
    res = await r.panel.post(
        f"/api/v1/missions/{mid}/commands",
        json={"client_command_id": str(new_id()), "skill": "takeoff", "params": {"alt_m": 30}},
        headers=H,
    )
    cid = res.json()["command"]["id"]
    await wait_order_status(r, mid, cid, "running")
    res = await r.commander.post(f"/commander/v1/missions/{mid}/commands/{cid}/cancel")
    assert res.status_code == 404 and res.json()["code"] == "COMMAND_NOT_FOUND"
    assert await status_is(r, mid, cid, "running")


async def test_stop_cancels_the_running_order(r: Rig) -> None:
    mid = await r.ready_mission()
    d = driver(r, mid)
    await take_off(r, mid, d)
    r.onboard.order_outcome = "running"
    no_rate_limit(r)
    lat, lon = r.at(50.0, 0.0)
    out = await order(r, d, "goto", {"lat": lat, "lon": lon, "alt_m": 40})
    await wait_order_status(r, mid, out["command_id"], "running")
    stop = await r.panel.post(
        f"/api/v1/vehicles/{VID}/actions/stop", json={"client_action_id": str(new_id())}, headers=H
    )
    assert stop.status_code == 202, stop.text
    await wait_order_status(r, mid, out["command_id"], "cancelled")
    cmd = (await commands(r, mid))[out["command_id"]]
    assert cmd["status_reason"] == "OPERATOR_STOP"


# --- what the onboard computer answers -------------------------------------------------


async def test_late_and_out_of_order_acks_change_nothing(r: Rig) -> None:
    mid = await r.ready_mission()
    d = driver(r, mid)
    await take_off(r, mid, d)
    got = silent_onboard(r, SkillCommand)
    no_rate_limit(r)
    lat, lon = r.at(40.0, 0.0)
    out = await order(r, d, "goto", {"lat": lat, "lon": lon, "alt_m": 40})
    cid = out["command_id"]
    await r.until(lambda: got, 5, "goto onboard")
    await onboard_ack(r, cid, "running")
    await wait_order_status(r, mid, cid, "running")
    await onboard_ack(r, cid, "accepted")  # older than what we have
    await asyncio.sleep(0.3)
    assert await status_is(r, mid, cid, "running")
    await onboard_ack(r, cid, "done")
    await wait_order_status(r, mid, cid, "done")
    await onboard_ack(r, cid, "failed")  # after the end
    await asyncio.sleep(0.3)
    assert await status_is(r, mid, cid, "done")
    assert await r.status(mid) == "ACTIVE"


async def test_onboard_refusal_of_an_allowed_order_raises_a_gate_mismatch(r: Rig) -> None:
    mid = await r.ready_mission()
    d = driver(r, mid)
    await take_off(r, mid, d)
    got = silent_onboard(r, SkillCommand)
    no_rate_limit(r)
    lat, lon = r.at(40.0, 0.0)
    out = await order(r, d, "goto", {"lat": lat, "lon": lon, "alt_m": 40})
    cid = out["command_id"]
    await r.until(lambda: got, 5, "goto onboard")
    onboard_gate = GateDecision(
        decision="REJECT",
        reason_codes=["OUTSIDE_AREA"],
        rule_ids=["R-AREA"],
        message_sv="Utanför området.",
        details={},
        side="onboard",
    )
    await onboard_ack(r, cid, "rejected", reason_codes=["OUTSIDE_AREA"], gate=onboard_gate)
    await wait_order_status(r, mid, cid, "rejected_onboard")
    cmd = (await commands(r, mid))[cid]
    assert cmd["status_reason"] == "OUTSIDE_AREA"
    alerts = [e.data for e in _bus(r) if e.type == "alert"]
    assert [(a["code"], a["command_id"]) for a in alerts] == [("GATE_MISMATCH", cid)]
    async with r.flow.repo.engine.connect() as conn:
        phases = (
            await conn.execute(
                text("SELECT phase FROM gate_decisions WHERE command_id = :c ORDER BY created_at"),
                {"c": cid},
            )
        ).scalars()
        assert list(phases) == ["submit", "onboard"]


def _bus(r: Rig) -> list[Any]:
    return list(r.g.bus.replay(r.g.bus.server_epoch, 0) or [])


async def test_takeoff_refused_onboard_keeps_the_mission_ready(r: Rig) -> None:
    mid = await r.ready_mission()
    got = silent_onboard(r, SkillCommand)
    d = driver(r, mid)
    assert (await r.clearance(mid)).status_code == 200
    out = (await d.order("takeoff", {"alt_m": 40})).json()
    assert out["decision"] == "ALLOW", out
    await r.until(lambda: got, 5, "take-off onboard")
    await onboard_ack(r, out["command_id"], "rejected", reason_codes=["ONB_STALE_ORDER"])
    await wait_order_status(r, mid, out["command_id"], "rejected_onboard")
    await asyncio.sleep(0.3)
    assert await r.status(mid) == "READY"


async def test_takeoff_that_fails_after_acceptance_lands_the_mission(r: Rig) -> None:
    mid = await r.ready_mission()
    got = silent_onboard(r, SkillCommand)
    d = driver(r, mid)
    assert (await r.clearance(mid)).status_code == 200
    out = (await d.order("takeoff", {"alt_m": 40})).json()
    await r.until(lambda: got, 5, "take-off onboard")
    await onboard_ack(r, out["command_id"], "accepted")
    await r.wait_status(mid, "ACTIVE")
    await onboard_ack(r, out["command_id"], "failed", reason_codes=["ONB_TAKEOFF_FAILED"])
    await wait_order_status(r, mid, out["command_id"], "failed")
    await r.wait_status(mid, "LANDED")  # it never left the ground


def _expired(r: Rig, mid: str, message: CommandCancel | SkillCommand) -> Expired:
    """What SD-link reports for a reliable message that got no Ack in time."""
    envelope = Envelope(
        msg_id=new_id(),
        kind=type(message).__name__,
        sender="ground:main",
        vehicle_id=VID,
        mission_id=uuid.UUID(mid),
        seq=1,
        sent_utc=r.clock.utc_now(),
        ttl_ms=None,
        reliable=True,
        corr_id=None,
        body=message.model_dump(mode="json"),
    )
    return Expired(envelope, "LINK_NO_ACK")


async def test_expired_order_is_marked_but_an_expired_cancel_is_not(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``SkillCommand`` without an Ack expires its order, but only while it is still
    ``dispatched``; an expired message of another kind (a ``CommandCancel`` naming the
    same order) leaves the order alone."""
    mid = await r.ready_mission()
    original = r.link.send_message
    got: list[SkillCommand] = []

    async def lost_on_the_way(vehicle_id: str, message: Any, **kw: Any) -> Any:
        if isinstance(message, SkillCommand):
            got.append(message)
            return new_id()  # queued, but it never arrives
        return await original(vehicle_id, message, **kw)

    monkeypatch.setattr(r.link, "send_message", lost_on_the_way)
    d = driver(r, mid)
    assert (await r.clearance(mid)).status_code == 200
    out = await order(r, d, "takeoff", {"alt_m": 40})
    assert got and r.onboard.orders == []
    sent = got[0]
    cancel = CommandCancel(command_id=sent.command_id)
    await r.flow.on_expired(VID, _expired(r, mid, cancel))
    assert await status_is(r, mid, out["command_id"], "dispatched")
    await r.flow.on_expired(VID, _expired(r, mid, sent))
    cmd = (await commands(r, mid))[out["command_id"]]
    assert (cmd["status"], cmd["status_reason"]) == ("expired", "LINK_NO_ACK")

    second = await order(r, d, "report", {"text_sv": "klar"})
    assert await status_is(r, mid, second["command_id"], "done")
    late = SkillCommand(
        command_id=uuid.UUID(second["command_id"]),
        skill="report",
        params={"text_sv": "klar"},
        issuer="brain:ground",
        rationale=None,
        charter_version=1,
        based_on=None,
        approval=None,
    )
    await r.flow.on_expired(VID, _expired(r, mid, late))
    assert await status_is(r, mid, second["command_id"], "done")  # only from dispatched


# --- without SD-link -------------------------------------------------------------------


async def test_without_sdlink_the_clearance_is_kept_on_the_ground(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ground station running without SD-link (``flow.link`` unset): the clearance is
    recorded but not sent, and the gate refuses the take-off."""
    mid = await r.ready_mission()
    clearances_before = len(r.onboard.clearances)
    monkeypatch.setattr(r.flow, "link", None)
    res = await r.clearance(mid)
    assert res.status_code == 200, res.text
    assert (await r.mission(mid))["clearance"] is not None
    await asyncio.sleep(0.3)
    assert len(r.onboard.clearances) == clearances_before
    d = driver(r, mid)
    out = await order(r, d, "takeoff", {"alt_m": 40}, "REJECT")  # the gate needs SD-link
    assert "LINK_DOWN" in out["reason_codes"]
    assert r.onboard.orders == []


async def test_cancel_without_sdlink_sends_nothing(r: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    mid = await r.ready_mission()
    d = driver(r, mid)
    await take_off(r, mid, d)
    r.onboard.order_outcome = "running"
    lat, lon = r.at(50.0, 0.0)
    out = await order(r, d, "goto", {"lat": lat, "lon": lon, "alt_m": 40})
    await wait_order_status(r, mid, out["command_id"], "running")
    sent = record_link(r, monkeypatch)
    monkeypatch.setattr(r.flow, "link", None)
    res = await r.commander.post(
        f"/commander/v1/missions/{mid}/commands/{out['command_id']}/cancel"
    )
    assert res.status_code == 202 and res.json()["status"] == "cancelled", res.text
    assert sent == []


async def test_order_the_link_cannot_queue_expires(r: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    mid = await r.ready_mission()
    original = r.link.send_message

    async def busy(vehicle_id: str, message: Any, **kw: Any) -> Any:
        if isinstance(message, SkillCommand):
            return None  # e.g. LinkBusyError: the queue towards the vehicle is full
        return await original(vehicle_id, message, **kw)

    monkeypatch.setattr(r.link, "send_message", busy)
    assert (await r.clearance(mid)).status_code == 200
    d = driver(r, mid)
    out = await order(r, d, "takeoff", {"alt_m": 40})
    cmd = (await commands(r, mid))[out["command_id"]]
    assert (cmd["status"], cmd["status_reason"]) == ("expired", "LINK_DOWN")
    assert await r.status(mid) == "READY"


# --- charter proposals as orders -------------------------------------------------------


async def test_charter_proposal_orders_that_cannot_be_offered(r: Rig) -> None:
    mid = await r.ready_mission()
    d = driver(r, mid)
    far = await order(
        r,
        d,
        "propose_charter_change",
        {"base_version": 1, "area": r.square(100, east=900), "reason_sv": "Långt bort"},
        "REJECT",
    )
    assert {"HOME_OUTSIDE_AREA", "AREA_BEYOND_RANGE"} <= set(far["reason_codes"])
    assert far["status"] == "rejected_by_gate"
    bad = await order(r, d, "propose_charter_change", {"area": "inte en polygon"}, "REJECT")
    assert bad["reason_codes"] == ["SCHEMA_INVALID"]
    off = await r.panel.put(f"/api/v1/missions/{mid}/ai", json={"enabled": False}, headers=H)
    assert off.status_code == 200, off.text
    disabled = await order(
        r,
        d,
        "propose_charter_change",
        {"base_version": 1, "area": r.square(100), "reason_sv": "Mindre"},
        "REJECT",
    )
    assert disabled["reason_codes"] == ["AI_DISABLED"]
    rows = await commands(r, mid)
    assert {rows[o["command_id"]]["status_reason"] for o in (far, bad, disabled)} == {
        "CHARTER_INVALID"
    }
    assert [c for c in await pending_cards(r, mid) if c["kind"] == "charter"] == []


# --- the database loses or changes a row under the order -------------------------------


def _calls_then_none(real: Any, keep: int) -> Any:
    """``real`` for the first ``keep`` calls, then ``None`` (the row is gone)."""
    calls = {"n": 0}

    async def fake(*args: Any, **kw: Any) -> Any:
        calls["n"] += 1
        return await real(*args, **kw) if calls["n"] <= keep else None

    return fake


async def test_order_row_gone_after_the_write_sends_nothing(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    mid = await r.ready_mission()
    d = driver(r, mid)
    monkeypatch.setattr(r.flow.repo, "command", _calls_then_none(r.flow.repo.command, 0))
    with pytest.raises(RuntimeError, match="vanished"):
        await d.order("report", {"text_sv": "x"})
    no_rate_limit(r)
    with pytest.raises(RuntimeError, match="vanished"):
        await d.order(
            "propose_charter_change",
            {"base_version": 1, "area": r.square(100), "reason_sv": "Mindre"},
        )
    chat = (await r.panel.get(f"/api/v1/missions/{mid}/chat")).json()["items"]
    assert [c for c in chat if c["role"] == "brain"] == []  # the report was not carried out


async def test_repeated_id_without_a_stored_decision_is_a_reject(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    mid = await r.ready_mission()
    d = driver(r, mid)
    cid = str(new_id())
    first = await d.order("report", {"text_sv": "x"}, command_id=cid)
    assert first.json()["decision"] == "ALLOW", first.text

    async def no_gate(_command_id: str) -> None:
        return None

    monkeypatch.setattr(r.flow.repo, "first_gate", no_gate)
    again = await d.order("report", {"text_sv": "x"}, command_id=cid)
    assert again.status_code == 200, again.text
    assert (again.json()["decision"], again.json()["reason_codes"]) == ("REJECT", [])


async def test_card_row_gone_after_the_write_is_not_announced(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    mid = await r.ready_mission()
    await approve_each(r, mid)
    d = driver(r, mid)

    async def gone(_approval_id: str) -> None:
        return None

    monkeypatch.setattr(r.flow.repo, "approval", gone)
    seen = len(_bus(r))
    out = await order(r, d, "takeoff", {"alt_m": 40}, "PENDING_APPROVAL")
    assert out["approval_id"] is not None
    types = [e.type for e in _bus(r)[seen:]]
    assert "command.updated" in types and "approval.requested" not in types


async def test_click_that_loses_the_race_for_the_card_sends_nothing(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The card was answered elsewhere between the check and the write."""
    mid = await r.ready_mission()
    await approve_each(r, mid)
    d = driver(r, mid)
    assert (await r.clearance(mid)).status_code == 200
    out = await order(r, d, "takeoff", {"alt_m": 40}, "PENDING_APPROVAL")

    async def lost(*_args: Any, **_kw: Any) -> bool:
        return False

    monkeypatch.setattr(r.flow.repo, "decide_approval", lost)
    res = await r.approve(out["approval_id"])
    assert res.status_code == 409 and res.json()["code"] == "SUPERSEDED", res.text
    await asyncio.sleep(0.3)
    assert r.onboard.orders == []
    assert (await r.mission(mid))["clearance"] is not None  # not consumed


async def _card_gone_after_decision(r: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Once the decision is written, the card row reads back as missing."""
    decided = {"yes": False}
    decide, read = r.flow.repo.decide_approval, r.flow.repo.approval

    async def decide_then_gone(*args: Any, **kw: Any) -> bool:
        ok = await decide(*args, **kw)
        decided["yes"] = True
        return ok

    async def approval(approval_id: str) -> Any:
        return None if decided["yes"] else await read(approval_id)

    monkeypatch.setattr(r.flow.repo, "decide_approval", decide_then_gone)
    monkeypatch.setattr(r.flow.repo, "approval", approval)


async def test_card_row_gone_after_the_decision(r: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    mid = await r.ready_mission()
    await approve_each(r, mid)
    d = driver(r, mid)
    assert (await r.clearance(mid)).status_code == 200
    first = await order(r, d, "takeoff", {"alt_m": 40}, "PENDING_APPROVAL")
    await _card_gone_after_decision(r, monkeypatch)
    with pytest.raises(RuntimeError, match="vanished"):
        await r.panel.post(
            f"/api/v1/approvals/{first['approval_id']}/decision",
            json={"decision": "reject", "subject_version": 1},
            headers=H,
        )
    assert await status_is(r, mid, first["command_id"], "rejected_by_operator")
    monkeypatch.undo()
    second = await order(r, d, "takeoff", {"alt_m": 40}, "PENDING_APPROVAL")
    await _card_gone_after_decision(r, monkeypatch)
    with pytest.raises(RuntimeError, match="vanished"):
        await r.approve(second["approval_id"])
    await asyncio.sleep(0.3)
    assert r.onboard.orders == []  # approved in the database, but never sent


async def test_approved_order_whose_row_reads_back_missing_still_goes(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    mid = await r.ready_mission()
    await approve_each(r, mid)
    d = driver(r, mid)
    assert (await r.clearance(mid)).status_code == 200
    out = await order(r, d, "takeoff", {"alt_m": 40}, "PENDING_APPROVAL")
    seen = len(_bus(r))
    monkeypatch.setattr(r.flow.repo, "command", _calls_then_none(r.flow.repo.command, 1))
    res = await r.approve(out["approval_id"])
    assert res.status_code == 200 and res.json()["status"] == "approved", res.text
    await r.until(lambda: r.onboard.orders, 5, "take-off onboard")
    assert r.onboard.orders[-1].approval is not None
    updates = [e for e in _bus(r)[seen:] if e.type == "command.updated"]
    assert updates == []  # nothing to announce without the row


async def test_status_change_whose_row_reads_back_missing_is_not_published(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    mid = await r.ready_mission()
    d = driver(r, mid)
    await take_off(r, mid, d)
    r.onboard.order_outcome = "running"
    lat, lon = r.at(50.0, 0.0)
    out = await order(r, d, "goto", {"lat": lat, "lon": lon, "alt_m": 40})
    await wait_order_status(r, mid, out["command_id"], "running")
    seen = len(_bus(r))
    monkeypatch.setattr(r.flow.repo, "command", _calls_then_none(r.flow.repo.command, 1))
    res = await r.commander.post(
        f"/commander/v1/missions/{mid}/commands/{out['command_id']}/cancel"
    )
    assert res.status_code == 202, res.text
    assert res.json()["status"] == "running"  # the answer is the row as it was read
    monkeypatch.undo()
    assert await status_is(r, mid, out["command_id"], "cancelled")
    assert [e for e in _bus(r)[seen:] if e.type == "command.updated"] == []


async def test_ack_that_changes_no_row_moves_nothing(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    mid = await r.ready_mission()
    got = silent_onboard(r, SkillCommand)
    d = driver(r, mid)
    assert (await r.clearance(mid)).status_code == 200
    out = await order(r, d, "takeoff", {"alt_m": 40})
    await r.until(lambda: got, 5, "take-off onboard")

    async def unchanged(*_args: Any, **_kw: Any) -> None:
        return None

    monkeypatch.setattr(r.flow.repo, "set_command_status", unchanged)
    await onboard_ack(r, out["command_id"], "accepted")
    await asyncio.sleep(0.5)
    monkeypatch.undo()
    assert await r.status(mid) == "READY"  # no TAKEOFF_ACCEPTED
    status = (await commands(r, mid))[out["command_id"]]["status"]
    assert status in ("dispatched", "received")


async def test_waiting_order_without_its_card_is_still_cancelled(r: Rig) -> None:
    mid = await r.ready_mission()
    await approve_each(r, mid)
    d = driver(r, mid)
    out = await order(r, d, "takeoff", {"alt_m": 40}, "PENDING_APPROVAL")
    async with r.flow.repo.engine.begin() as conn:
        await conn.execute(
            text("UPDATE commands SET approval_id = NULL WHERE id = :id"),
            {"id": out["command_id"]},
        )
    res = await r.panel.post(f"/api/v1/commands/{out['command_id']}/cancel", headers=H)
    assert res.status_code == 202 and res.json()["status"] == "cancelled", res.text
    (card,) = [c for c in await pending_cards(r, mid) if c["id"] == out["approval_id"]]
    assert card["status"] == "pending"  # no card was named, so none was touched


async def test_survey_by_sector_fills_in_what_was_seen(r: Rig) -> None:
    """The track flown during a sector survey (at its height) becomes the sector's
    covered area; the vehicle hovers in the middle of sector A here."""
    mid, _sector = await ready_with_sector(r)
    d = driver(r, mid)
    await take_off(r, mid, d)
    r.onboard.order_outcome = "running"
    out = await order(r, d, "survey_sector", {"sector_id": "A", "alt_m": 40, "spacing_m": 20})
    await wait_order_status(r, mid, out["command_id"], "running")
    assert r.fc is not None
    r.fc.lat, r.fc.lon = r.at(50.0, 0.0)

    async def covered() -> dict[str, Any] | None:
        features = (await r.mission(mid))["sectors"]["features"]
        return next(f for f in features if f["id"] == "A")["properties"]["covered"]

    def bounds(area: dict[str, Any]) -> tuple[float, float, float, float]:
        lons = [p[0] for poly in area["coordinates"] for ring in poly for p in ring]
        lats = [p[1] for poly in area["coordinates"] for ring in poly for p in ring]
        return min(lons), min(lats), max(lons), max(lats)

    disc_sw, disc_ne = r.at(50.0 - 11.0, -11.0), r.at(50.0 + 11.0, 11.0)

    async def hover_seen() -> bool:
        area = await covered()
        if area is None:
            return False
        west, south, east, north = bounds(area)
        return west <= disc_sw[1] and east >= disc_ne[1] and south <= disc_sw[0] <= north

    # The disc of 20 / 2 + 2 m around the hover point is counted ...
    await r.until(hover_seen, 20, "hover point covered")
    seen = await covered()
    assert seen is not None and seen["type"] == "MultiPolygon"
    # ... and clipped to sector A (east 10..90 m, north -40..40 m); points flown before
    # the move may add a sliver at its edge.
    west, south, east, north = bounds(seen)
    south_west, north_east = r.at(10.0, -40.0), r.at(90.0, 40.0)
    eps = 1e-7
    assert south_west[1] - eps <= west and east <= north_east[1] + eps
    assert south_west[0] - eps <= south and north <= north_east[0] + eps
    assert south <= disc_sw[0] and north >= disc_ne[0]


# --- the order path changes between the gate's ALLOW and the dispatch -------------------


def _after_the_gate(r: Rig, monkeypatch: pytest.MonkeyPatch, change: Any) -> None:
    """Run ``change()`` once the order and its ALLOW are written, before the dispatch
    reads the charter and SD-link again."""
    add = r.flow.repo.add_command

    async def add_then_change(*args: Any, **kw: Any) -> Any:
        out = await add(*args, **kw)
        change()
        return out

    monkeypatch.setattr(r.flow.repo, "add_command", add_then_change)


async def test_charter_gone_at_dispatch_expires_the_order(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    mid = await r.ready_mission()
    assert (await r.clearance(mid)).status_code == 200
    gone = {"yes": False}
    read = r.flow.active_charter_row

    async def charter_row(mission: Any) -> Any:
        return None if gone["yes"] else await read(mission)

    monkeypatch.setattr(r.flow, "active_charter_row", charter_row)
    _after_the_gate(r, monkeypatch, lambda: gone.update(yes=True))
    out = await order(r, driver(r, mid), "takeoff", {"alt_m": 40})
    cmd = (await commands(r, mid))[out["command_id"]]
    assert (cmd["status"], cmd["status_reason"]) == ("expired", "LINK_DOWN")
    await asyncio.sleep(0.3)
    assert r.onboard.orders == []


async def test_sdlink_gone_at_dispatch_expires_the_order(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    mid = await r.ready_mission()
    assert (await r.clearance(mid)).status_code == 200
    sent = record_link(r, monkeypatch)
    _after_the_gate(r, monkeypatch, lambda: setattr(r.flow, "link", None))
    monkeypatch.setattr(r.flow, "link", r.flow.link)  # restored after the test
    out = await order(r, driver(r, mid), "takeoff", {"alt_m": 40})
    cmd = (await commands(r, mid))[out["command_id"]]
    assert (cmd["status"], cmd["status_reason"]) == ("expired", "LINK_DOWN")
    assert [m for m in sent if isinstance(m, SkillCommand)] == []
    await asyncio.sleep(0.3)
    assert r.onboard.orders == []


async def test_sector_gone_from_the_charter_at_dispatch_rejects_the_order(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    mid, _sector = await ready_with_sector(r)
    d = driver(r, mid)
    await take_off(r, mid, d)
    changed = {"yes": False}
    read = r.flow.active_charter_row

    async def charter_row(mission: Any) -> Any:
        row = await read(mission)
        if row is None or not changed["yes"]:
            return row
        return row.model_copy(update={"body": row.body.model_copy(update={"sectors": []})})

    monkeypatch.setattr(r.flow, "active_charter_row", charter_row)
    _after_the_gate(r, monkeypatch, lambda: changed.update(yes=True))
    out = await order(r, d, "survey_sector", {"sector_id": "A", "alt_m": 40})
    cmd = (await commands(r, mid))[out["command_id"]]
    assert (cmd["status"], cmd["status_reason"]) == ("rejected_by_gate", "SCHEMA_INVALID")
    await asyncio.sleep(0.3)
    assert [o for o in r.onboard.orders if o.skill == "survey_sector"] == []


async def test_approved_card_for_a_ground_skill_runs_on_the_ground(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a registry (injected for this test only) in which ``report`` needs approval,
    the click carries the report out on the ground: done, a chat line, nothing onboard."""
    registry = dict(SKILL_REGISTRY.as_mapping())
    registry["report"] = registry["report"].model_copy(update={"requires_approval_in_b": True})
    monkeypatch.setattr(r.flow, "registry", registry)
    mid = await r.ready_mission()
    d = driver(r, mid)
    out = await order(r, d, "report", {"text_sv": "Efter klicket."}, "PENDING_APPROVAL")
    assert out["reason_codes"] == ["NEEDS_APPROVAL_SKILL"] and out["status"] == "pending_approval"
    chat = (await r.panel.get(f"/api/v1/missions/{mid}/chat")).json()["items"]
    assert [c for c in chat if c["role"] == "brain"] == []  # not before the click
    (card,) = await pending_cards(r, mid)
    assert card["summary_sv"] == "Rapportera"
    ok = await r.approve(out["approval_id"])
    assert ok.status_code == 200 and ok.json()["status"] == "approved", ok.text
    cmd = (await commands(r, mid))[out["command_id"]]
    assert (cmd["status"], cmd["status_reason"]) == ("done", None)
    chat = (await r.panel.get(f"/api/v1/missions/{mid}/chat")).json()["items"]
    lines = [c for c in chat if c["role"] == "brain"]
    assert [c["text"] for c in lines] == ["Efter klicket."]
    assert lines[0]["refs"]["command_id"] == out["command_id"]
    await asyncio.sleep(0.3)
    assert r.onboard.orders == []
