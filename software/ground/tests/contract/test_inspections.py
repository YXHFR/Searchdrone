"""TN-7: the closer look ("Titta närmare") on the ground, end to end over the real SD-link
server with the fake drone.

The pilot's "Titta närmare" goes through the gate and moves the candidate to
``inspecting`` (acceptance 22); the drone's questions and reports open the panel's boxes;
a person's decision ends the look and goes onboard (``InspectOutcome``); without one the
system puts the candidate back in the queue. The pilot picks the operation mode when
approving a charter, and the ground stamps whether a look may fly over its candidate.
"""

from __future__ import annotations

import datetime as dt
import sys
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flow_rig import VID, H, Rig, rig  # pyright: ignore[reportMissingImports]
from test_candidate_decisions import (  # pyright: ignore[reportMissingImports]
    decide,
    person,
    view,
)
from test_candidates import person as seen_at  # pyright: ignore[reportMissingImports]
from test_inspect_order import flying_with  # pyright: ignore[reportMissingImports]
from test_orders import (  # pyright: ignore[reportMissingImports]
    driver,
    no_rate_limit,
    order,
    wait_order_status,
)

from sd_ground.api.commander import to_commander
from sd_ground.services import commands, inspections
from sd_safety.floor_pack import write_floor_pack
from sd_schemas.base import LatLon, new_id
from sd_schemas.inspect import AWAIT_DECISION_S, InspectProgress
from sd_schemas.link import InspectLower, InspectOutcome, InspectStay, OnboardEvent

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]


@pytest.fixture
async def r(migrated_db: Db, tmp_path: Path) -> AsyncIterator[Rig]:
    async with rig(migrated_db.name, migrated_db.app_password, tmp_path) as built:
        yield built


async def card(c: httpx.AsyncClient, cid: str) -> dict[str, Any]:
    res = await c.get(f"/api/v1/candidates/{cid}")
    assert res.status_code == 200, res.text
    return res.json()


async def looks(r: Rig, mid: str) -> list[dict[str, Any]]:
    res = await r.panel.get(f"/api/v1/missions/{mid}/inspections")
    assert res.status_code == 200, res.text
    return res.json()


async def event(r: Rig, kind: str, **data: Any) -> None:
    """An ``inspect.*`` event from the drone, handled by the ground (the panel event
    after it shows on the bus once the ground has handled both, in order)."""
    assert r.onboard.session is not None
    before = r.bus_types().count("executor.find_hold_ended")
    await r.onboard.session.send(OnboardEvent(type=kind, severity="info", data=data))
    await r.onboard.session.send(OnboardEvent(type="executor.find_hold_ended", severity="info"))

    def handled() -> bool:
        return r.bus_types().count("executor.find_hold_ended") > before

    await r.until(handled, 5, kind)


async def started(r: Rig, n: int = 1) -> tuple[str, str, Any]:
    """In the air with ``n`` ready candidates, the pilot looks at the first:
    (mission, candidate, the order sent onboard)."""
    r.onboard.order_outcome = "running"
    dets = [seen_at(r, 20 * i, 100) for i in range(n)]
    mid, _ = await flying_with(r, *dets, mode="exercise")
    cid = str(dets[0].detection_id)
    async with person(r, "pilot", "kalle") as pilot:
        res = await decide(pilot, cid, "inspect", "röd jacka?")
        assert res.status_code == 200, res.text
        assert res.json()["card"]["state"] == "inspecting"
    await r.until(lambda: any(o.skill == "inspect_detection" for o in r.onboard.orders), 5, "sent")
    (sent,) = [o for o in r.onboard.orders if o.skill == "inspect_detection"]
    return mid, cid, sent


def done(sent: Any, **kw: Any) -> dict[str, Any]:
    return {
        "command_id": str(sent.command_id),
        "candidate_id": sent.params["candidate_id"],
        "end": "clear",
        "steps": 3,
        "lowest_above_ground_m": 22.0,
        "still_ids": [],
        "clear": True,
    } | kw


async def test_acceptance_22_the_pilot_looks_closer(r: Rig) -> None:
    mid, cid, sent = await started(r, n=2)
    # Over the candidate in exercise: the order has no height of its own.
    assert sent.params["candidate_id"] == cid
    assert sent.issuer == "operator:kalle"
    (look,) = await looks(r, mid)
    assert look["phase"] == "flying" and look["outcome"] is None
    assert look["operation_mode"] == "exercise" and look["overflight"] is True
    assert look["started_by"] == "operator:kalle" and look["seq"] == 1
    assert "inspection.updated" in r.bus_types()
    async with person(r, "pilot", "kalle") as pilot:
        d = await card(pilot, cid)
        assert [x["decision"] for x in d["decisions"]] == ["inspect"]
        # One look at a time: the second candidate is refused with the gate's code.
        other = next(
            c["candidate_id"]
            for c in (await r.panel.get(f"/api/v1/missions/{mid}/candidates")).json()["items"]
            if c["candidate_id"] != cid
        )
        res = await decide(pilot, other, "inspect")
        assert res.status_code == 409 and res.json()["code"] == "INSPECT_REFUSED", res.text
        assert {"code": "INSPECT_BUSY"} in res.json()["errors"]
        # Not twice on the same candidate either (it is no longer ready).
        res = await decide(pilot, cid, "inspect")
        assert res.status_code == 409 and res.json()["code"] == "TRANSITION_NOT_ALLOWED"
    async with person(r, "observer", "olle") as observer:
        res = await decide(observer, other, "inspect")
        assert res.status_code == 403 and res.json()["code"] == "FORBIDDEN_ROLE"
        res = await decide(observer, cid, "resume")
        assert res.status_code == 403


async def test_going_lower_needs_the_pilots_yes(r: Rig) -> None:
    mid, _, sent = await started(r)
    cmd = str(sent.command_id)
    res = await r.panel.post(f"/api/v1/inspections/{cmd}/lower", headers=H)
    assert res.status_code == 409 and res.json()["code"] == "NO_LOWER_OFFER"
    offer = {"command_id": cmd, "now_above_ground_m": 31.0, "lowest_above_ground_m": 12.0}
    await event(r, "inspect.lower_offer", **offer, reason="laser")
    (look,) = await looks(r, mid)
    assert look["phase"] == "lower_offer"
    assert look["lower_offer"]["lowest_above_ground_m"] == 12.0
    assert look["lower_offer_until"] is not None
    res = await r.panel.post(f"/api/v1/inspections/{cmd}/lower", headers=H)
    assert res.status_code == 200 and res.json()["phase"] == "flying", res.text
    await r.until(lambda: r.onboard.looks, 5, "InspectLower")
    assert isinstance(r.onboard.looks[0], InspectLower)
    assert str(r.onboard.looks[0].command_id) == cmd
    # Unanswered, the question goes away by itself.
    await event(r, "inspect.lower_offer", **offer, reason="map")
    r.flow.looks[cmd].lower_offer_until = r.clock.utc_now() - dt.timedelta(seconds=1)
    await inspections.tick(r.flow, await r.flow.open_mission_of(mid))
    assert (await looks(r, mid))[0]["phase"] == "flying"
    res = await r.panel.post(f"/api/v1/inspections/{mid}/lower", headers=H)
    assert res.status_code == 404


async def test_the_pilots_no_to_going_lower_reaches_the_drone(r: Rig) -> None:
    mid, _, sent = await started(r)
    cmd = str(sent.command_id)
    res = await r.panel.post(f"/api/v1/inspections/{cmd}/stay", headers=H)
    assert res.status_code == 409 and res.json()["code"] == "NO_LOWER_OFFER"
    offer = {"command_id": cmd, "now_above_ground_m": 31.0, "lowest_above_ground_m": 12.0}
    await event(r, "inspect.lower_offer", **offer, reason="laser")
    async with person(r, "observer", "olle") as observer:  # only the pilot answers
        res = await observer.post(f"/api/v1/inspections/{cmd}/stay")
        assert res.status_code == 403 and res.json()["code"] == "FORBIDDEN_ROLE"
        res = await observer.post(f"/api/v1/inspections/{cmd}/lower")
        assert res.status_code == 403
    assert (await looks(r, mid))[0]["phase"] == "lower_offer"
    assert r.onboard.looks == []
    res = await r.panel.post(f"/api/v1/inspections/{cmd}/stay", headers=H)
    assert res.status_code == 200 and res.json()["phase"] == "flying", res.text
    await r.until(lambda: r.onboard.looks, 5, "InspectStay")
    (no,) = r.onboard.looks
    assert isinstance(no, InspectStay) and not isinstance(no, InspectLower)
    assert str(no.command_id) == cmd
    # Answered: a second answer has no question to go to.
    res = await r.panel.post(f"/api/v1/inspections/{cmd}/lower", headers=H)
    assert res.status_code == 409 and res.json()["code"] == "NO_LOWER_OFFER"
    # The drone ends the look where it is, and the wait for the decision begins.
    await event(r, "inspect.done", **done(sent, end="lower_declined", clear=False))
    (look,) = await looks(r, mid)
    assert look["phase"] == "await" and look["done"]["end"] == "lower_declined"
    res = await r.panel.post(f"/api/v1/inspections/{mid}/stay", headers=H)
    assert res.status_code == 404


@pytest.mark.parametrize(
    ("decision", "state", "outcome"),
    [
        ("reject", "rejected", "rejected"),
        ("unsure", "ready", "unsure"),
        ("resume", "ready", "resume"),
    ],
)
async def test_the_decision_while_the_drone_waits(
    r: Rig, decision: str, state: str, outcome: str
) -> None:
    mid, cid, sent = await started(r)
    await event(r, "inspect.done", **done(sent))
    (look,) = await looks(r, mid)
    assert look["phase"] == "await" and look["done"]["steps"] == 3
    assert look["await_until"] is not None
    async with person(r, "pilot", "kalle") as pilot:
        res = await decide(pilot, cid, decision)
        assert res.status_code == 200, res.text
        assert res.json()["card"]["state"] == state
    await r.until(lambda: r.onboard.looks, 5, "InspectOutcome")
    (told,) = r.onboard.looks
    assert isinstance(told, InspectOutcome) and told.outcome == outcome
    (look,) = await looks(r, mid)
    assert look["phase"] == "ended" and look["outcome"] == outcome


async def test_no_decision_puts_the_candidate_back_in_the_queue(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    mid, cid, sent = await started(r)
    await event(r, "inspect.done", **done(sent, end="floor", clear=False))
    later = r.clock.utc_now() + dt.timedelta(seconds=AWAIT_DECISION_S + 1)
    monkeypatch.setattr(r.g.clock, "utc_now", lambda: later)
    await inspections.tick(r.flow, await r.flow.open_mission_of(mid))
    (look,) = await looks(r, mid)
    assert look["outcome"] == "timeout" and look["detail"] is None
    assert (await card(r.panel, cid))["card"]["state"] == "ready"
    chat = (await r.panel.get(f"/api/v1/missions/{mid}/chat")).json()["items"]
    assert any("Inget beslut om K1" in m["text"] for m in chat)
    assert not r.onboard.looks  # the drone has its own timer


async def test_a_look_the_ground_lost_track_of_ends(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a restart the ground has only the start row: once even the slowest look
    would be over, it ends."""
    mid, cid, sent = await started(r)
    r.flow.looks.clear()
    mission = await r.flow.open_mission_of(mid)
    await inspections.tick(r.flow, mission)
    assert (await looks(r, mid))[0]["phase"] == "flying"
    later = r.clock.utc_now() + dt.timedelta(hours=1)
    monkeypatch.setattr(r.g.clock, "utc_now", lambda: later)
    await inspections.tick(r.flow, mission)
    (look,) = await looks(r, mid)
    assert look["outcome"] == "timeout" and look["detail"] == "lost"
    assert (await card(r.panel, cid))["card"]["state"] == "ready"
    assert str(sent.command_id) not in r.flow.looks


@pytest.mark.parametrize(
    ("reason", "outcome"),
    [("stop", "aborted"), ("refused_onboard", "refused_onboard"), ("failed", "failed")],
)
async def test_an_aborted_look(r: Rig, reason: str, outcome: str) -> None:
    mid, cid, sent = await started(r)
    await event(
        r,
        "inspect.aborted",
        command_id=str(sent.command_id),
        candidate_id=cid,
        reason=reason,
    )
    (look,) = await looks(r, mid)
    assert look["outcome"] == outcome and look["detail"] == reason
    assert (await card(r.panel, cid))["card"]["state"] == "ready"
    # A late or malformed report changes nothing.
    await event(r, "inspect.done", **done(sent))
    await event(r, "inspect.done", command_id="nonsense")
    await event(r, "inspect.laser_lost", command_id=str(sent.command_id))
    assert (await looks(r, mid))[0]["outcome"] == outcome


async def test_a_look_the_drone_drops_itself_ends_with_its_reason(r: Rig) -> None:
    """STOP during a look (TN10): the drone answers the look's order "superseded"
    (``ONB_CONTROL_HELD``) before its ``inspect.aborted``. That is no other order
    replacing the look: the look ends aborted, with the drone's reason."""
    mid, cid, sent = await started(r)
    assert r.onboard.session is not None
    await r.onboard.session.ack(sent.command_id, "superseded", reason_codes=["ONB_CONTROL_HELD"])
    await wait_order_status(r, mid, str(sent.command_id), "superseded")
    assert (await looks(r, mid))[0]["outcome"] is None
    await event(
        r, "inspect.aborted", command_id=str(sent.command_id), candidate_id=cid, reason="stop"
    )
    (look,) = await looks(r, mid)
    assert (look["outcome"], look["detail"]) == ("aborted", "stop"), look
    assert (await card(r.panel, cid))["card"]["state"] == "ready"


@pytest.mark.parametrize(
    ("latch", "reason"),
    [("OPERATOR_STOP", "stop"), ("OPERATOR_RTL", "home"), ("BATTERY_RTL", "battery")],
)
async def test_the_grounds_stop_before_the_drones_report_ends_with_the_drones_reason(
    r: Rig, latch: str, reason: str
) -> None:
    """STOP during a look (TN10): the ground cancels the open orders (``cancel_open``)
    before the drone's ``inspect.aborted``, which waits for the ground's notice to name
    the mode change. The look ends at once with the drone's word for it, never the
    latch's code, and the report that comes after changes nothing."""
    mid, cid, sent = await started(r)
    await commands.cancel_open(r.flow, VID, latch)
    (look,) = await looks(r, mid)
    assert (look["outcome"], look["detail"]) == ("aborted", reason), look
    assert (await card(r.panel, cid))["card"]["state"] == "ready"
    await event(
        r, "inspect.aborted", command_id=str(sent.command_id), candidate_id=cid, reason=reason
    )
    assert [(x["outcome"], x["detail"]) for x in await looks(r, mid)] == [("aborted", reason)]


async def test_another_order_replaces_the_look(r: Rig) -> None:
    mid, cid, sent = await started(r)
    await event(r, "inspect.done", **done(sent))
    res = await r.panel.post(
        f"/api/v1/missions/{mid}/commands",
        json={"client_command_id": str(new_id()), "skill": "hold", "params": {}},
        headers=H,
    )
    assert res.status_code == 201, res.text

    async def ended() -> bool:
        return (await looks(r, mid))[0]["outcome"] is not None

    await r.until(ended, 5, "the look replaced once the drone took the order")
    (look,) = await looks(r, mid)
    assert look["outcome"] == "resumed_by_pilot" and look["detail"] == "hold"
    assert (await card(r.panel, cid))["card"]["state"] == "ready"


async def test_an_order_the_drone_refuses_leaves_the_look_open(r: Rig) -> None:
    """Only an order the drone took ends the look: one it refuses (or the link loses)
    leaves the drone looking, and the pilot's decision still reaches it."""
    mid, cid, sent = await started(r)
    await event(r, "inspect.done", **done(sent))
    r.onboard.order_outcome = "rejected"
    hold = await r.panel.post(
        f"/api/v1/missions/{mid}/commands",
        json={"client_command_id": str(new_id()), "skill": "hold", "params": {}},
        headers=H,
    )
    assert hold.status_code == 201, hold.text
    await wait_order_status(r, mid, hold.json()["command"]["id"], "rejected_onboard")
    (look,) = await looks(r, mid)
    assert look["outcome"] is None and look["phase"] == "await"
    assert (await card(r.panel, cid))["card"]["state"] == "inspecting"
    async with person(r, "pilot", "kalle") as pilot:
        res = await decide(pilot, cid, "reject")
        assert res.status_code == 200, res.text
    await r.until(lambda: r.onboard.looks, 5, "InspectOutcome")
    (told,) = r.onboard.looks
    assert isinstance(told, InspectOutcome) and told.outcome == "rejected"


async def test_a_late_answer_to_an_order_sent_before_the_look_leaves_it_open(r: Rig) -> None:
    """The drone's answer to an order sent before the look arrives after the look started
    (the link's round trip): that order is one the look replaced onboard, so the look
    stays open and the pilot's decision still reaches the drone. An order sent after the
    look started still replaces it."""
    r.onboard.order_outcome = "running"
    det = seen_at(r, 0, 100)
    mid, _ = await flying_with(r, det, mode="exercise")
    cid = str(det.detection_id)
    r.onboard.order_outcome = "silent"
    hold = await r.panel.post(
        f"/api/v1/missions/{mid}/commands",
        json={"client_command_id": str(new_id()), "skill": "hold", "params": {}},
        headers=H,
    )
    assert hold.status_code == 201, hold.text
    hold_id = hold.json()["command"]["id"]
    await r.until(lambda: any(str(o.command_id) == hold_id for o in r.onboard.orders), 5, "hold")
    r.onboard.order_outcome = "running"
    async with person(r, "pilot", "kalle") as pilot:
        res = await decide(pilot, cid, "inspect")
        assert res.status_code == 200, res.text
    await r.until(lambda: any(o.skill == "inspect_detection" for o in r.onboard.orders), 5, "sent")
    assert r.onboard.session is not None
    for stage in ("accepted", "running", "done"):
        await r.onboard.session.ack(uuid.UUID(hold_id), stage)
        await wait_order_status(r, mid, hold_id, stage)
    (look,) = await looks(r, mid)
    assert look["outcome"] is None
    assert (await card(r.panel, cid))["card"]["state"] == "inspecting"
    # An order sent now replaces the look once the drone takes it.
    again = await r.panel.post(
        f"/api/v1/missions/{mid}/commands",
        json={"client_command_id": str(new_id()), "skill": "hold", "params": {}},
        headers=H,
    )
    assert again.status_code == 201, again.text

    async def ended() -> bool:
        return (await looks(r, mid))[0]["outcome"] is not None

    await r.until(ended, 5, "the look replaced by the order sent after it")
    assert (await looks(r, mid))[0]["outcome"] == "resumed_by_pilot"


async def test_during_a_look_the_ai_does_not_move_the_drone(r: Rig) -> None:
    """Owner decision: during a look only the pilot moves the drone. The ground refuses
    the AI's orders that would take the drone from the look (``INSPECT_BUSY``), in the
    estimate too; flying home and the skills on the ground still pass. Once the pilot has
    decided, the AI's orders go again."""
    mid, cid, sent = await started(r)
    await event(r, "inspect.done", **done(sent))
    d = driver(r, mid)
    lat, lon = r.at(0.0, 60.0)
    goto = {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 6}
    sent_before = len(r.onboard.orders)
    for skill, params in (("goto", goto), ("hold", {})):
        out = await order(r, d, skill, params, want="REJECT")
        assert out["reason_codes"] == ["INSPECT_BUSY"], out
    situation = await d.situation()
    body = {
        "command_id": str(new_id()),
        "skill": "goto",
        "params": goto,
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
    assert res.json()["gate"]["reason_codes"] == ["INSPECT_BUSY"], res.text
    await order(r, d, "report", {"text_sv": "Väntar på piloten."})
    (look,) = await looks(r, mid)
    assert look["outcome"] is None
    assert len(r.onboard.orders) == sent_before
    async with person(r, "pilot", "kalle") as pilot:
        res = await decide(pilot, cid, "reject")
        assert res.status_code == 200, res.text
    assert (await looks(r, mid))[0]["outcome"] == "rejected"
    await order(r, d, "goto", goto)


async def test_during_a_look_the_ai_does_not_cancel_it(r: Rig) -> None:
    """Owner decision: during a look only the pilot moves the drone. A look started from
    the AI's card stays the AI's order, yet the AI's cancel would end it onboard (INSPECT
    to HOLD) and free its later orders: the ground refuses it (``INSPECT_BUSY``), as it
    does the cancel of any of its orders but the stills it may take during the look. The
    pilot's cancel still ends the look."""
    r.onboard.order_outcome = "running"
    det = seen_at(r, 0, 100)
    mid, d = await flying_with(r, det, mode="exercise")
    out = await order(r, d, "inspect_detection", {"candidate_id": "K1"}, want="PENDING_APPROVAL")
    res = await r.approve(str(out["approval_id"]))
    assert res.status_code == 200, res.text
    look_id = str(out["command_id"])
    await wait_order_status(r, mid, look_id, "running")
    sent_before = len(r.onboard.orders)
    res = await r.commander.post(f"/commander/v1/missions/{mid}/commands/{look_id}/cancel")
    assert res.status_code == 409, res.text
    assert res.json()["code"] == "INSPECT_BUSY", res.text
    (look,) = await looks(r, mid)
    assert look["outcome"] is None
    cmd = await r.flow.repo.command(look_id)
    assert cmd is not None and str(cmd.status) == "running"
    # The stills do not move the drone: the AI may take them and cancel them.
    stills = await order(r, d, "capture_stills", {"count": 1})
    stills_id = str(stills["command_id"])
    await wait_order_status(r, mid, stills_id, "running")
    res = await r.commander.post(f"/commander/v1/missions/{mid}/commands/{stills_id}/cancel")
    assert res.status_code == 202, res.text
    assert res.json()["status"] == "cancelled", res.text
    assert (await looks(r, mid))[0]["outcome"] is None
    assert len(r.onboard.orders) == sent_before + 1
    # The pilot's cancel ends the look.
    res = await r.panel.post(f"/api/v1/commands/{look_id}/cancel", headers=H)
    assert res.status_code == 202, res.text
    (look,) = await looks(r, mid)
    assert look["outcome"] == "aborted" and look["detail"] == "CANCELLED"
    assert (await card(r.panel, str(det.detection_id)))["card"]["state"] == "ready"


async def test_during_a_look_the_ai_may_still_fly_home(r: Rig) -> None:
    mid, _, sent = await started(r)
    await event(r, "inspect.done", **done(sent))
    no_rate_limit(r)
    res = await driver(r, mid).order("return_home", {})
    assert res.status_code == 201, res.text
    assert res.json()["decision"] != "REJECT", res.text


@pytest.mark.parametrize("decision", ["unsure", "reject", "confirm"])
async def test_during_a_look_an_observers_decision_only_tells_the_pilot(
    r: Rig, decision: str
) -> None:
    """Owner decision: during a look only the pilot moves the drone. An observer's mark
    is recorded and shown, the candidate stays ``inspecting`` and nothing goes onboard;
    the pilot's decision after it ends the look."""
    mid, cid, sent = await started(r)
    await event(r, "inspect.done", **done(sent))
    async with person(r, "observer", "olle") as observer:
        if decision == "confirm":
            await view(observer, cid)
        res = await decide(observer, cid, decision)
        assert res.status_code == 200, res.text
        assert res.json()["card"]["state"] == "inspecting"
        assert res.json()["decisions"][-1]["decision"] == decision
    (look,) = await looks(r, mid)
    assert look["outcome"] is None and look["phase"] == "await"
    assert not r.onboard.looks
    chat = (await r.panel.get(f"/api/v1/missions/{mid}/chat")).json()["items"]
    assert any("Piloten bestämmer" in m["text"] for m in chat)
    # A note for the pilot, never a decision for the AI.
    bus = r.g.bus.replay(r.g.bus.server_epoch, 0) or []
    (mark,) = [
        e
        for e in bus
        if e.type == "candidate.decided" and e.data["decision"]["decision"] == decision
    ]
    assert mark.data["decision"]["noted"] is True
    assert to_commander(mark) is None
    async with person(r, "pilot", "kalle") as pilot:
        res = await decide(pilot, cid, "unsure")
        assert res.status_code == 200, res.text
        assert res.json()["card"]["state"] == "ready"
    await r.until(lambda: r.onboard.looks, 5, "InspectOutcome")
    (told,) = r.onboard.looks
    assert isinstance(told, InspectOutcome) and told.outcome == "unsure"
    assert (await looks(r, mid))[0]["outcome"] == "unsure"


async def test_the_drone_refusing_the_order_ends_the_look(r: Rig) -> None:
    det = seen_at(r, 0, 100)
    mid, _ = await flying_with(r, det, mode="exercise")
    r.onboard.order_outcome = "rejected"
    cid = str(det.detection_id)
    async with person(r, "pilot", "kalle") as pilot:
        res = await decide(pilot, cid, "inspect")
        assert res.status_code == 200, res.text
    await r.until(lambda: r.onboard.orders, 5, "sent")
    cmd = str(r.onboard.orders[-1].command_id)
    await wait_order_status(r, mid, cmd, "rejected_onboard")
    (look,) = await looks(r, mid)
    assert look["outcome"] == "refused_onboard"
    assert (await card(r.panel, cid))["card"]["state"] == "ready"


async def test_cancelling_the_order_aborts_the_look(r: Rig) -> None:
    mid, cid, sent = await started(r)
    res = await r.panel.post(f"/api/v1/commands/{sent.command_id}/cancel", headers=H)
    assert res.status_code == 202, res.text
    (look,) = await looks(r, mid)
    assert look["outcome"] == "aborted" and look["detail"] == "CANCELLED"
    assert (await card(r.panel, cid))["card"]["state"] == "ready"


async def test_the_progress_reaches_the_panel_at_most_twice_a_second(r: Rig) -> None:
    _, cid, sent = await started(r)
    r.onboard.inspect = InspectProgress.model_validate(
        {"command_id": sent.command_id, "candidate_id": cid, "step": 2, "phase": "step"}
    )

    def shown() -> int:
        return r.bus_types().count("candidate.inspect_progress")

    await r.until(lambda: shown() >= 2, 5, "progress")
    first = r.clock.monotonic_ms()
    await r.until(lambda: shown() >= 4, 5, "more progress")
    # 5 Hz of snapshots, at most 2 Hz on the topic.
    assert r.clock.monotonic_ms() - first >= inspections.PROGRESS_MIN_MS


async def test_the_ais_card_starts_the_look_on_the_pilots_click(r: Rig) -> None:
    r.onboard.order_outcome = "running"
    det = seen_at(r, 0, 100)
    mid, d = await flying_with(r, det, mode="exercise")
    out = await order(r, d, "inspect_detection", {"candidate_id": "K1"}, want="PENDING_APPROVAL")
    res = await r.approve(str(out["approval_id"]))
    assert res.status_code == 200, res.text
    assert res.json()["status"] == "approved", res.text
    (look,) = await looks(r, mid)
    assert look["started_by"].startswith("operator:")
    assert look["candidate_id"] == str(det.detection_id)


async def charter_again(r: Rig, mid: str, **approval: Any) -> Any:
    """A new charter version, approved by the pilot (with ``approval``'s fields)."""
    installed = len(r.onboard.charters)
    res = await r.panel.post(
        f"/api/v1/missions/{mid}/charters",
        json={
            "area": r.square(140.0),
            "reason_sv": "Mindre",
            "base_version": r.onboard.charters[-1].charter.version,
        },
        headers=H,
    )
    assert res.status_code == 201, res.text
    body = res.json()
    res = await r.approve(str(body["approval_id"]), body["version"], **approval)
    assert res.status_code == 200, res.text
    await r.until(lambda: len(r.onboard.charters) > installed, 10, "charter sent onboard")
    return r.onboard.charters[-1].charter


async def test_the_operation_mode_and_the_permit(r: Rig) -> None:
    """Sim and exercise may fly over the candidate; live only with the permit, which the
    pilot sets (granting it counts from the next charter approval; revoking it at once, see
    ``test_revoking_the_permit_stops_overflight_at_once``). Without a pick on the card the
    mode in force stays."""
    res = await r.panel.get("/api/v1/permit")
    assert res.json() == {"specific_category_ref": "", "allow_overflight_live": False}
    bad = {"specific_category_ref": "", "allow_overflight_live": True}
    assert (await r.panel.put("/api/v1/permit", json=bad, headers=H)).status_code == 422
    mid = await r.ready_mission(mode="live")
    charter = r.onboard.charters[-1].charter
    assert charter.operation_mode == "live" and charter.overflight_allowed is False
    good = {"specific_category_ref": "OP-123", "allow_overflight_live": True}
    res = await r.panel.put("/api/v1/permit", json=good, headers=H)
    assert res.status_code == 200 and res.json() == good
    assert (await r.panel.get("/api/v1/permit")).json() == good
    again = await charter_again(r, mid)
    assert again.operation_mode == "live" and again.overflight_allowed is True
    res = await r.panel.put("/api/v1/permit", json={}, headers=H)
    assert res.status_code == 200
    exercise = await charter_again(r, mid, operation_mode="exercise")
    assert exercise.operation_mode == "exercise" and exercise.overflight_allowed is True


def floor_pack(r: Rig, name: str, *, tree_dm: int = 0) -> str:
    """A small map pack in the ground's ``<data_dir>/floor/<name>/``; its hash."""
    side = 40
    pack = write_floor_pack(
        r.flow.data_root() / "floor" / name,
        name=name,
        origin=LatLon(lat=-35.363261, lon=149.165230),  # CMAC, SITL's public test field
        dtm_cm=np.full((side, side), 584_00, dtype=np.int32),
        chm_dm=np.full((side, side), tree_dm, dtype=np.int16),
        scan_year=np.full((1, 1), 2020, dtype=np.uint16),
        scan_cell_m=side,
        attribution="Testdata (CC0)",
    )
    return pack.sha256


async def proposal(r: Rig, mid: str) -> dict[str, Any]:
    res = await r.panel.post(
        f"/api/v1/missions/{mid}/charters",
        json={
            "area": r.square(140.0),
            "reason_sv": "Mindre",
            "base_version": r.onboard.charters[-1].charter.version,
        },
        headers=H,
    )
    assert res.status_code == 201, res.text
    return res.json()


async def test_the_pilot_picks_the_floor_pack_on_the_charter_card(r: Rig) -> None:
    """The list holds the whole packs only, with their source credit; the pick names the
    hash the list showed, and anything else is refused (the card stays waiting). Without
    a pick the pack in force stays; "Ingen höjdkarta" clears it."""
    sha = floor_pack(r, "area-a")
    broken = r.flow.data_root() / "floor" / "broken"
    floor_pack(r, "broken")
    (broken / "chm_dm.npy").write_bytes(b"not a grid")
    (r.flow.data_root() / "floor" / ".area-b.writing").mkdir()  # a write in progress
    async with person(r, "observer", "olle") as observer:
        res = await observer.get("/api/v1/floor_packs")
    assert res.status_code == 200, res.text
    assert res.json() == [
        {
            "floor_pack_id": "area-a",
            "floor_pack_sha256": sha,
            "attribution": "Testdata (CC0)",
            "origin": {"lat": -35.363261, "lon": 149.16523},
            "size_east_m": 40,
            "size_north_m": 40,
        }
    ]
    mid = await r.ready_mission(mode="exercise")
    assert r.onboard.charters[-1].charter.floor_pack_id is None  # a first charter: none
    card = await proposal(r, mid)
    for choice, code in [
        ({"floor_pack_id": "nowhere", "floor_pack_sha256": sha}, "FLOOR_PACK_UNAVAILABLE"),
        ({"floor_pack_id": "broken", "floor_pack_sha256": sha}, "FLOOR_PACK_UNAVAILABLE"),
        ({"floor_pack_id": "area-a", "floor_pack_sha256": "0" * 64}, "FLOOR_PACK_CHANGED"),
    ]:
        res = await r.approve(str(card["approval_id"]), card["version"], floor_pack=choice)
        assert res.status_code == 409 and res.json()["code"] == code, res.text
    for bad in ({"floor_pack_id": "area-a"}, {"floor_pack_id": "../x", "floor_pack_sha256": sha}):
        res = await r.approve(str(card["approval_id"]), card["version"], floor_pack=bad)
        assert res.status_code == 422, res.text
    installed = len(r.onboard.charters)
    good = {"floor_pack_id": "area-a", "floor_pack_sha256": sha}
    res = await r.approve(str(card["approval_id"]), card["version"], floor_pack=good)
    assert res.status_code == 200 and res.json()["status"] == "approved", res.text
    await r.until(lambda: len(r.onboard.charters) > installed, 10, "charter sent onboard")
    picked = r.onboard.charters[-1].charter
    assert (picked.floor_pack_id, picked.floor_pack_sha256) == ("area-a", sha)
    res = await r.panel.get(f"/api/v1/missions/{mid}/charters")
    (stored,) = [c for c in res.json() if c["status"] == "approved"]
    assert stored["body"]["floor_pack_id"] == "area-a"
    # No pick on the next card: the pack in force stays.
    kept = await charter_again(r, mid)
    assert (kept.floor_pack_id, kept.floor_pack_sha256) == ("area-a", sha)
    none = {"floor_pack_id": None, "floor_pack_sha256": None}
    cleared = await charter_again(r, mid, floor_pack=none)
    assert cleared.floor_pack_id is None and cleared.floor_pack_sha256 is None
    again = await charter_again(r, mid)  # and none stays none
    assert again.floor_pack_id is None


async def test_revoking_the_permit_stops_overflight_at_once(r: Rig) -> None:
    """The live charter was approved with the permit; the pilot revokes it. From then on
    no look starts (the drone's charter still allows overflight) until a new charter:
    neither the pilot's "Titta närmare" nor the click on a card made before."""
    good = {"specific_category_ref": "OP-123", "allow_overflight_live": True}
    assert (await r.panel.put("/api/v1/permit", json=good, headers=H)).status_code == 200
    r.onboard.order_outcome = "running"
    det = seen_at(r, 0, 100)
    mid, d = await flying_with(r, det, mode="live")
    assert r.onboard.charters[-1].charter.overflight_allowed is True
    out = await order(r, d, "inspect_detection", {"candidate_id": "K1"}, want="PENDING_APPROVAL")
    assert (await r.panel.put("/api/v1/permit", json={}, headers=H)).status_code == 200
    res = await r.approve(str(out["approval_id"]))
    assert res.status_code == 200, res.text
    assert res.json()["status"] == "rejected", res.text
    cmd = await r.flow.repo.command(out["command_id"])
    assert cmd is not None and str(cmd.status) == "rejected_by_gate"
    async with person(r, "pilot", "kalle") as pilot:
        res = await decide(pilot, str(det.detection_id), "inspect")
        assert res.status_code == 409 and res.json()["code"] == "INSPECT_REFUSED", res.text
        assert {"code": "OVERFLIGHT_NOT_ALLOWED"} in res.json()["errors"]
    assert not [o for o in r.onboard.orders if o.skill == "inspect_detection"]
    assert await looks(r, mid) == []
    # No cylinder is suspended for a look under the revoked permit either.
    mission = await r.flow.open_mission_of(mid)
    ctx = await r.flow.gate_context(mission)
    facts = await inspections.gate_facts(
        r.flow, mission, ctx, {"candidate_id": str(det.detection_id)}
    )
    assert facts.suspended_keepout_ids == ()


async def test_a_permit_revoked_just_before_the_order_goes_stops_it(r: Rig) -> None:
    """The gate said yes under the permit, which the pilot revoked before the order went:
    the look does not start and the order is not sent."""
    good = {"specific_category_ref": "OP-123", "allow_overflight_live": True}
    assert (await r.panel.put("/api/v1/permit", json=good, headers=H)).status_code == 200
    det = seen_at(r, 0, 100)
    mid, d = await flying_with(r, det, mode="live")
    out = await order(r, d, "inspect_detection", {"candidate_id": "K1"}, want="PENDING_APPROVAL")
    assert (await r.panel.put("/api/v1/permit", json={}, headers=H)).status_code == 200
    mission = await r.flow.open_mission_of(mid)
    cmd = await r.flow.repo.command(out["command_id"])
    assert cmd is not None
    await commands._dispatch(r.flow, mission, cmd, approval=None)  # pyright: ignore[reportPrivateUsage]
    cmd = await r.flow.repo.command(out["command_id"])
    assert cmd is not None and str(cmd.status) == "rejected_by_gate"
    assert cmd.status_reason == "OVERFLIGHT_NOT_ALLOWED"
    assert not [o for o in r.onboard.orders if o.skill == "inspect_detection"]
    assert await looks(r, mid) == []


async def test_a_candidate_closed_just_before_the_order_goes_stops_it(r: Rig) -> None:
    """The gate saw it open, but a person decided before the order went: the look does
    not start and the order is not sent (a race the gate alone cannot close)."""
    det = seen_at(r, 0, 100)
    mid, d = await flying_with(r, det, mode="exercise")
    out = await order(r, d, "inspect_detection", {"candidate_id": "K1"}, want="PENDING_APPROVAL")
    async with person(r, "pilot", "kalle") as pilot:
        res = await decide(pilot, str(det.detection_id), "reject")
        assert res.status_code == 200, res.text
    mission = await r.flow.open_mission_of(mid)
    cmd = await r.flow.repo.command(out["command_id"])
    assert cmd is not None
    await commands._dispatch(r.flow, mission, cmd, approval=None)  # pyright: ignore[reportPrivateUsage]
    cmd = await r.flow.repo.command(out["command_id"])
    assert cmd is not None and str(cmd.status) == "rejected_by_gate"
    assert cmd.status_reason == "INSPECT_TARGET_CLOSED"
    assert not [o for o in r.onboard.orders if o.skill == "inspect_detection"]
    assert await looks(r, mid) == []
