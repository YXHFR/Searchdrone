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
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flow_rig import H, Rig, rig  # pyright: ignore[reportMissingImports]
from test_candidate_decisions import decide, person  # pyright: ignore[reportMissingImports]
from test_candidates import person as seen_at  # pyright: ignore[reportMissingImports]
from test_inspect_order import flying_with  # pyright: ignore[reportMissingImports]
from test_orders import order, wait_order_status  # pyright: ignore[reportMissingImports]

from sd_ground.services import inspections
from sd_schemas.base import new_id
from sd_schemas.inspect import AWAIT_DECISION_S, InspectProgress
from sd_schemas.link import InspectLower, InspectOutcome, OnboardEvent

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


async def test_another_order_replaces_the_look(r: Rig) -> None:
    mid, cid, sent = await started(r)
    await event(r, "inspect.done", **done(sent))
    res = await r.panel.post(
        f"/api/v1/missions/{mid}/commands",
        json={"client_command_id": str(new_id()), "skill": "hold", "params": {}},
        headers=H,
    )
    assert res.status_code == 201, res.text
    (look,) = await looks(r, mid)
    assert look["outcome"] == "resumed_by_pilot" and look["detail"] == "hold"
    assert (await card(r.panel, cid))["card"]["state"] == "ready"


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
    pilot sets (it counts from the next charter approval). Without a pick on the card the
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
