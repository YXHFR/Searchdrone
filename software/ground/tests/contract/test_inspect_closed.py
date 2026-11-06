"""The end of a closer look as the AI hears it, over the real SD-link server with the fake
drone (``services.inspections``).

The look's "done" is not its end: the drone hovers over the candidate until a person
decides, and the ground refuses the AI's moves meanwhile (INSPECT_BUSY). Its end reaches
the AI once, as a skill event of the look's order with status "closed" that says the
outcome and what the drone does next. A search the drone flies on after the look ends as
that search's own order (the "linked event" model): its row stays superseded, its sector's
coverage still counts, and its end is a skill event for the original order.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import sys
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from brain_rig import (  # pyright: ignore[reportMissingImports]
    loopback_listener,
    run_brain,
    serve_commander,
)
from flow_rig import Rig, rig  # pyright: ignore[reportMissingImports]
from test_candidate_decisions import decide, person  # pyright: ignore[reportMissingImports]
from test_candidates import person as seen_at  # pyright: ignore[reportMissingImports]
from test_inspect_order import (  # pyright: ignore[reportMissingImports]
    flying_with,
    post,
    revision,
    send,
)
from test_inspections import done, event, looks  # pyright: ignore[reportMissingImports]
from test_orders import (  # pyright: ignore[reportMissingImports]
    no_rate_limit,
    order,
    wait_order_status,
)

from sd_ground.api.commander import to_commander
from sd_ground.db.inspections import InspectionRepo
from sd_ground.services import inspections
from sd_schemas.base import new_id
from sd_schemas.executor import ExecutorState
from sd_schemas.inspect import AWAIT_DECISION_S
from sd_schemas.mission import Trigger

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]

SK09 = Path(__file__).resolve().parents[4] / "simulation/scenarios/scripts/SK09.yaml"


@pytest.fixture
async def r(migrated_db: Db, tmp_path: Path) -> AsyncIterator[Rig]:
    async with rig(migrated_db.name, migrated_db.app_password, tmp_path) as built:
        yield built


def sectors(r: Rig) -> list[dict[str, Any]]:
    """Sector A east of home, B west of it; home is in neither."""
    return [
        {"sector_id": "A", "name": "Öster", "area": r.square(30.0, east=60.0), "order_idx": 0},
        {"sector_id": "B", "name": "Väster", "area": r.square(30.0, east=-60.0), "order_idx": 1},
    ]


def bus(r: Rig, kind: str) -> list[Any]:
    return [e for e in r.g.bus.replay(r.g.bus.server_epoch, 0) or [] if e.type == kind]


def closed(sent: Any, next_: str, **kw: Any) -> dict[str, Any]:
    """The drone's ``inspect.closed`` for the look ``sent``."""
    return {
        "command_id": str(sent.command_id),
        "candidate_id": sent.params["candidate_id"],
        "cause": "returned" if next_ == "resume" else "confirmed",
        "next": next_,
    } | kw


async def ai_look(r: Rig, mid: str, d: Any) -> Any:
    """The AI's card for a closer look at K1, approved by the pilot: the order sent."""
    out = await order(r, d, "inspect_detection", {"candidate_id": "K1"}, want="PENDING_APPROVAL")
    res = await r.approve(str(out["approval_id"]))
    assert res.status_code == 200, res.text
    await r.until(lambda: any(o.skill == "inspect_detection" for o in r.onboard.orders), 5, "look")
    return next(o for o in r.onboard.orders if o.skill == "inspect_detection")


async def test_the_look_closes_once_for_the_ai_with_what_comes_next(r: Rig) -> None:
    r.onboard.order_outcome = "running"
    det = seen_at(r, 0, 100)
    mid, d = await flying_with(r, det, mode="exercise")
    sent = await ai_look(r, mid, d)
    await event(r, "inspect.done", **done(sent))
    # While the drone waits: the open look in the AI's situation, and its moves are refused
    # with the hint to wait for the close event.
    line = (await d.situation())["candidates_line"]
    assert "Titt på K1 pågår: drönaren väntar på operatörens beslut om K1." in line
    assert "inspect_detection closed" in line
    lat, lon = r.at(0.0, 60.0)
    out = await order(r, d, "goto", {"lat": lat, "lon": lon, "alt_m": 40}, want="REJECT")
    assert out["reason_codes"] == ["INSPECT_BUSY"]
    assert out["details"]["retry_after"] == "inspect_closed"
    assert out["details"]["candidate"] == "K1" and out["details"]["phase"] == "await"
    assert "vänta" in out["message_sv"].lower()
    async with person(r, "pilot", "kalle") as pilot:
        res = await decide(pilot, str(det.detection_id), "reject")
        assert res.status_code == 200, res.text
    # The ground has ended the look; the drone's report says what comes next. Until it
    # comes the drone may still be in the look: the AI's moves stay refused.
    assert not bus(r, "inspection.closed")
    out = await order(r, d, "goto", {"lat": lat, "lon": lon, "alt_m": 40}, want="REJECT")
    assert out["reason_codes"] == ["INSPECT_BUSY"]
    line = (await d.situation())["candidates_line"]
    assert "Titt på K1 avslutas" in line
    await event(r, "inspect.closed", **closed(sent, "hold", cause="returned"))
    await event(r, "inspect.closed", **closed(sent, "hold"))  # a resend: told once
    (e,) = bus(r, "inspection.closed")
    out = to_commander(e)
    assert out is not None and out["type"] == "event.skill"
    body = out["body"]
    assert body["command_id"] == str(sent.command_id)
    assert body["skill"] == "inspect_detection" and body["status"] == "closed"
    details = body["details"]
    assert details["outcome"] == "rejected" and details["next"] == "hold"
    assert details["candidate"] == "K1" and details["resumed_command_id"] is None
    assert "K1" in body["message_sv"] and "står still" in body["message_sv"]
    assert e.data["issuer_kind"] == "brain" and body["issuer_kind"] == "brain"
    assert "Titt på" not in ((await d.situation())["candidates_line"] or "")
    await r.g.flush_events()
    res = await r.panel.get(f"/api/v1/missions/{mid}/events", params={"type": "inspection.closed"})
    assert res.status_code == 200 and len(res.json()) == 1, res.text


async def test_without_the_drones_report_the_look_closes_after_a_while(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    r.onboard.order_outcome = "running"
    det = seen_at(r, 0, 100)
    mid, d = await flying_with(r, det, mode="exercise")
    sent = await ai_look(r, mid, d)
    await event(r, "inspect.done", **done(sent))
    async with person(r, "pilot", "kalle") as pilot:
        res = await decide(pilot, str(det.detection_id), "unsure")
        assert res.status_code == 200, res.text
    mission = await r.flow.open_mission_of(mid)
    await inspections.tick(r.flow, mission)
    assert not bus(r, "inspection.closed")  # still waiting for the drone
    lat, lon = r.at(0.0, 60.0)
    out = await order(r, d, "goto", {"lat": lat, "lon": lon, "alt_m": 40}, want="REJECT")
    assert out["reason_codes"] == ["INSPECT_BUSY"]  # the drone may still be in the look
    later = r.clock.utc_now() + dt.timedelta(seconds=inspections.CLOSED_WAIT_S + 1)
    monkeypatch.setattr(r.g.clock, "utc_now", lambda: later)
    await inspections.tick(r.flow, mission)
    (e,) = bus(r, "inspection.closed")
    assert e.data["details"]["outcome"] == "unsure" and e.data["details"]["next"] is None
    # Not knowing what the drone does, the AI looks or asks before it orders anything.
    assert "fråga operatören" in e.data["message_sv"]
    assert await inspections.busy_look(r.flow, mid) is None
    # A report after that is only logged.
    await event(r, "inspect.closed", **closed(sent, "hold"))
    assert len(bus(r, "inspection.closed")) == 1
    assert str(sent.command_id) not in r.flow.looks


async def test_the_search_flown_on_after_a_look_is_the_ais_order_to_its_end(r: Rig) -> None:
    """The linked event model: the drone flies the AI's search on with its own resume; the
    ground follows it as the original order (active, coverage) without reviving the row,
    and its end reaches the AI as that order's skill event."""
    r.onboard.order_outcome = "running"
    det = seen_at(r, 0, 100)
    mid, d = await flying_with(r, det, mode="exercise", sectors=sectors(r))
    survey = await order(r, d, "survey_sector", {"sector_id": "A", "alt_m": 40, "spacing_m": 20})
    sid = str(survey["command_id"])
    await wait_order_status(r, mid, sid, "running")
    sent = await ai_look(r, mid, d)
    assert r.onboard.session is not None
    await r.onboard.session.ack(uuid.UUID(sid), "superseded")  # the look broke it off
    await wait_order_status(r, mid, sid, "superseded")
    await event(r, "inspect.done", **done(sent))
    async with person(r, "pilot", "kalle") as pilot:
        res = await decide(pilot, str(det.detection_id), "reject")
        assert res.status_code == 200, res.text
    resume_id = new_id()
    r.onboard.own_resume()  # the executor's own resume, unknown here and never acked
    await event(
        r,
        "inspect.closed",
        **closed(
            sent,
            "resume",
            breakpoint="survey",
            resume_command_id=str(resume_id),
            resumed_command_id=sid,
        ),
    )
    (e,) = bus(r, "inspection.closed")
    body = to_commander(e)["body"]  # type: ignore[index]
    assert body["details"]["next"] == "resume" and body["details"]["breakpoint"] == "survey"
    assert body["details"]["resumed_command_id"] == sid
    assert body["details"]["sector_id"] == "A" and "sektor A" in body["message_sv"]

    # The resumed run is the AI's active order, its row unchanged.
    async def active() -> bool:
        ac = (await d.situation())["active_command"]
        return ac is not None and ac["id"] == sid

    await r.until(active, 5, "the search flown on as the active order")
    ac = (await d.situation())["active_command"]
    assert ac["status"] == "superseded" and ac["params"]["sector_id"] == "A"

    # Its progress moves sector A, though the snapshot names the resume, not the search.
    r.onboard.set_state(ExecutorState.SURVEY, progress_pct=40.0)

    async def at_40() -> bool:
        rows = {s["sector_id"]: s for s in (await d.situation())["sectors"]}
        return rows["A"]["progress"] == pytest.approx(0.4)

    await r.until(at_40, 5, "sector A at 40 % by the search flown on")

    # Coverage: the drone over sector A now counts for it (its row is superseded).
    assert r.fc is not None
    r.fc.lat, r.fc.lon = r.at(60.0, 0.0)

    async def covered() -> bool:
        features = (await r.mission(mid))["sectors"]["features"]
        return next(f for f in features if f["id"] == "A")["properties"]["covered"] is not None

    await r.until(covered, 20, "sector A covered by the search flown on")
    features = (await r.mission(mid))["sectors"]["features"]
    assert next(f for f in features if f["id"] == "B")["properties"]["covered"] is None

    # Its end: a skill event for the original order, marked as the run after the look.
    await event(
        r,
        "inspect.resume_ended",
        command_id=str(sent.command_id),
        resumed_command_id=sid,
        status="done",
    )
    (e,) = bus(r, "inspection.resume_ended")
    out = to_commander(e)
    assert out is not None and out["type"] == "event.skill"
    body = out["body"]
    assert body["command_id"] == sid and body["skill"] == "survey_sector"
    assert body["status"] == "done" and body["sector_id"] == "A"
    assert body["details"]["resumed_after_look"] == str(sent.command_id)
    rows = {s["sector_id"]: s for s in (await d.situation())["sectors"]}
    assert rows["A"]["status"] == "done" and rows["A"]["progress"] == 1.0
    assert (await r.flow.repo.command(sid)).status == "superseded"  # not revived
    assert str(sent.command_id) not in r.flow.looks


async def _flown_on_a(r: Rig) -> tuple[str, str, Any, Any, uuid.UUID]:
    """A look broke off the AI's search of sector A and the drone flies it on by its own
    resume (its state names the resume). The mission, the search, the look's order, the
    AI's situation and the resume's id."""
    r.onboard.order_outcome = "running"
    det = seen_at(r, 0, 100)
    mid, d = await flying_with(r, det, mode="exercise", sectors=sectors(r))
    survey = await order(r, d, "survey_sector", {"sector_id": "A", "alt_m": 40, "spacing_m": 20})
    sid = str(survey["command_id"])
    await wait_order_status(r, mid, sid, "running")
    sent = await ai_look(r, mid, d)
    assert r.onboard.session is not None
    await r.onboard.session.ack(uuid.UUID(sid), "superseded")
    await wait_order_status(r, mid, sid, "superseded")
    async with person(r, "pilot", "kalle") as pilot:
        res = await decide(pilot, str(det.detection_id), "reject")
        assert res.status_code == 200, res.text
    resume_id = new_id()
    r.onboard.own_resume()
    await event(
        r,
        "inspect.closed",
        **closed(
            sent,
            "resume",
            breakpoint="survey",
            resume_command_id=str(resume_id),
            resumed_command_id=sid,
        ),
    )

    async def active_is_search() -> bool:
        ac = (await d.situation())["active_command"]
        return ac is not None and ac["id"] == sid

    await r.until(active_is_search, 5, "the search flown on as the active order")
    assert r.flow.looks[str(sent.command_id)].resume_seen
    return mid, sid, sent, d, resume_id


async def test_the_link_to_the_search_flown_on_ends_when_the_drone_leaves_it(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The drone's state names another order than its resume: the search no longer counts
    as flown on (active order, coverage), its late end is still told, and the look leaves
    memory a while later without it."""
    mid, sid, sent, d, _ = await _flown_on_a(r)
    r.onboard.idle()  # e.g. the pilot took over; no resume_ended came

    async def left() -> bool:
        return (await d.situation())["active_command"] is None

    await r.until(left, 5, "the search no longer the active order")
    assert inspections.flown_on(r.flow, mid) == []
    look = r.flow.looks[str(sent.command_id)]
    assert look.resume_left_at is not None
    # Its end, late: still the AI's event for the search.
    await event(
        r,
        "inspect.resume_ended",
        command_id=str(sent.command_id),
        resumed_command_id=sid,
        status="superseded",
    )
    (e,) = bus(r, "inspection.resume_ended")
    assert e.data["command_id"] == sid and e.data["status"] == "superseded"
    assert str(sent.command_id) not in r.flow.looks


async def test_a_search_left_without_its_end_leaves_memory_after_a_while(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    mid, _sid, sent, d, _ = await _flown_on_a(r)
    r.onboard.idle()

    async def left() -> bool:
        return (await d.situation())["active_command"] is None

    await r.until(left, 5, "the search no longer the active order")
    mission = await r.flow.open_mission_of(mid)
    await inspections.tick(r.flow, mission)
    assert str(sent.command_id) in r.flow.looks  # a late resume_ended may still come
    later = r.clock.utc_now() + dt.timedelta(seconds=inspections.CLOSED_WAIT_S + 1)
    monkeypatch.setattr(r.g.clock, "utc_now", lambda: later)
    await inspections.tick(r.flow, mission)
    assert str(sent.command_id) not in r.flow.looks
    assert not bus(r, "inspection.resume_ended")


async def test_a_resume_end_for_another_search_is_only_logged(r: Rig) -> None:
    """The run's end must name the search the look's end named (and of its mission)."""
    _mid, _sid, sent, d, _ = await _flown_on_a(r)
    await event(
        r,
        "inspect.resume_ended",
        command_id=str(sent.command_id),
        resumed_command_id=str(new_id()),
        status="done",
    )
    assert not bus(r, "inspection.resume_ended")
    rows = {s["sector_id"]: s for s in (await d.situation())["sectors"]}
    assert rows["A"]["status"] != "done"


async def test_the_link_ends_with_the_mission(r: Rig) -> None:
    mid, _sid, sent, _d, _ = await _flown_on_a(r)
    mission = await r.flow.open_mission_of(mid)
    await r.flow.close(mission, Trigger.CLOSE_MANUAL, "manual")  # forgets its looks
    assert inspections.flown_on(r.flow, mid) == []
    assert str(sent.command_id) not in r.flow.looks


async def _snapshots_after(r: Rig, n: int = 3) -> None:
    """``n`` more state snapshots from the drone, the last ones taken in by the ground."""
    seq = r.onboard.state_seq
    await r.until(lambda: r.onboard.state_seq > seq + n, 5, f"{n} more snapshots")


async def _resume_ended(r: Rig, sent: Any, resumed: str, status: str = "done") -> None:
    await event(
        r,
        "inspect.resume_ended",
        command_id=str(sent.command_id),
        resumed_command_id=resumed,
        status=status,
    )


async def test_the_executors_own_resume_is_followed_to_its_end_past_the_wait(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As the real executor: its own resume is never acked, so its state names it by
    skill only. The search stays flown on however long it runs, and its end is told."""
    mid, sid, sent, d, _ = await _flown_on_a(r)
    later = r.clock.utc_now() + dt.timedelta(seconds=3 * inspections.CLOSED_WAIT_S)
    monkeypatch.setattr(r.g.clock, "utc_now", lambda: later)
    await _snapshots_after(r)
    mission = await r.flow.open_mission_of(mid)
    await inspections.tick(r.flow, mission)
    (flown,) = inspections.flown_on(r.flow, mid)
    assert flown.command_id == sid
    assert (await d.situation())["active_command"]["id"] == sid
    await _resume_ended(r, sent, sid)
    (e,) = bus(r, "inspection.resume_ended")
    assert e.data["command_id"] == sid and e.data["status"] == "done"


async def test_a_resume_the_drone_never_shows_ends_the_link_after_the_wait(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The look's end says the drone flies the search on, but its state never names the
    resume (it ended at once, its end lost): after ``CLOSED_WAIT_S`` the search no longer
    counts as flown on."""
    r.onboard.order_outcome = "running"
    det = seen_at(r, 0, 100)
    mid, d = await flying_with(r, det, mode="exercise", sectors=sectors(r))
    survey = await order(r, d, "survey_sector", {"sector_id": "A", "alt_m": 40, "spacing_m": 20})
    sid = str(survey["command_id"])
    await wait_order_status(r, mid, sid, "running")
    sent = await ai_look(r, mid, d)
    assert r.onboard.session is not None
    await r.onboard.session.ack(uuid.UUID(sid), "superseded")
    await wait_order_status(r, mid, sid, "superseded")
    async with person(r, "pilot", "kalle") as pilot:
        res = await decide(pilot, str(det.detection_id), "reject")
        assert res.status_code == 200, res.text
    r.onboard.idle()
    await event(
        r,
        "inspect.closed",
        **closed(
            sent,
            "resume",
            breakpoint="survey",
            resume_command_id=str(new_id()),
            resumed_command_id=sid,
        ),
    )
    await _snapshots_after(r)
    look = r.flow.looks[str(sent.command_id)]
    assert not look.resume_seen and look.resume_left_at is None  # not yet
    later = r.clock.utc_now() + dt.timedelta(seconds=inspections.CLOSED_WAIT_S + 1)
    monkeypatch.setattr(r.g.clock, "utc_now", lambda: later)
    await _snapshots_after(r)
    assert look.resume_left_at is not None
    assert inspections.flown_on(r.flow, mid) == []


def _command_as(r: Rig, monkeypatch: pytest.MonkeyPatch, cid: str, real: str, **kw: Any) -> None:
    """The ground's order ``cid`` read as its order ``real`` with ``kw`` changed."""
    read = r.flow.repo.command

    async def command(command_id: str) -> Any:
        if command_id != cid:
            return await read(command_id)
        row = await read(real)
        return None if row is None else row.model_copy(update={"id": uuid.UUID(cid)} | kw)

    monkeypatch.setattr(r.flow.repo, "command", command)


async def test_a_resume_end_naming_another_search_of_the_mission_is_only_logged(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another superseded search of the same mission, not the one the look's end named:
    the named search alone decides."""
    mid, sid, sent, d, _ = await _flown_on_a(r)
    other = str(new_id())
    _command_as(r, monkeypatch, other, sid)  # exists, same mission, superseded
    await _resume_ended(r, sent, other)
    assert not bus(r, "inspection.resume_ended")
    look = r.flow.looks[str(sent.command_id)]
    assert not look.resume_ended and inspections.flown_on(r.flow, mid) != []
    rows = {s["sector_id"]: s for s in (await d.situation())["sectors"]}
    assert rows["A"]["status"] != "done"
    await _resume_ended(r, sent, sid)  # its own end is still told
    (e,) = bus(r, "inspection.resume_ended")
    assert e.data["command_id"] == sid


async def test_a_resume_end_whose_search_is_of_another_mission_is_only_logged(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The search the look's end named, but an order of another mission: the mission alone
    decides."""
    mid, sid, sent, _d, _ = await _flown_on_a(r)
    _command_as(r, monkeypatch, sid, sid, mission_id=new_id())
    await _resume_ended(r, sent, sid)
    assert not bus(r, "inspection.resume_ended")
    look = r.flow.looks[str(sent.command_id)]
    assert not look.resume_ended and inspections.flown_on(r.flow, mid) != []


async def test_after_a_ground_restart_the_end_of_the_search_broken_off_is_told_once(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ground lost the look's end from memory (a restart): the end of the search the
    look broke off (superseded, of its mission, older than the look) is told, once;
    another search is only logged."""
    _mid, sid, sent, _d, _ = await _flown_on_a(r)
    r.flow.looks.clear()  # as after a restart
    newer = str(new_id())
    later = r.clock.utc_now() + dt.timedelta(minutes=5)
    _command_as(r, monkeypatch, newer, sid, created_at=later)  # ordered after the look
    await _resume_ended(r, sent, newer)
    assert not bus(r, "inspection.resume_ended")
    await _resume_ended(r, sent, sid)
    await _resume_ended(r, sent, sid)  # a resend
    (e,) = bus(r, "inspection.resume_ended")
    assert e.data["command_id"] == sid


async def test_while_the_ground_ends_a_look_the_ais_moves_stay_refused(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """From its end in the database until the AI has heard it, through every await in
    between: the look holds the AI's moves, said as a look being closed."""
    r.onboard.order_outcome = "running"
    det = seen_at(r, 0, 100)
    mid, d = await flying_with(r, det, mode="exercise")
    await ai_look(r, mid, d)
    written, go = asyncio.Event(), asyncio.Event()
    end = InspectionRepo.end

    async def slow_end(self: InspectionRepo, *a: Any, **kw: Any) -> bool:
        ended = await end(self, *a, **kw)
        written.set()
        await go.wait()
        return ended

    monkeypatch.setattr(InspectionRepo, "end", slow_end)
    async with person(r, "pilot", "kalle") as pilot:
        deciding = asyncio.ensure_future(decide(pilot, str(det.detection_id), "reject"))
        await asyncio.wait_for(written.wait(), 5)
        assert await inspections.open_look(r.flow, mid) is None  # ended in the database
        assert await inspections.busy_look(r.flow, mid) is not None
        lat, lon = r.at(0.0, 60.0)
        out = await order(r, d, "goto", {"lat": lat, "lon": lon, "alt_m": 40}, want="REJECT")
        assert out["reason_codes"] == ["INSPECT_BUSY"]
        assert out["message_sv"].startswith(inspections.CLOSING_SV)
        go.set()
        res = await deciding
        assert res.status_code == 200, res.text


# --- the race from SK09, with the real AI on its script -----------------------------------


@pytest.fixture
async def stack(migrated_db: Db, tmp_path: Path) -> AsyncIterator[tuple[Rig, int]]:
    sock = loopback_listener()
    port = sock.getsockname()[1]
    async with (
        rig(
            migrated_db.name,
            migrated_db.app_password,
            tmp_path,
            http={"commander_bind": f"127.0.0.1:{port}"},
        ) as r,
        serve_commander(r, sock),
    ):
        yield r, port


async def _sk09_look(r: Rig, mid: str) -> tuple[Any, Any]:
    """With SK09's script on the real AI: a HÖG candidate, the AI's look at it approved
    and sent, and the drone's "done" (it now waits for the operator). The detection and
    the look's order."""
    det = seen_at(r, 0, 100)
    await send(r, det)
    found = {"detection_id": str(det.detection_id), "distance_m": 0.0}
    res = await post(
        r,
        mid,
        revision(
            r,
            candidate_id=str(det.detection_id),
            state="ready",
            lat=det.lat,
            lon=det.lon,
            sigma95_m=det.sigma95_m,
            priority=0.9,
            observations=[found],
        ),
    )
    assert res.status_code == 200, res.text

    # The AI's card for the look; the pilot approves it.
    async def card() -> str | None:
        res = await r.panel.get(f"/api/v1/missions/{mid}/approvals")
        pending = [a for a in res.json() if a["status"] == "pending"] if res.is_success else []
        return str(pending[0]["id"]) if pending else None

    await r.until(card, 15, "the AI's card for the look")
    res = await r.approve(str(await card()))
    assert res.status_code == 200, res.text
    await r.until(lambda: any(o.skill == "inspect_detection" for o in r.onboard.orders), 5, "look")
    sent = next(o for o in r.onboard.orders if o.skill == "inspect_detection")
    await event(r, "inspect.done", **done(sent))
    assert r.onboard.session is not None
    await r.onboard.session.ack(sent.command_id, "done")
    await r.until(lambda: said(r, mid, "Väntar på operatörens beslut"), 10, "the AI waits")
    return det, sent


async def said(r: Rig, mid: str, text: str) -> bool:
    res = await r.panel.get(f"/api/v1/missions/{mid}/chat")
    return any(text in m["text"] for m in res.json()["items"])


def surveys(r: Rig) -> list[Any]:
    return [o for o in r.onboard.orders if o.skill == "survey_sector"]


async def test_sk09_orders_the_search_again_once_the_slow_decision_closes_the_look(
    stack: tuple[Rig, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SK09's script on the real AI: the operator decides 40 s after the look's "done"
    (when the AI used to order sector B into INSPECT_BUSY); the drone holds after the
    look, so sector A goes onboard again once the look has closed."""
    r, port = stack
    r.onboard.order_outcome = "running"
    # The setup's own orders and candidates go to the loopback APIs on their bound address.
    assert r.search is not None
    for client in (r.commander, r.search):
        client.base_url = httpx.URL(f"http://127.0.0.1:{port}")
    mid, _ = await flying_with(r, mode="exercise", sectors=sectors(r))
    no_rate_limit(r)  # the setup's takeoff is not the AI's
    config = {"backend": {"kind": "scripted", "scripted_file": str(SK09)}, "profile": "sim"}
    async with run_brain(r, port, **config):
        det, sent = await _sk09_look(r, mid)
        assert not surveys(r)
        # The operator decides 40 s later (the drone waits up to AWAIT_DECISION_S).
        later = r.clock.utc_now() + dt.timedelta(seconds=40)
        assert later < r.clock.utc_now() + dt.timedelta(seconds=AWAIT_DECISION_S)
        monkeypatch.setattr(r.g.clock, "utc_now", lambda: later)
        assert (await looks(r, mid))[0]["outcome"] is None
        async with person(r, "pilot", "kalle") as pilot:
            res = await decide(pilot, str(det.detection_id), "reject")
            assert res.status_code == 200, res.text
        await event(r, "inspect.closed", **closed(sent, "hold", cause="returned"))
        await r.until(lambda: bool(surveys(r)), 15, "sector A ordered after the look closed")
        (survey,) = surveys(r)
        assert survey.params["polygon"] == sectors(r)[0]["area"]
        rows = (await r.panel.get(f"/api/v1/missions/{mid}/commands")).json()
        assert not [o for o in rows if "INSPECT_BUSY" in (o.get("status_reason") or "")]


async def test_sk09_flies_sector_a_on_after_the_look_and_then_searches_sector_b(
    stack: tuple[Rig, int],
) -> None:
    """SK09's script on the real AI, as in SITL: the look breaks off the search of sector
    A and the drone flies it on by itself. The AI orders nothing on the close (the rest of
    sector A may hold the wrong person when the look was at a false detection), and orders
    sector B once the search flown on is done."""
    r, port = stack
    r.onboard.order_outcome = "running"
    assert r.search is not None
    for client in (r.commander, r.search):
        client.base_url = httpx.URL(f"http://127.0.0.1:{port}")
    mid, d = await flying_with(r, mode="exercise", sectors=sectors(r))
    search_a = await order(r, d, "survey_sector", {"sector_id": "A", "alt_m": 50, "spacing_m": 45})
    sid = str(search_a["command_id"])
    await wait_order_status(r, mid, sid, "running")
    no_rate_limit(r)  # the setup's orders are not the AI's
    config = {"backend": {"kind": "scripted", "scripted_file": str(SK09)}, "profile": "sim"}
    async with run_brain(r, port, **config):
        det, sent = await _sk09_look(r, mid)
        assert r.onboard.session is not None
        await r.onboard.session.ack(uuid.UUID(sid), "superseded")  # the look broke it off
        async with person(r, "pilot", "kalle") as pilot:
            res = await decide(pilot, str(det.detection_id), "reject")
            assert res.status_code == 200, res.text
        resume_id = new_id()
        r.onboard.own_resume()
        await event(
            r,
            "inspect.closed",
            **closed(
                sent,
                "resume",
                breakpoint="survey",
                resume_command_id=str(resume_id),
                resumed_command_id=sid,
            ),
        )
        await r.until(
            lambda: said(r, mid, "Drönaren söker vidare i sektor A"), 10, "the AI flies A on"
        )
        assert len(surveys(r)) == 1  # only the setup's search of sector A
        await event(
            r,
            "inspect.resume_ended",
            command_id=str(sent.command_id),
            resumed_command_id=sid,
            status="done",
        )
        await r.until(lambda: len(surveys(r)) == 2, 15, "sector B ordered after sector A")
        assert surveys(r)[-1].params["polygon"] == sectors(r)[1]["area"]
