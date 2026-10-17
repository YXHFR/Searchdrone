"""The mission flow against a real database, a fake flight controller and a fake
onboard computer (B1-12 acceptance points 1-12 and 14; the SITL tests cover 13).

Every test gets its own migrated ``searchdrone_test_*`` database. The AI is played by
``helpers/script_driver.py`` on ``/commander/v1``; the pilot by the panel API.
"""

from __future__ import annotations

import asyncio
import re
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))

from flow_rig import VID, H, Rig, rig  # pyright: ignore[reportMissingImports]
from script_driver import ScriptDriver  # pyright: ignore[reportMissingImports]

from sd_mav.mission import FENCE
from sd_schemas.base import new_id
from sd_schemas.executor import ExecutorState

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
async def r(migrated_db: Db, tmp_path: Path) -> AsyncIterator[Rig]:
    async with rig(migrated_db.name, migrated_db.app_password, tmp_path) as built:
        yield built


def driver(r: Rig, mission_id: str) -> ScriptDriver:
    p1 = r.at(60.0, 40.0)
    return ScriptDriver(
        r.commander,
        mission_id,
        {"p1": p1, "box": r.square(150.0), "small": r.square(80.0)},
    )


async def take_off(r: Rig, mid: str, d: ScriptDriver) -> dict[str, Any]:
    res = await r.clearance(mid)
    assert res.status_code == 200, res.text
    await r.until(lambda: r.onboard.clearances, 5, "clearance onboard")
    res = await d.order("takeoff", {"alt_m": 40})
    assert res.status_code == 201, res.text
    out = res.json()
    assert out["decision"] == "ALLOW", out
    r.fly(40.0)
    await r.until(lambda: r.flow.in_flight(VID), 5, "in the air for the ground")
    return out


# --- 1: the states ---------------------------------------------------------------------


async def test_t19_whole_mission(r: Rig) -> None:
    """DRAFT -> CHARTER_PROPOSED -> READY -> ACTIVE -> PAUSED -> ACTIVE -> RETURNING ->
    LANDED -> CLOSED, with the files of the mission at the end."""
    mid = await r.open_mission()
    assert await r.status(mid) == "DRAFT"
    d = driver(r, mid)
    res = await d.charter("box")
    assert res.status_code == 201, res.text
    proposal = res.json()
    assert proposal["validation"]["ok"] is True
    assert await r.status(mid) == "CHARTER_PROPOSED"
    res = await r.approve(proposal["approval_id"], 1)
    assert res.status_code == 200, res.text
    await r.wait_status(mid, "READY", 30)
    await r.wait_snapshot(ExecutorState.PREFLIGHT)
    assert len(r.fc.missions[FENCE]) >= 3  # the polygon is in the flight controller
    charter = (await r.mission(mid))["charter"]
    assert charter["fence_verified_at"] is not None and charter["install_status"] == "installed"
    assert charter["body"]["autonomy_mode"] == "WITHIN_CHARTER"

    await take_off(r, mid, d)
    await r.wait_status(mid, "ACTIVE")
    assert r.onboard.orders[-1].skill == "takeoff"

    stop = await r.panel.post(
        f"/api/v1/vehicles/{VID}/actions/stop", json={"client_action_id": str(new_id())}, headers=H
    )
    assert stop.status_code == 202, stop.text
    await r.wait_status(mid, "PAUSED")
    assert (await r.mission(mid))["mission"]["paused_by"] == "operator"

    res = await r.panel.post(
        f"/api/v1/vehicles/{VID}/actions/resume",
        json={"client_action_id": str(new_id())},
        headers=H,
    )
    assert res.status_code == 202, res.text
    await r.wait_status(mid, "ACTIVE")

    r.onboard.set_state(ExecutorState.TRANSIT)
    await asyncio.sleep(3.1)  # the gate's rate limit: one AI action per 3 s
    res = await d.order("return_home")
    assert res.status_code == 201 and res.json()["decision"] == "ALLOW", res.text
    await r.wait_status(mid, "RETURNING")
    assert (await r.mission(mid))["mission"]["returning_cause"] == "brain"

    r.land()
    await r.wait_status(mid, "LANDED", 30)
    await r.wait_status(mid, "CLOSED", 20)
    files = await r.panel.get(f"/api/v1/missions/{mid}/files")
    assert files.status_code == 200
    names = {f["rel_path"]: f for f in files.json()}
    assert {"mission.json", "events.jsonl", "exports/track.geojson"} <= set(names)
    assert all(re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) for f in names.values())
    one = names["mission.json"]
    got = await r.panel.get(f"/api/v1/missions/{mid}/files/{one['id']}")
    assert got.status_code == 200 and b'"charters"' in got.content


async def test_forbidden_transitions_are_409(r: Rig) -> None:
    mid = await r.ready_mission()
    d = driver(r, mid)
    await take_off(r, mid, d)
    await r.wait_status(mid, "ACTIVE")
    end = await r.panel.post(f"/api/v1/missions/{mid}/end", json={"reason": "x"}, headers=H)
    assert end.status_code == 409 and end.json()["code"] == "MISSION_IN_FLIGHT"
    r.land()
    await r.wait_status(mid, "LANDED", 30)
    res = await d.charter("small")
    assert res.status_code == 409 and res.json()["code"] == "MISSION_STATE_INVALID"
    end = await r.panel.post(f"/api/v1/missions/{mid}/end", json={"reason": "klart"}, headers=H)
    assert end.status_code == 200 and end.json()["status"] == "CLOSED"
    chat = await r.panel.post(
        f"/api/v1/missions/{mid}/chat",
        json={"client_msg_id": str(new_id()), "text": "hej"},
        headers=H,
    )
    assert chat.status_code == 409 and chat.json()["code"] == "MISSION_CLOSED"
    order = await d.order("report", {"text_sv": "hej"})
    assert order.status_code == 409 and order.json()["code"] == "MISSION_CLOSED"


# --- 2: the charter --------------------------------------------------------------------


async def test_charter_only_by_the_pilots_click(r: Rig) -> None:
    mid = await r.open_mission()
    d = driver(r, mid)
    proposal = (await d.charter("box")).json()
    approval = proposal["approval_id"]
    # The AI has no route for a decision, and the commander token is refused on the panel.
    res = await r.commander.post(
        f"/commander/v1/approvals/{approval}/decision",
        json={"decision": "approve", "subject_version": 1},
    )
    assert res.status_code == 404
    res = await r.panel.post(
        f"/api/v1/approvals/{approval}/decision",
        json={"decision": "approve", "subject_version": 1},
        headers={**H, "Authorization": "Bearer brain-token-for-flow-tests-0123456789ab"},
    )
    assert res.status_code in (401, 403)
    stale = await r.approve(approval, 7)
    assert stale.status_code == 409 and stale.json()["code"] == "SUPERSEDED"
    ok = await r.approve(approval, 1, autonomy_mode="APPROVE_EACH")
    assert ok.status_code == 200 and ok.json()["status"] == "approved"
    detail = await r.mission(mid)
    assert detail["mission"]["autonomy_mode"] == "APPROVE_EACH"
    assert detail["charter"]["approved_by"] == "operator:ci-harness"


async def test_charter_validation_and_forbidden_fields(r: Rig) -> None:
    mid = await r.open_mission()
    d = driver(r, mid)
    far = await d.charter(r.square(100, east=900))
    assert far.status_code == 422
    codes = {e["code"] for e in far.json()["validation"]["errors"]}
    assert {"HOME_OUTSIDE_AREA", "AREA_BEYOND_RANGE"} <= codes
    for field, code in (
        ("autonomy_mode", "AUTONOMY_CHANGE_FORBIDDEN"),
        ("limits", "LIMITS_CHANGE_FORBIDDEN"),
    ):
        body = {"area": r.square(100), "reason_sv": "x", field: "WITHIN_CHARTER"}
        res = await r.commander.post(f"/commander/v1/missions/{mid}/charter-proposals", json=body)
        assert res.status_code == 422 and res.json()["code"] == code
    assert await r.status(mid) == "DRAFT"  # nothing was shown


# --- 3: the fence ----------------------------------------------------------------------


async def test_fence_params_wrong_blocks_takeoff(r: Rig) -> None:
    r.fc.params["FENCE_ACTION"] = 0  # someone changed the flight controller
    mid = await r.open_mission()
    d = driver(r, mid)
    proposal = (await d.charter("box")).json()
    assert (await r.approve(proposal["approval_id"], 1)).status_code == 200
    await r.until(lambda: "alert" in r.bus_types(), 10, "alarm")
    await asyncio.sleep(1.0)
    detail = await r.mission(mid)
    assert detail["charter"]["fence_verified_at"] is None
    assert detail["mission"]["status"] != "READY"
    assert r.fc.params["FENCE_ACTION"] == 0  # read, never written
    events = await r.panel.get(f"/api/v1/missions/{mid}/events?type=fc.safety_params")
    assert events.json(), "fc.safety_params event"
    await r.clearance(mid)
    res = await d.order("takeoff", {"alt_m": 40})
    assert "FENCE_NOT_VERIFIED" in res.json()["reason_codes"]


async def test_raised_limits_move_the_fence(r: Rig) -> None:
    mid = await r.open_mission()
    res = await r.panel.put(
        f"/api/v1/missions/{mid}/limits",
        json={
            "max_alt_agl_m": 200,
            "max_speed_ms": 20,
            "max_range_from_pilot_m": 1000,
            "battery_rtl_pct": 20,
        },
        headers=H,
    )
    assert res.status_code == 200, res.text
    too_high = await r.panel.put(
        f"/api/v1/missions/{mid}/limits",
        json={
            "max_alt_agl_m": 400,
            "max_speed_ms": 15,
            "max_range_from_pilot_m": 500,
            "battery_rtl_pct": 15,
        },
        headers=H,
    )
    assert too_high.status_code == 422 and too_high.json()["code"] == "LIMIT_ABOVE_ABSOLUTE"
    # The AI has no way to change limits.
    res = await r.commander.put(f"/commander/v1/missions/{mid}/limits", json={})
    assert res.status_code == 404
    d = driver(r, mid)
    proposal = (await d.charter("box", alt_max_agl_m=180)).json()
    assert (await r.approve(proposal["approval_id"], 1)).status_code == 200
    await r.wait_status(mid, "READY", 30)
    assert r.fc.params["FENCE_ALT_MAX"] == 205
    assert r.fc.params["FENCE_RADIUS"] == 1030
    assert r.fc.params["LOIT_SPEED_MS"] == 20
    assert r.fc.params["FENCE_ENABLE"] == 1 and r.fc.params["FENCE_ACTION"] == 3


# --- 4, 5, 6, 7: clearance, gate decisions, cards, autonomy -----------------------------


async def test_clearance_is_per_takeoff_and_logged(r: Rig) -> None:
    mid = await r.ready_mission()
    d = driver(r, mid)
    no = await d.order("takeoff", {"alt_m": 40})
    assert "CLEARANCE_REQUIRED" in no.json()["reason_codes"]
    out = await take_off(r, mid, d)
    rows = await r.panel.get(f"/api/v1/missions/{mid}/events?type=clearance.confirmed")
    assert rows.json() and rows.json()[0]["payload"]["by"] == "operator:ci-harness"
    gate = await r.panel.get(
        f"/api/v1/missions/{mid}/gate_decisions?command_id={out['command_id']}"
    )
    assert [g["decision"] for g in gate.json()] == ["ALLOW"]
    detail = await r.mission(mid)
    assert detail["clearance"] is None  # consumed by the take-off


async def test_command_card_lifecycle(r: Rig) -> None:
    mid = await r.ready_mission()
    put = await r.panel.put(
        f"/api/v1/missions/{mid}/autonomy", json={"mode": "APPROVE_EACH"}, headers=H
    )
    assert put.status_code == 200 and put.json()["autonomy_mode"] == "APPROVE_EACH"
    assert (
        await r.commander.put(f"/commander/v1/missions/{mid}/autonomy", json={})
    ).status_code == 404
    d = driver(r, mid)
    await r.clearance(mid)
    res = await d.order("takeoff", {"alt_m": 40})
    out = res.json()
    assert out["decision"] == "PENDING_APPROVAL" and out["status"] == "pending_approval", out
    approval = out["approval_id"]
    stale = await r.approve(approval, 2)
    assert stale.status_code == 409 and stale.json()["code"] == "SUPERSEDED"
    rej = await r.panel.post(
        f"/api/v1/approvals/{approval}/decision",
        json={"decision": "reject", "subject_version": 1},
        headers=H,
    )
    assert rej.status_code == 200
    cmd = await r.panel.get(f"/api/v1/missions/{mid}/commands")
    assert cmd.json()[-1]["status"] == "rejected_by_operator"
    # A second card times out.
    await asyncio.sleep(3.1)  # the gate's rate limit
    out = (await d.order("takeoff", {"alt_m": 40})).json()
    assert out["decision"] == "PENDING_APPROVAL", out
    r.clock.skew_s = 61
    await r.until(
        lambda: _status_of(r, mid, out["command_id"], "approval_expired"), 5, "card expired"
    )
    r.clock.skew_s = 0
    # A third is approved: the gate runs again, the order goes out with its hash.
    await asyncio.sleep(3.1)
    out = (await d.order("takeoff", {"alt_m": 40})).json()
    ok = await r.approve(out["approval_id"], 1)
    assert ok.status_code == 200 and ok.json()["status"] == "approved", ok.text
    await r.until(lambda: r.onboard.orders, 5, "order onboard")
    sent = r.onboard.orders[-1]
    assert sent.approval is not None and str(sent.approval.approval_id) == out["approval_id"]


async def _status_of(r: Rig, mid: str, command_id: str, want: str) -> bool:
    cmds = (await r.panel.get(f"/api/v1/missions/{mid}/commands")).json()
    return any(c["id"] == command_id and c["status"] == want for c in cmds)


async def test_autonomy_change_invalidates_clearance(r: Rig) -> None:
    mid = await r.ready_mission()
    await r.clearance(mid)
    assert (await r.mission(mid))["clearance"] is not None
    put = await r.panel.put(
        f"/api/v1/missions/{mid}/autonomy", json={"mode": "APPROVE_EACH"}, headers=H
    )
    assert put.status_code == 200
    detail = await r.mission(mid)
    assert detail["clearance"] is None
    assert detail["charter"]["version"] == 2 and detail["charter"]["fence_verified_at"] is not None


# --- 8, 9: pilot position, ground skills -------------------------------------------------


async def test_pilot_position_is_home_and_ground_skills(r: Rig) -> None:
    mid = await r.ready_mission()
    detail = await r.mission(mid)
    body = detail["charter"]["body"]
    assert body["pilot_position"]["lat"] == pytest.approx(body["home"]["lat"])
    d = driver(r, mid)
    rep = await d.order("report", {"text_sv": "Jag ser ingen här."})
    assert rep.json()["decision"] == "ALLOW" and rep.json()["status"] == "done"
    chat = (await r.panel.get(f"/api/v1/missions/{mid}/chat")).json()["items"]
    assert chat[-1]["role"] == "brain" and chat[-1]["text"] == "Jag ser ingen här."
    lat, lon = r.at(20, 20)
    mark = await d.order("mark_location", {"lat": lat, "lon": lon, "label": "Jacka"})
    assert mark.json()["status"] == "done"
    pins = (await r.panel.get(f"/api/v1/missions/{mid}/annotations")).json()
    assert [p["label"] for p in pins] == ["M1"]
    prop = await d.order(
        "propose_charter_change",
        {"base_version": 1, "area": r.square(100), "reason_sv": "Mindre"},
    )
    out = prop.json()
    assert out["decision"] == "PENDING_APPROVAL" and out["reason_codes"] == ["NEEDS_APPROVAL_SKILL"]
    cards = (await r.panel.get(f"/api/v1/missions/{mid}/approvals?status=pending")).json()
    assert [c["kind"] for c in cards] == ["charter"]


# --- 12: STOP and FLY HOME words in the chat --------------------------------------------


async def _say(r: Rig, mid: str, text: str, msg_id: str | None = None) -> Any:
    res = await r.panel.post(
        f"/api/v1/missions/{mid}/chat",
        json={"client_msg_id": msg_id or str(new_id()), "text": text},
        headers=H,
    )
    assert res.status_code == 201, res.text
    return res.json()


async def test_stop_words_without_the_ai(r: Rig) -> None:
    mid = await r.ready_mission()
    d = driver(r, mid)
    await take_off(r, mid, d)
    await r.wait_status(mid, "ACTIVE")
    assert r.g.brain_state == "down"  # no AI process at all
    before = r.bus_types().count("vehicle.action")
    t0 = asyncio.get_running_loop().time()
    line = await _say(r, mid, "Stopp!")
    assert line["parsed_intent"] == "stop" and line["handled_by"] == "ground"
    await r.until(lambda: r.bus_types().count("vehicle.action") > before, 1.0, "action within 1 s")
    assert asyncio.get_running_loop().time() - t0 < 1.0
    await r.wait_status(mid, "PAUSED")
    plain = await _say(r, mid, "fortsätt")
    assert plain["parsed_intent"] is None
    await asyncio.sleep(0.5)
    assert r.actions.latches.get(VID) is not None  # the chat never releases
    home = await _say(r, mid, "flyg hem")
    assert home["parsed_intent"] == "return_home"
    await r.until(lambda: r.actions.latches.get(VID).action == "return_home", 2, "home latch")


async def test_chat_intent_rules(r: Rig) -> None:
    mid = await r.ready_mission()
    d = driver(r, mid)
    await take_off(r, mid, d)
    await r.wait_status(mid, "ACTIVE")
    url = f"/commander/v1/missions/{mid}/chat-intents"

    def report(message_id: str, intent: str = "stop", engine: str = "rule") -> dict[str, Any]:
        return {
            "request_id": str(new_id()),
            "message_id": message_id,
            "intent": intent,
            "probability": 0.97,
            "engine": engine,
            "location": "local",
            "latency_ms": 120,
        }

    assert (await r.commander.post(url, json=report(str(new_id())))).status_code == 404
    brain_line = await r.commander.post(
        f"/commander/v1/missions/{mid}/messages",
        json={"client_msg_id": str(new_id()), "text": "stopp"},
    )
    bid = brain_line.json()["message_id"]
    res = await r.commander.post(url, json=report(bid))
    assert res.status_code == 403 and res.json()["code"] == "NOT_PILOT_LINE"
    old = await _say(r, mid, "kan du stanna där borta")
    r.clock.skew_s = 11
    res = await r.commander.post(url, json=report(old["id"]))
    assert res.status_code == 409
    r.clock.skew_s = 0
    # A model's reading of "flyg hem" only stops the vehicle where it is (owner decision).
    line = await _say(r, mid, "nu får det räcka, åk tillbaka")
    res = await r.commander.post(url, json=report(line["id"], "return_home", "local"))
    assert res.status_code == 202, res.text
    await r.until(lambda: r.actions.latches.get(VID) is not None, 2, "latch")
    assert r.actions.latches.get(VID).action == "stop"  # type: ignore[union-attr]
    chat = (await r.panel.get(f"/api/v1/missions/{mid}/chat")).json()["items"]
    assert chat[-1]["role"] == "system" and "står still" in chat[-1]["text"]


async def test_same_line_acts_once(r: Rig) -> None:
    mid = await r.ready_mission()
    d = driver(r, mid)
    await take_off(r, mid, d)
    await r.wait_status(mid, "ACTIVE")
    msg = str(new_id())
    await _say(r, mid, "stopp", msg)
    upper = msg.upper()  # the same id written differently must not act again
    res = await r.commander.post(
        f"/commander/v1/missions/{mid}/chat-intents",
        json={
            "request_id": str(new_id()),
            "message_id": upper,
            "intent": "stop",
            "probability": 1.0,
            "engine": "rule",
            "location": "local",
            "latency_ms": 5,
        },
    )
    assert res.status_code == 202, res.text
    await asyncio.sleep(0.5)
    stops = [e for e in r.g.bus.replay(r.g.bus.server_epoch, 0) or [] if e.type == "latch.set"]
    assert len(stops) == 1


# --- 14: no automatic approval anywhere ---------------------------------------------------


_AUTO = re.compile(r"auto_?(approve|confirm|clear)|skip_?(clearance|approval|gate)", re.IGNORECASE)


def test_no_auto_approval_paths() -> None:
    roots = [
        ROOT / "software/ground/src",
        ROOT / "software/common/safety/src",
        ROOT / "software/brain/src",
        ROOT / "software/onboard/src",
        ROOT / "software/common/schemas/src",
    ]
    hits = []
    for root in roots:
        for path in root.rglob("*.py"):
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if _AUTO.search(line):
                    hits.append(f"{path.relative_to(ROOT)}:{n}")
    for path in (ROOT / "config").glob("*.example.toml"):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _AUTO.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{n}")
    assert hits == []
