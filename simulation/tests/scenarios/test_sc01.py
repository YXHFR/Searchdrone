"""SC01, the demo (D1-D10) end to end, and the pilot's manual takeover, on SITL.

* SC01 (``just e2e``): the whole stack as ``just demo`` starts it, as separate programs
  (``stack_rig``), driven only through the panel API like the operator: the pilot asks
  for the flight in the chat, the AI proposes the charter (a card within 2 s), the pilot
  approves it and confirms "clear around the drone" (exactly two clicks), the AI takes
  off to 40 m, flies to P1, searches the 100 x 100 m box there (coverage >= 95 % in
  PostGIS), is refused 200 m by the gate mid-search (the only refusal; the height stays
  40 m), flies home and lands within 3 m; the mission closes by itself. Afterwards the
  database (F-08) and the mission folder (F-09) are checked.
* SC01_takeover: mid-search the pilot takes over, moves the drone 20 m with the sticks,
  lets go (the ground station stops sending after 1 s: the drone stands in LOITER) and
  hands back; the AI continues the search from where it stopped.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import itertools
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
from pymavlink import mavutil

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario_rig import (  # pyright: ignore[reportMissingImports]
    FIXTURE,
    fresh_db,
    migrated_db,
    passwords,
    pin,
    scenario,
    sim,
    survey_box_enu,
)
from stack_rig import VID, Pilot, live_stack  # pyright: ignore[reportMissingImports]

from sd_safety.targets import Plane, distance_m
from sd_sim.invariants import check_all
from sd_sim.mp4 import probe

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.brain]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]

ALT_M = 40.0
SPACING_M = 30.0
MIN_COVERAGE = 0.95
SURVEY_LIMIT_S = 240.0
CARD_LIMIT_S = 2.0  # the scripted commander's charter card (08 D3)
D7_AFTER_S = 5.0  # into the search before "climb to 200 m" (past the AI's rate limit)
DEMO_LIMIT_S = 600.0  # D5-D9
CLOSE_LIMIT_S = 30.0  # LANDED -> CLOSED 10 s after disarming, plus the folder
CMAC = FIXTURE["home"]
D3_TEXT = (
    "Flyg till P1 och sök av en ruta på 100 gånger 100 meter där, på 40 meters höjd. "
    "Landa hemma sen."
)
D7_TEXT = "Gå upp till 200 meter."


def pin_at(east: float, north: float) -> tuple[float, float]:
    return Plane(float(CMAC["lat"]), float(CMAC["lon"])).latlon(east, north)


@pytest.mark.e2e
@pytest.mark.budget_s(900)
async def test_sc01_demo_end_to_end() -> None:
    """The demo D1-D10 on the whole stack started as ``just demo`` starts it."""
    async with live_stack("sc01-demo", "SC01") as p:
        # D1: every part ready (the launcher waited for it; the health says so too).
        health = await p.get("/api/v1/health")
        assert health["ready"] is True, health
        for part in ("db", "mavlink", "sdlink", "onboard", "brain"):
            assert health["parts"][part] == "ok", health
        # D2: the panel's built files are served on the panel port.
        index = await p.http.get("/")
        assert index.status_code == 200 and "<html" in index.text.lower()

        # D3: the pilot's request; the AI's charter card within 2 s.
        mid = await p.open_mission()
        await p.say(mid, D3_TEXT)
        cards: list[dict[str, Any]] = []

        async def card() -> bool:
            pending = await p.get(f"/api/v1/missions/{mid}/approvals", status="pending")
            cards[:] = [a for a in pending if a["kind"] == "charter"]
            return bool(cards)

        took = await p.until(card, 10, "the AI's charter card")
        assert took <= CARD_LIMIT_S, f"the charter card took {took:.1f} s"
        assert await p.status(mid) == "CHARTER_PROPOSED"

        # D4: click 1, the charter (mode B is the default).
        await p.post(
            f"/api/v1/approvals/{cards[0]['id']}/decision",
            {"decision": "approve", "subject_version": cards[0]["subject_version"]},
        )
        await p.until(lambda: _is(p, mid, "READY"), 90, "READY")
        detail = await p.get(f"/api/v1/missions/{mid}")
        charter = detail["charter"]
        assert charter["approved_at"] and charter["approved_by"].startswith("operator:")
        assert detail["mission"]["autonomy_mode"] == "WITHIN_CHARTER"  # the charter's copy
        assert charter["fence_verified_at"], charter
        await p.until(lambda: _preflight_ok(p), 90, "onboard preflight")

        # D5: click 2, "clear around the drone": the AI's take-off (a rule, no model).
        await p.post(f"/api/v1/missions/{mid}/clearance", {})
        takeoff = await p.command(mid, "takeoff", ("accepted", "running", "done"), 30)
        assert takeoff["issuer_kind"] == "brain", takeoff
        t_takeoff = time.monotonic()
        await p.until(lambda: abs(p.telemetry["alt_rel_home_m"] - ALT_M) <= 2.0, 30, "40 m")
        await p.command(mid, "takeoff", ("done",), 30)

        # D6: at P1 and standing still for 2 s (in the track: the AI reports and starts
        # the search 3 s after the goto's end).
        await p.command(mid, "goto", ("done",), 120)
        p1 = pin("P1")
        await p.until(lambda: _still_at(p.track, p1, 5.0, 2.0), 6, "still within 5 m of P1 for 2 s")

        # D7 mid-search: "climb to 200 m" is refused by the gate, the height stays.
        await p.command(mid, "survey_sector", ("running",), 60)
        survey_from = dt.datetime.now(dt.UTC)
        t_survey = time.monotonic()
        await asyncio.sleep(D7_AFTER_S)
        await p.say(mid, D7_TEXT)

        async def refused() -> bool:
            return any(g["decision"] == "REJECT" for g in await p.gate(mid))

        await p.until(refused, 15, "the gate's refusal of 200 m")

        async def explained() -> bool:
            return any(m["role"] == "brain" and "Grinden" in m["text"] for m in await p.chat(mid))

        await p.until(explained, 30, "the refusal explained in Swedish in the chat")
        await p.command(mid, "survey_sector", ("done",), SURVEY_LIMIT_S)
        survey_to = dt.datetime.now(dt.UTC)
        t_survey_done = time.monotonic()
        assert t_survey_done - t_survey <= SURVEY_LIMIT_S

        # D9: home, landed, disarmed; D5-D9 in at most 10 min.
        await p.command(mid, "return_home", ("done",), 300)
        await p.until(lambda: not p.telemetry["armed"], 120, "disarmed after landing")
        t_disarmed = time.monotonic()
        assert t_disarmed - t_takeoff <= DEMO_LIMIT_S, f"D5-D9 took {t_disarmed - t_takeoff:.0f} s"
        home = (float(CMAC["lat"]), float(CMAC["lon"]))
        last = p.telemetry
        assert distance_m(*home, last["lat"], last["lon"]) <= 3.0
        await p.until(lambda: _is(p, mid, "CLOSED"), CLOSE_LIMIT_S, "CLOSED after disarming")

        # The orders and the gate: D7's refusal is the only one, and nothing over the
        # charter's ceiling went on board.
        cmds = await p.commands(mid)
        gates = await p.gate(mid)
        rejects = [g for g in gates if g["decision"] == "REJECT"]
        assert len(rejects) == 1, rejects
        assert "ALT_ABOVE_CHARTER" in rejects[0]["reason_codes"], rejects[0]
        assert rejects[0]["message_sv"], rejects[0]
        too_high = [
            c
            for c in cmds
            if float(c["params"].get("alt_m", 0)) > float(charter["body"]["alt_max_agl_m"])
        ]
        assert [c["status"] for c in too_high] == ["rejected_by_gate"], too_high
        assert {c["skill"] for c in cmds} >= {"takeoff", "goto", "survey_sector", "return_home"}

        # The track: 40 +- 2 m from P1 to the end of the search (D7 changed nothing), and
        # the invariants over the whole flight.
        flying = [s for s in p.track if t_survey - D7_AFTER_S <= s.t <= t_survey_done]
        assert flying and all(abs(s.alt_m - ALT_M) <= 2.0 for s in flying), (
            min(s.alt_m for s in flying),
            max(s.alt_m for s in flying),
        )
        charters = await p.get(f"/api/v1/missions/{mid}/charters")
        events = await p.events(mid)
        by_cmd = {
            c["id"]: await p.get(f"/api/v1/missions/{mid}/gate_decisions", command_id=c["id"])
            for c in cmds
        }
        broken = check_all(p.track, charters, cmds, by_cmd, events)
        assert not broken, broken

        # Exactly two clicks by the operator: the charter and "clear around the drone".
        clicks = [e for e in events if e["type"] in ("charter.approved", "clearance.confirmed")]
        assert sorted(e["type"] for e in clicks) == ["charter.approved", "clearance.confirmed"]
        decided = [
            a for a in await p.get(f"/api/v1/missions/{mid}/approvals") if a["status"] != "pending"
        ]
        assert len(decided) == 1 and decided[0]["kind"] == "charter", decided
        states = [e["payload"]["to"] for e in events if e["type"] == "mission.state_changed"]
        assert states[-3:] == ["RETURNING", "LANDED", "CLOSED"], states

        # D10: the API gives the mission with its charter, chat, orders, gate, events, track.
        chat = await p.chat(mid)
        operator_lines = [m for m in chat if m["role"] == "operator"]
        assert len(operator_lines) == 2 and len([m for m in chat if m["role"] == "brain"]) >= 2
        assert any("Uppdraget är klart." in m["text"] for m in chat)
        track = await p.get(f"/api/v1/missions/{mid}/track")
        assert track, "no track"
        await _database_checks(p, mid, charter, survey_from, survey_to)
        await _mission_folder_checks(p, mid)


async def _is(p: Pilot, mid: str, status: str) -> bool:
    return await p.status(mid) == status


async def _preflight_ok(p: Pilot) -> bool:
    live = await p.get(f"/api/v1/vehicles/{VID}/state")
    snap = live.get("snapshot")
    return snap is not None and bool(snap.get("preflight_ok"))


def _still_at(track: list[Any], at: tuple[float, float], radius_m: float, hold_s: float) -> bool:
    """Some stretch of ``hold_s`` seconds in which every sample is within ``radius_m``
    of ``at`` and slower than 0.5 m/s."""
    start: float | None = None
    for s in track:
        ok = distance_m(*at, s.lat, s.lon) <= radius_m and s.groundspeed_ms < 0.5
        if not ok:
            start = None
            continue
        start = s.t if start is None else start
        if s.t - start >= hold_s:
            return True
    return False


async def _database_checks(
    p: Pilot, mid: str, charter: dict[str, Any], survey_from: dt.datetime, survey_to: dt.datetime
) -> None:
    """F-08 and D8 in PostGIS: CLOSED; telemetry at 2 Hz without a gap over 3 s from
    take-off to landing; every order has a gate decision; every row of the mission has
    the same vehicle; no armed sample outside the charter's area; coverage >= 95 %."""
    async with p.db() as db:
        uid = uuid.UUID(mid)
        row = await db.fetchrow("SELECT status, vehicle_id FROM missions WHERE id = $1", uid)
        assert row is not None and row["status"] == "CLOSED", row
        for table in (
            "charters",
            "chat_messages",
            "commands",
            "gate_decisions",
            "events",
            "telemetry",
        ):
            got = await db.fetch(
                f"SELECT DISTINCT vehicle_id FROM {table} WHERE mission_id = $1",  # noqa: S608 - fixed names
                uid,
            )
            assert [r["vehicle_id"] for r in got] == [VID], (table, got)
        times = [
            r["t"]
            for r in await db.fetch(
                "SELECT t FROM telemetry WHERE mission_id = $1 AND armed ORDER BY t", uid
            )
        ]
        assert len(times) > 10
        gaps = [(b - a).total_seconds() for a, b in itertools.pairwise(times)]
        span = (times[-1] - times[0]).total_seconds()
        assert max(gaps) <= 3.0, f"telemetry gap {max(gaps):.1f} s"
        assert len(times) / span >= 1.9, f"telemetry {len(times) / span:.2f} Hz"
        unchecked = await db.fetchval(
            "SELECT count(*) FROM commands c WHERE c.mission_id = $1 AND NOT EXISTS "
            "(SELECT 1 FROM gate_decisions g WHERE g.command_id = c.id)",
            uid,
        )
        assert unchecked == 0
        outside = await db.fetchval(
            "SELECT count(*) FROM telemetry t, charters c WHERE t.mission_id = $1 AND t.armed "
            "AND c.mission_id = $1 AND c.status = 'approved' "
            "AND NOT ST_Within(ST_Force2D(t.pos), c.area)",
            uid,
        )
        assert outside == 0, f"{outside} armed samples outside the charter's area"
        covered = await db.fetchval(
            COVERAGE_SQL,
            uid,
            survey_from,
            survey_to,
            ALT_M - 2.0,
            ALT_M + 2.0,
            _box_wkt(),
            SPACING_M / 0.8 / 2,
        )
        assert covered is not None and covered >= MIN_COVERAGE, f"coverage {covered}"


async def _mission_folder_checks(p: Pilot, mid: str) -> None:
    """F-09: flight.tlog, the simulator's *.BIN, events.jsonl and mission.json; with
    MediaMTX running, the recorded video is H.264 and covers the flight but 5 s."""
    folder = p.mission_dir(mid)
    tlog = folder / "flight.tlog"
    assert tlog.is_file() and tlog.stat().st_size > 0, f"{tlog} missing"
    first = mavutil.mavlink_connection(str(tlog)).recv_match(blocking=False)
    assert first is not None, "flight.tlog has no MAVLink message"
    assert list((folder / "dataflash").glob("*.BIN")), f"no *.BIN in {folder}"
    assert (folder / "events.jsonl").stat().st_size > 0
    export = json.loads((folder / "mission.json").read_text(encoding="utf-8"))
    assert str(export["mission"]["id"]) == mid
    if not p.stack.video_on:
        assert os.environ.get("SD_SIM__REQUIRE_VIDEO") != "1", "MediaMTX kör inte"
        return
    mission = (await p.get(f"/api/v1/missions/{mid}"))["mission"]
    flight_s = (
        dt.datetime.fromisoformat(mission["landed_at"])
        - dt.datetime.fromisoformat(mission["activated_at"])
    ).total_seconds()
    videos = sorted((folder / "video").glob("*.mp4"))
    assert videos, f"no video in {folder}"
    tracks = [probe(v).video() for v in videos]
    assert all(t is not None and t.codec == "avc1" for t in tracks), tracks
    length = sum(t.seconds for t in tracks if t is not None)
    assert length >= flight_s - 5.0, f"video {length:.0f} s, flight {flight_s:.0f} s"


def _box_wkt() -> str:
    ring = [pin_at(e, n) for e, n in survey_box_enu()]
    ring.append(ring[0])
    return "POLYGON((" + ", ".join(f"{lon} {lat}" for lat, lon in ring) + "))"


# Covered share of the box in UTM zone 55 S (CMAC): the track flown during the search
# inside the height band, buffered by half the swath ($7: lane spacing / 0.8 / 2).
COVERAGE_SQL = """
WITH pts AS (
  SELECT ST_Transform(ST_Force2D(pos), 32755) AS p, t FROM telemetry
  WHERE mission_id = $1 AND t BETWEEN $2 AND $3 AND alt_rel_home_m BETWEEN $4 AND $5
), line AS (SELECT ST_MakeLine(p ORDER BY t) AS g FROM pts),
box AS (SELECT ST_Transform(ST_GeomFromText($6, 4326), 32755) AS g)
SELECT ST_Area(ST_Intersection(ST_Buffer(line.g, $7), box.g)) / ST_Area(box.g)
FROM line, box
"""


@pytest.mark.budget_s(660)
async def test_sc01_takeover_on_sitl(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne_with_ai("SC01_takeover", behaviour={"idle_airborne_tick_s": 5})
        await sc.wait_command("survey_sector", "running", 120, issuer_kind="brain")
        await asyncio.sleep(20.0)  # some of the search flown

        res = await sc.press("takeover")
        assert res.status_code == 202, res.text
        takeover_id = res.json()["action_id"]
        await sc.until(lambda: sc.mode() == "LOITER", 3, "LOITER after the takeover")
        await sc.wait_command("survey_sector", "superseded", 10, issuer_kind="brain")
        # The notice reached the onboard side before the ground's LOITER: it is the
        # pilot's take-over there, not a stop nobody asked for (no stop latch).
        state = await sc.onboard_state()
        for _ in range(10):
            if state["held_by"] == "operator_manual":
                break
            await asyncio.sleep(0.5)
            state = await sc.onboard_state()
        assert state["held_by"] == "operator_manual" and not state["stop_latch"], state

        # The pilot pushes the stick forward at 10 Hz until the drone has moved 20 m.
        start = (sc.tel.lat, sc.tel.lon)
        moved = 0.0
        t0 = time.monotonic()
        while moved < 20.0:
            assert time.monotonic() - t0 < 30, f"the sticks moved the drone only {moved:.1f} m"
            assert await sc.sticks(pitch=0.6, roll=0.0, takeover_id=takeover_id)
            await asyncio.sleep(0.1)
            moved = distance_m(*start, sc.tel.lat, sc.tel.lon)
        # Let go: no more values; after 1 s the ground station stops sending and the
        # drone stands still in LOITER.
        await sc.until(lambda: sc.tel.groundspeed_ms < 0.5, 10, "standing after letting go")
        here = (sc.tel.lat, sc.tel.lon)
        await sc.hold_for(
            5.0,
            lambda: sc.mode() == "LOITER" and distance_m(*here, sc.tel.lat, sc.tel.lon) < 2.0,
            "stands in LOITER without stick values",
        )
        alt = sc.tel.alt_rel_home_m
        assert abs(alt - ALT_M) <= 2.0  # the flight controller held the height
        # No AI order while the pilot steers.
        assert not [
            c for c in await sc.commands() if c["issuer_kind"] == "brain" and c["skill"] == "resume"
        ]

        res = await sc.press("handback")
        assert res.status_code == 202, res.text
        # The AI's resume carries the search on from where it stopped, to its end.
        await sc.wait_command("resume", "done", 30 + SURVEY_LIMIT_S, issuer_kind="brain")
        await sc.wait_command("return_home", "done", 240, issuer_kind="brain")
        await sc.until(lambda: not sc.tel.armed, 60, "disarmed after landing")
        assert sc.from_home_m() <= 3.0
        await sc.check_invariants()
