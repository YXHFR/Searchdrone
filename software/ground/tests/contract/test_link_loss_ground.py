"""Link loss on the ground: the confirmed finds' keep-out list for the onboard computer,
and the forecast the panel gets while SD-link is down (``services/link_loss.py``).

A fake onboard computer over the real SD-link server and a migrated database
(``helpers/flow_rig.py``); the candidates are written as the search service writes them
(append-only, a new row per revision).
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))

from flow_rig import VID, Rig, rig  # pyright: ignore[reportMissingImports]

from sd_ground.services import link_loss
from sd_safety.constants import PERSON_KEEPOUT_M
from sd_safety.link_loss import FindPlan, choose_mode
from sd_schemas.executor import ExecutorState

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]


@pytest.fixture
async def r(migrated_db: Db, tmp_path: Path) -> AsyncIterator[Rig]:
    async with rig(migrated_db.name, migrated_db.app_password, tmp_path) as built:
        yield built


async def candidate(
    r: Rig, mission: str, lat: float, lon: float, *, state: str, rev: int, cid: str | None = None
) -> str:
    """One revision of a candidate, as sd_app writes it."""
    async with r.flow.repo.engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "INSERT INTO candidates (candidate_id, rev, mission_id, seq, state, geom, "
                    "sigma95_m, n_seen, n_in_view, best_detector_score, priority, "
                    "first_seen_utc, last_seen_utc) VALUES (COALESCE(CAST(:cid AS uuid), "
                    "uuidv7()), :rev, CAST(:m AS uuid), 1, :state, "
                    "ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), 5, 3, 3, 0.8, 0.9, now(), "
                    "now()) RETURNING candidate_id::text"
                ),
                {"cid": cid, "rev": rev, "m": mission, "state": state, "lon": lon, "lat": lat},
            )
        ).one()
    return str(row[0])


def installs(r: Rig) -> list[Any]:
    return list(r.onboard.keepouts)


async def test_keepout_install_confirmed(r: Rig) -> None:
    rig_ = r
    mid = await rig_.ready_mission()
    lat, lon = rig_.at(60.0, 0.0)
    cid = await candidate(rig_, mid, lat, lon, state="ready", rev=1)
    await candidate(rig_, mid, lat, lon, state="confirmed", rev=2, cid=cid)

    def installed(source: str) -> Callable[[], bool]:
        def check() -> bool:
            last = installs(rig_)[-1] if installs(rig_) else None
            return last is not None and [(str(k.source_id), k.source) for k in last.keepouts] == [
                (cid, source)
            ]

        return check

    confirmed = installed("confirmed")

    await rig_.until(confirmed, 10, "the confirmed find installed")
    first = installs(rig_)[-1]
    k = first.keepouts[0]
    assert k.source == "confirmed"
    assert k.radius_m == pytest.approx(PERSON_KEEPOUT_M + 5.0)
    assert (k.lat, k.lon) == pytest.approx((lat, lon))

    # Reopened (a newer revision that is not confirmed): still kept out, as an open
    # candidate with a detector score of at least 0.5, under a higher seq.
    await candidate(rig_, mid, lat, lon, state="inspecting", rev=3, cid=cid)
    await rig_.until(installed("candidate"), 10, "the reopened find kept as a candidate")
    assert installs(rig_)[-1].install_seq > first.install_seq
    # Rejected: an empty list.
    await candidate(rig_, mid, lat, lon, state="rejected", rev=4, cid=cid)
    await rig_.until(lambda: installs(rig_)[-1].keepouts == [], 10, "the rejected find gone")

    # Confirmed again, then a new SD-link session: the list again at its Hello.
    await candidate(rig_, mid, lat, lon, state="confirmed", rev=5, cid=cid)
    await rig_.until(confirmed, 10, "confirmed again")
    before = len(installs(rig_))
    seq = installs(rig_)[-1].install_seq
    await rig_.onboard.close()
    rig_.onboard.restart()
    await rig_.onboard.start(rig_.home)
    await rig_.until(lambda: len(installs(rig_)) > before, 10, "resent at Hello")
    again = installs(rig_)[-1]
    assert again.install_seq >= seq
    assert [str(x.source_id) for x in again.keepouts] == [cid]


async def test_refused_keepout_list_alarms_and_gets_in(r: Rig) -> None:
    """The onboard computer kept a higher ``install_seq`` than the ground's clock gives
    (ONB_STALE_ORDER): the operator gets the red alarm at once, the ground sends the list
    again with a higher number until it is in, and says so."""
    rig_ = r
    rig_.onboard.keepout_seq = int(rig_.flow.now().timestamp() * 1000) + 5 * 3_600_000
    mid = await rig_.ready_mission()
    lat, lon = rig_.at(60.0, 0.0)
    cid = await candidate(rig_, mid, lat, lon, state="confirmed", rev=1)

    def alerted(code: str) -> Callable[[], bool]:
        def check() -> bool:
            events = rig_.g.bus.replay(rig_.g.bus.server_epoch, 0) or []
            return any(e.data.get("code") == code for e in events)

        return check

    await rig_.until(alerted("KEEPOUTS_NOT_INSTALLED"), 10, "the red alarm")

    def in_force() -> bool:
        last = installs(rig_)[-1] if installs(rig_) else None
        return last is not None and [str(k.source_id) for k in last.keepouts] == [cid]

    await rig_.until(in_force, 30, "the list in force on board")
    assert installs(rig_)[-1].install_seq >= rig_.onboard.keepout_seq
    await rig_.until(alerted("KEEPOUTS_INSTALLED"), 10, "the all-clear")
    events = rig_.g.bus.replay(rig_.g.bus.server_epoch, 0) or []
    red = [e for e in events if e.data.get("code") == "KEEPOUTS_NOT_INSTALLED"]
    assert len(red) == 1 and red[0].data["severity"] == "critical"
    assert "förbudszonerna" in red[0].data["message_sv"]


async def test_ground_plan_matches_onboard(r: Rig) -> None:
    """The forecast is ``choose_mode`` on the last snapshot and the installed list: the
    same function and inputs as the onboard rule."""
    rig_ = r
    mid = await rig_.ready_mission()
    lat, lon = rig_.at(70.0, 0.0)
    await candidate(rig_, mid, lat, lon, state="confirmed", rev=1)
    await rig_.until(lambda: any(i.keepouts for i in installs(rig_)), 10, "keep-out installed")
    install = installs(rig_)[-1]
    rig_.fly(40.0)
    rig_.onboard.position = rig_.at(40.0, 0.0)
    rig_.onboard.alt_rel_home_m = 40.0
    rig_.onboard.set_state(ExecutorState.SURVEY)
    await rig_.wait_snapshot(ExecutorState.SURVEY)
    snap, _ = rig_.flow.snapshot(VID)
    assert snap is not None

    await rig_.onboard.close()  # SD-link gone
    await rig_.until(lambda: link_loss.plan_of(rig_.flow, VID) is not None, 10, "forecast")
    plan = link_loss.plan_of(rig_.flow, VID)
    assert plan is not None and plan.mode == "find"
    mission = await rig_.flow.mission(mid)
    charter = await rig_.flow.active_charter(mission)
    assert charter is not None
    onboard = choose_mode(
        state=snap.executor_state,
        lat=snap.position.lat,
        lon=snap.position.lon,
        alt_rel_home_m=snap.alt_rel_home_m,
        nav_ok=True,
        keepouts=install.keepouts,
        charter=charter,
    )
    assert isinstance(onboard, FindPlan)
    assert plan.keepout_id == onboard.keepout_id
    assert (plan.hover_lat, plan.hover_lon) == pytest.approx((onboard.hover_lat, onboard.hover_lon))
    assert plan.candidate_seq == 1
    assert plan.hover_until_utc == plan.home_at_utc
    assert (plan.home_at_utc - plan.lost_at_utc).total_seconds() == pytest.approx(123, abs=0.01)
    live = (await rig_.panel.get(f"/api/v1/vehicles/{VID}/state")).json()
    assert live["link_loss_plan"]["mode"] == "find"
    alerts = [e.data.get("code") for e in rig_.g.bus.replay(rig_.g.bus.server_epoch, 0) or []]
    assert {"FIND_HOLD", "FIND_HOLD_ALL"} & set(alerts)

    # The link is back and the drone hovers: the plan is gone, the pilot is told.
    rig_.onboard.set_state(ExecutorState.HOLD, hold_reason="find_hold")
    rig_.onboard.accept_charters = False  # the fake would go to PREFLIGHT at the resend
    rig_.onboard.restart()
    await rig_.onboard.start(rig_.home)
    await rig_.until(lambda: link_loss.plan_of(rig_.flow, VID) is None, 10, "plan cleared")

    def back() -> bool:
        events = rig_.g.bus.replay(rig_.g.bus.server_epoch, 0) or []
        return any(e.data.get("code") == "LINK_BACK_HOLD" for e in events)

    await rig_.until(back, 10, "LINK_BACK_HOLD")


async def test_far_find_is_the_usual_rule(r: Rig) -> None:
    rig_ = r
    mid = await rig_.ready_mission()
    lat, lon = rig_.at(140.0, 140.0)  # ~200 m away: outside the find radius
    await candidate(rig_, mid, lat, lon, state="confirmed", rev=1)
    await rig_.until(lambda: any(i.keepouts for i in installs(rig_)), 10, "keep-out installed")
    rig_.fly(40.0)
    rig_.onboard.position = rig_.at(0.0, 0.0)
    rig_.onboard.set_state(ExecutorState.SURVEY)
    await rig_.wait_snapshot(ExecutorState.SURVEY)
    await rig_.onboard.close()
    await rig_.until(lambda: link_loss.plan_of(rig_.flow, VID) is not None, 10, "forecast")
    plan = link_loss.plan_of(rig_.flow, VID)
    assert plan is not None and plan.mode == "normal" and plan.hold_at_utc is not None
    assert (plan.hold_at_utc - plan.lost_at_utc).total_seconds() == pytest.approx(23, abs=0.01)
    assert (plan.home_at_utc - plan.lost_at_utc).total_seconds() == pytest.approx(53, abs=0.01)


async def _detection(r: Rig, mission: str, cid: str, lat: float, lon: float, score: float) -> str:
    """A detection of the candidate ``cid`` (a still, the detection, the observation)."""
    point = f"POINT({lon} {lat})"
    square = (
        f"POLYGON(({lon - 1e-4} {lat - 1e-4},{lon + 1e-4} {lat - 1e-4},"
        f"{lon + 1e-4} {lat + 1e-4},{lon - 1e-4} {lat + 1e-4},{lon - 1e-4} {lat - 1e-4}))"
    )
    async with r.flow.repo.engine.begin() as conn:
        still = (
            await conn.execute(
                text(
                    "INSERT INTO perception_stills (mission_id, vehicle_id, still_id, "
                    "captured_utc, sensor, pose, footprint, px_per_m, detector) VALUES "
                    "(CAST(:m AS uuid), :v, uuidv7(), now(), 'rgb', '{}', "
                    "ST_GeomFromText(:sq, 4326), 50, '{}') RETURNING still_id::text"
                ),
                {"m": mission, "v": VID, "sq": square},
            )
        ).one()[0]
        det = (
            await conn.execute(
                text(
                    "INSERT INTO detections (mission_id, vehicle_id, detection_id, still_id, "
                    "score, bbox, geom, sigma95_m, px_size) VALUES (CAST(:m AS uuid), :v, "
                    "uuidv7(), CAST(:s AS uuid), :score, '{}', ST_GeomFromText(:p, 4326), 4.5, "
                    "90) RETURNING detection_id::text"
                ),
                {"m": mission, "v": VID, "s": still, "score": score, "p": point},
            )
        ).one()[0]
        await conn.execute(
            text(
                "INSERT INTO candidate_observations (candidate_id, detection_id, distance_m) "
                "VALUES (CAST(:c AS uuid), CAST(:d AS uuid), 0)"
            ),
            {"c": cid, "d": det},
        )
    return str(det)


async def test_a_rejected_candidate_releases_its_detections(r: Rig) -> None:
    """Owner decision 5 (2026-11-03): a cylinder the drone made from its own detections
    goes when the operator rejects the candidate. The list names the rejected candidates'
    detections at least ``KEEPOUT_SCORE_MIN`` sure (``released``), none while it is open."""
    mid = await r.ready_mission()
    lat, lon = r.at(60.0, 0.0)
    cid = await candidate(r, mid, lat, lon, state="ready", rev=1)
    sure = await _detection(r, mid, cid, lat, lon, 0.6)
    await _detection(r, mid, cid, lat, lon, 0.3)  # never a cylinder on board

    def released() -> list[str]:
        last = installs(r)[-1] if installs(r) else None
        return [] if last is None else [str(d) for d in last.released]

    await r.until(lambda: bool(installs(r)), 10, "a first list")
    assert released() == []  # nothing rejected yet
    await candidate(r, mid, lat, lon, state="rejected", rev=2, cid=cid)
    await r.until(lambda: released() == [sure], 10, "the rejected candidate's detection")
    # Reopened (a newer revision that is not rejected): kept out again, not released.
    await candidate(r, mid, lat, lon, state="inspecting", rev=3, cid=cid)
    await r.until(lambda: released() == [], 10, "reopened: no longer released")
