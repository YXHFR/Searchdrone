"""Searching an area: the mission, the resume arithmetic and hold/resume coverage."""

from __future__ import annotations

from dataclasses import replace

from onboard_helpers import PLANE, at, charter, rig, square

from sd_geo import coverage_fraction
from sd_mav.mission import MISSION, MissionItem
from sd_mav.modes import CopterMode
from sd_onboard.skills.survey_sector import next_index, progress_pct
from sd_schemas.base import new_id
from sd_schemas.charter import Sector
from sd_schemas.executor import ExecutorState as S
from sd_schemas.link import OperatorRelease

BOX = square(40, east=80, north=40)


def test_resume_index_math() -> None:
    # Original mission: seq 2 is waypoint 0.
    assert next_index(2, 0, 0) == 0
    assert next_index(5, 0, 0) == 3
    # Resumed at waypoint 4: seq 2 is interrupted_at, seq 3 is waypoint 4.
    assert next_index(2, 4, 1) == 4
    assert next_index(3, 4, 1) == 4
    assert next_index(6, 4, 1) == 7
    assert progress_pct(0, 8) == 0 and progress_pct(1, 8) == 0
    assert round(progress_pct(4, 8), 1) == 42.9 and progress_pct(8, 8) == 100


async def test_survey_mission_shape_and_hold_resume_coverage() -> None:
    async with rig() as r:
        await r.takeoff(40)
        start = len(r.fc.track)
        cmd = await r.send(
            r.order("survey_sector", {"polygon": BOX, "spacing_m": 20, "speed_ms": 6}), 1
        )
        assert r.ex.state == S.SURVEY
        items = r.fc.missions[MISSION]
        assert [i.command for i in items[:2]] == [16, 178] and items[1].param2 == 6
        assert [i.command for i in items[-2:]] == [19, 20] and items[-2].param1 == 30
        assert r.fc.mission_current >= 1
        await r.run(25)
        hold = await r.send(r.order("hold", {}))
        assert r.ex.state == S.HOLD and r.up.stages(cmd.command_id)[-1] == "superseded"
        rp = r.ex.resume_point
        assert rp is not None and rp.kind == "survey" and rp.next_index > 0
        assert r.up.stages(hold.command_id)[-1] in ("running", "done")
        await r.run(30)
        assert r.fc.mode == CopterMode.LOITER
        resume = await r.send(r.order("resume", {}), 1)
        assert r.ex.state == S.SURVEY
        items = r.fc.missions[MISSION]
        assert (items[2].x, items[2].y) == (
            round(rp.interrupted_at.lat * 1e7),
            round(rp.interrupted_at.lon * 1e7),
        )
        await r.until(lambda: r.ex.state == S.HOLD, 300, step=0.05)
        assert r.up.stages(resume.command_id)[-1] == "done"
        assert r.ex.resume_point is None  # done: nothing left to resume
        corners = [PLANE.xy(lat, lon) for lon, lat in BOX["coordinates"][0][:-1]]
        track = [(*PLANE.xy(la, lo), alt) for la, lo, alt in r.fc.track[start:]]
        assert coverage_fraction(corners, track, 40, 20) >= 0.95


async def test_survey_by_sector_id() -> None:
    sector = Sector(sector_id="A", name="A", area=BOX, order_idx=0)
    async with rig() as r:
        await r.install(charter(sectors=[sector]))
        await r.takeoff()
        cmd = await r.send(r.order("survey_sector", {"sector_id": "A"}), 1)
        assert r.ex.state == S.SURVEY, r.up.acks[-2:]
        unknown = await r.send(r.order("survey_sector", {"sector_id": "B"}), 0.5)
        assert r.up.codes(unknown.command_id) == ("ONB_UNKNOWN_SECTOR",)
        assert "running" in r.up.stages(cmd.command_id)


async def test_survey_other_pattern_not_implemented() -> None:
    async with rig() as r:
        await r.takeoff()
        cmd = await r.send(r.order("survey_sector", {"polygon": BOX, "pattern": "spiral"}))
        assert r.up.codes(cmd.command_id) == ("SKILL_NOT_IMPLEMENTED",)


async def test_survey_too_large() -> None:
    async with rig() as r:
        await r.install(charter(area=square(300)))
        await r.takeoff()
        cmd = await r.send(r.order("survey_sector", {"polygon": square(290), "spacing_m": 10}))
        assert r.up.codes(cmd.command_id) == ("ONB_SURVEY_TOO_LARGE",)


async def test_mission_readback_mismatch_stays_in_loiter() -> None:
    async with rig() as r:
        await r.takeoff()
        store = r.fc._store

        def keep_something_else(item: MissionItem) -> MissionItem:
            kept = store(item)
            return replace(kept, x=kept.x + 100) if item.seq == 3 else kept

        r.fc._store = keep_something_else  # type: ignore[method-assign]
        cmd = await r.send(r.order("survey_sector", {"polygon": BOX}), 3)
        assert r.up.codes(cmd.command_id) == ("ONB_MISSION_READBACK_MISMATCH",)
        assert r.ex.state == S.HOLD
        assert r.fc.mode == CopterMode.LOITER  # never switched to AUTO


async def test_resume_point_lifetime() -> None:
    async with rig() as r:
        await r.takeoff()
        lat, lon = at(120, 0)
        await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 5}))
        await r.run(3)
        await r.send(r.order("hold", {}))
        assert r.ex.resume_point is not None and r.ex.resume_point.kind == "goto"
        r.ex.on_override(r.notice("stop"))  # kept through STOP
        assert r.ex.resume_point is not None
        r.fc.set_mode_external(CopterMode.BRAKE)
        await r.run(1)
        r.ex.on_release(OperatorRelease(action_seq=r._seq, operator_id="operator:pilot"), new_id())
        await r.run(2)
        lat, lon = at(0, 60)
        await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40}))
        assert r.ex.resume_point is None  # a new move replaces it
        await r.send(r.order("return_home"))
        await r.until(lambda: r.ex.state == S.LANDED, 200, step=0.05)
        assert r.ex.resume_point is None
