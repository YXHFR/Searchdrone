"""Following a drawn path: the mission, progress along the line, hover at the end,
the pilot's approval, STOP on the way and resume."""

from __future__ import annotations

import math
from typing import Any

import pytest
from onboard_helpers import PLANE, at, rig

from sd_mav.mission import MISSION
from sd_mav.modes import CopterMode
from sd_onboard.skills.follow_path import path_progress_pct
from sd_schemas.approval import command_hash
from sd_schemas.base import LatLon, new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.executor import SurveyResumePoint
from sd_schemas.link import ApprovalRef
from sd_schemas.skills import FollowPathParams

CORNERS = [(0.0, 60.0), (60.0, 90.0), (120.0, 60.0), (120.0, 0.0), (60.0, -40.0)]


def params(corners: list[tuple[float, float]] = CORNERS, **extra: Any) -> dict[str, Any]:
    points = [{"lat": at(e, n)[0], "lon": at(e, n)[1]} for e, n in corners]
    return {"points": points, "alt_m": 30, "speed_ms": 6, "label": "pa-1", **extra}


def test_progress_along_the_line() -> None:
    pts = [LatLon(lat=at(e, n)[0], lon=at(e, n)[1]) for e, n in [(0, 0), (100, 0), (100, 100)]]
    assert path_progress_pct(PLANE, pts, 0, *at(-50, 0)) == 0  # flying to the first point
    assert path_progress_pct(PLANE, pts, 1, *at(50, 0)) == pytest.approx(25, abs=0.1)
    assert path_progress_pct(PLANE, pts, 2, *at(100, 50)) == pytest.approx(75, abs=0.1)
    assert path_progress_pct(PLANE, pts, 2, *at(100, 300)) == pytest.approx(100, abs=0.1)
    assert path_progress_pct(PLANE, pts, 3, *at(0, 0)) == 100


async def test_flies_every_point_then_hovers_at_the_end() -> None:
    async with rig() as r:
        await r.takeoff(40)
        cmd = await r.send(r.order("follow_path", params()), 1)
        assert r.ex.state == S.TRANSIT, r.up.acks[-2:]
        items = r.fc.missions[MISSION]
        # home, speed, the five points, loiter, return-to-launch
        assert [i.command for i in items] == [16, 178, 16, 16, 16, 16, 16, 19, 20]
        assert items[1].param2 == 6
        for item, (e, n) in zip(items[2:7], CORNERS, strict=True):
            lat, lon = at(e, n)
            assert (item.x, item.y, item.z) == (round(lat * 1e7), round(lon * 1e7), 30)
        seen: list[float] = []
        closest = [math.inf] * len(CORNERS)
        start = len(r.fc.track)

        def watch() -> bool:
            snap = r.ex.snapshot()
            if snap.progress_pct is not None:
                seen.append(snap.progress_pct)
            return r.ex.state == S.HOLD

        await r.until(watch, 300, step=0.05)
        assert r.up.stages(cmd.command_id)[-1] == "done"
        assert r.fc.mode == CopterMode.LOITER  # it hovers at the last point
        for lat, lon, _alt in r.fc.track[start:]:
            x, y = PLANE.xy(lat, lon)
            for i, (e, n) in enumerate(CORNERS):
                closest[i] = min(closest[i], math.dist((x, y), (e, n)))
        assert max(closest) <= 3.0, closest
        assert seen == sorted(seen) and seen[-1] > 90  # grows along the line
        x, y = PLANE.xy(r.fc.lat, r.fc.lon)
        assert math.dist((x, y), CORNERS[-1]) <= 3.0
        assert r.ex.resume_point is None


async def test_the_ais_path_needs_the_pilots_approval_of_these_points() -> None:
    async with rig() as r:
        await r.takeoff(40)
        p = params()
        missing = await r.send(r.order("follow_path", p, issuer="brain:ground"))
        assert r.up.codes(missing.command_id) == ("ONB_APPROVAL_MISSING",)
        other = params(CORNERS[:3])
        wrong = ApprovalRef(
            approval_id=new_id(),
            command_hash=command_hash("follow_path", FollowPathParams(**other), 1),
        )
        swapped = await r.send(r.order("follow_path", p, issuer="brain:ground", approval=wrong))
        assert r.up.codes(swapped.command_id) == ("ONB_APPROVAL_MISMATCH",)
        good = ApprovalRef(
            approval_id=new_id(),
            command_hash=command_hash("follow_path", FollowPathParams(**p), 1),
        )
        ok = await r.send(r.order("follow_path", p, issuer="brain:ground", approval=good), 1)
        assert "running" in r.up.stages(ok.command_id)


async def test_hold_on_the_way_then_resume_flies_the_rest() -> None:
    async with rig() as r:
        await r.takeoff(40)
        cmd = await r.send(r.order("follow_path", params()), 1)
        await r.until(lambda: (r.ex.snapshot().progress_pct or 0) > 30, 120, step=0.05)
        await r.send(r.order("hold", {}))
        assert r.ex.state == S.HOLD and r.up.stages(cmd.command_id)[-1] == "superseded"
        rp = r.ex.resume_point
        assert isinstance(rp, SurveyResumePoint) and 0 < rp.next_index < len(CORNERS)
        resume = await r.send(r.order("resume", {}), 1)
        await r.until(lambda: r.ex.state == S.HOLD, 300, step=0.05)
        assert r.up.stages(resume.command_id)[-1] == "done"
        x, y = PLANE.xy(r.fc.lat, r.fc.lon)
        assert math.dist((x, y), CORNERS[-1]) <= 3.0


async def test_outside_the_charter_is_refused_onboard() -> None:
    async with rig() as r:
        await r.takeoff(40)
        cmd = await r.send(r.order("follow_path", params([(0.0, 60.0), (0.0, 400.0)])))
        assert "OUTSIDE_CHARTER" in r.up.codes(cmd.command_id)
        assert r.ex.state == S.HOLD
