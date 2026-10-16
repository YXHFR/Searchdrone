"""A restarted sd-onboard: HELD in the air whatever was saved, IDLE on the ground."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from onboard_helpers import at, rig

from sd_mav.modes import CopterMode
from sd_onboard.io.persistence import StateStore
from sd_schemas.base import LatLon, new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.executor import SurveyResumePoint
from sd_schemas.link import OperatorRelease


def _saved(path: Path, state: str) -> None:
    lat, lon = at(50, 50)
    rp = SurveyResumePoint(
        command_id=new_id(),
        waypoints=[LatLon(lat=lat, lon=lon), LatLon(lat=lat + 1e-4, lon=lon)],
        alt_m=40,
        speed_ms=7,
        base_index=0,
        seq_offset=0,
        next_index=1,
        interrupted_at=LatLon(lat=lat, lon=lon),
    )
    path.mkdir(parents=True, exist_ok=True)
    (path / "executor_state.json").write_text(
        json.dumps(
            {
                "schema": 2,
                "vehicle_id": "sim-01",
                "mission_id": None,
                "boot_id": 7,
                "state": state,
                "hold_reason": None,
                "held_by": None,
                "expected_mode": "AUTO",
                "charter_id": None,
                "charter_version": None,
                "active_command_id": None,
                "resume_point": rp.model_dump(mode="json"),
                "saved_mono_ms": 1,
                "saved_utc": "2026-10-15T10:00:00Z",
            }
        )
    )


async def test_restart_airborne_goes_held(tmp_path: Path) -> None:
    _saved(tmp_path, "SURVEY")
    async with rig(start=False) as r:
        r.ex.store = StateStore(tmp_path, r.clock)
        r.fc.armed, r.fc.alt_rel, r.fc.mode = True, 40.0, CopterMode.AUTO
        r.fc.target = (r.fc.lat, r.fc.lon, 40.0)
        r.background.append(asyncio.ensure_future(r.ex.run(r.stop)))
        await r.until(lambda: r.ex._started, 10)
        assert r.ex.state == S.HELD and r.ex.held_by == "onboard_restart"
        assert r.ex.boot_id == 8
        snap = r.ex.snapshot()
        assert snap.resume_available and snap.state_epoch == 8
        ref = new_id()
        r.ex.on_release(OperatorRelease(action_seq=0, operator_id="operator:pilot"), ref)
        assert r.up.codes(ref) == ("ONB_RELEASE_MODE_UNSUPPORTED",)
        assert "executor.onboard_restart" in r.up.kinds()
        saved = json.loads((tmp_path / "executor_state.json").read_text())
        assert saved["state"] == "HELD" and saved["boot_id"] == 8


async def test_restart_on_the_ground_is_idle(tmp_path: Path) -> None:
    _saved(tmp_path, "SURVEY")
    async with rig(start=False) as r:
        r.ex.store = StateStore(tmp_path, r.clock)
        r.background.append(asyncio.ensure_future(r.ex.run(r.stop)))
        await r.until(lambda: r.ex._started, 10)
        assert r.ex.state == S.IDLE and r.ex.resume_point is None
        assert r.ex.hello().boot_id == 8


def test_state_file_is_written_atomically_and_private(tmp_path: Path) -> None:
    from sd_testing import FakeClock

    store = StateStore(tmp_path / "s", FakeClock())
    store.save(
        {
            "vehicle_id": "sim-01",
            "mission_id": None,
            "boot_id": 1,
            "state": "IDLE",
            "hold_reason": None,
            "held_by": None,
            "expected_mode": None,
            "charter_id": None,
            "charter_version": None,
            "active_command_id": None,
            "resume_point": None,
        }
    )
    path = tmp_path / "s" / "executor_state.json"
    assert path.stat().st_mode & 0o077 == 0
    assert not (tmp_path / "s" / "executor_state.json.tmp").exists()
    loaded = store.load()
    assert loaded is not None and loaded.state == S.IDLE and loaded.saved_utc.endswith("Z")
    path.write_text("{broken")
    assert store.load() is None
