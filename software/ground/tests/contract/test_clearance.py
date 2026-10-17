"""``POST /missions/{id}/clearance`` ("clear around the drone") against a real database."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))

from flow_rig import VID, Rig, rig  # pyright: ignore[reportMissingImports]

from sd_ground.services import clearance as clearance_service
from sd_ground.services.missions import FlowError

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]


@pytest.fixture
async def r(migrated_db: Db, tmp_path: Path) -> AsyncIterator[Rig]:
    async with rig(migrated_db.name, migrated_db.app_password, tmp_path) as built:
        yield built


async def test_clearance_from_a_remote_listener_is_refused(r: Rig) -> None:
    mid = await r.ready_mission()
    mission = await r.flow.mission(mid)
    with pytest.raises(FlowError) as exc:
        await clearance_service.confirm(r.flow, mission, by="operator:far", origin="remote")
    assert (exc.value.status, exc.value.code) == (403, "CLEARANCE_REMOTE")
    assert (await r.mission(mid))["clearance"] is None
    assert r.onboard.clearances == []


async def test_clearance_needs_an_approved_charter(r: Rig) -> None:
    mid = await r.open_mission()
    res = await r.clearance(mid)
    assert res.status_code == 409 and res.json()["code"] == "CHARTER_NOT_APPROVED", res.text
    assert r.flow.clearances.get(VID) is None


async def test_clearance_row_gone_after_the_write(r: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    mid = await r.ready_mission()

    async def gone(_clearance_id: str) -> None:
        return None

    monkeypatch.setattr(r.flow.repo, "clearance", gone)
    with pytest.raises(RuntimeError, match="vanished"):
        await r.clearance(mid)
