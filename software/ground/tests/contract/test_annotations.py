"""The pilot's pins and drawn areas on the map, against a real database."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))

from flow_rig import H, Rig, rig  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]


@pytest.fixture
async def r(migrated_db: Db, tmp_path: Path) -> AsyncIterator[Rig]:
    async with rig(migrated_db.name, migrated_db.app_password, tmp_path) as built:
        yield built


async def _add(r: Rig, mid: str, kind: str, label: str, geometry: dict[str, Any]) -> Any:
    return await r.panel.post(
        f"/api/v1/missions/{mid}/annotations",
        json={"kind": kind, "label": label, "geometry": geometry},
        headers=H,
    )


async def test_pins_and_areas_are_added_named_once_and_deleted(r: Rig) -> None:
    mid = await r.open_mission()
    lat, lon = r.at(20.0, 30.0)
    pin = await _add(r, mid, "pin", "P1", {"type": "Point", "coordinates": [lon, lat]})
    assert pin.status_code == 201, pin.text
    assert pin.json()["kind"] == "pin" and pin.json()["created_by"] == "operator:ci-harness"
    assert "annotation.created" in r.bus_types()
    same = await _add(r, mid, "pin", "P1", {"type": "Point", "coordinates": [lon, lat]})
    assert same.status_code == 409 and same.json()["code"] == "LABEL_TAKEN", same.text
    area = await _add(r, mid, "area", "Skogen", r.square(80.0))
    assert area.status_code == 201, area.text
    assert area.json()["area_m2"] == pytest.approx(160.0 * 160.0, rel=0.02)
    listed = (await r.panel.get(f"/api/v1/missions/{mid}/annotations")).json()
    assert sorted(a["label"] for a in listed) == ["P1", "Skogen"]

    gone = await r.panel.delete(f"/api/v1/annotations/{pin.json()['id']}", headers=H)
    assert gone.status_code == 204, gone.text
    assert "annotation.deleted" in r.bus_types()
    again = await r.panel.delete(f"/api/v1/annotations/{pin.json()['id']}", headers=H)
    assert again.status_code == 404 and again.json()["code"] == "ANNOTATION_NOT_FOUND"
    listed = (await r.panel.get(f"/api/v1/missions/{mid}/annotations")).json()
    assert [a["label"] for a in listed] == ["Skogen"]


async def test_a_drawn_area_becomes_the_charters_search_area(r: Rig) -> None:
    mid = await r.open_mission()
    lat, lon = r.at(0.0, 0.0)
    pin = await _add(r, mid, "pin", "P1", {"type": "Point", "coordinates": [lon, lat]})
    area = await _add(r, mid, "area", "Ängen", r.square(120.0))
    by_pin = await r.panel.post(
        f"/api/v1/missions/{mid}/charters",
        json={"area_annotation_id": pin.json()["id"], "reason_sv": "Punkten"},
        headers=H,
    )
    assert by_pin.status_code == 422, by_pin.text
    assert by_pin.json()["code"] == "AREA_INVALID"
    assert [e["field"] for e in by_pin.json()["errors"]] == ["area_annotation_id"]
    by_area = await r.panel.post(
        f"/api/v1/missions/{mid}/charters",
        json={"area_annotation_id": area.json()["id"], "reason_sv": "Ängen"},
        headers=H,
    )
    assert by_area.status_code == 201, by_area.text
    assert by_area.json()["body"]["area"] == area.json()["geometry"]
