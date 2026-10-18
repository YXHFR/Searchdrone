"""SK-RED's evaluation: finds matched to the world's facit, the map and the record."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sd_safety.targets import Plane
from sd_sim import fixture, search_report, targets, world


def _truth() -> dict[str, Any]:
    return world.world_truth(world.WORLD_NAME, targets.layout("SK-RED", 11))


def _find(fid: str, label: str, east: float, north: float, home: dict[str, Any]) -> dict[str, Any]:
    lat, lon = Plane(home["lat"], home["lon"]).latlon(east, north)
    return {"find_id": fid, "label": label, "lat": lat, "lon": lon}


def test_finds_are_matched_to_the_facit() -> None:
    truth = _truth()
    home = truth["home"]
    blocks = [o for o in truth["objects"] if o["truth"]["cls"] == "red_block"]
    assert len(blocks) == 3
    finds = [
        _find("a", "red_block", blocks[0]["east_m"] + 0.6, blocks[0]["north_m"], home),
        _find("b", "red_block", blocks[1]["east_m"], blocks[1]["north_m"] - 2.5, home),
        _find("far", "red_block", blocks[2]["east_m"] + 9.0, blocks[2]["north_m"], home),
        _find("car", "car", blocks[2]["east_m"], blocks[2]["north_m"], home),  # other class
    ]
    result = search_report.match_finds(truth, finds, "red_block", 3.0)
    assert [m.target for m in result.matches] == [blocks[0]["name"], blocks[1]["name"]]
    assert [m.error_m for m in result.matches] == pytest.approx([0.6, 2.5], abs=0.02)
    assert result.missed == [blocks[2]["name"]] and result.false_finds == ["far"]
    assert not result.ok
    assert result.as_dict()["max_error_m"] == pytest.approx(2.5, abs=0.02)


def test_one_find_matches_one_target() -> None:
    truth = _truth()
    home = truth["home"]
    b = next(o for o in truth["objects"] if o["truth"]["cls"] == "red_block")
    finds = [
        _find("near", "red_block", b["east_m"] + 0.2, b["north_m"], home),
        _find("also_near", "red_block", b["east_m"] - 0.5, b["north_m"], home),
    ]
    result = search_report.match_finds(truth, finds, "red_block", 3.0)
    assert [(m.target, m.find_id) for m in result.matches] == [(b["name"], "near")]
    assert result.false_finds == ["also_near"]


def test_map_and_record_are_written(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    truth = _truth()
    result = search_report.match_finds(truth, [], "red_block", 3.0)
    box = targets.search_box(fixture.load())
    out = search_report.draw_map(truth, [], result, [], tmp_path / "search" / "map.png", box, "t")
    assert out.is_file() and out.stat().st_size > 0
    path = search_report.write(tmp_path, {"result": result.as_dict()})
    assert json.loads(path.read_text(encoding="utf-8"))["result"]["missed"] == sorted(
        o["name"] for o in truth["objects"] if o["truth"]["cls"] == "red_block"
    )
