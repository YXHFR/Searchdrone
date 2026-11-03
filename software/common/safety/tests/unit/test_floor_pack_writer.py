"""write_floor_pack: what it writes is what load_floor_pack reads, and nothing half done."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from floor_fixtures import FLIGHT, ORIGIN, START_AMSL, at, make_pack

from sd_safety.floor import FloorPack, floor_at
from sd_safety.floor_pack import (
    FloorPackError,
    PackLine,
    load_floor_pack,
    write_floor_pack,
)


def _write(folder: Path, pack: FloorPack | None = None, **kw: Any) -> FloorPack:
    pack = pack or make_pack(tree_m=24.0, scan_year=2021)
    args: dict[str, Any] = {
        "name": "example-area",
        "origin": pack.origin,
        "dtm_cm": pack.dtm_cm,
        "chm_dm": pack.chm_dm,
        "scan_year": pack.scan_year,
        "scan_cell_m": pack.scan_cell_m,
        "attribution": "synthetic test data",
    }
    args.update(kw)
    return write_floor_pack(folder, **args)


def _floor(pack: FloorPack, east: float = 0.0, north: float = 0.0) -> object:
    p = at(east, north)
    return floor_at(pack, p.lat, p.lon, flight_date=FLIGHT, laser=None, start_alt_amsl_m=START_AMSL)


def test_round_trip(tmp_path: Path) -> None:
    pack = make_pack(
        tree_m=24.0,
        scan_year=2021,
        trees={(3.0, 3.0): -1},
        lines=(((-20.0, 20.0), (20.0, 20.0)), ((10.0, -10.0),)),
    )
    lines = [
        PackLine((at(-20.0, 20.0), at(20.0, 20.0)), {"source": "osm", "power": "line"}),
        PackLine((at(10.0, -10.0),), {"power": "pole"}),
    ]
    about = {
        "script": "fetch_floor_map.py",
        "script_version": 1,
        "margin_m": 20.5,
        "osm_lines": True,
        "copc": ["a.copc.laz", "b.copc.laz"],
    }
    returned = _write(tmp_path / "pack", pack, lines=lines, about=about)
    folder = tmp_path / "pack"
    loaded = load_floor_pack(folder)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pack"]  # nothing left beside it
    assert returned.sha256 == loaded.sha256
    assert loaded.sha256 == hashlib.sha256((folder / "floor_pack.toml").read_bytes()).hexdigest()
    assert (loaded.name, loaded.origin, loaded.attribution) == (
        "example-area",
        ORIGIN,
        "synthetic test data",
    )
    np.testing.assert_array_equal(loaded.dtm_cm, pack.dtm_cm)
    np.testing.assert_array_equal(loaded.chm_dm, pack.chm_dm)
    np.testing.assert_array_equal(loaded.scan_year, pack.scan_year)
    corners = [[pt for xy in line for pt in xy] for line in loaded.lines]
    assert corners[0] == pytest.approx([-20.0, 20.0, 20.0, 20.0], abs=1e-6)
    assert corners[1] == pytest.approx([10.0, -10.0, 10.0, -10.0], abs=1e-6)  # a pole
    meta = tomllib.loads((folder / "floor_pack.toml").read_text(encoding="utf-8"))
    assert meta["about"] == about
    geo = json.loads((folder / "lines.geojson").read_text(encoding="utf-8"))
    assert [f["properties"] for f in geo["features"]] == [
        {"source": "osm", "power": "line"},
        {"power": "pole"},
    ]
    for east, north in ((0.0, 0.0), (3.0, 3.0), (0.0, -27.0), (25.0, -25.0)):
        assert _floor(loaded, east, north) == _floor(pack, east, north)


def test_a_pack_without_lines_or_about(tmp_path: Path) -> None:
    loaded = _write(tmp_path / "pack")
    assert loaded.lines == ()
    meta = tomllib.loads((tmp_path / "pack" / "floor_pack.toml").read_text(encoding="utf-8"))
    assert meta["about"] == {}


def test_any_text_survives(tmp_path: Path) -> None:
    text = '© Lantmäteriet "grid 1"\\ CC BY 4.0\n\ttab\x7fdel\x01'
    loaded = _write(tmp_path / "pack", attribution=text, about={"note": text})
    assert loaded.attribution == text


def test_a_new_pack_takes_the_old_ones_place(tmp_path: Path) -> None:
    folder = tmp_path / "pack"
    first = _write(folder)
    (folder / "stray.txt").write_text("from before", encoding="utf-8")
    # Left over from an interrupted write: cleared, never mixed in.
    (tmp_path / ".pack.writing").mkdir()
    (tmp_path / ".pack.writing" / "chm_dm.npy").write_bytes(b"half")
    (tmp_path / ".pack.old").mkdir()
    second = _write(folder, make_pack(tree_m=5.0, scan_year=2024))
    assert second.sha256 != first.sha256
    assert load_floor_pack(folder).sha256 == second.sha256
    assert not (folder / "stray.txt").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pack"]


def test_a_refused_pack_leaves_the_old_one(tmp_path: Path) -> None:
    folder = tmp_path / "pack"
    first = _write(folder)
    with pytest.raises(FloorPackError):
        _write(folder, about={"bad key": "x"})
    assert load_floor_pack(folder).sha256 == first.sha256


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"name": "Has Spaces"}, "name"),
        ({"name": "../up"}, "name"),
        ({"dtm_cm": np.zeros((60, 60), np.float32)}, "int32"),
        ({"chm_dm": np.full((60, 60), -2, np.int16)}, "below -1"),
        ({"dtm_cm": np.full((60, 60), 900_001, np.int32)}, "outside"),
        ({"scan_cell_m": 0}, "scan_cell_m"),
        ({"lines": [PackLine(())]}, "at least one point"),
        ({"about": {"Upper": "x"}}, "a key"),
        ({"about": {"x": float("nan")}}, "not finite"),
        ({"about": {"x": ["a", 1]}}, "only strings"),
    ],
)
def test_refused(tmp_path: Path, kw: dict[str, Any], message: str) -> None:
    with pytest.raises(FloorPackError, match=message):
        _write(tmp_path / "pack", **kw)
    assert not (tmp_path / "pack").exists()
