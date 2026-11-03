"""The height floor (spec "Titta närmare" §6): the owner's worked examples, the margins,
the laser, power lines, paths and the pack on disk."""

from __future__ import annotations

import builtins
import hashlib
import json
import socket
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from floor_fixtures import FLIGHT, GROUND_M, ORIGIN, START_AMSL, at, make_pack, write_pack
from numpy.typing import NDArray

from sd_safety.floor import (
    FloorPack,
    FloorPackError,
    FloorResult,
    LaserState,
    floor_along,
    floor_at,
    load_floor_pack,
)
from sd_schemas.base import LatLon
from sd_schemas.rangefinder import LaserState as RangefinderLaserState

TF02 = 0.0  # tested minimum range, m
GRF250 = 10.0


def floor(
    pack: FloorPack | None,
    east: float = 0.0,
    north: float = 0.0,
    laser: LaserState | None = None,
    *,
    flight_date: date = FLIGHT,
    start_alt_amsl_m: float = START_AMSL,
    laser_lost: bool = False,
    extra_lines: Sequence[Sequence[LatLon]] = (),
) -> FloorResult:
    p = at(east, north)
    return floor_at(
        pack,
        p.lat,
        p.lon,
        flight_date=flight_date,
        laser=laser,
        start_alt_amsl_m=start_alt_amsl_m,
        laser_lost=laser_lost,
        extra_lines=extra_lines,
    )


def seen(surface_above_ground_m: float, min_range_m: float) -> LaserState:
    """A confirmed laser that saw a surface this high above the ground."""
    return LaserState(
        confirmed=True,
        surface_amsl_m=GROUND_M + surface_above_ground_m,
        min_range_m=min_range_m,
    )


UNCONFIRMED = LaserState(confirmed=False, surface_amsl_m=None, min_range_m=GRF250)


def agl(value: float) -> float:
    return round(value - GROUND_M, 6)


# --- the worked examples in spec §6.3 (acceptance test 1) -----------------------------------


def test_open_meadow() -> None:
    pack = make_pack(tree_m=0.4, scan_year=2019)
    r = floor(pack)
    assert r.ground_amsl_m == pytest.approx(GROUND_M)
    assert r.tree_m == pytest.approx(5.9, abs=0.05)  # 0.4 + 2 + 7 years x 0.5
    assert agl(r.floor_lowest_amsl_m) == pytest.approx(20.9, abs=0.05)
    assert agl(r.floor_auto_amsl_m) == pytest.approx(30.0, abs=0.05)
    assert (r.reason, r.laser_confirmed) == ("map", False)
    tf02 = floor(pack, laser=seen(0.0, TF02))
    assert agl(tf02.floor_lowest_amsl_m) == pytest.approx(10.9, abs=0.05)
    assert agl(tf02.floor_auto_amsl_m) == pytest.approx(30.0, abs=0.05)
    assert (tf02.reason, tf02.laser_confirmed) == ("map", True)  # the map's +5 binds
    grf = floor(pack, laser=seen(0.0, GRF250))
    assert agl(grf.floor_lowest_amsl_m) == pytest.approx(12.0, abs=0.05)  # the surface + 12
    assert (grf.reason, grf.laser_confirmed) == ("laser", True)


def test_spruce_forest() -> None:
    pack = make_pack(tree_m=24.0, scan_year=2021)
    r = floor(pack)
    assert r.tree_m == pytest.approx(28.5, abs=0.05)
    assert agl(r.floor_lowest_amsl_m) == pytest.approx(43.5, abs=0.05)
    assert agl(r.floor_auto_amsl_m) == pytest.approx(43.5, abs=0.05)
    laser = floor(pack, laser=seen(24.0, TF02))  # the laser sees the tops, under 28.5
    assert agl(laser.floor_lowest_amsl_m) == pytest.approx(33.5, abs=0.05)
    assert agl(laser.floor_auto_amsl_m) == pytest.approx(43.5, abs=0.05)


def test_unknown_is_thirty_metre_forest() -> None:
    pack = make_pack(trees={(0.0, 0.0): -1}, scan_year=0)
    r = floor(pack)
    assert r.tree_m == pytest.approx(37.0, abs=0.05)  # 30 + 2 + 5
    assert agl(r.floor_lowest_amsl_m) == pytest.approx(52.0, abs=0.05)
    assert agl(r.floor_auto_amsl_m) == pytest.approx(52.0, abs=0.05)
    assert r.reason == "unknown"
    laser = floor(pack, laser=seen(0.0, TF02))
    assert agl(laser.floor_lowest_amsl_m) == pytest.approx(42.0, abs=0.05)
    assert laser.reason == "unknown"


def test_near_a_power_line_no_descent() -> None:
    line = ((-100.0, 20.0), (100.0, 20.0))  # 20 m north of the point
    r = floor(make_pack(tree_m=0.4, lines=(line,)))
    assert r.floor_lowest_amsl_m == r.floor_auto_amsl_m == START_AMSL
    assert r.reason == "powerline"
    assert floor(make_pack(tree_m=0.4, lines=(line,)), north=-10.5).reason == "map"  # 30.5 m


def test_a_power_line_cannot_lower_the_floor() -> None:
    """Started lower than the trees allow: the trees still win."""
    pack = make_pack(tree_m=24.0, scan_year=2021, lines=(((-50.0, 0.0), (50.0, 0.0)),))
    r = floor(pack, start_alt_amsl_m=GROUND_M + 10.0)
    assert agl(r.floor_lowest_amsl_m) == pytest.approx(43.5, abs=0.05)
    assert r.reason == "powerline"


def test_lines_the_pilot_drew_count() -> None:
    drawn = [at(-100.0, -25.0), at(100.0, -25.0)]
    r = floor(make_pack(), extra_lines=[drawn])
    assert (r.reason, r.floor_lowest_amsl_m) == ("powerline", START_AMSL)


def test_a_line_of_one_point_is_a_pole_with_its_zone() -> None:
    """One point (a pylon) the pilot drew, or in the pack, keeps the 30 m zone around it."""
    pole = [at(0.0, 25.0)]
    assert floor(make_pack(), extra_lines=[pole]).reason == "powerline"
    assert floor(make_pack(), north=-10.0, extra_lines=[pole]).reason == "map"  # 35 m
    assert floor(make_pack(lines=(((0.0, 25.0),),))).reason == "powerline"


def test_a_line_of_no_points_is_refused() -> None:
    with pytest.raises(FloorPackError, match="at least one point"):
        floor(make_pack(), extra_lines=[[]])


# --- the margins -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scan_year", "tree_m"),
    [
        (2026, 10.0 + 2.0),  # scanned this year: no growth yet
        (2027, 37.0),  # a scan year after the flight: an error, so unknown
        (2022, 10.0 + 2.0 + 2.0),
        (2016, 10.0 + 2.0 + 5.0),  # ten years: the growth is at its cap
        (2015, 37.0),  # eleven years: too old, unknown
        (0, 10.0 + 2.0 + 5.0),  # an unknown year: the most growth
    ],
)
def test_growth_since_the_scan(scan_year: int, tree_m: float) -> None:
    r = floor(make_pack(tree_m=10.0, scan_year=scan_year))
    assert r.tree_m == pytest.approx(tree_m)


def test_scan_year_per_square() -> None:
    """The scan year comes from the square the tree stands in, not P's."""
    pack = make_pack(size=40, tree_m=10.0, scan_cell_m=20)
    pack.scan_year[0, 1] = 2010  # north-east square: too old
    assert floor(pack, -8.0, -8.0).tree_m == pytest.approx(15.5)  # 10 + 2 + 3.5
    assert floor(pack, 8.0, 8.0).tree_m == pytest.approx(37.0)


def test_highest_tree_within_ten_metres() -> None:
    near = make_pack(trees={(9.6, 0.0): 20.0})  # its cell reaches to 9.0 m
    assert near.chm_dm[30, 39] == 200
    assert floor(near).tree_m == pytest.approx(20.0 + 2.0 + 3.5)
    edge = make_pack(trees={(10.5, 0.2): 20.0})  # its cell starts 10.0 m away
    assert floor(edge).tree_m == pytest.approx(25.5)
    far = make_pack(trees={(11.5, 0.0): 20.0})  # 11 m to its nearest point
    assert floor(far).tree_m == pytest.approx(5.5)  # only the bare ground's margins
    diagonal = make_pack(trees={(8.1, 8.1): 20.0})  # 8 m each way: 11.3 m
    assert floor(diagonal).tree_m == pytest.approx(5.5)
    inside = make_pack(trees={(7.6, 7.6): 20.0})  # 7 m each way: 9.9 m
    assert floor(inside).tree_m == pytest.approx(25.5)


def test_a_tree_on_higher_ground_counts_from_its_ground() -> None:
    pack = make_pack(trees={(5.0, 0.0): 20.0}, ground={(5.0, 0.0): GROUND_M + 3.0})
    r = floor(pack)
    assert r.ground_amsl_m == pytest.approx(GROUND_M)
    assert r.tree_m == pytest.approx(3.0 + 25.5)


def test_a_tree_lower_down_counts_from_ground_at_p() -> None:
    pack = make_pack(trees={(5.0, 0.0): 20.0}, ground={(5.0, 0.0): GROUND_M - 3.0})
    assert floor(pack).tree_m == pytest.approx(25.5)


def test_near_the_edge_of_the_pack_is_unknown() -> None:
    pack = make_pack(size=60)
    r = floor(pack, 25.0, 0.0)  # 5 m from the east edge
    assert r.tree_m == pytest.approx(37.0)
    assert r.reason == "unknown"
    assert floor(pack, 15.0, 0.0).reason == "map"


def test_a_cell_without_ground_is_unknown_on_the_highest_ground_near() -> None:
    pack = make_pack(ground={(3.0, 0.0): float("nan"), (-3.0, 0.0): GROUND_M + 2.0})
    r = floor(pack)
    assert r.tree_m == pytest.approx(2.0 + 37.0)
    assert r.reason == "unknown"


def test_all_unknown_around() -> None:
    pack = make_pack(size=30)
    pack.chm_dm[:] = -1
    assert floor(pack).tree_m == pytest.approx(37.0)


# --- no map --------------------------------------------------------------------------------


def test_without_a_pack_no_descent() -> None:
    r = floor(None, laser=seen(0.0, GRF250))
    assert r == FloorResult(None, None, START_AMSL, START_AMSL, "no_map", False)


@pytest.mark.parametrize(("east", "north"), [(31.0, 0.0), (0.0, -30.5), (-1000.0, 0.0)])
def test_outside_the_pack_no_descent(east: float, north: float) -> None:
    r = floor(make_pack(size=60), east, north)
    assert (r.reason, r.floor_lowest_amsl_m, r.ground_amsl_m) == ("no_map", START_AMSL, None)


def test_an_old_scan_keeps_a_taller_tree() -> None:
    """Unknown is 30 m forest at the least, not instead: a 38 m spruce scanned 12 years
    ago still counts with its own height and the most growth."""
    r = floor(make_pack(tree_m=38.0, scan_year=2014))
    assert r.tree_m == pytest.approx(38.0 + 2.0 + 5.0)
    assert r.reason == "unknown"
    assert floor(make_pack(tree_m=20.0, scan_year=2014)).tree_m == pytest.approx(37.0)


def test_a_cell_without_ground_keeps_a_taller_tree() -> None:
    pack = make_pack(trees={(3.0, 0.0): 38.0}, ground={(3.0, 0.0): float("nan")})
    assert floor(pack).tree_m == pytest.approx(38.0 + 2.0 + 5.0)


@pytest.mark.parametrize("chm_dm", [-2, -50, -32768])
def test_any_negative_tree_height_is_unknown(chm_dm: int) -> None:
    """Another nodata value than -1 (int16's lowest, say) is no height, not a negative tree."""
    pack = make_pack()
    pack = replace(pack, chm_dm=np.full(pack.chm_dm.shape, chm_dm, np.int16))
    r = floor(pack)
    assert r.tree_m == pytest.approx(37.0)
    assert agl(r.floor_lowest_amsl_m) == pytest.approx(52.0)
    assert r.reason == "unknown"


def test_no_ground_height_at_p_no_descent() -> None:
    r = floor(make_pack(ground={(0.0, 0.0): float("nan")}))
    assert r.reason == "no_map"


# --- the laser (acceptance test 3 for fixed cases) ---------------------------------------


def test_the_laser_sees_a_tree_the_map_missed() -> None:
    pack = make_pack(tree_m=0.4)
    r = floor(pack, laser=seen(18.0, TF02))  # a tree the map has as meadow
    assert agl(r.floor_lowest_amsl_m) == pytest.approx(23.0)
    assert agl(r.floor_auto_amsl_m) == pytest.approx(30.0)
    assert r.reason == "laser"
    high = floor(pack, laser=seen(40.0, TF02))
    assert agl(high.floor_auto_amsl_m) == pytest.approx(45.0)  # the auto floor rises too


def test_the_laser_cannot_go_below_the_map() -> None:
    pack = make_pack(tree_m=24.0, scan_year=2021)
    r = floor(pack, laser=seen(2.0, TF02))  # a gap in the forest
    assert agl(r.floor_lowest_amsl_m) == pytest.approx(33.5)  # the map's lowest
    assert r.reason == "map"


def test_an_unconfirmed_laser_does_not_count() -> None:
    r = floor(make_pack(tree_m=0.4), laser=UNCONFIRMED)
    assert agl(r.floor_lowest_amsl_m) == pytest.approx(20.9, abs=0.05)
    assert (r.reason, r.laser_confirmed) == ("map", False)


def test_a_lost_laser_counts_as_none() -> None:
    pack = make_pack(tree_m=0.4)
    for laser in (None, UNCONFIRMED, seen(0.0, GRF250)):
        r = floor(pack, laser=laser, laser_lost=True)
        assert agl(r.floor_lowest_amsl_m) == pytest.approx(20.9, abs=0.05)
        assert (r.reason, r.laser_confirmed) == ("laser_lost", False)


def test_a_power_line_beats_the_laser_and_its_loss() -> None:
    pack = make_pack(lines=(((-50.0, 5.0), (50.0, 5.0)),))
    assert floor(pack, laser=seen(0.0, TF02)).reason == "powerline"
    assert floor(pack, laser_lost=True).reason == "powerline"


@pytest.mark.parametrize(
    ("surface", "min_range"),
    [
        (float("nan"), TF02),
        (float("inf"), TF02),
        (float("-inf"), TF02),
        (GROUND_M, float("nan")),
        (GROUND_M, float("inf")),
    ],
)
def test_a_reading_that_is_not_a_number_is_never_a_laser_state(
    surface: float, min_range: float
) -> None:
    """``max`` drops a NaN that is not its first argument: it would leave the map's lowest
    floor (trees + 5) in place of trees + 15. The laser's state refuses one outright, so
    the floor never sees it (the laser reader then has no confirmed state to give)."""
    with pytest.raises(ValueError, match="finite"):
        LaserState(confirmed=True, surface_amsl_m=surface, min_range_m=min_range)
    assert LaserState is RangefinderLaserState  # one type, the schemas'


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_start_height_or_point_that_is_not_a_number_is_refused(bad: float) -> None:
    """A NaN start height would turn the power-line rule off unseen: refused instead."""
    pack = make_pack(lines=(((-50.0, 0.0), (50.0, 0.0)),))
    for p in (pack, None):
        with pytest.raises(ValueError, match="start_alt_amsl_m"):
            floor(p, start_alt_amsl_m=bad)
        with pytest.raises(ValueError, match="start_alt_amsl_m"):
            floor_along(p, [at(0, 0)], flight_date=FLIGHT, laser=None, start_alt_amsl_m=bad)
    with pytest.raises(ValueError, match="lat"):
        floor_at(pack, bad, 18.2, flight_date=FLIGHT, laser=None, start_alt_amsl_m=99.0)
    with pytest.raises(ValueError, match="lon"):
        floor_at(pack, 59.3, bad, flight_date=FLIGHT, laser=None, start_alt_amsl_m=99.0)


def test_surface_set_exactly_when_confirmed() -> None:
    with pytest.raises(ValueError, match="confirmed"):
        LaserState(confirmed=True, surface_amsl_m=None, min_range_m=0.0)
    with pytest.raises(ValueError, match="confirmed"):
        LaserState(confirmed=False, surface_amsl_m=GROUND_M, min_range_m=0.0)
    assert LaserState(confirmed=True, surface_amsl_m=1.0, min_range_m=0.0, distance_m=3.0)


# --- along a path ------------------------------------------------------------------------


def test_floor_along_finds_a_narrow_tree_between_far_points() -> None:
    pack = make_pack(size=300, tree_m=0.4, trees={(13.0, 9.0): 30.0})
    path = [at(-100.0, 0.0), at(100.0, 0.0)]
    r = floor_along(pack, path, flight_date=FLIGHT, laser=None, start_alt_amsl_m=START_AMSL)
    assert r.tree_m == pytest.approx(30.0 + 2.0 + 3.5)
    ends = [floor(pack, -100.0, 0.0), floor(pack, 100.0, 0.0)]
    assert all(e.tree_m == pytest.approx(5.9) for e in ends)


def test_floor_along_takes_the_highest_auto_floor_anywhere() -> None:
    """The laser lets the drone lower over the tree than the auto floor there, and a line
    holds it at the start height at the other end: the lowest floor is the line's, the
    auto floor the tree's."""
    pack = make_pack(
        size=300, tree_m=0.4, trees={(-60.0, 0.0): 40.0}, lines=(((60.0, -100.0), (60.0, 100.0)),)
    )
    path = [at(-60.0, 0.0), at(60.0, 0.0)]
    start = GROUND_M + 55.0
    laser = seen(40.0, TF02)
    r = floor_along(pack, path, flight_date=FLIGHT, laser=laser, start_alt_amsl_m=start)
    assert r.floor_lowest_amsl_m == pytest.approx(start)
    assert r.reason == "powerline"
    assert agl(r.floor_auto_amsl_m) == pytest.approx(40.0 + 5.5 + 15.0)
    tree = floor(pack, -60.0, laser=laser, start_alt_amsl_m=start)
    assert agl(tree.floor_lowest_amsl_m) == pytest.approx(40.0 + 5.5 + 5.0)


def test_floor_along_one_point_is_floor_at() -> None:
    pack = make_pack(tree_m=12.0)
    for lost in (False, True):
        one = floor_along(
            pack,
            [at(3.0, 4.0)],
            flight_date=FLIGHT,
            laser=seen(0.0, TF02),
            start_alt_amsl_m=START_AMSL,
            laser_lost=lost,
        )
        assert one == floor(pack, 3.0, 4.0, laser=seen(0.0, TF02), laser_lost=lost)


def test_floor_along_the_laser_counts_only_near_the_drone() -> None:
    """The laser saw the meadow under the drone at the path's start; 10 m on it says
    nothing any more."""
    pack = make_pack(size=120, tree_m=0.4)
    near = floor_along(
        pack,
        [at(0.0, 0.0), at(9.9, 0.0)],
        flight_date=FLIGHT,
        laser=seen(0.0, GRF250),
        start_alt_amsl_m=START_AMSL,
    )
    assert near.laser_confirmed
    assert agl(near.floor_lowest_amsl_m) == pytest.approx(12.0)
    far = floor_along(
        pack,
        [at(0.0, 0.0), at(40.0, 0.0)],
        flight_date=FLIGHT,
        laser=seen(0.0, GRF250),
        start_alt_amsl_m=START_AMSL,
    )
    assert far.laser_confirmed is False
    assert agl(far.floor_lowest_amsl_m) == pytest.approx(20.9, abs=0.05)


def test_floor_along_without_a_pack_or_a_path() -> None:
    r = floor_along(None, [at(0, 0)], flight_date=FLIGHT, laser=None, start_alt_amsl_m=99.0)
    assert (r.reason, r.floor_lowest_amsl_m) == ("no_map", 99.0)
    with pytest.raises(ValueError, match="one point"):
        floor_along(make_pack(), [], flight_date=FLIGHT, laser=None, start_alt_amsl_m=99.0)


def test_pure_once_the_pack_is_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """No file, clock or socket once the pack is in memory, and the same answer each time."""
    pack = make_pack(tree_m=7.0, trees={(2.0, 2.0): -1}, lines=(((-50.0, 40.0), (50.0, 40.0)),))
    path = [at(0.0, 0.0), at(15.0, 15.0)]
    laser = seen(5.0, GRF250)

    def run() -> tuple[FloorResult, FloorResult]:
        at_point = floor(pack, 1.0, 1.0, laser=laser)
        along = floor_along(pack, path, flight_date=FLIGHT, laser=laser, start_alt_amsl_m=99.0)
        return at_point, along

    first = run()

    def forbidden(*_a: object, **_k: object) -> None:
        raise AssertionError("the floor must not read clocks or do I/O")

    for name in ("time", "monotonic", "monotonic_ns", "perf_counter", "time_ns"):
        monkeypatch.setattr(time, name, forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    assert run() == first == run()


# --- the pack -------------------------------------------------------------------------------


def test_pack_round_trip(tmp_path: Path) -> None:
    pack = make_pack(tree_m=24.0, scan_year=2021)
    folder = write_pack(tmp_path / "p", pack, lines=[[at(-50.0, 20.0), at(50.0, 20.0)]])
    loaded = load_floor_pack(folder)
    assert loaded.name == "synthetic"
    assert loaded.attribution == "synthetic test data"
    assert loaded.sha256 == hashlib.sha256((folder / "floor_pack.toml").read_bytes()).hexdigest()
    assert loaded.origin == ORIGIN
    np.testing.assert_array_equal(loaded.chm_dm, pack.chm_dm)
    assert len(loaded.lines) == 1
    assert loaded.lines[0][0] == pytest.approx((-50.0, 20.0), abs=1e-6)
    assert floor(loaded).reason == "powerline"
    assert floor(loaded, north=-20.0) == floor(pack, north=-20.0)


def test_pack_in_memory_read_only_and_as_hashed(tmp_path: Path) -> None:
    """The grids are the bytes whose hash was checked: a later write to the files on disk
    changes nothing, and the grids cannot be written through."""
    folder = write_pack(tmp_path / "p", make_pack(tree_m=24.0, scan_year=2021))
    loaded = load_floor_pack(folder)
    before = floor(loaded)
    np.save(folder / "chm_dm.npy", np.zeros((60, 60), np.int16))
    np.save(folder / "dtm_cm.npy", np.zeros((60, 60), np.int32))
    assert floor(loaded) == before
    for grid in (loaded.dtm_cm, loaded.chm_dm, loaded.scan_year):
        with pytest.raises(ValueError, match="read-only"):
            grid[0, 0] = 1


def test_pack_multilinestring(tmp_path: Path) -> None:
    folder = write_pack(tmp_path / "p", make_pack())
    multi = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": [
                        [[p.lon, p.lat] for p in (at(0, 0), at(5, 0))],
                        [[p.lon, p.lat] for p in (at(0, 9), at(5, 9), at(9, 9))],
                    ],
                },
            }
        ],
    }
    _rewrite(folder, "lines.geojson", json.dumps(multi).encode())
    assert [len(line) for line in load_floor_pack(folder).lines] == [2, 3]


def _rewrite(folder: Path, name: str, data: bytes) -> None:
    """Replace a file and its hash, so only the change itself is under test."""
    (folder / name).write_bytes(data)
    toml = folder / "floor_pack.toml"
    text = toml.read_text(encoding="utf-8").splitlines()
    sha = hashlib.sha256(data).hexdigest()
    toml.write_text(
        "\n".join(f'"{name}" = "{sha}"' if line.startswith(f'"{name}"') else line for line in text)
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda f: (f / "chm_dm.npy").write_bytes(b"altered"), "sha256"),
        (lambda f: (f / "scan_year.npy").unlink(), "not a readable"),
        (lambda f: (f / "floor_pack.toml").unlink(), "not a readable"),
        (lambda f: _toml(f, "format = 1", "format = 2"), "format"),
        (lambda f: _toml(f, "size_m = [60, 60]", "size_m = [60, 61]"), "size"),
        (lambda f: _toml(f, "scan_cell_m = 2500", "scan_cell_m = 0"), "scan_cell_m"),
        (lambda f: _toml(f, "lat = 59.3", "lat = 95.0"), "not a readable"),
        (lambda f: _toml(f, '[sha256]\n"dtm_cm.npy"', '[sha256]\n"dtm.npy"'), "not a readable"),
        (lambda f: _rewrite(f, "lines.geojson", b'{"type": "Feature"}'), "FeatureCollection"),
        (lambda f: _rewrite(f, "lines.geojson", b"not json"), "not a readable"),
        (
            lambda f: _rewrite(
                f,
                "lines.geojson",
                b'{"type": "FeatureCollection", "features": [{"type": "Feature", '
                b'"geometry": {"type": "Point", "coordinates": [18.2, 59.3]}}]}',
            ),
            "Point",
        ),
        (
            lambda f: _rewrite(
                f,
                "lines.geojson",
                b'{"type": "FeatureCollection", "features": [{"type": "Feature", '
                b'"geometry": {"type": "LineString", "coordinates": []}}]}',
            ),
            "at least one point",
        ),
        (lambda f: _rewrite(f, "dtm_cm.npy", _npy(np.zeros((60, 60), np.float32))), "int32"),
        (lambda f: _rewrite(f, "chm_dm.npy", _npy(_grid(np.int16, -2))), "below -1"),
        (lambda f: _rewrite(f, "chm_dm.npy", _npy(_grid(np.int16, -32768))), "below -1"),
        (lambda f: _rewrite(f, "dtm_cm.npy", _npy(_grid(np.int32, -999_900))), "outside"),
        (lambda f: _rewrite(f, "dtm_cm.npy", _npy(_grid(np.int32, 900_001))), "outside"),
        (
            lambda f: _rewrite(f, "dtm_cm.npy", _npy(_grid(np.int32, np.iinfo(np.int32).min))),
            "no ground height",
        ),
    ],
)
def test_pack_refused(tmp_path: Path, change: Callable[[Path], object], message: str) -> None:
    folder = write_pack(tmp_path / "p", make_pack())
    change(folder)
    with pytest.raises(FloorPackError, match=message):
        load_floor_pack(folder)


def _toml(folder: Path, old: str, new: str) -> None:
    path = folder / "floor_pack.toml"
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


def _grid(dtype: type[np.integer[Any]], value: int) -> NDArray[Any]:
    return np.full((60, 60), value, dtype)


def _npy(array: NDArray[Any]) -> bytes:
    buf = BytesIO()
    np.save(buf, array)
    return buf.getvalue()


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"dtm_cm": np.zeros((4, 4), np.int16)}, "int32"),
        ({"dtm_cm": np.zeros((0, 0), np.int32)}, "int32"),
        ({"chm_dm": np.zeros((4, 5), np.int16)}, "int16"),
        ({"chm_dm": np.zeros((4, 4), np.int32)}, "int16"),
        ({"scan_cell_m": 0}, "scan_cell_m"),
        ({"scan_year": np.zeros((2, 2), np.uint16)}, "scan_year"),
        ({"scan_year": np.zeros((1, 1), np.int16)}, "scan_year"),
        ({"lines": ((),)}, "at least one point"),
    ],
)
def test_pack_in_memory_refused(kw: dict[str, object], message: str) -> None:
    args: dict[str, object] = {
        "name": "x",
        "sha256": "0" * 64,
        "origin": ORIGIN,
        "dtm_cm": np.zeros((4, 4), np.int32),
        "chm_dm": np.zeros((4, 4), np.int16),
        "scan_year": np.zeros((1, 1), np.uint16),
        "scan_cell_m": 2500,
    }
    args.update(kw)
    with pytest.raises(FloorPackError, match=message):
        FloorPack(**args)  # pyright: ignore[reportArgumentType]


def test_flight_date_is_a_date() -> None:
    """The flight's year is what counts (a scan in 2019, a flight on new year's eve)."""
    pack = make_pack(tree_m=0.4, scan_year=2019)
    assert floor(pack, flight_date=date(2026, 12, 31)).tree_m == pytest.approx(5.9)
