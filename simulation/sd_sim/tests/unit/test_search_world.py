"""A search scenario's world: the same seed gives the same world, inside the charter."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from shapely.geometry import LineString, Point, Polygon

from sd_sim import search_world
from sd_sim.search_world import SCENARIOS, WorldError, load, scenario_path, seeds

SCENARIO_IDS = ("SK01", "SK02", "SK03", "SK04", "SK06", "SK09")


def test_seeds_parse_ranges_and_lists() -> None:
    assert seeds("1-3") == [1, 2, 3]
    assert seeds("4, 2,9") == [4, 2, 9]
    assert len(seeds("1-50")) == 50


@pytest.mark.parametrize("bad", ["", "0", "a", "1-x", "3,3", "1-201", "-2"])
def test_seeds_refuse_bad_input(bad: str) -> None:
    with pytest.raises(WorldError):
        seeds(bad)


@pytest.mark.parametrize("scenario", SCENARIO_IDS)
def test_every_scenario_loads_inside_its_charter(scenario: str) -> None:
    for seed in range(1, 11):
        w = load(scenario, seed)
        assert w.scenario == scenario and w.seed == seed
        assert w.fixture_path == scenario_path(scenario)
        for t in w.things:
            assert w.nearest(t.east, t.north, 0.01) is not None
            assert math.hypot(t.east, t.north) < 500  # inside the pilot's range
        assert sum(t.target for t in w.things) == (0 if scenario == "SK03" else 1)


def test_the_same_seed_gives_the_same_world_and_another_seed_moves_it() -> None:
    a, b, c = load("SK01", 1), load("SK01", 1), load("SK01", 2)
    assert a.things == b.things
    assert a.target is not None and c.target is not None
    assert (a.target.east, a.target.north) != (c.target.east, c.target.north)


def test_the_target_stays_within_its_jitter() -> None:
    for seed in range(1, 31):
        t = load("SK01", seed).target
        assert t is not None
        assert math.hypot(t.east - 210, t.north - 210) <= 40.0 + 1e-6


def test_sk06_person_lies_20_to_80_m_from_the_trail_inside_the_sectors() -> None:
    for seed in range(1, 31):
        w = load("SK06", seed)
        t = w.target
        assert t is not None
        d = LineString(w.trail).distance(Point(t.east, t.north))
        assert 19.5 <= d <= 80.5, (seed, d)
        assert Polygon(w.areas["all_sectors"]).contains(Point(t.east, t.north))
        assert w.sector_order == ("A", "B", "C", "D")


def test_the_trail_is_a_terrain_path_in_lon_lat() -> None:
    w = load("SK06", 1)
    (feature,) = w.terrain()
    assert feature["properties"]["kind"] == "path"
    lon, lat = feature["geometry"]["coordinates"][0]
    assert (lat, lon) == pytest.approx(w.latlon(*w.trail[0]))
    assert load("SK01", 1).terrain() == []


def test_the_profile_draft_has_last_seen_but_no_truth() -> None:
    w = load("SK01", 1)
    draft = w.profile_draft("2026-10-23T10:00:00+00:00")
    assert draft["last_seen"]["at_utc"].startswith("2026-10-23")
    assert "people" not in draft and "target" not in str(draft)


def _write(tmp_path: Path, search: str) -> Path:
    base = scenario_path("SK01").read_text(encoding="utf-8")
    head = base[: base.index("search:")]
    path = tmp_path / "SK99_test.yaml"
    path.write_text(head + search, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "search",
    [
        "search:\n  unknown: 1\n",
        "search:\n  people:\n    - { at_enu: [0, 0], upper: mauve }\n",
        "search:\n  people:\n    - { at_enu: [0, 0], pose: flying }\n",
        "search:\n  people:\n    - { at_enu: [0, 0], target: true }\n"
        "    - { at_enu: [9, 9], target: true }\n",
        "search:\n  decoys:\n    - { kind: car, at_enu: [0, 0] }\n",
        "search:\n  people:\n    - { near_trail_m: [20, 80] }\n",
        "search:\n  people:\n    - { at_enu: [0, 0], area: nowhere }\n",
        "search:\n  people:\n    - { at_enu: [900, 900] }\n",
    ],
)
def test_bad_scenarios_are_refused(tmp_path: Path, search: str) -> None:
    with pytest.raises(WorldError):
        load(_write(tmp_path, search), 1)


def test_scenario_names_are_unique() -> None:
    assert len(list(SCENARIOS.glob("SK01_*.yaml"))) == 1
    with pytest.raises(WorldError):
        search_world.scenario_path("SK77")
