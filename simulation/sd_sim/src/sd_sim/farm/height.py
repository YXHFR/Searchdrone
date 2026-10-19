"""The round's flight height, clear of the tallest ground and tree top it flies over.

The seed draws a height (``scenario.HEIGHTS_M``), but over a world with trees that height
can be under a tree top. So a round flies at least ``CLEARANCE_M`` over the highest ground
or tree top where it can fly (``flight_area``): home, the search box's pin, the box and the
straight way home between them, widened by ``BESIDE_M``. The clearance is the site worlds'
fixtures' own (``site_world.search_floor_m``, in whole 5 m), and the height is never above
the test charter's ceiling. When both cannot hold, the round is skipped before the
simulator starts, with the reason in ``round.json``. A decision maker that picks its own
heights gets a charter whose lowest height is the round's (``charter_floor``).

Heights are metres above home, the same as the script's ``alt_m``. The round records the
height it drew and the one it flies, never a tree. A raised height still tells the area's
highest top to within the 5 m step.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import shapely
import yaml

from sd_sim import fixture, paths, site, site_world, terrain, world
from sd_sim.farm import scenario
from sd_sim.farm.round import RoundError, SkippedError
from sd_sim.targets import search_box

CLEARANCE_M = site_world.SEARCH_CLEARANCE_M
BESIDE_M = 15.0  # half the search's line spacing: how far the camera's path passes a line
CHARTER = "charter_test"
SKIPPED = (
    f"varvet hoppas över: marken och träden under flygbanan går inte att hålla "
    f"{CLEARANCE_M:g} m under ramens tak"
)
LAKE_FOREST_CROWN = 0.15  # crown radius over height, as sd_lake_forest.sdf.jinja draws it


def flight_area(fix: fixture.Fixture) -> shapely.Geometry:
    """Where the round can fly, in metres east/north of home: the hull of home, the search
    box's pin and the box, with ``BESIDE_M`` around it. The hull also holds a straight
    return home (RTL) from anywhere in the box, not only the way out."""
    pin = fix.pins[fix.survey_box.center_pin].xy
    points = shapely.MultiPoint([(0.0, 0.0), pin, *fix.survey_box.corners()])
    return shapely.buffer(shapely.convex_hull(points), BESIDE_M)


def charter_floor(height_m: float) -> dict[str, float]:
    """The test charter's overrides that keep a decision maker that picks its own heights
    (sd-brain) at the round's height or above: the gate rejects any order below it."""
    return {"alt_min_agl_m": height_m, "default_alt_agl_m": height_m}


def lake_forest_top(area: shapely.Geometry) -> float:
    """The highest tree top whose crown reaches ``area`` in ``sd_lake_forest`` (flat ground)."""
    top = 0.0
    for t in world.trees():
        if shapely.distance(area, shapely.Point(t.x, t.y)) <= LAKE_FOREST_CROWN * t.height:
            top = max(top, world.ground_z(t.x, t.y) + t.height)
    return top


def clear_of(world_name: str, fix: fixture.Fixture, repo: Path = paths.REPO) -> tuple[float, float]:
    """The highest ground or tree top under and beside the flight in ``world_name`` and the
    test charter's ceiling, in metres above home. ``sd_lake_forest`` flies ``fix``; a site
    world flies its own fixture (``site_world.fixture_yaml``)."""
    if world_name == world.WORLD_NAME:
        return lake_forest_top(flight_area(fix)), fix.charters[CHARTER].alt_max_agl_m
    sites = site.world_sites(repo)
    if world_name not in sites:
        raise RoundError(f"okänd värld {world_name!r}: vet inte hur högt det är där")
    s = site.load(sites[world_name], repo)
    ground = terrain.ground_grid(s)
    try:
        text = site_world.fixture_yaml(s, ground, repo)
    except ValueError:
        raise SkippedError(SKIPPED) from None  # its own message names the tallest top
    own = fixture.parse(yaml.safe_load(text))
    top = site_world.tallest(s, ground, flight_area(own))
    return top, own.charters[CHARTER].alt_max_agl_m


def flown_m(drawn_m: float, top_m: float, ceiling_m: float) -> float:
    """The drawn height, raised to the clearance over ``top_m``; ``SkippedError`` when that
    is not below ``ceiling_m``. The message names no tree height. The ceiling is the charter's
    height above ground and the height is above home: the same on the flat
    ``sd_lake_forest``, the only world the Gazebo round flies so far."""
    lowest = site_world.search_floor_m(top_m)
    if lowest >= ceiling_m:  # a charter's lowest height must be below its ceiling
        raise SkippedError(f"{SKIPPED} (ramens tak är {ceiling_m:g} m)")
    return max(drawn_m, lowest)


def draw(
    seed: int,
    worlds: Sequence[str],
    fix: fixture.Fixture | None = None,
    repo: Path = paths.REPO,
) -> scenario.FarmScenario:
    """The seed's round inside the fixture's search box, at a height clear of the trees."""
    fix = fixture.load() if fix is None else fix
    sc = scenario.sample(seed, worlds, search_box(fix), fix.home.lat)
    top, ceiling = clear_of(sc.world, fix, repo)
    height = flown_m(sc.height_m, top, ceiling)
    return dataclasses.replace(
        sc, height_m=height, extra={**sc.extra, "height_drawn_m": sc.height_m}
    )
