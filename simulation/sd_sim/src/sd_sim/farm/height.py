"""The round's flight height, clear of the tallest ground and tree top it flies over.

The seed draws a height (``scenario.HEIGHTS_M``), but over a world with trees that height
can be under a tree top. So a round flies at least ``CLEARANCE_M`` over the highest ground
or tree top under and beside its path: home to the search box's pin and the search box,
widened by ``BESIDE_M``. The clearance is the one the site worlds' fixtures keep
(``site_world.SEARCH_CLEARANCE_M``), rounded up to whole ``STEP_M``. The height is never
above the test charter's ceiling. When the clearance and the ceiling cannot both hold, the
round is skipped before the simulator starts, with the reason in ``round.json``.

Heights are metres above home, the same as the script's ``alt_m``. Tree heights stay here.
The round records only the height it drew and the one it flies.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from pathlib import Path

import shapely

from sd_sim import fixture, paths, site, site_world, terrain, world
from sd_sim.farm import scenario
from sd_sim.farm.round import RoundError, SkippedError
from sd_sim.targets import search_box

CLEARANCE_M = site_world.SEARCH_CLEARANCE_M
BESIDE_M = 15.0  # half the search's line spacing: how far the camera's path passes a line
STEP_M = 5.0
CHARTER = "charter_test"
LAKE_FOREST_CROWN = 0.15  # crown radius over height, as sd_lake_forest.sdf.jinja draws it


def flight_area(fix: fixture.Fixture) -> shapely.Geometry:
    """Where the round flies, in metres east/north of home: the line from home to the
    search box's pin, the search box, and ``BESIDE_M`` around both."""
    pin = fix.pins[fix.survey_box.center_pin].xy
    path = shapely.union(
        shapely.LineString([(0.0, 0.0), pin]), shapely.Polygon(fix.survey_box.corners())
    )
    return shapely.buffer(path, BESIDE_M)


def lake_forest_top(area: shapely.Geometry) -> float:
    """The highest tree top whose crown reaches ``area`` in ``sd_lake_forest`` (flat ground)."""
    top = 0.0
    for t in world.trees():
        if shapely.distance(area, shapely.Point(t.x, t.y)) <= LAKE_FOREST_CROWN * t.height:
            top = max(top, world.ground_z(t.x, t.y) + t.height)
    return top


def obstacle_top(world_name: str, area: shapely.Geometry, repo: Path = paths.REPO) -> float:
    """The highest ground or tree top in ``area``, in metres above home."""
    if world_name == world.WORLD_NAME:
        return lake_forest_top(area)
    sites = site.world_sites(repo)
    if world_name not in sites:
        raise RoundError(f"okänd värld {world_name!r}: vet inte hur högt det är där")
    s = site.load(sites[world_name], repo)
    return site_world.tallest(s, terrain.ground_grid(s), area)


def floor_m(top_m: float) -> float:
    """The lowest height that keeps ``CLEARANCE_M`` over ``top_m``, in whole ``STEP_M``."""
    return math.ceil((top_m + CLEARANCE_M) / STEP_M) * STEP_M


def flown_m(drawn_m: float, top_m: float, ceiling_m: float) -> float:
    """The drawn height, raised to the clearance over ``top_m``; ``SkippedError`` when that
    is above ``ceiling_m``. The message names no tree height."""
    lowest = floor_m(top_m)
    if lowest > ceiling_m:
        raise SkippedError(
            f"varvet hoppas över: marken och träden under flygbanan kräver mer än "
            f"ramens tak på {ceiling_m:g} m för att hålla {CLEARANCE_M:g} m marginal"
        )
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
    top = obstacle_top(sc.world, flight_area(fix), repo)
    height = flown_m(sc.height_m, top, fix.charters[CHARTER].alt_max_agl_m)
    return dataclasses.replace(
        sc, height_m=height, extra={**sc.extra, "height_drawn_m": sc.height_m}
    )
