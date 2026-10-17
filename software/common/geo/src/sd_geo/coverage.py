"""How much of a search polygon a flown track has covered (local east/north metres)."""

from __future__ import annotations

import math
from collections.abc import Sequence

from shapely import Point, prepared
from shapely import Polygon as ShapelyPolygon
from shapely.strtree import STRtree

__all__ = ["ALT_TOLERANCE_M", "CELL_M", "REACH_EXTRA_M", "coverage_fraction"]

CELL_M = 5.0  # the polygon is cut into cells of this size
ALT_TOLERANCE_M = 3.0  # track samples further than this from the search height do not count
REACH_EXTRA_M = 2.0  # a cell is covered within spacing/2 + this of a counted sample


def coverage_fraction(
    polygon: Sequence[tuple[float, float]],
    track: Sequence[tuple[float, float, float]],
    alt_target_m: float,
    spacing_m: float,
) -> float:
    """The share (0..1) of 5 x 5 m cells inside ``polygon`` whose centre lies within
    ``spacing_m / 2 + 2 m`` of a track sample (east, north, height above home) flown
    within 3 m of ``alt_target_m``."""
    shape = ShapelyPolygon(polygon)
    if shape.is_empty or shape.area <= 0:
        raise ValueError("the polygon has no area")
    reach = spacing_m / 2 + REACH_EXTRA_M
    samples = [Point(x, y) for x, y, alt in track if abs(alt - alt_target_m) <= ALT_TOLERANCE_M]
    inside = prepared.prep(shape)
    x0, y0, x1, y1 = shape.bounds
    cells: list[Point] = []
    for i in range(math.ceil((x1 - x0) / CELL_M)):
        for j in range(math.ceil((y1 - y0) / CELL_M)):
            centre = Point(x0 + (i + 0.5) * CELL_M, y0 + (j + 0.5) * CELL_M)
            if inside.contains(centre):
                cells.append(centre)
    if not cells:
        return 0.0
    if not samples:
        return 0.0
    tree = STRtree(samples)
    covered = sum(1 for c in cells if tree.query(c, predicate="dwithin", distance=reach).size)
    return covered / len(cells)
