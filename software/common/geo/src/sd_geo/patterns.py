"""The lawnmower search pattern over a polygon in a local east/north plane (metres).

Lines run in the direction ``heading_deg`` (compass degrees: 0 = north, 90 = east,
clockwise). They are ``spacing_m`` apart, the outer ones half a spacing inside the
polygon, and every second line is flown backwards. Concave polygons can cut a line in
several pieces; they are flown in order along the line.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from shapely import LineString, MultiLineString
from shapely import Polygon as ShapelyPolygon
from shapely.affinity import rotate

__all__ = ["MAX_CORNERS", "PatternError", "Xy", "lawnmower"]

Xy = tuple[float, float]
MAX_CORNERS = 20  # a search polygon has at most this many corners


class PatternError(ValueError):
    """The polygon or the spacing cannot give a pattern."""


def _longest_side_heading(corners: Sequence[Xy]) -> float:
    best, heading = -1.0, 0.0
    for i, (x1, y1) in enumerate(corners):
        x2, y2 = corners[(i + 1) % len(corners)]
        length = math.hypot(x2 - x1, y2 - y1)
        if length > best:
            best = length
            heading = math.degrees(math.atan2(x2 - x1, y2 - y1)) % 180.0
    return heading


def lawnmower(
    polygon: Sequence[Xy],
    spacing_m: float,
    heading_deg: float | None,
    start_near: Xy,
) -> list[Xy]:
    """The waypoints (east, north) of the pattern, starting at the corner nearest
    ``start_near``. ``heading_deg = None`` runs the lines along the longest side."""
    corners = list(polygon)
    if len(corners) > 1 and corners[0] == corners[-1]:
        corners = corners[:-1]
    if not 3 <= len(corners) <= MAX_CORNERS:
        raise PatternError(f"a search polygon has 3..{MAX_CORNERS} corners, not {len(corners)}")
    if not (math.isfinite(spacing_m) and spacing_m > 0):
        raise PatternError(f"spacing {spacing_m}")
    shape = ShapelyPolygon(corners)
    if not shape.is_valid or shape.area <= 0:
        raise PatternError("the polygon crosses itself or has no area")
    heading = _longest_side_heading(corners) if heading_deg is None else heading_deg
    theta = 90.0 - heading  # the line direction as an angle counter-clockwise from east
    # Rotate so that the lines run along x.
    turned = rotate(shape, -theta, origin=(0.0, 0.0))
    _, y_min, _, y_max = turned.bounds
    x_min, _, x_max, _ = turned.bounds
    width = y_max - y_min
    inset = spacing_m / 2
    if width <= 2 * inset:
        ys = [y_min + width / 2]
    else:
        n = max(1, math.ceil((width - 2 * inset) / spacing_m) + 1)
        d = (width - 2 * inset) / (n - 1) if n > 1 else 0.0
        ys = [y_min + inset + i * d for i in range(n)] if n > 1 else [y_min + width / 2]
    lines: list[list[Xy]] = []
    for y in ys:
        cut = turned.intersection(LineString([(x_min - 1.0, y), (x_max + 1.0, y)]))
        pieces = _pieces(cut)
        if not pieces:
            continue
        line: list[Xy] = []
        for a, b in sorted(pieces):
            line += [(a, y), (b, y)]
        lines.append(line)
    if not lines:
        raise PatternError("no search line fits inside the polygon")
    # Start at one of the four ends of the first or the last line.
    candidates = [
        _zigzag(order, flip_first) for order in (lines, lines[::-1]) for flip_first in (False, True)
    ]
    best = min(candidates, key=lambda c: math.dist(_rotate(c[0], theta), start_near))
    return [_rotate(p, theta) for p in best]


def _zigzag(lines: list[list[Xy]], flip_first: bool) -> list[Xy]:
    """The lines in this order, every second one flown backwards."""
    out: list[Xy] = []
    for i, line in enumerate(lines):
        out += line[::-1] if (i % 2 == 1) != flip_first else line
    return out


def _pieces(geom: object) -> list[tuple[float, float]]:
    if isinstance(geom, LineString):
        segments = [geom] if not geom.is_empty else []
    elif isinstance(geom, MultiLineString):
        segments = [g for g in geom.geoms if not g.is_empty]
    else:
        segments = [g for g in getattr(geom, "geoms", []) if isinstance(g, LineString)]
    out: list[tuple[float, float]] = []
    for seg in segments:
        xs = [c[0] for c in seg.coords]
        if max(xs) - min(xs) > 1e-9:
            out.append((min(xs), max(xs)))
    return out


def _rotate(p: Xy, deg: float) -> Xy:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)
