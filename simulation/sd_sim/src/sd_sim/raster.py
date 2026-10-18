"""Polygons onto regular grids (numpy only), for the generated terrain and its textures.

A grid is given by the coordinates of its cell centres as two 2-D arrays (``east``,
``north``) from ``numpy.meshgrid``, row 0 northernmost. ``coverage`` fills polygons with
the even-odd rule by scanlines, so holes (islands in a lake, clearings in a forest)
come out right. Everything is deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import MultiPolygon, Polygon

Grid = NDArray[np.float64]


def _rings(geom: Any) -> list[NDArray[np.float64]]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    elif hasattr(geom, "geoms"):
        polys = [g for g in geom.geoms if isinstance(g, Polygon | MultiPolygon)]
        return [r for p in polys for r in _rings(p)]
    else:
        return []
    out = []
    for poly in polys:
        out.append(np.asarray(poly.exterior.coords, dtype=np.float64))
        out.extend(np.asarray(ring.coords, dtype=np.float64) for ring in poly.interiors)
    return out


def frame(east: Grid, north: Grid) -> tuple[float, float, float, int, int]:
    """(west edge, north edge, step, width, height) of a grid of cell centres."""
    step = float(east[0, 1] - east[0, 0])
    return (
        float(east[0, 0]) - step / 2,
        float(north[0, 0]) + step / 2,
        step,
        east.shape[1],
        east.shape[0],
    )


def mask(
    geom: Any, west: float, north: float, step: float, width: int, height: int
) -> NDArray[np.bool_]:
    """Cells whose centre lies inside ``geom``."""
    diff = np.zeros((height, width + 1), dtype=np.int32)
    for ring in _rings(geom):
        px = (ring[:, 0] - west) / step - 0.5
        py = (north - ring[:, 1]) / step - 0.5
        x1, y1, x2, y2 = px[:-1], py[:-1], px[1:], py[1:]
        keep = y1 != y2
        for ax, ay, bx, by in zip(x1[keep], y1[keep], x2[keep], y2[keep], strict=True):
            lo, hi = (ay, by) if ay < by else (by, ay)
            r0 = max(int(np.ceil(lo)), 0)
            r1 = min(int(np.ceil(hi)), height)  # rows r with lo <= r < hi
            if r1 <= r0:
                continue
            rows = np.arange(r0, r1)
            xs = ax + (rows - ay) * (bx - ax) / (by - ay)
            cols = np.clip(np.ceil(xs).astype(np.int64), 0, width)
            np.add.at(diff, (rows, cols), 1)
    return (np.cumsum(diff[:, :width], axis=1) % 2).astype(bool)


def coverage(geom: Any, east: Grid, north: Grid) -> Grid:
    """1.0 where a cell centre is inside ``geom``, else 0.0."""
    return mask(geom, *frame(east, north)).astype(np.float64)


def box_blur(values: Grid, radius: int) -> Grid:
    """Mean over a (2r+1) square, edges clamped; separable and exact."""
    if radius <= 0:
        return values
    out = values
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius + 1, radius)
        padded = np.pad(out, pad, mode="edge")
        csum = np.cumsum(padded, axis=axis)
        n = out.shape[axis]
        hi = np.take(csum, np.arange(2 * radius + 1, 2 * radius + 1 + n), axis=axis)
        lo = np.take(csum, np.arange(0, n), axis=axis)
        out = (hi - lo) / (2 * radius + 1)
    return out


def soften(values: Grid, width_cells: float) -> Grid:
    """A smooth edge about ``width_cells`` wide (two box blurs, close to a Gaussian)."""
    radius = max(round(width_cells / 4), 0)
    return box_blur(box_blur(values, radius), radius)
