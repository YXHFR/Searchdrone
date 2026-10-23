"""The probability map's grid: square cells in a local east/north plane with its origin at
the place the person was last seen (the IPP).

Row ``j`` is the cells ``y0 + j * cell`` to ``y0 + (j + 1) * cell`` metres north of the IPP,
column ``i`` likewise east; ``p[j, i]``. The grid covers the charter's area plus a margin,
never more than ``GRID_MAX_M`` on a side. Projection uses the same WGS84 tangent plane as
the rest of the system (``sd_schemas.geo``), so a cell here is the cell the gate sees.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Final

import numpy as np
from numpy.typing import NDArray

from sd_schemas.geo import tangent_radii, wrap_lon
from sd_schemas.probmap import ProbabilityGrid

__all__ = [
    "GRID_CELL_M",
    "GRID_MARGIN_M",
    "GRID_MAX_M",
    "NO_AREA_HALF_M",
    "Floats",
    "Grid",
    "grid_around",
]

GRID_CELL_M: Final = 25.0
GRID_MARGIN_M: Final = 500.0  # around the charter's area
GRID_MAX_M: Final = 4000.0  # at most 160 x 160 cells
NO_AREA_HALF_M: Final = 2000.0  # no charter yet: a square around the IPP
SUB: Final = 4  # sub-points per cell side for integrals and overlaps

Floats = NDArray[np.float64]


@dataclass(frozen=True)
class Grid:
    lat0: float  # the IPP
    lon0: float
    x0: float  # south-west corner, metres east of the IPP
    y0: float  # metres north of the IPP
    nx: int
    ny: int
    cell: float = GRID_CELL_M

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def size(self) -> int:
        return self.nx * self.ny

    @cached_property
    def _radii(self) -> tuple[float, float]:
        return tangent_radii(self.lat0)

    def xy(self, lat: float, lon: float) -> tuple[float, float]:
        east_r, north_r = self._radii
        return (
            math.radians(wrap_lon(lon - self.lon0)) * east_r,
            math.radians(lat - self.lat0) * north_r,
        )

    def latlon(self, x: float, y: float) -> tuple[float, float]:
        east_r, north_r = self._radii
        return (
            self.lat0 + math.degrees(y / north_r),
            wrap_lon(self.lon0 + math.degrees(x / east_r)),
        )

    def ring_xy(self, ring: Iterable[Sequence[float]]) -> list[tuple[float, float]]:
        """A GeoJSON ring ([lon, lat] pairs) in the plane."""
        return [self.xy(float(p[1]), float(p[0])) for p in ring]

    @cached_property
    def centers(self) -> tuple[Floats, Floats]:
        """(x, y) of every cell's centre, each of shape ``(ny, nx)``."""
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.cell
        ys = self.y0 + (np.arange(self.ny) + 0.5) * self.cell
        gx, gy = np.meshgrid(xs, ys)
        return np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64)

    @cached_property
    def subpoints(self) -> tuple[Floats, Floats]:
        """``SUB`` x ``SUB`` points per cell, shape ``(ny, nx, SUB * SUB)``."""
        off = (np.arange(SUB) + 0.5) / SUB * self.cell - self.cell / 2
        dx, dy = np.meshgrid(off, off)
        cx, cy = self.centers
        return (
            cx[..., None] + dx.ravel()[None, None, :],
            cy[..., None] + dy.ravel()[None, None, :],
        )

    def cell_of(self, x: float, y: float) -> tuple[int, int] | None:
        """(row, column) of the cell holding a point, or ``None`` outside the grid."""
        i = math.floor((x - self.x0) / self.cell)
        j = math.floor((y - self.y0) / self.cell)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return j, i
        return None

    def window(
        self, xmin: float, ymin: float, xmax: float, ymax: float
    ) -> tuple[slice, slice] | None:
        """The rows and columns of the cells a box touches, or ``None`` if it misses."""
        i0 = max(0, math.floor((xmin - self.x0) / self.cell))
        i1 = min(self.nx, math.ceil((xmax - self.x0) / self.cell))
        j0 = max(0, math.floor((ymin - self.y0) / self.cell))
        j1 = min(self.ny, math.ceil((ymax - self.y0) / self.cell))
        if i0 >= i1 or j0 >= j1:
            return None
        return slice(j0, j1), slice(i0, i1)

    def bounds_lonlat(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) of the grid."""
        south, west = self.latlon(self.x0, self.y0)
        north, east = self.latlon(self.x0 + self.nx * self.cell, self.y0 + self.ny * self.cell)
        return west, south, east, north

    def model(self) -> ProbabilityGrid:
        west, south, east, north = self.bounds_lonlat()
        return ProbabilityGrid(
            ipp_lat=self.lat0,
            ipp_lon=self.lon0,
            x0_m=self.x0,
            y0_m=self.y0,
            nx=self.nx,
            ny=self.ny,
            cell_m=self.cell,
            west=west,
            south=south,
            east=east,
            north=north,
        )


def _span(lo: float, hi: float, limit: float) -> tuple[float, float]:
    """``[lo, hi]`` cut to ``limit`` around its centre."""
    if hi - lo <= limit:
        return lo, hi
    mid = (lo + hi) / 2
    return mid - limit / 2, mid + limit / 2


def grid_around(
    lat0: float,
    lon0: float,
    area: Sequence[tuple[float, float]] | None,
    *,
    cell: float = GRID_CELL_M,
    margin: float = GRID_MARGIN_M,
    limit: float = GRID_MAX_M,
) -> Grid:
    """The grid for an IPP and the charter's area (plane points; ``None``: no charter
    yet, a square around the IPP). Corners snap to whole cells from the IPP, so the same
    inputs always give the same cells."""
    if area:
        xs = [p[0] for p in area]
        ys = [p[1] for p in area]
        xlo, xhi = _span(min(xs) - margin, max(xs) + margin, limit)
        ylo, yhi = _span(min(ys) - margin, max(ys) + margin, limit)
    else:
        half = min(NO_AREA_HALF_M, limit / 2)
        xlo, xhi, ylo, yhi = -half, half, -half, half
    i0, i1 = math.floor(xlo / cell), math.ceil(xhi / cell)
    j0, j1 = math.floor(ylo / cell), math.ceil(yhi / cell)
    cap = max(1, round(limit / cell))
    return Grid(
        lat0=lat0,
        lon0=lon0,
        x0=i0 * cell,
        y0=j0 * cell,
        nx=max(1, min(cap, i1 - i0)),
        ny=max(1, min(cap, j1 - j0)),
        cell=cell,
    )
