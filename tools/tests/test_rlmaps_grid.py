"""The numpy half of ``just rl-maps`` (tools/rlmaps/grid.py): no network, no rasterio."""

from __future__ import annotations

import numpy as np
import pytest
from rlmaps import grid

N = grid.GRID_CELLS


def _classes(fill: int) -> np.ndarray:
    return np.full((N * grid.SUB, N * grid.SUB), fill, dtype=np.uint8)


def test_place_round_trip_puts_the_centre_in_the_middle() -> None:
    place = grid.Place(62.0, 15.0)
    xy = place.to_grid_m(np.array([15.0]), np.array([62.0]))[0]
    assert xy == pytest.approx([grid.SIZE_M / 2, grid.SIZE_M / 2])
    west, south, east, north = place.bounds()
    corner = place.to_grid_m(np.array([west]), np.array([south]))[0]
    assert corner == pytest.approx([0.0, 0.0], abs=1e-6)
    far = place.to_grid_m(np.array([east]), np.array([north]))[0]
    assert far == pytest.approx([grid.SIZE_M, grid.SIZE_M], abs=1e-6)


def test_sample_points_follow_the_grid_rows_north() -> None:
    lon, lat = grid.Place(60.0, 18.0).sample_points()
    assert lon.shape == lat.shape == (N * grid.SUB, N * grid.SUB)
    assert (np.diff(lat[:, 0]) > 0).all()  # row 0 south
    assert (np.diff(lon[0, :]) > 0).all()  # column 0 west


def test_shares_and_canopy() -> None:
    classes = _classes(grid.TREE)
    classes[: grid.SUB, : grid.SUB] = grid.WATER  # one cell of water
    shares = grid.class_shares(classes)
    assert shares[grid.WATER][0, 0] == 1.0
    assert shares[grid.TREE][0, 0] == 0.0
    arrays = grid.site_arrays(shares, np.zeros((N, N), dtype=bool))
    assert arrays["water"][0, 0] and arrays["water"].sum() == 1
    assert arrays["canopy"][0, 0] == 0.0
    assert arrays["canopy"][1, 1] == pytest.approx(grid.CANOPY[grid.TREE])


def test_rasterise_marks_the_cells_a_line_crosses() -> None:
    line = np.array([[12.5, 12.5], [12.5 + 4 * grid.CELL_M, 12.5]])  # five cells east
    mask = grid.rasterise([line])
    assert mask[0, :5].all() and mask.sum() == 5
    outside = np.array([[-500.0, -500.0], [-400.0, -400.0]])
    assert not grid.rasterise([outside]).any()


def test_acceptable_says_why_not() -> None:
    track = np.zeros((N, N), dtype=bool)
    track[N // 2, :] = True
    assert grid.acceptable(grid.class_shares(_classes(grid.TREE)), track) is None
    assert grid.acceptable(grid.class_shares(_classes(grid.WATER)), track) is not None
    assert grid.acceptable(grid.class_shares(_classes(grid.BUILT)), track) is not None
    assert grid.acceptable(grid.class_shares(_classes(grid.CROP)), track) is not None
    no_track = np.zeros((N, N), dtype=bool)
    assert grid.acceptable(grid.class_shares(_classes(grid.TREE)), no_track) is not None
