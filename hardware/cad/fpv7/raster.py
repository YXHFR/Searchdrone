"""Triangle rasterisation shared by the pictures (render.py) and the drag estimate
(aero.py): which cell centres of a grid a triangle covers, and its depth there."""

from __future__ import annotations

import math

import numpy as np


def triangle(
    px: np.ndarray, py: np.ndarray, pz: np.ndarray, shape: tuple[int, int]
) -> tuple[slice, slice, np.ndarray, np.ndarray] | None:
    """The cells of a ``shape`` (rows, columns) grid whose centres lie inside the triangle
    with corners (``px``, ``py``) in cell units: the window (rows, columns) round it, a
    mask of the covered cells in that window and the depth ``pz`` interpolated there.
    ``None`` if it covers no cell centre."""
    h, w = shape
    x0, x1 = max(math.floor(px.min()), 0), min(math.ceil(px.max()), w - 1)
    y0, y1 = max(math.floor(py.min()), 0), min(math.ceil(py.max()), h - 1)
    if x0 > x1 or y0 > y1:
        return None
    (ax, bx, cx), (ay, by, cy) = px, py
    area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    if abs(area) < 1e-12:
        return None
    xs = np.arange(x0, x1 + 1) + 0.5
    ys = np.arange(y0, y1 + 1)[:, None] + 0.5
    w0 = ((bx - xs) * (cy - ys) - (cx - xs) * (by - ys)) / area
    w1 = ((cx - xs) * (ay - ys) - (ax - xs) * (cy - ys)) / area
    w2 = 1 - w0 - w1
    inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
    if not inside.any():
        return None
    z = w0 * pz[0] + w1 * pz[1] + w2 * pz[2]
    return slice(y0, y1 + 1), slice(x0, x1 + 1), inside, z
