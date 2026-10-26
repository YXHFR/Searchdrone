"""One walker by the same rules as the map: for the simulator's true path.

Each step the walker may stop (for good), turn 45 degrees or pick any heading, turn along
a track it stands on or towards one it is next to, and then walks at the terrain's speed.
Water turns it back. Positions are in cells (row, column) as floats; a walker that leaves
the grid is gone, and its position is NaN from then on.
"""

from __future__ import annotations

import math

import numpy as np

from sd_motion.model import DT_S
from sd_motion.params import Params
from sd_motion.terrain import DIRS, Terrain


def _along_track(heading: int, axis: int, rng: np.random.Generator) -> int:
    d = (heading - axis) % 8
    if d in (0, 1, 7):
        return axis
    if d in (3, 4, 5):
        return axis + 4
    return axis if rng.random() < 0.5 else axis + 4


def walk(
    terrain: Terrain,
    params: Params,
    start_rc: tuple[float, float],
    t_s: float,
    minutes: int,
    rng: np.random.Generator,
    moving: bool = True,
) -> np.ndarray:
    """Positions at each whole minute from ``t_s`` seconds after last seen, (minutes + 1, 2)."""
    h, w = terrain.shape
    out = np.full((minutes + 1, 2), np.nan)
    r, c = start_rc
    heading = int(rng.integers(8))
    per_min = round(60.0 / DT_S)
    a = params.turn45 * DT_S / 60.0
    b = params.turn_any * DT_S / 60.0
    k = params.k_track * DT_S / 60.0
    out[0] = r, c
    for minute in range(1, minutes + 1):
        for _ in range(per_min if moving else 0):
            cell = int(r), int(c)
            s0 = params.still_moving(t_s / 3600.0)
            keep = params.still_moving((t_s + DT_S) / 3600.0) / s0 if s0 > 0.0 else 0.0
            if terrain.trap[cell]:
                keep **= params.trap
            t_s += DT_S
            if rng.random() >= keep:
                moving = False
                break
            u = rng.random()
            if u < a:
                heading = (heading + (1 if rng.random() < 0.5 else -1)) % 8
            elif u < a + b:
                heading = int(rng.integers(8))
            if rng.random() < k:
                if terrain.track_dir[cell] >= 0:
                    heading = _along_track(heading, int(terrain.track_dir[cell]), rng)
                elif terrain.near_track_dir[cell] >= 0:
                    heading = int(terrain.near_track_dir[cell])
            dr, dc = DIRS[heading]
            step = params.speed_mps * terrain.speed_factor[cell] * DT_S / terrain.cell_m
            step /= math.hypot(dr, dc)
            nr, nc = r + dr * step, c + dc * step
            if not (0.0 <= nr < h and 0.0 <= nc < w):
                return out  # off the map: gone
            if terrain.wall[int(nr), int(nc)]:
                heading = (heading + 4) % 8
            else:
                r, c = nr, nc
        out[minute] = r, c
    return out
