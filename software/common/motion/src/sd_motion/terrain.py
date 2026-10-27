"""What the ground does to a walker: speed, water walls, tracks and traps.

Built once per map. Row 0 is the southernmost row, so north is +row.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Headings N, NE, E, SE, S, SW, W, NW as (row, column) steps.
DIRS: tuple[tuple[int, int], ...] = (
    (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
)  # fmt: skip

# ESA WorldCover classes and the speed factor off a track (Tobler-style guesses).
WORLDCOVER_SPEED = {
    10: 0.45,  # tree cover
    20: 0.25,  # shrubland
    30: 0.6,  # grassland
    40: 0.5,  # cropland
    50: 0.8,  # built-up
    60: 0.6,  # bare / sparse vegetation
    70: 0.3,  # snow and ice
    80: 0.0,  # permanent water: a wall
    90: 0.2,  # herbaceous wetland
    95: 0.2,  # mangroves
    100: 0.6,  # moss and lichen
}
TRAP_CLASSES = (20, 90, 95)
TRACK_SPEED = 1.0
FOREST_SPEED = 0.45
OPEN_SPEED = 0.6
NEAR_TRACK_CELLS = 2


def _heading(dr: int, dc: int) -> int:
    return DIRS.index((int(np.sign(dr)), int(np.sign(dc))))


def _track_dir(track: np.ndarray) -> np.ndarray:
    """Per track cell, the axis (0..3: N-S, NE-SW, E-W, SE-NW) with the most track
    neighbours; -1 off the track or on a lone track cell."""
    h, w = track.shape
    padded = np.pad(track, 1)
    counts = np.zeros((4, h, w), dtype=np.int8)
    for k in range(4):
        for dr, dc in (DIRS[k], DIRS[k + 4]):
            counts[k] += padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
    best = counts.argmax(axis=0)
    return np.where(track & (counts.max(axis=0) > 0), best, -1).astype(np.int8)


def _near_track_dir(track: np.ndarray, wall: np.ndarray) -> np.ndarray:
    """Per cell off the track, the heading to the nearest track cell within
    ``NEAR_TRACK_CELLS`` (Chebyshev distance, ties to the first found); -1 if none."""
    h, w = track.shape
    out = np.full((h, w), -1, dtype=np.int8)
    best = np.full((h, w), np.inf)
    n = NEAR_TRACK_CELLS
    padded = np.pad(track, n)
    offsets = [(dr, dc) for dr in range(-n, n + 1) for dc in range(-n, n + 1) if dr or dc]
    offsets.sort(key=lambda o: (max(abs(o[0]), abs(o[1])), abs(o[0]) + abs(o[1])))
    for dr, dc in offsets:
        there = padded[n + dr : n + dr + h, n + dc : n + dc + w]
        dist = float(np.hypot(dr, dc))
        better = there & (dist < best)
        out[better] = _heading(dr, dc)
        best[better] = dist
    out[track | wall] = -1
    return out


def _cells(index: np.ndarray, n: int) -> tuple[tuple[int, np.ndarray], ...]:
    flat = index.ravel()
    return tuple((i, np.flatnonzero(flat == i)) for i in range(n) if (flat == i).any())


@dataclass(frozen=True)
class Terrain:
    speed_factor: np.ndarray  # (H, W) 0..1; 0 is a wall (water)
    track_dir: np.ndarray  # (H, W) int8: track axis 0..3, -1 off the track
    near_track_dir: np.ndarray  # (H, W) int8: heading 0..7 to a nearby track, or -1
    trap: np.ndarray  # (H, W) bool: shrub and wetland, where people get stuck
    cell_m: float
    wall: np.ndarray = field(init=False)  # (H, W) 1.0 on water
    # Per track axis / heading to a track that occurs: (index, flat cell indices).
    on_track: tuple[tuple[int, np.ndarray], ...] = field(init=False)
    near_track: tuple[tuple[int, np.ndarray], ...] = field(init=False)

    def __post_init__(self) -> None:
        shape = self.speed_factor.shape
        for name in ("track_dir", "near_track_dir", "trap"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} does not match speed_factor {shape}")
        if self.cell_m <= 0:
            raise ValueError("cell_m must be positive")
        object.__setattr__(self, "wall", (self.speed_factor <= 0.0).astype(float))
        object.__setattr__(self, "on_track", _cells(self.track_dir, 4))
        object.__setattr__(self, "near_track", _cells(self.near_track_dir, 8))

    @property
    def shape(self) -> tuple[int, int]:
        h, w = self.speed_factor.shape
        return h, w

    @classmethod
    def from_masks(
        cls,
        speed_factor: np.ndarray,
        track: np.ndarray,
        trap: np.ndarray | None,
        cell_m: float,
    ) -> Terrain:
        track = track.astype(bool) & (speed_factor > 0.0)
        speed = np.where(track, TRACK_SPEED, speed_factor).astype(float)
        wall = speed <= 0.0
        return cls(
            speed_factor=speed,
            track_dir=_track_dir(track),
            near_track_dir=_near_track_dir(track, wall),
            trap=np.zeros(speed.shape, bool) if trap is None else trap.astype(bool) & ~track,
            cell_m=cell_m,
        )

    @classmethod
    def from_classes(cls, worldcover: np.ndarray, track: np.ndarray, cell_m: float) -> Terrain:
        """From ESA WorldCover classes and a track/road mask. Unknown classes are open."""
        speed = np.full(worldcover.shape, OPEN_SPEED)
        for cls_id, factor in WORLDCOVER_SPEED.items():
            speed[worldcover == cls_id] = factor
        return cls.from_masks(speed, track, np.isin(worldcover, TRAP_CLASSES), cell_m)

    @classmethod
    def from_cover(
        cls, forest: np.ndarray, track: np.ndarray, water: np.ndarray, cell_m: float
    ) -> Terrain:
        """From forest, track and water masks only (no shrub or wetland, so no traps)."""
        speed = np.where(forest.astype(bool), FOREST_SPEED, OPEN_SPEED)
        speed = np.where(water.astype(bool), 0.0, speed)
        return cls.from_masks(speed, track, None, cell_m)

    @classmethod
    def open(cls, shape: tuple[int, int], cell_m: float, speed: float = OPEN_SPEED) -> Terrain:
        """Flat open ground with no tracks."""
        return cls.from_masks(np.full(shape, speed), np.zeros(shape, bool), None, cell_m)
