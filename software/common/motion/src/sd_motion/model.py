"""The motion model: where a lost person walks while we search.

The map has two layers: mass that has *stopped*, and mass *on the move* in one of eight
headings (N, NE, E, SE, S, SW, W, NW). Each time step moves a share of the walkers one
cell along their heading at the terrain's speed, lets some turn (45 degrees, or any
heading), turns some onto a track they stand on or next to, and stops some, as the
category's mobility curve says. Water is a wall: walkers turn back. Mass that walks off
the grid joins the rest of the world (``row``). Everything is deterministic numpy.

A sweep that found nothing scales both layers down where it looked (Bayes). Searched
cells fill up again only from walkers, never from the stopped layer: someone who sat
down where we looked is still not there.

Grids are row-major with row 0 the southernmost, as in ``sd_schemas.probmap``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from sd_motion.params import Params
from sd_motion.terrain import DIRS, Terrain

DT_S = 15.0
CHECK_STEPS = 240  # how often (in steps) to ask whether anyone still walks
STILL_EPS = 1e-6  # less moving mass than this (one in a million): nobody walks
_DIAG = math.sqrt(2.0)


@dataclass(frozen=True)
class MotionState:
    stopped: np.ndarray  # (H, W)
    moving: np.ndarray  # (8, H, W)
    row: float  # the rest of the world
    t_s: float  # seconds since last seen

    @property
    def p(self) -> np.ndarray:
        """Where the person may be: both layers together."""
        return self.stopped + self.moving.sum(axis=0)

    def total(self) -> float:
        return float(self.stopped.sum() + self.moving.sum() + self.row)


def _track_weights() -> np.ndarray:
    """For a track on axis k: the share of heading h that turns to heading g, (4, 8, 8) as
    [k, g, h]. Walkers turn to the nearer end of the track; square-on walkers split."""
    w = np.zeros((4, 8, 8))
    for k in range(4):
        for h in range(8):
            d = (h - k) % 8
            if d in (1, 7):
                w[k, k, h] = 1.0
            elif d in (3, 5):
                w[k, k + 4, h] = 1.0
            elif d in (2, 6):
                w[k, k, h] = w[k, k + 4, h] = 0.5
    return w


_TRACK_W = _track_weights()
_TRACK_LEAVE = _TRACK_W.sum(axis=1)  # [k, h]: 1 if heading h turns on axis k
_LEFT = np.roll(np.arange(8), 1)  # the heading 45 degrees anticlockwise of each
_RIGHT = np.roll(np.arange(8), -1)


# The steps run on flat grids with a one-cell empty border, so a step to a neighbour is a
# fixed offset into one contiguous array (fast), and a step onto the border leaves the map.


def _pad(a: np.ndarray, dtype: type = np.float32) -> np.ndarray:
    pad = [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)]
    return np.pad(a.astype(dtype), pad).reshape(*a.shape[:-2], -1)


def _unpad(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return a.reshape(*a.shape[:-1], h + 2, w + 2)[..., 1:-1, 1:-1].astype(np.float64)


def _padded_index(idx: np.ndarray, w: int) -> np.ndarray:
    return (idx // w + 1) * (w + 2) + idx % w + 1


@dataclass(frozen=True)
class _Kernel:
    """What steps of one length need, worked out once per ``advance`` (flat, padded)."""

    dt_s: float
    offsets: tuple[int, ...]  # flat offset of a step in each heading
    stay: np.ndarray  # (8, L) share of the walkers that stays in its cell
    land: np.ndarray  # (8, L) share that steps and lands on land
    bounce: tuple[tuple[np.ndarray, np.ndarray], ...]  # per heading: cells, share turning back
    edge: tuple[tuple[np.ndarray, np.ndarray], ...]  # per heading: cells, share off the map
    on_track: tuple[tuple[int, np.ndarray], ...]
    near_track: tuple[tuple[int, np.ndarray], ...]
    trap: np.ndarray  # (L,) bool
    turn45: float
    turn_any: float
    track: float
    trapped: bool  # some trap cells and a trap factor other than 1


def _kernel(terrain: Terrain, params: Params, dt_s: float) -> _Kernel:
    hh, ww = terrain.shape
    n = (hh + 2) * (ww + 2)
    speed = _pad(params.speed_mps * terrain.speed_factor)
    wall = _pad(terrain.wall)
    inside = _pad(np.ones(terrain.shape))
    land = inside - wall
    offsets = tuple(dr * (ww + 2) + dc for dr, dc in DIRS)
    go = np.empty((8, n))
    on_land = np.zeros((8, n))
    bounce = []
    edge = []
    for h, (dr, dc) in enumerate(DIRS):
        length = terrain.cell_m * (_DIAG if dr and dc else 1.0)
        go[h] = np.clip(speed * dt_s / length, 0.0, 1.0)
        o = offsets[h]
        src, dst = (slice(0, n - o), slice(o, n)) if o > 0 else (slice(-o, n), slice(0, n + o))
        on_land[h, src] = go[h, src] * land[dst]
        back = np.zeros(n)
        back[src] = go[h, src] * wall[dst]
        cells = np.flatnonzero(back)
        bounce.append((cells, back[cells].astype(np.float32)))
        off = np.zeros(n)
        off[src] = go[h, src] * (1.0 - inside[dst])
        cells = np.flatnonzero(off)
        edge.append((cells, off[cells]))  # float64: the world's share is summed exactly
    return _Kernel(
        dt_s=dt_s,
        offsets=offsets,
        stay=(1.0 - go).astype(np.float32),
        land=on_land.astype(np.float32),
        bounce=tuple(bounce),
        edge=tuple(edge),
        on_track=tuple((k, _padded_index(i, ww)) for k, i in terrain.on_track),
        near_track=tuple((g, _padded_index(i, ww)) for g, i in terrain.near_track),
        trap=_pad(terrain.trap).astype(bool),
        turn45=min(params.turn45 * dt_s / 60.0, 0.5),
        turn_any=min(params.turn_any * dt_s / 60.0, 0.5),
        track=min(params.k_track * dt_s / 60.0, 1.0),
        trapped=bool(terrain.trap.any()) and params.trap != 1.0,
    )


class _Flat:
    """The state while stepping, with work buffers reused every step: fresh large arrays
    cost more than the arithmetic."""

    def __init__(self, state: MotionState) -> None:
        self.stopped = _pad(state.stopped, np.float64)
        self.moving = _pad(state.moving)
        self.row = state.row
        self.t_s = state.t_s
        self.turned = np.empty_like(self.moving)
        self.walked = np.empty_like(self.moving)
        self.scratch = np.empty_like(self.moving)
        self.total = np.empty(self.moving.shape[1:], np.float32)
        self.cell = np.empty_like(self.total)
        self.stop = np.empty_like(self.stopped)

    def state(self, shape: tuple[int, int]) -> MotionState:
        return MotionState(
            stopped=_unpad(self.stopped, shape),
            moving=_unpad(self.moving, shape),
            row=self.row,
            t_s=self.t_s,
        )


def _step(s: _Flat, params: Params, k: _Kernel) -> None:
    moving, turned, walked, tmp, total, cell = (
        s.moving, s.turned, s.walked, s.scratch, s.total, s.cell
    )  # fmt: skip
    moving.sum(axis=0, out=total)

    # 1. Stop, as the mobility curve says (faster in traps).
    s0 = params.still_moving(s.t_s / 3600.0)
    s1 = params.still_moving((s.t_s + k.dt_s) / 3600.0)
    ratio = min(max(s1 / s0, 0.0), 1.0) if s0 > 0.0 else 0.0
    if k.trapped:
        moving *= np.where(k.trap, ratio**params.trap, ratio).astype(np.float32)
        s.stopped += total
        moving.sum(axis=0, out=total)
        s.stopped -= total
        ratio = 1.0
    else:
        np.multiply(total, 1.0 - ratio, out=s.stop)
        s.stopped += s.stop

    # 2. Turn: 45 degrees either way, or any heading (the stop share folded in).
    a, b = k.turn45, k.turn_any
    side = 0.5 * a * ratio
    np.multiply(moving, ratio * (1.0 - a - b), out=turned)
    np.multiply(moving, side, out=tmp)
    turned[1:] += tmp[:-1]
    turned[0] += tmp[-1]
    turned[:-1] += tmp[1:]
    turned[-1] += tmp[0]
    np.multiply(total, ratio * b / 8.0, out=cell)
    turned += cell

    # 3. Follow tracks: along a track, towards a nearby one.
    if k.track > 0.0:
        for axis, idx in k.on_track:
            sub = turned[:, idx]
            moved = k.track * sub
            turned[:, idx] = sub - moved * _TRACK_LEAVE[axis][:, None] + _TRACK_W[axis] @ moved
        for g, idx in k.near_track:
            sub = turned[:, idx]
            moved = k.track * sub
            sub -= moved
            sub[g] += moved.sum(axis=0)
            turned[:, idx] = sub

    # 4. Walk one cell; water turns walkers back, the edge loses them to the world.
    np.multiply(turned, k.stay, out=walked)
    for h, (cells, share) in enumerate(k.bounce):
        walked[(h + 4) % 8, cells] += turned[h, cells] * share
    n = cell.shape[0]
    lost = 0.0
    for h, o in enumerate(k.offsets):
        np.multiply(turned[h], k.land[h], out=cell)
        if o > 0:
            walked[h, o:] += cell[: n - o]
        else:
            walked[h, : n + o] += cell[-o:]
        cells, share = k.edge[h]
        lost += float(turned[h, cells] @ share)

    s.moving, s.walked = walked, moving
    s.row += lost
    s.t_s += k.dt_s


def _run(
    state: MotionState, params: Params, k: _Kernel, steps: int, shape: tuple[int, int]
) -> MotionState:
    s = _Flat(state)
    for i in range(steps):
        if i % CHECK_STEPS == 0 and s.moving.sum(dtype=np.float64) < STILL_EPS:
            # Nobody is walking any more: the rest of the steps change nothing.
            s.stopped += s.moving.sum(axis=0, dtype=np.float64)
            s.moving[...] = 0.0
            s.t_s += (steps - i) * k.dt_s
            break
        _step(s, params, k)
    return s.state(shape)


def step(state: MotionState, terrain: Terrain, params: Params, dt_s: float = DT_S) -> MotionState:
    """One time step: stop, turn, follow tracks, walk."""
    return _run(state, params, _kernel(terrain, params, dt_s), 1, terrain.shape)


def advance(
    state: MotionState, terrain: Terrain, params: Params, seconds: float, dt_s: float = DT_S
) -> MotionState:
    """Run the model forward ``seconds`` (whole steps, then one shorter)."""
    if seconds < 0:
        raise ValueError("cannot run the model backwards")
    whole, rest = divmod(seconds, dt_s)
    if whole:
        state = _run(state, params, _kernel(terrain, params, dt_s), int(whole), terrain.shape)
    if rest > 1e-9:
        state = step(state, terrain, params, rest)
    return state


def negative(state: MotionState, pod: np.ndarray) -> MotionState:
    """A sweep with detection chance ``pod`` per cell found nothing (Bayes)."""
    seen = float((state.p * pod).sum())
    if seen >= 1.0 - 1e-12:
        raise ValueError("the sweep was certain to find the person; nothing is left")
    scale = 1.0 / (1.0 - seen)
    miss = 1.0 - pod
    return replace(
        state,
        stopped=state.stopped * miss * scale,
        moving=state.moving * miss[None] * scale,
        row=state.row * scale,
    )


def _heading_weights(heading_deg: float | None) -> np.ndarray:
    if heading_deg is None:
        return np.full(8, 1.0 / 8.0)
    angles = np.arange(8) * 45.0
    w = 0.75 + 0.25 * np.cos(np.radians(angles - heading_deg))  # 1 ahead, 0.5 behind
    return w / w.sum()


def start_state(
    terrain: Terrain,
    params: Params,
    ipp_rc: tuple[int, int],
    t_s: float,
    rings: np.ndarray,
    rings_row: float,
    heading_deg: float | None = None,
    blend: float = 0.5,
    moving_now: float | None = None,
) -> MotionState:
    """The map when the search starts, ``t_s`` seconds after the person was last seen.

    Half (``blend``) comes from walking the model out from the last seen point, half from
    the distance rings (``rings``, summing to ``1 - rings_row``): one model alone is easily
    wrong. The ring half is split into stopped and moving by the mobility curve, or by
    ``moving_now`` when the profile says better (someone lying down: 0).
    """
    shape = terrain.shape
    if rings.shape != shape:
        raise ValueError(f"rings {rings.shape} do not match the terrain {shape}")
    r, c = ipp_rc
    moving0 = params.still_moving(0.0) if moving_now is None else min(moving_now, 1.0)
    walked = MotionState(stopped=np.zeros(shape), moving=np.zeros((8, *shape)), row=0.0, t_s=0.0)
    walked.stopped[r, c] = 1.0 - moving0
    walked.moving[:, r, c] = moving0 * _heading_weights(heading_deg)
    if moving_now is not None:
        params = replace(params, mobility=((0.0, moving0), (1e6, moving0)))
    walked = advance(walked, terrain, params, t_s)
    s_now = params.still_moving(t_s / 3600.0) if moving_now is None else moving0
    land = 1.0 - terrain.wall
    ring_p = rings * land
    lost = float(rings.sum() - ring_p.sum())
    stopped = blend * walked.stopped + (1 - blend) * (1 - s_now) * ring_p
    moving = blend * walked.moving + (1 - blend) * (s_now / 8.0) * ring_p[None]
    row = blend * walked.row + (1 - blend) * (rings_row + lost)
    return MotionState(stopped=stopped, moving=moving, row=row, t_s=t_s)
