"""Movement in the map: the person may keep walking while we search (``sd_motion``).

With movement on, the map is no longer the prior with the passes applied in order; it is a
state that runs in time. It starts at the first moment the map is needed, from the prior
blended with a walk out from the last seen point; it walks forward to each pass's time and
takes the pass's Bayes update there; and it walks on to "now". Searched cells fill up
again only from people still on the move.

Each step is cached: a new still costs a few seconds of walking, not the whole search.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime

import numpy as np

from sd_motion import (
    DT_S,
    MotionState,
    Params,
    Terrain,
    advance,
    for_category,
    negative,
    start_state,
    walk_out,
    walk_params,
)
from sd_schemas.search import SearchProfile
from sd_search.probmap.grid import Floats, Grid
from sd_search.probmap.terrain import TerrainSource

__all__ = ["MotionCache", "TimedPass", "motion_params", "motion_terrain", "run_motion"]

STILL_STATES = frozenset({"sitting", "lying"})


@dataclass(frozen=True)
class TimedPass:
    key: tuple[object, ...]  # changes whenever the pass's POD changes
    at_utc: datetime | None  # the middle of the pass; None: no capture times
    pod: Floats


@dataclass
class MotionCache:
    start_key: tuple[object, ...] | None = None
    start: MotionState | None = None
    walk_key: tuple[object, ...] | None = None
    walked: MotionState | None = None  # the walk out from the last seen point, kept going
    chain: list[tuple[tuple[object, ...], MotionState]] = field(
        default_factory=list[tuple[tuple[object, ...], MotionState]]
    )
    now: MotionState | None = None  # the last "now", after the end of the chain


def motion_params(profile: SearchProfile) -> tuple[Params, float | None]:
    """The category's numbers, and the share on the move now when the profile says so.
    Reduced mobility halves the speed; someone only ever seen sitting or lying is not
    walking (``moving_now = 0``)."""
    params = for_category(profile.subject_category)
    if profile.mobility == "reduced":
        params = replace(params, speed_mps=params.speed_mps / 2)
    states = set(profile.likely_state) - {"unknown"}
    moving_now = 0.0 if states and states <= STILL_STATES else None
    return params, moving_now


def motion_terrain(terrain: TerrainSource, grid: Grid, canopy: Floats) -> Terrain:
    """Tracks and roads, forest and water from the map's terrain source."""
    half = grid.cell / 2
    track = np.minimum(terrain.distance_to("track", grid), terrain.distance_to("road", grid))
    water = terrain.distance_to("water", grid) <= 0.0  # inside a lake, not beside a stream
    return Terrain.from_cover(canopy > 0.0, track <= half, water, grid.cell)


def _seconds(at: datetime, since: datetime) -> float:
    return max(0.0, (at - since).total_seconds())


def _weighted(state: MotionState, weight: Floats) -> MotionState:
    """The operator's areas on the walked map: ``excluded`` stays empty, ``high`` weighs
    more; the map's share of the total is kept."""
    before = float(state.p.sum())
    stopped, moving = state.stopped * weight, state.moving * weight[None]
    after = float(stopped.sum() + moving.sum())
    if after <= 0.0:
        return replace(state, stopped=stopped, moving=moving, row=1.0)
    scale = before / after
    return replace(state, stopped=stopped * scale, moving=moving * scale)


def run_motion(
    cache: MotionCache,
    key: tuple[object, ...],
    *,
    prior_p: Floats,
    prior_row: float,
    area_weight: Floats,
    terrain: Terrain,
    params: Params,
    moving_now: float | None,
    heading_deg: float | None,
    last_seen_utc: datetime,
    passes: Sequence[TimedPass],
    now_utc: datetime,
    ipp_rc: tuple[int, int],
) -> MotionState:
    """The map at ``now_utc``: started, walked through every pass, then walked to now."""
    # The start is anchored at the first pass (or at now, before any pass), never at
    # when the map was first built: the same inputs give the same map in any builder.
    first = min((p.at_utc for p in passes if p.at_utc is not None), default=now_utc)
    t0 = _seconds(min(first, now_utc), last_seen_utc)
    start_key = (*key, t0)
    if cache.start_key != start_key or cache.start is None:
        # Before the first pass the start moves with now; the walk out from the last
        # seen point then goes on from where it was instead of starting over.
        # It is kept at whole steps, the way one long walk would step.
        whole = math.floor(t0 / DT_S) * DT_S
        walking = walk_params(params, moving_now)
        if cache.walk_key != key or cache.walked is None or cache.walked.t_s > whole:
            cache.walk_key = key
            cache.walked = walk_out(terrain, params, ipp_rc, whole, heading_deg, moving_now)
        else:
            cache.walked = _walk_to(cache.walked, terrain, walking, whole)
        start = start_state(
            terrain, params, ipp_rc, t0, prior_p, prior_row, heading_deg,
            moving_now=moving_now, walked=_walk_to(cache.walked, terrain, walking, t0),
        )  # fmt: skip
        cache.start_key, cache.start = start_key, _weighted(start, area_weight)
        cache.chain, cache.now = [], None
    state = cache.start
    if len(cache.chain) > len(passes):
        del cache.chain[len(passes) :]
        cache.now = None
    for i, ps in enumerate(passes):
        if i < len(cache.chain) and cache.chain[i][0] == ps.key:
            state = cache.chain[i][1]
            continue
        del cache.chain[i:]
        cache.now = None
        if ps.at_utc is not None:
            state = _walk_to(state, terrain, params, _seconds(ps.at_utc, last_seen_utc))
        state = negative(state, np.clip(ps.pod, 0.0, 1.0))
        cache.chain.append((ps.key, state))
    t_now = _seconds(now_utc, last_seen_utc)
    if cache.now is not None and cache.now.t_s <= t_now:
        state = cache.now  # the chain is unchanged: walk on from the last "now"
    state = _walk_to(state, terrain, params, t_now)
    cache.now = state
    return state


def _walk_to(state: MotionState, terrain: Terrain, params: Params, t_s: float) -> MotionState:
    seconds = t_s - state.t_s
    if seconds <= 0.0 or math.isclose(seconds, 0.0, abs_tol=1e-6):
        return state
    return advance(state, terrain, params, seconds)
