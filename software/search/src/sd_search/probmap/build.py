"""The map from its inputs: prior plus the passes so far, in the order they began.

``MapBuilder.build`` takes everything the map depends on (the approved profile, the
charter's area and sectors, the operator's areas, the stills and the candidates) and
gives the same map for the same inputs. The prior and each finished pass's POD are
cached, so a new still costs one pass's work.

Candidates hold the map back: a cell where a candidate waits for a person's decision
(``new``, ``assessing``, ``ready``, ``inspecting``) is not counted down by the pass that
saw it, until the candidate is rejected. A confirmed candidate freezes the map.
"""

from __future__ import annotations

import base64
import math
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from sd_schemas.probmap import MapNote, PoaHint, SectorScore
from sd_schemas.search import SearchProfile
from sd_search.probmap.bayes import HeadingPrior, MapState, OperatorArea, RingTerrainMap
from sd_search.probmap.grid import Floats, Grid, grid_around
from sd_search.probmap.pod import DEFAULT_DETECTOR, DETECTORS, Still, pass_pod
from sd_search.probmap.rings import CHILD_CATEGORIES, rings_for
from sd_search.probmap.sectors import PlanSector, sector_scores
from sd_search.probmap.terrain import TerrainSource

__all__ = ["PENDING", "Built", "MapBuilder", "MapCandidate", "MapInputs", "encode_f32"]

PENDING: Final = frozenset({"new", "assessing", "ready", "inspecting"})


@dataclass(frozen=True)
class MapCandidate:
    candidate_id: str
    state: str
    lat: float
    lon: float
    sigma95_m: float
    still_ids: frozenset[str]
    merged_into: str | None = None


@dataclass(frozen=True)
class MapInputs:
    profile: SearchProfile
    area: Sequence[Sequence[float]] | None  # the charter area's ring, [lon, lat]
    sectors: Sequence[tuple[str, Sequence[Sequence[float]]]]
    areas: Sequence[tuple[PoaHint, Sequence[Sequence[float]]]]  # (hint, ring)
    stills: Sequence[Still]  # footprints in lon/lat: ``Still.footprint`` as (lon, lat)
    candidates: Sequence[MapCandidate]
    alt_m: float
    speed_ms: float


@dataclass(frozen=True)
class Built:
    state: MapState
    sectors: list[SectorScore]
    notes: list[MapNote]
    event_seq: int
    confirmed: bool


@dataclass
class _Cache:
    prior_key: tuple[object, ...] | None = None
    prior: MapState | None = None
    canopy: Floats | None = None
    passes: dict[tuple[object, ...], Floats] = field(
        default_factory=dict[tuple[object, ...], Floats]
    )


def encode_f32(a: Floats) -> str:
    """base64 of zlib-compressed little-endian float32."""
    return base64.b64encode(zlib.compress(a.astype("<f4").tobytes(), 6)).decode("ascii")


def _resolve(c: MapCandidate, by_id: dict[str, MapCandidate]) -> str:
    """The state that decides a candidate's hold: a merged one follows the one it joined."""
    seen: set[str] = set()
    while c.state == "merged" and c.merged_into in by_id and c.candidate_id not in seen:
        seen.add(c.candidate_id)
        c = by_id[c.merged_into]
    return "new" if c.state == "merged" else c.state  # a lost target: keep holding


class MapBuilder:
    def __init__(self, terrain: TerrainSource) -> None:
        self.terrain = terrain
        self.model = RingTerrainMap()
        self.detector = DETECTORS[DEFAULT_DETECTOR]
        self._cache = _Cache()

    def grid_for(self, inputs: MapInputs) -> Grid:
        ls = inputs.profile.last_seen
        probe = Grid(ls.lat, ls.lon, 0, 0, 1, 1)
        area = None if not inputs.area else probe.ring_xy(inputs.area)
        return grid_around(ls.lat, ls.lon, area)

    def build(self, inputs: MapInputs) -> Built:
        grid = self.grid_for(inputs)
        state, canopy = self._prior(grid, inputs)
        by_id = {c.candidate_id: c for c in inputs.candidates}
        confirmed = any(c.state == "confirmed" for c in inputs.candidates)
        holds = [c for c in inputs.candidates if _resolve(c, by_id) in PENDING]

        passes: dict[str, list[Still]] = {}
        for s in sorted(inputs.stills, key=lambda s: s.seq):
            passes.setdefault(s.pass_id, []).append(
                Still(
                    s.still_id,
                    s.seq,
                    s.pass_id,
                    tuple(grid.xy(lat, lon) for lon, lat in s.footprint),
                    s.px_per_m,
                    s.detector_ok,
                )
            )
        for pass_id, stills in passes.items():
            ids = {s.still_id for s in stills}
            hold = sorted(
                {cell for c in holds if c.still_ids & ids for cell in self._cells(grid, c)}
            )
            key = (grid, pass_id, tuple(s.still_id for s in stills), tuple(hold))
            pod = self._cache.passes.get(key)
            if pod is None:
                pod = pass_pod(grid, stills, canopy, self.detector, hold)
                self._cache.passes = {
                    k: v for k, v in self._cache.passes.items() if k[1] != pass_id
                }
                self._cache.passes[key] = pod
            state = self.model.update_negative(state, pod)
        sectors = sector_scores(
            state,
            [PlanSector(sid, tuple(grid.ring_xy(ring))[:-1]) for sid, ring in inputs.sectors],
            canopy,
            self.detector,
            alt_m=inputs.alt_m,
            speed_ms=inputs.speed_ms,
        )
        return Built(
            state=state,
            sectors=sectors,
            notes=self._notes(inputs),
            event_seq=max((s.seq for s in inputs.stills), default=0),
            confirmed=confirmed,
        )

    def _prior(self, grid: Grid, inputs: MapInputs) -> tuple[MapState, Floats]:
        prof = inputs.profile
        areas = tuple(
            OperatorArea(hint, tuple(grid.ring_xy(ring))[:-1]) for hint, ring in inputs.areas
        )
        subject = HeadingPrior(prof.subject_category, prof.last_seen.heading_deg)
        key = (grid, subject, areas)
        cache = self._cache
        if cache.prior_key != key or cache.prior is None or cache.canopy is None:
            prior = self.model.prior(subject, self.terrain, areas, grid)
            canopy = self.terrain.canopy(grid)
            self._cache = _Cache(prior_key=key, prior=prior, canopy=canopy)
            return prior, canopy
        return cache.prior, cache.canopy

    @staticmethod
    def _cells(grid: Grid, c: MapCandidate) -> list[tuple[int, int]]:
        x, y = grid.xy(c.lat, c.lon)
        reach = c.sigma95_m + grid.cell * math.sqrt(2) / 2
        win = grid.window(x - reach, y - reach, x + reach, y + reach)
        if win is None:
            return []
        cx, cy = grid.centers
        near = np.hypot(cx[win] - x, cy[win] - y) <= reach
        out = [
            (int(j) + win[0].start, int(i) + win[1].start)
            for j, i in zip(*np.nonzero(near), strict=True)
        ]
        own = grid.cell_of(x, y)
        if own is not None and own not in out:
            out.append(own)
        return out

    def _notes(self, inputs: MapInputs) -> list[MapNote]:
        notes: list[MapNote] = []
        if not self.terrain.available:
            notes.append("no_terrain")
        category = inputs.profile.subject_category
        if category in CHILD_CATEGORIES:
            notes.append("child_statistics")
        elif rings_for(category).uncertain:
            notes.append("uncertain_statistics")
        if not inputs.area:
            notes.append("no_charter")
        return notes
