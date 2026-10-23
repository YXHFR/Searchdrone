"""The search service's probability map loop: inputs from the ground, versions back.

Every round (at most every ``MAP_EVERY_S``) and for each open mission with an approved
search profile, ``MapService`` reads what the map depends on: the charter's area and
sectors and the operator's areas, the stills after its cursor, and the candidates with
the stills they were seen in. It builds the map (``probmap.build``) and writes a new
version when the map changed:

* at once for a new profile version, a changed area or charter, a candidate a person
  decided on, or a new pass (a new order began, so the one before ended);
* otherwise at most every ``SAVE_EVERY_S`` while stills come in.

A confirmed candidate writes one last version with ``frozen = True``; the ground refuses
anything after it. The map only ranks where to look: nothing here can give an order.
A failed read or a 409 makes the mission wait a round, the candidate loop goes on.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final, cast

from sd_schemas.probmap import MapReason, ProbabilityMapInputs, ProbabilityMapWrite, SearchStill
from sd_schemas.search import SearchProfile, SearchTrack
from sd_search.client import ConflictError, GroundError, SearchApi
from sd_search.probmap.build import Built, MapBuilder, MapCandidate, MapInputs, encode_f32
from sd_search.probmap.pod import Still
from sd_search.probmap.terrain import TerrainSource
from sd_search.profile import ProfileSource

__all__ = ["MAP_EVERY_S", "SAVE_EVERY_S", "MapService", "map_candidate", "map_still"]

log = logging.getLogger("sd_search.mapper")
MAP_EVERY_S: Final = 2.0
SAVE_EVERY_S: Final = 30.0
STILLS_PAGE: Final = 500


def map_still(s: SearchStill) -> Still:
    """A still as the map takes it in (footprint as lon/lat; the builder projects it)."""
    return Still(
        still_id=str(s.still_id),
        seq=s.seq,
        pass_id="" if s.command_id is None else str(s.command_id),
        footprint=tuple((p.lon, p.lat) for p in s.footprint),
        px_per_m=s.px_per_m,
        detector_ok=s.detector_ok,
    )


def map_candidate(t: SearchTrack) -> MapCandidate:
    c = t.candidate
    return MapCandidate(
        candidate_id=str(c.candidate_id),
        state=c.state.value,
        lat=c.lat,
        lon=c.lon,
        sigma95_m=c.sigma95_m,
        still_ids=frozenset(str(d.still_id) for d in t.sightings),
        merged_into=None if c.merged_into is None else str(c.merged_into),
    )


@dataclass
class _Mission:
    builder: MapBuilder
    cursor: int = 0
    stills: list[Still] = field(default_factory=list[Still])
    written_at: float | None = None
    last_key: tuple[object, ...] | None = None
    passes: frozenset[str] = frozenset()
    frozen: bool = False


class MapService:
    def __init__(
        self,
        api: SearchApi,
        profiles: ProfileSource,
        terrain: TerrainSource,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.api = api
        self.profiles = profiles
        self.terrain = terrain
        self.monotonic = monotonic
        self.missions: dict[str, _Mission] = {}
        self._round_at: float | None = None

    async def step(self, open_ids: list[str], *, force: bool = False) -> int:
        """One round over the open missions; the number of versions written."""
        now = self.monotonic()
        if not force and self._round_at is not None and now - self._round_at < MAP_EVERY_S:
            return 0
        self._round_at = now
        for gone in set(self.missions) - set(open_ids):
            del self.missions[gone]
        written = 0
        for mid in open_ids:
            try:
                written += await self._mission_step(mid)
            except ConflictError:
                log.info("kartan ändrades på marken: läser om uppdraget %s", mid)
                self.missions.pop(mid, None)
            except GroundError as exc:
                log.warning("sannolikhetskartan för %s: %s", mid, exc)
            except Exception:  # the map is advice: a fault in it never stops the candidates
                log.exception("sannolikhetskartan för %s kunde inte byggas", mid)
                self.missions.pop(mid, None)
        return written

    async def _mission_step(self, mission_id: str) -> int:
        m = self.missions.setdefault(mission_id, _Mission(MapBuilder(self.terrain)))
        if m.frozen:
            return 0
        profile = await self.profiles.current(mission_id)
        if profile is None:
            return 0  # no approved profile: no map (the matching waits the same way)
        inputs = await self.api.map_inputs(mission_id)
        if inputs.latest is not None and inputs.latest.frozen:
            m.frozen = True
            return 0
        while True:
            page = await self.api.stills(mission_id, m.cursor, STILLS_PAGE)
            for s in page:
                m.cursor = max(m.cursor, s.seq)
                m.stills.append(map_still(s))
            if len(page) < STILLS_PAGE:
                break
        tracks = await self.api.tracks(mission_id)
        candidates = [map_candidate(t) for t in tracks]
        # The numbers run in a worker thread: the candidate loop never waits for them.
        built = await asyncio.to_thread(
            m.builder.build, _inputs(profile, inputs, m.stills, candidates)
        )
        seen = (m.last_key, m.passes)
        reason = self._reason(m, profile, inputs, built, candidates)
        if reason is None:
            return 0
        state = built.state
        if built.confirmed:
            state = m.builder.model.freeze(state, "confirmed")
        latest = 0 if inputs.latest is None else inputs.latest.version
        body = ProbabilityMapWrite(
            version=latest + 1,
            reason=reason,
            grid=state.grid.model(),
            row=state.row,
            frozen=state.frozen,
            event_seq=built.event_seq,
            profile_version=profile.profile_version,
            sectors=built.sectors,
            notes=built.notes,
            p_z=encode_f32(state.p),
            pod_z=encode_f32(state.pod_cum),
        )
        try:
            written = await self.api.write_map(mission_id, body)
        except BaseException:
            m.last_key, m.passes = seen  # not on the ground: the next round tries again
            raise
        m.written_at = self.monotonic()
        m.frozen = written.frozen
        return 1

    def _reason(
        self,
        m: _Mission,
        profile: SearchProfile,
        inputs: ProbabilityMapInputs,
        built: Built,
        candidates: list[MapCandidate],
    ) -> MapReason | None:
        setup = (
            profile.profile_version,
            None if inputs.area is None else str(inputs.area),
            tuple((s, str(a)) for s, a in inputs.sectors),
            tuple((str(a.annotation_id), a.poa_hint) for a in inputs.areas),
        )
        decided = tuple(sorted((c.candidate_id, c.state) for c in candidates))
        passes = frozenset(s.pass_id for s in m.stills)
        key = (setup, decided, built.event_seq)
        before, m.last_key = m.last_key, key
        new_passes, m.passes = passes - m.passes, passes
        if built.confirmed:
            return "confirmed"
        if before is None:
            return "prior" if inputs.latest is None else "search"
        if before == key:
            return None
        if before[0] != setup:
            old = cast("tuple[object, ...]", before[0])
            if old[0] != setup[0]:
                return "profile"
            return "areas" if old[3] != setup[3] else "charter"
        if before[1] != decided:
            return "candidate"
        if new_passes and len(m.passes) > 1:
            return "search"  # an order began: the one before it has ended
        due = m.written_at is None or self.monotonic() - m.written_at >= SAVE_EVERY_S
        if due:
            return "search"
        m.last_key = before  # not written yet: compare with what the ground has
        return None


def _inputs(
    profile: SearchProfile,
    inputs: ProbabilityMapInputs,
    stills: list[Still],
    candidates: list[MapCandidate],
) -> MapInputs:
    return MapInputs(
        profile=profile,
        area=None if inputs.area is None else inputs.area["coordinates"][0],
        sectors=[(sid, poly["coordinates"][0]) for sid, poly in inputs.sectors],
        areas=[(a.poa_hint, a.polygon["coordinates"][0]) for a in inputs.areas],
        stills=stills,
        candidates=candidates,
        alt_m=inputs.alt_m,
        speed_ms=inputs.speed_ms,
    )
