"""The search service's loop: detections in, candidates and the code's assessment out.

Each round (``SearchService.step``) reads the open missions, and for each the target
detections after the last one it fused (``after_seq``, with a short look-back so a
detection committed out of order is not skipped; fusion ignores the ones it has seen).
New sightings go through fusion; every candidate they touched is scored (the colours of
its crops against the approved profile) and written as one batch, all or none.

A restart reads the candidates back with the detections they observed and goes on from
there: the same detection never makes a second observation (fusion's ``seen`` set and the
database's unique index), and nothing that arrived while the service was down is lost.
A 409 (a person decided, or another revision came first) makes the service read the
mission back and fuse again. If sd-search dies, the rest of the system goes on: it has
no part in flying, and the panel shows ``parts.search`` as down.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from PIL import Image

from sd_schemas.link import BBox
from sd_schemas.search import (
    SEARCH_BATCH_MAX,
    CandidateAssessment,
    CandidateBatch,
    CandidateRevision,
    CandidateState,
    FieldVerdict,
    SearchDetection,
    SearchHeartbeat,
    SearchProfile,
    SearchTrack,
)
from sd_search.client import ConflictError, GroundError, SearchApi
from sd_search.color import CROPS_MAX, CropView, assess_colors, box_in_crop
from sd_search.fusion import Fusion, Sighting, Track
from sd_search.priority import priority
from sd_search.profile import ProfileSource
from sd_search.score import match_score, person_probability

__all__ = ["FEED_LIMIT", "LOOKBACK", "SearchService", "sighting", "track_of"]

log = logging.getLogger("sd_search")
FEED_LIMIT = 200
LOOKBACK = 64  # detections re-read behind the cursor (out-of-order commits)
CROP_CACHE = 256
HEARTBEAT_S = 5.0


def sighting(d: SearchDetection) -> Sighting:
    return Sighting(
        detection_id=str(d.detection_id),
        seq=d.seq,
        still_id=str(d.still_id),
        label=d.label,
        lat=d.lat,
        lon=d.lon,
        sigma95_m=d.sigma95_m,
        score=d.score,
        px_size=d.px_size,
        captured_utc=d.captured_utc,
        has_crop=d.has_crop,
    )


def track_of(t: SearchTrack) -> Track:
    c = t.candidate
    return Track(
        candidate_id=str(c.candidate_id),
        label=t.label,
        state=c.state,
        lat=c.lat,
        lon=c.lon,
        sigma95_m=c.sigma95_m,
        first_seen_utc=c.first_seen_utc,
        last_seen_utc=c.last_seen_utc,
        sightings=[sighting(d) for d in t.sightings],
        merged_into=None if c.merged_into is None else str(c.merged_into),
        rev=c.rev,
        seq=c.seq,
        decided=t.decided,
        refined_person_score=c.refined_person_score,
    )


@dataclass
class _Mission:
    fusion: Fusion
    cursor: int  # the highest detection seq fused
    boxes: dict[str, tuple[BBox, BBox | None]] = field(
        default_factory=dict[str, tuple[BBox, BBox | None]]
    )  # detection id -> (box in the still, crop square in the still)
    assessed: dict[str, tuple[int, tuple[str, ...]]] = field(
        default_factory=dict[str, tuple[int, tuple[str, ...]]]
    )  # candidate id -> (profile version, crops) of its last code assessment
    fields: dict[str, dict[str, FieldVerdict]] = field(
        default_factory=dict[str, dict[str, FieldVerdict]]
    )  # candidate id -> the code's verdicts


class SearchService:
    def __init__(
        self,
        api: SearchApi,
        profiles: ProfileSource,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.api = api
        self.profiles = profiles
        self.monotonic = monotonic
        self.missions: dict[str, _Mission] = {}
        self._crops: OrderedDict[str, bytes | None] = OrderedDict()
        self._beat_at = 0.0

    # --- the loop --------------------------------------------------------------------

    async def run(self, stop: asyncio.Event, poll_s: float = 0.25) -> None:
        while not stop.is_set():
            try:
                await self.beat()
                await self.step()
            except GroundError as exc:
                log.warning("markstationen svarar inte: %s", exc)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), poll_s)

    async def beat(self, *, force: bool = False) -> None:
        now = self.monotonic()
        if force or now - self._beat_at >= HEARTBEAT_S:
            await self.api.heartbeat(SearchHeartbeat(cloud="off", missions=len(self.missions)))
            self._beat_at = now

    async def step(self) -> int:
        """One round over the open missions; the number of new sightings fused."""
        open_ids = await self.api.missions()
        for gone in set(self.missions) - set(open_ids):
            del self.missions[gone]
        fused = 0
        for mid in open_ids:
            try:
                fused += await self._mission_step(mid)
            except ConflictError:
                log.info("kandidaterna ändrades på marken: läser om uppdraget")
                self.missions.pop(mid, None)
            except GroundError:
                self.missions.pop(mid, None)  # fused but maybe not written: read it back
                raise
        return fused

    async def _load(self, mission_id: str) -> _Mission:
        tracks = await self.api.tracks(mission_id)
        m = _Mission(
            fusion=Fusion([track_of(t) for t in tracks]),
            cursor=max((d.seq for t in tracks for d in t.sightings), default=0),
        )
        for t in tracks:
            for d in t.sightings:
                m.boxes[str(d.detection_id)] = (d.bbox, d.crop_box)
        self.missions[mission_id] = m
        return m

    async def _mission_step(self, mission_id: str) -> int:
        m = self.missions.get(mission_id) or await self._load(mission_id)
        profile = await self.profiles.current(mission_id)
        version = 0 if profile is None else profile.profile_version
        fresh = await self.api.detections(mission_id, max(0, m.cursor - LOOKBACK), FEED_LIMIT)
        dirty: dict[str, list[tuple[str, float]]] = {}
        fused = 0
        for d in fresh:
            m.cursor = max(m.cursor, d.seq)
            step = m.fusion.add(sighting(d))
            if step is None:
                continue
            fused += 1
            m.boxes[str(d.detection_id)] = (d.bbox, d.crop_box)
            dirty.setdefault(step.candidate_id, []).append((str(d.detection_id), step.distance_m))
            for merged in step.merged:
                dirty.setdefault(merged, [])
        for t in m.fusion.tracks.values():  # a new profile version: score the open ones again
            if t.open and m.assessed.get(t.candidate_id, (version,))[0] != version:
                dirty.setdefault(t.candidate_id, [])
        if not dirty:
            return 0
        revisions: list[CandidateRevision] = []
        for cid, observations in dirty.items():
            t = m.fusion.tracks[cid]
            revisions.append(await self._revision(m, t, profile, version, observations))
        for i in range(0, len(revisions), SEARCH_BATCH_MAX):
            chunk = revisions[i : i + SEARCH_BATCH_MAX]
            written = await self.api.write(mission_id, CandidateBatch(revisions=chunk))
            for w in written:
                t = m.fusion.tracks[str(w.candidate_id)]
                t.rev, t.seq, t.state = w.rev, w.seq, w.state
        for r in revisions:
            if r.assessment is not None:
                m.assessed[str(r.candidate_id)] = (
                    version,
                    tuple(str(c) for c in r.assessment.crops_used),
                )
        return fused

    # --- one candidate -----------------------------------------------------------------

    async def _crop(self, detection_id: str) -> bytes | None:
        if detection_id in self._crops:
            self._crops.move_to_end(detection_id)
            return self._crops[detection_id]
        data = await self.api.crop(detection_id)
        self._crops[detection_id] = data
        while len(self._crops) > CROP_CACHE:
            self._crops.popitem(last=False)
        return data

    async def _assess(
        self, m: _Mission, t: Track, profile: SearchProfile | None, version: int
    ) -> CandidateAssessment | None:
        with_crop = sorted(
            (s for s in t.sightings if s.has_crop), key=lambda s: (-s.px_size, s.detection_id)
        )[:CROPS_MAX]
        key = (version, tuple(s.detection_id for s in with_crop))
        if m.assessed.get(t.candidate_id) == key:
            return None  # nothing new to look at
        started = self.monotonic()
        views: list[CropView] = []
        for s in with_crop:
            data = await self._crop(s.detection_id)
            bbox, crop_box = m.boxes.get(s.detection_id, (None, None))
            if data is None or bbox is None or crop_box is None:
                continue
            w, h = _jpeg_size(data)
            views.append(
                CropView(s.detection_id, s.px_size, data, box_in_crop(bbox, crop_box, w, h))
            )
        person_p = person_probability(t.best_detector_score, t.refined_person_score)
        a = assess_colors(
            t.candidate_id,
            profile,
            views,
            person_p=person_p,
            latency_ms=round((self.monotonic() - started) * 1000),
        )
        m.assessed[t.candidate_id] = key
        m.fields[t.candidate_id] = dict(a.fields)
        return a

    async def _revision(
        self,
        m: _Mission,
        t: Track,
        profile: SearchProfile | None,
        version: int,
        observations: list[tuple[str, float]],
    ) -> CandidateRevision:
        assessment = None if not t.open else await self._assess(m, t, profile, version)
        fields = m.fields.get(t.candidate_id, {})
        match = None if profile is None else match_score(profile, fields)
        p = priority(
            person_probability(t.best_detector_score, t.refined_person_score),
            match,
            multi_frame=t.multi_frame,
            refined=t.refined_person_score,
        )
        if t.state is CandidateState.MERGED:
            state = "merged"
        elif t.held:
            state = None  # a person or the pilot has it: keep their state
        else:
            state = "ready" if t.candidate_id in m.assessed else "new"
        return CandidateRevision.model_validate(
            {
                "candidate_id": t.candidate_id,
                "base_rev": t.rev,
                "state": state,
                "lat": t.lat,
                "lon": t.lon,
                "sigma95_m": t.sigma95_m,
                "n_seen": max(1, t.n_seen),
                "best_detector_score": t.best_detector_score,
                "refined_person_score": t.refined_person_score,
                "priority": p,
                "merged_into": t.merged_into if state == "merged" else None,
                "first_seen_utc": t.first_seen_utc,
                "last_seen_utc": t.last_seen_utc,
                "observations": [
                    {"detection_id": did, "distance_m": round(dist, 2)}
                    for did, dist in observations
                ],
                "assessment": assessment,
            }
        )


def _jpeg_size(data: bytes) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.size
    except (OSError, ValueError, Image.DecompressionBombError):
        return (0, 0)
