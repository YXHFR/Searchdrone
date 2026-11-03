"""Fusion: detections of a target class become candidates (places where someone may be).

A detection joins the nearest active candidate of its class within
``FUSION_RADIUS_M + max(sigma_new, sigma_candidate)`` (at most ``FUSION_RADIUS_MAX_M``),
otherwise it starts a new candidate. A single sighting is never dropped. The position is
the mean of the sightings weighted by score / sigma^2, and the 95 % radius shrinks with
more sightings but never below ``SIGMA_MIN_M``.

Confirmed and rejected candidates take no new sightings: a detection next to one starts a
new candidate (the panel shows it as "near rejected K7"). Two active candidates that end
up closer than the radius are merged: the older stays, the younger becomes ``merged``
with ``merged_into`` set; its observations stay where they are. Nothing merges once a
person has decided on either, or while one is being inspected.

A closer look (``inspect_detection``) flies over one candidate and photographs it from
lower and lower. Its picture of that target joins the target beyond the fusion radius
(``force_candidate_id``): it starts no new candidate and merges nothing. Which picture
that is the search service decides (``sd_search.app.look_joins``): at most one per still,
the nearest to where the target was when the order went out, and only within
``look_radius_m``. Being sharper and taken from lower, the close-ups have a smaller
sigma, so they weigh more in the refit and the position tightens. A forced join is not
held to the fusion radius, so it may be a second person: the 95 % radius of a candidate
with its closer look's pictures is therefore never smaller than the distance of its
farthest sighting from the new position, and its keep-out cylinder (30 m + sigma) keeps
the drone at least 30 m from every place it was seen.

Pure and deterministic: the same detections in the same order give the same candidates.
A new candidate's id is its first detection's id, so a restart that replays the same
detections cannot create a second one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime

from sd_schemas.geo import plane_distance_m as distance_m
from sd_schemas.inspect import match_radius_m
from sd_schemas.search import (
    FUSION_RADIUS_M,
    FUSION_RADIUS_MAX_M,
    CandidateState,
    fusion_radius_m,
)

__all__ = [
    "FUSION_RADIUS_M",
    "FUSION_RADIUS_MAX_M",
    "LOOK_JOIN_MAX_M",
    "SIGMA_MIN_M",
    "Fusion",
    "Sighting",
    "Step",
    "Track",
    "fusion_radius_m",
    "look_radius_m",
]

SIGMA_MIN_M = 2.0
_SIGMA_FLOOR_M = 0.5  # a sighting's sigma below this counts as this (no division by zero)
_SCORE_FLOOR = 0.05  # a sighting with score 0 still has a little weight
# How far from the target's position at order time a closer look's picture may lie and
# still be forced onto it: ``match_radius_m`` (the drone's own rule, max(15 m, 2 sigma))
# but at most the 30 m keep-out (PERSON_KEEPOUT_M), which is the drone's rule up to an
# order sigma of 15 m. Farther out, the joined picture would stretch the target's
# cylinder past 30 m + 30 m, while as a candidate of its own it gets a 30 m + sigma
# cylinder around it: tighter, and right when it is someone else.
LOOK_JOIN_MAX_M = 30.0  # = sd_safety.constants.PERSON_KEEPOUT_M (a test checks)

# States a person (or the pilot through the gate) owns: fusion never changes or merges them.
HUMAN_STATES = frozenset(
    {CandidateState.INSPECTING, CandidateState.CONFIRMED, CandidateState.REJECTED}
)
# Candidates that take no new sightings.
CLOSED_STATES = frozenset(
    {CandidateState.CONFIRMED, CandidateState.REJECTED, CandidateState.MERGED}
)


@dataclass(frozen=True)
class Sighting:
    """One stored detection of a target class, as sd-search reads it from the ground."""

    detection_id: str
    seq: int
    still_id: str
    label: str
    lat: float
    lon: float
    sigma95_m: float
    score: float
    px_size: float
    captured_utc: datetime
    has_crop: bool = False
    command_id: str | None = None  # the order its still was taken during
    px_per_m: float | None = None  # its still's detail (which camera, at what height)
    look_of: str | None = None  # a closer look's still: the candidate the look was at


def look_radius_m(sigma95_m: float) -> float:
    """How near the target's position at order time (whose uncertainty was ``sigma95_m``)
    a closer look's picture must lie to be forced onto the target."""
    return min(match_radius_m(sigma95_m), LOOK_JOIN_MAX_M)


@dataclass
class Track:
    """A candidate as fusion holds it. ``rev`` is the revision in force on the ground
    (0 = not stored yet) and ``seq`` its display number once the ground gave one."""

    candidate_id: str
    label: str
    state: CandidateState
    lat: float
    lon: float
    sigma95_m: float
    first_seen_utc: datetime
    last_seen_utc: datetime
    sightings: list[Sighting] = field(default_factory=list[Sighting])
    merged_into: str | None = None
    rev: int = 0
    seq: int | None = None
    decided: bool = False  # a person has recorded a decision on it
    refined_person_score: float | None = None

    @property
    def n_seen(self) -> int:
        return len({s.still_id for s in self.sightings})

    @property
    def multi_frame(self) -> bool:
        return self.n_seen >= 2

    @property
    def best_detector_score(self) -> float:
        return max((s.score for s in self.sightings), default=0.0)

    @property
    def open(self) -> bool:
        return self.state not in CLOSED_STATES

    @property
    def held(self) -> bool:
        """A person or the pilot has it: fusion may add sightings but never merge it."""
        return self.decided or self.state in HUMAN_STATES

    def refit(self) -> None:
        """Position and radius from every sighting (weight score / sigma^2). With its
        closer look's pictures the radius also reaches every sighting (rounded up)."""
        if not self.sightings:
            return
        wsum = lat = lon = inv = 0.0
        for s in self.sightings:
            sig = max(_SIGMA_FLOOR_M, s.sigma95_m)
            w = max(_SCORE_FLOOR, s.score) / (sig * sig)
            wsum += w
            lat += w * s.lat
            lon += w * s.lon
            inv += 1.0 / (sig * sig)
        self.lat, self.lon = lat / wsum, lon / wsum
        sigma = round(max(SIGMA_MIN_M, min(1000.0, 1.0 / math.sqrt(inv))), 2)
        if any(s.look_of == self.candidate_id for s in self.sightings):
            spread = max(distance_m(self.lat, self.lon, s.lat, s.lon) for s in self.sightings)
            sigma = max(sigma, min(1000.0, math.ceil(spread * 100) / 100))
        self.sigma95_m = sigma
        self.first_seen_utc = min(s.captured_utc for s in self.sightings)
        self.last_seen_utc = max(s.captured_utc for s in self.sightings)


@dataclass(frozen=True)
class Step:
    """What one sighting changed: the candidate it joined (``created`` if new), its
    distance from that candidate before the refit, and candidates merged by it."""

    candidate_id: str
    created: bool
    distance_m: float
    merged: tuple[str, ...] = ()


def _age_key(t: Track) -> tuple[datetime, str]:
    return (t.first_seen_utc, t.candidate_id)


class Fusion:
    """The candidates of one mission. ``tracks`` keeps creation order."""

    def __init__(self, tracks: list[Track] | None = None) -> None:
        self.tracks: dict[str, Track] = {}
        self.seen: set[str] = set()  # detection ids already fused
        for t in tracks or []:
            self.tracks[t.candidate_id] = t
            self.seen.update(s.detection_id for s in t.sightings)

    def add(self, s: Sighting, force_candidate_id: str | None = None) -> Step | None:
        """Fuse one sighting; ``None`` if this detection was fused before.

        ``force_candidate_id``: a closer look's picture of its target. It joins that
        candidate, when it is open and of the sighting's class, however far it lies, and
        merges nothing; otherwise the sighting is fused as usual. The caller limits how
        far (``look_radius_m``) and how many (one per still)."""
        if s.detection_id in self.seen:
            return None
        self.seen.add(s.detection_id)
        target = None if force_candidate_id is None else self.tracks.get(force_candidate_id)
        if target is not None and target.open and target.label == s.label:
            d = distance_m(target.lat, target.lon, s.lat, s.lon)
            target.sightings.append(replace(s, look_of=target.candidate_id))
            target.refit()
            return Step(target.candidate_id, created=False, distance_m=d)
        best: tuple[float, datetime, str] | None = None
        for t in self.tracks.values():
            if t.label != s.label or not t.open:
                continue
            d = distance_m(t.lat, t.lon, s.lat, s.lon)
            if d <= fusion_radius_m(s.sigma95_m, t.sigma95_m):
                key = (d, *_age_key(t))
                if best is None or key < best:
                    best = key
        if best is None:
            t = Track(
                candidate_id=s.detection_id,
                label=s.label,
                state=CandidateState.NEW,
                lat=s.lat,
                lon=s.lon,
                sigma95_m=s.sigma95_m,
                first_seen_utc=s.captured_utc,
                last_seen_utc=s.captured_utc,
                sightings=[s],
            )
            t.refit()
            self.tracks[t.candidate_id] = t
            return Step(t.candidate_id, created=True, distance_m=0.0, merged=self._merge(t))
        t = self.tracks[best[2]]
        t.sightings.append(s)
        t.refit()
        return Step(t.candidate_id, created=False, distance_m=best[0], merged=self._merge(t))

    def _merge(self, t: Track) -> tuple[str, ...]:
        """Merge ``t`` with every open candidate of its class now within the radius."""
        merged: list[str] = []
        while not t.held and t.open:
            near = [
                o
                for o in self.tracks.values()
                if o is not t
                and o.label == t.label
                and o.open
                and not o.held
                and distance_m(o.lat, o.lon, t.lat, t.lon)
                < fusion_radius_m(o.sigma95_m, t.sigma95_m)
            ]
            if not near:
                break
            other = min(near, key=lambda o: (distance_m(o.lat, o.lon, t.lat, t.lon), *_age_key(o)))
            older, younger = sorted((t, other), key=_age_key)
            younger.state = CandidateState.MERGED
            younger.merged_into = older.candidate_id
            merged.append(younger.candidate_id)
            t = older
        return tuple(merged)
