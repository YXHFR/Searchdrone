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

Pure and deterministic: the same detections in the same order give the same candidates.
A new candidate's id is its first detection's id, so a restart that replays the same
detections cannot create a second one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from sd_schemas.search import CandidateState
from sd_search.geo import distance_m

__all__ = [
    "FUSION_RADIUS_M",
    "FUSION_RADIUS_MAX_M",
    "SIGMA_MIN_M",
    "Fusion",
    "Sighting",
    "Step",
    "Track",
    "fusion_radius_m",
]

FUSION_RADIUS_M = 4.0
FUSION_RADIUS_MAX_M = 12.0
SIGMA_MIN_M = 2.0
_SIGMA_FLOOR_M = 0.5  # a sighting's sigma below this counts as this (no division by zero)
_SCORE_FLOOR = 0.05  # a sighting with score 0 still has a little weight

# States a person (or the pilot through the gate) owns: fusion never changes or merges them.
HUMAN_STATES = frozenset(
    {CandidateState.INSPECTING, CandidateState.CONFIRMED, CandidateState.REJECTED}
)
# Candidates that take no new sightings.
CLOSED_STATES = frozenset(
    {CandidateState.CONFIRMED, CandidateState.REJECTED, CandidateState.MERGED}
)


def fusion_radius_m(sigma_a: float, sigma_b: float) -> float:
    return min(FUSION_RADIUS_MAX_M, FUSION_RADIUS_M + max(sigma_a, sigma_b))


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
        """Position and radius from every sighting (weight score / sigma^2)."""
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
        self.sigma95_m = round(max(SIGMA_MIN_M, min(1000.0, 1.0 / math.sqrt(inv))), 2)
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

    def add(self, s: Sighting) -> Step | None:
        """Fuse one sighting; ``None`` if this detection was fused before."""
        if s.detection_id in self.seen:
            return None
        self.seen.add(s.detection_id)
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
