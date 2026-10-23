"""``FakeGround``: the search API's rules in memory (revisions, one observation per
detection, closed and held candidates), for the service tests."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from sd_schemas.link import BBox
from sd_schemas.probmap import (
    ProbabilityMapInputs,
    ProbabilityMapLatest,
    ProbabilityMapWrite,
    ProbabilityMapWritten,
    SearchStill,
)
from sd_schemas.search import (
    Candidate,
    CandidateBatch,
    CandidateState,
    CandidateWritten,
    SearchDetection,
    SearchHeartbeat,
    SearchProfile,
    SearchTrack,
)
from sd_search.client import ConflictError

CLOSED = {CandidateState.CONFIRMED, CandidateState.REJECTED, CandidateState.MERGED}


@dataclass
class FakeGround:
    mission_id: str
    stored: list[SearchDetection] = field(default_factory=list[SearchDetection])
    crops: dict[str, bytes] = field(default_factory=dict[str, bytes])
    revs: dict[str, list[Candidate]] = field(default_factory=dict[str, list[Candidate]])
    observed: dict[str, str] = field(default_factory=dict[str, str])  # detection -> candidate
    assessments: list[Any] = field(default_factory=list[Any])
    decided: set[str] = field(default_factory=set[str])
    profile_body: SearchProfile | None = None
    beats: list[SearchHeartbeat] = field(default_factory=list[SearchHeartbeat])
    writes: int = 0
    inputs: ProbabilityMapInputs = field(default_factory=lambda: ProbabilityMapInputs(area=None))
    still_rows: list[SearchStill] = field(default_factory=list[SearchStill])
    maps: list[ProbabilityMapWrite] = field(default_factory=list[ProbabilityMapWrite])

    def add_detection(self, **kw: Any) -> SearchDetection:
        d = SearchDetection.model_validate(
            {
                "seq": len(self.stored) + 1,
                "bbox": BBox(x=100, y=100, w=20, h=20),
                "crop_box": BBox(x=70, y=70, w=80, h=80),
                "has_crop": False,
                "label": "person",
                "score": 0.6,
                "sigma95_m": 3.0,
                "px_size": 20.0,
            }
            | kw
        )
        self.stored.append(d)
        return d

    def latest(self, cid: str) -> Candidate:
        return self.revs[cid][-1]

    def decide(self, cid: str, state: CandidateState) -> None:
        """A person's decision: a new revision in their state."""
        c = self.latest(cid)
        self.revs[cid].append(c.model_copy(update={"rev": c.rev + 1, "state": state}))
        self.decided.add(cid)

    # --- SearchApi -----------------------------------------------------------------------

    async def heartbeat(self, beat: SearchHeartbeat) -> None:
        self.beats.append(beat)

    async def missions(self) -> list[str]:
        return [self.mission_id]

    async def detections(
        self, mission_id: str, after_seq: int, limit: int
    ) -> list[SearchDetection]:
        return [d for d in self.stored if d.seq > after_seq][:limit]

    async def crop(self, detection_id: str) -> bytes | None:
        return self.crops.get(detection_id)

    async def tracks(self, mission_id: str) -> list[SearchTrack]:
        by_id = {str(d.detection_id): d for d in self.stored}
        out: list[SearchTrack] = []
        for cid, revs in self.revs.items():
            seen = [by_id[d] for d, c in self.observed.items() if c == cid]
            out.append(
                SearchTrack(
                    candidate=revs[-1],
                    label=seen[0].label if seen else "unknown",
                    decided=cid in self.decided,
                    sightings=sorted(seen, key=lambda d: d.seq),
                )
            )
        return out

    async def write(self, mission_id: str, batch: CandidateBatch) -> list[CandidateWritten]:
        staged = {k: list(v) for k, v in self.revs.items()}
        observed = dict(self.observed)
        out: list[CandidateWritten] = []
        for r in batch.revisions:
            cid = str(r.candidate_id)
            prev = staged.get(cid)
            if r.base_rev == 0:
                if prev:
                    raise ConflictError("exists", 409)
                seq = 1 + max((v[0].seq for v in staged.values()), default=0)
                state = CandidateState(r.state)
            else:
                if not prev or prev[-1].rev != r.base_rev or prev[-1].state in CLOSED:
                    raise ConflictError("stale or closed", 409)
                if r.state is not None and prev[-1].state is CandidateState.INSPECTING:
                    raise ConflictError("held", 409)
                seq = prev[-1].seq
                state = prev[-1].state if r.state is None else CandidateState(r.state)
            for o in r.observations:
                if str(o.detection_id) in observed:
                    raise ConflictError("observed", 409)
                observed[str(o.detection_id)] = cid
            c = Candidate.model_validate(
                r.model_dump(exclude={"base_rev", "observations", "assessment", "state"})
                | {
                    "rev": r.base_rev + 1,
                    "mission_id": mission_id,
                    "seq": seq,
                    "state": state,
                    "n_in_view": r.n_seen,
                }
            )
            staged.setdefault(cid, []).append(c)
            out.append(CandidateWritten(candidate_id=UUID(cid), rev=c.rev, seq=seq, state=state))
        self.revs, self.observed = staged, observed
        self.assessments += [r.assessment for r in batch.revisions if r.assessment is not None]
        self.writes += 1
        return out

    async def profile(self, mission_id: str) -> SearchProfile | None:
        return self.profile_body

    async def map_inputs(self, mission_id: str) -> ProbabilityMapInputs:
        latest = None
        if self.maps:
            m = self.maps[-1]
            latest = ProbabilityMapLatest(version=m.version, frozen=m.frozen, event_seq=m.event_seq)
        return self.inputs.model_copy(update={"latest": latest})

    async def stills(self, mission_id: str, after_seq: int, limit: int) -> list[SearchStill]:
        return [s for s in self.still_rows if s.seq > after_seq][:limit]

    async def write_map(self, mission_id: str, body: ProbabilityMapWrite) -> ProbabilityMapWritten:
        if self.maps and (self.maps[-1].frozen or body.version != self.maps[-1].version + 1):
            raise ConflictError("frozen or stale", 409)
        if not self.maps and body.version != 1:
            raise ConflictError("stale", 409)
        self.maps.append(body)
        return ProbabilityMapWritten(version=body.version, frozen=body.frozen)


def utc(seconds: float) -> dt.datetime:
    return dt.datetime(2026, 10, 1, 12, 0, tzinfo=dt.UTC) + dt.timedelta(seconds=seconds)
