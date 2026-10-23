"""Candidates: what the search service writes and what the panel's queue shows.

``sd-search`` reads the target detections (``feed``), keeps its fusion in memory, and
writes candidates in batches (``write``); on a restart it reads them back (``tracks``) and
goes on from the last detection it observed. Every write is a new revision; the live
topic ``candidates`` carries ``candidate.new``, ``candidate.updated`` and
``candidate.assessed``. A person's decision (``candidate.decided``) is a later ticket's.

The queue is an order, never a filter: every candidate is listed, the merged ones only on
request, and a low priority only puts a candidate further down.
"""

from __future__ import annotations

from typing import Any

from sd_ground.api.deps import ApiStatusError
from sd_ground.db.candidates import CandidateRepo, CandidateWriteError
from sd_ground.services.missions import MissionFlow
from sd_schemas.geo import plane_distance_m
from sd_schemas.search import (
    PRIORITY_MEDIUM,
    CandidateBatch,
    CandidateCard,
    CandidateState,
    CandidateWritten,
    SearchDetection,
    SearchTrack,
    fusion_radius_m,
    priority_class,
)

__all__ = [
    "CARDS_MAX",
    "FEED_MAX",
    "card_list",
    "cards",
    "detail",
    "feed",
    "repo",
    "tracks",
    "write",
]

FEED_MAX = 500
CARDS_MAX = 200
_DECIDED = (CandidateState.CONFIRMED.value, CandidateState.REJECTED.value)


def repo(flow: MissionFlow) -> CandidateRepo:
    return CandidateRepo(flow.repo.engine)


async def feed(
    flow: MissionFlow, mission_id: str, after_seq: int, limit: int
) -> list[SearchDetection]:
    await flow.open_mission_of(mission_id)
    rows = await repo(flow).feed(mission_id, after_seq, min(limit, FEED_MAX))
    return [SearchDetection.model_validate(r) for r in rows]


async def tracks(flow: MissionFlow, mission_id: str) -> list[SearchTrack]:
    await flow.open_mission_of(mission_id)
    r = repo(flow)
    latest = await r.latest(mission_id)
    seen = await r.sightings(mission_id)
    decided = await r.decided([c["candidate_id"] for c in latest])
    out: list[SearchTrack] = []
    for c in latest:
        sightings = [SearchDetection.model_validate(d) for d in seen.get(c["candidate_id"], [])]
        out.append(
            SearchTrack.model_validate(
                {
                    "candidate": c,
                    "label": sightings[0].label if sightings else "unknown",
                    "decided": c["candidate_id"] in decided,
                    "sightings": sightings,
                }
            )
        )
    return out


async def write(
    flow: MissionFlow, mission_id: str, batch: CandidateBatch
) -> list[CandidateWritten]:
    mission = await flow.open_mission_of(mission_id)
    try:
        written = await repo(flow).write(mission_id, batch.revisions, flow.now())
    except CandidateWriteError as exc:
        raise ApiStatusError(409, exc.code, exc.detail_sv) from exc
    touched = {w.candidate_id for w in written}
    by_id = {str(c.candidate_id): c for c in await card_list(flow, mission_id, only=touched)}
    for w in written:
        card = by_id.get(w.candidate_id)
        if card is None:
            continue
        data = card.model_dump(mode="json")
        kind = "candidate.new" if w.created else "candidate.updated"
        flow.publish("candidates", kind, data, vehicle_id=mission.vehicle_id, mission_id=mission_id)
        if w.assessment_id is not None:
            flow.publish(
                "candidates",
                "candidate.assessed",
                data,
                vehicle_id=mission.vehicle_id,
                mission_id=mission_id,
            )
    return [
        CandidateWritten.model_validate(
            {"candidate_id": w.candidate_id, "rev": w.rev, "seq": w.seq, "state": w.state}
        )
        for w in written
    ]


def _near(c: dict[str, Any], decided: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: tuple[float, int] | None = None
    found = None
    for o in decided:
        if o["candidate_id"] == c["candidate_id"]:
            continue
        d = plane_distance_m(c["lat"], c["lon"], o["lat"], o["lon"])
        if d <= fusion_radius_m(c["sigma95_m"], o["sigma95_m"]) and (
            best is None or (d, o["seq"]) < best
        ):
            best, found = (d, o["seq"]), o
    return found


async def card_list(
    flow: MissionFlow, mission_id: str, *, only: set[str] | None = None
) -> list[CandidateCard]:
    """Every candidate of the mission as a card, by display number (or only those named)."""
    r = repo(flow)
    latest = await r.latest(mission_id)
    shown = [c for c in latest if only is None or c["candidate_id"] in only]
    assessed = await r.assessments([c["candidate_id"] for c in shown])
    thumbs = await r.thumbs(mission_id)
    approved, _ = await flow.repo.search_profile_rows(mission_id)
    profile = None if approved is None else int(approved["profile_version"])
    decided = [c for c in latest if c["state"] in _DECIDED]
    out: list[CandidateCard] = []
    for c in shown:
        a = next(
            (
                x["body"]
                for x in assessed.get(c["candidate_id"], [])
                if x["id"] == c["latest_assessment_id"]
            ),
            None,
        )
        near = None if c["state"] in _DECIDED else _near(c, decided)
        out.append(
            CandidateCard.model_validate(
                {
                    **{
                        k: c[k]
                        for k in (
                            "candidate_id",
                            "rev",
                            "seq",
                            "state",
                            "priority",
                            "lat",
                            "lon",
                            "sigma95_m",
                            "n_seen",
                            "n_in_view",
                            "best_detector_score",
                            "refined_person_score",
                            "merged_into",
                            "first_seen_utc",
                            "last_seen_utc",
                        )
                    },
                    "priority_class": priority_class(c["priority"]),
                    "thumb_detection_id": thumbs.get(c["candidate_id"]),
                    "engine": None if a is None else a["engine"],
                    "person": None if a is None else a["person"],
                    "fields": {} if a is None else a["fields"],
                    "flags": [] if a is None else a["flags"],
                    "assessed_profile_version": None if a is None else a["profile_version"],
                    "current_profile_version": profile,
                    "near_seq": None if near is None else near["seq"],
                    "near_state": None if near is None else near["state"],
                }
            )
        )
    return out


async def cards(
    flow: MissionFlow,
    mission_id: str,
    *,
    state: CandidateState | None,
    min_priority: float | None,
    before_seq: int | None,
    limit: int,
) -> dict[str, Any]:
    """The queue: by priority (highest first, then display number), merged candidates only
    when asked for. ``before_seq`` is the paging cursor: the ``seq`` of the last card of
    the previous page, and the next page holds the cards after it in the same order (so a
    later page never skips a card with a higher seq and a lower priority). The cursor is
    that card's priority now: if it changes between two pages, the next page can skip or
    repeat cards; the live topic ``candidates`` keeps the panel's list current."""
    await flow.mission(mission_id)
    every = await card_list(flow, mission_id)
    counts = {"review": 0, "low": 0, "confirmed": 0, "rejected": 0}
    for c in every:
        if c.state in (CandidateState.CONFIRMED, CandidateState.REJECTED):
            counts[c.state.value] += 1
        elif c.state is not CandidateState.MERGED:
            counts["review" if c.priority >= PRIORITY_MEDIUM else "low"] += 1

    def order(c: CandidateCard) -> tuple[float, int]:
        return (-c.priority, c.seq)

    after: tuple[float, int] | None = None
    if before_seq is not None:
        cursor = next((c for c in every if c.seq == before_seq), None)
        if cursor is None:
            raise ApiStatusError(404, "CANDIDATE_NOT_FOUND", "Kandidaten finns inte.")
        after = order(cursor)
    items = [
        c
        for c in every
        if (c.state is state if state is not None else c.state is not CandidateState.MERGED)
        and (min_priority is None or c.priority >= min_priority)
        and (after is None or order(c) > after)
    ]
    items.sort(key=order)
    return {
        "items": [c.model_dump(mode="json") for c in items[: min(limit, CARDS_MAX)]],
        "counts": counts,
    }


async def detail(flow: MissionFlow, candidate_id: str) -> dict[str, Any]:
    candidate_id = candidate_id.lower()  # the path accepts either case; the tables hold lower
    r = repo(flow)
    mission_id = await r.candidate_mission(candidate_id)
    if mission_id is None:
        raise ApiStatusError(404, "CANDIDATE_NOT_FOUND", "Kandidaten finns inte.")
    (card,) = await card_list(flow, mission_id, only={candidate_id})
    assessed = await r.assessments([candidate_id])
    return {
        "card": card.model_dump(mode="json"),
        "mission_id": mission_id,
        "observations": await r.observations(candidate_id),
        "assessments": [
            {"id": a["id"], "created_at": a["created_at"], **a["body"]}
            for a in assessed.get(candidate_id, [])
        ],
        "decisions": await r.decisions(candidate_id),
    }
