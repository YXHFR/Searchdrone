"""Candidates: what the search service writes and what the panel's queue shows.

``sd-search`` reads the target detections (``feed``), keeps its fusion in memory, and
writes candidates in batches (``write``); on a restart it reads them back (``tracks``) and
goes on from the last detection it observed. Every write is a new revision; the live
topic ``candidates`` carries ``candidate.new``, ``candidate.updated`` and
``candidate.assessed``.

Only a person decides (``decide``): confirm, reject, unsure, and for the pilot inspect
and reopen, along ``CANDIDATE_TRANSITIONS``; anything else is 409. A confirmation needs a
large view of one of the candidate's pictures opened by the same session first
(``open_view``, checked in the server's own log). Each decision is a new row, a system
line in the chat and ``candidate.decided`` on the live topic.

The queue is an order, never a filter: every candidate is listed, the merged ones only on
request, and a low priority only puts a candidate further down.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final
from xml.sax.saxutils import escape

from sd_ground.api.deps import ApiStatusError, not_built
from sd_ground.auth import Principal
from sd_ground.db.candidates import CandidateRepo, CandidateWriteError, Decided
from sd_ground.logging import get_logger
from sd_ground.services import chat as chat_service
from sd_ground.services import files as files_service
from sd_ground.services.missions import MissionFlow
from sd_schemas.base import new_id
from sd_schemas.geo import plane_distance_m
from sd_schemas.search import (
    PRIORITY_MEDIUM,
    CandidateBatch,
    CandidateCard,
    CandidateDecision,
    CandidateState,
    CandidateView,
    CandidateWritten,
    DecisionKind,
    SearchDetection,
    SearchTrack,
    fusion_radius_m,
    priority_class,
)

__all__ = [
    "CANDIDATE_TRANSITIONS",
    "CARDS_MAX",
    "FEED_MAX",
    "PILOT_ONLY",
    "card_list",
    "cards",
    "decide",
    "detail",
    "feed",
    "gpx",
    "open_view",
    "repo",
    "tracks",
    "view_file",
    "write",
]

log = get_logger("sd_ground.candidates")

FEED_MAX = 500
CARDS_MAX = 200
_DECIDED = (CandidateState.CONFIRMED.value, CandidateState.REJECTED.value)

_S = CandidateState
_OPEN = (_S.READY, _S.ASSESSING, _S.INSPECTING)
# decision -> {state in force: state after}; ``None`` after = the state stays (a note).
# Everything not listed is refused with 409. ``merged`` and the search service's own
# moves (new -> ready -> assessing) are not a person's and are not here.
CANDIDATE_TRANSITIONS: Final[
    Mapping[DecisionKind, Mapping[CandidateState, CandidateState | None]]
] = {
    "confirm": dict.fromkeys(_OPEN, _S.CONFIRMED),
    "reject": dict.fromkeys(_OPEN, _S.REJECTED),
    # Not reachable yet: ``decide`` answers 501 until the closer-look skill exists.
    "inspect": {_S.READY: _S.INSPECTING, _S.ASSESSING: _S.INSPECTING},
    "unsure": dict.fromkeys((_S.NEW, *_OPEN), None),
    "reopen": {_S.REJECTED: _S.READY, _S.CONFIRMED: _S.READY},
}
# The pilot and the observer may both confirm or reject a find (a second pair of eyes on
# the pictures); moving the drone closer and undoing a decision are the pilot's.
PILOT_ONLY: Final = frozenset({"inspect", "reopen"})
PEOPLE: Final = frozenset({"pilot", "observer"})


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


# --- a person's decision ------------------------------------------------------------------


def _person(who: Principal, test_rig: bool) -> str:
    """The session of a person in the panel. Service tokens never decide (E1), and a CLI
    token only on the test rig (its scripted operator)."""
    if who.role not in PEOPLE:
        raise ApiStatusError(403, "HUMAN_REQUIRED", "Bara en människa kan bedöma kandidater.")
    if (not who.interactive and not test_rig) or not who.session_id:
        raise ApiStatusError(
            403, "INTERACTIVE_SESSION_REQUIRED", "Det här måste göras av en människa i panelen."
        )
    if not who.issuer_ref.startswith("operator:"):
        # A decision and a view name the operator who made them (the tables check it); an
        # observer token from the CLI has no operator to name.
        raise ApiStatusError(403, "OPERATOR_REQUIRED", "Bara en operatör kan bedöma kandidater.")
    if not who.interactive:
        # Only the test rig's scripted operator gets here; say so where it shows.
        log.warning("candidate.decision_by_script", issuer_ref=who.issuer_ref)
    return who.session_id


def _next_state(body: CandidateDecision) -> Callable[[str], str | None]:
    allowed = CANDIDATE_TRANSITIONS[body.decision]

    def step(state: str) -> str | None:
        before = CandidateState(state)
        if before not in allowed:
            raise CandidateWriteError(
                "TRANSITION_NOT_ALLOWED", "Det beslutet går inte för kandidaten just nu."
            )
        if body.decision == "reopen" and before is _S.CONFIRMED and not body.note.strip():
            raise CandidateWriteError("NOTE_REQUIRED", "Skriv varför bekräftelsen ska ångras.")
        after = allowed[before]
        return None if after is None else after.value

    return step


def _chat_text(d: Decided, decision: DecisionKind, by: str) -> str:
    """The ground's own words only: a person's note goes in the message refs and the event
    log, never into a system line the commander reads as the ground's."""
    k = f"K{d.seq}"
    if decision == "confirm":
        return f"Fynd bekräftat: {k} (av {by})."
    if decision == "reject":
        return f"{k} avvisad av {by}. Sökningen fortsätter."
    if decision == "unsure":
        return f"{k} markerad som osäker av {by}."
    if d.before == _S.CONFIRMED.value:
        return f"Bekräftelsen av {k} ångrad av {by}. Anteckningen finns i händelseloggen."
    return f"{k} öppnad igen av {by}."


async def decide(
    flow: MissionFlow, candidate_id: str, body: CandidateDecision, who: Principal
) -> dict[str, Any]:
    """A person's decision. 403 for anyone but a person (and inspect/reopen for anyone
    but the pilot); 409 ``NOT_VIEWED`` for a confirmation before a large view,
    ``SUPERSEDED`` when the card is older than the candidate, ``TRANSITION_NOT_ALLOWED``
    for a move the table does not have. Returns the candidate's detail."""
    candidate_id = candidate_id.lower()
    session_id = _person(who, flow.g.test_rig)
    if body.decision in PILOT_ONLY and who.role != "pilot":
        raise ApiStatusError(403, "FORBIDDEN_ROLE", "Bara piloten får göra det här.")
    if body.decision == "inspect":
        raise not_built("B2-09")  # the drone's closer look; the state would wait forever
    r = repo(flow)
    mission_id = await r.candidate_mission(candidate_id)
    if mission_id is None:
        raise ApiStatusError(404, "CANDIDATE_NOT_FOUND", "Kandidaten finns inte.")
    mission = await flow.open_mission_of(mission_id)
    if body.decision == "confirm" and not await r.viewed(candidate_id, session_id):
        raise ApiStatusError(
            409, "NOT_VIEWED", "Öppna minst en bild i stor storlek innan du bekräftar."
        )
    try:
        d = await r.decide(
            candidate_id,
            decision=body.decision,
            note=body.note,
            subject_rev=body.subject_rev,
            operator_ref=who.issuer_ref,
            session_id=session_id,
            now=flow.now(),
            next_state=_next_state(body),
        )
    except CandidateWriteError as exc:
        raise ApiStatusError(409, exc.code, exc.detail_sv) from exc
    out = await detail(flow, candidate_id)
    payload = {
        "candidate_id": candidate_id,
        "seq": d.seq,
        "decision": body.decision,
        "before": d.before,
        "after": d.after,
        "rev": d.rev,
        "decision_id": d.decision_id,
        "by": who.issuer_ref,
    }
    log.info("candidate.decided", mission_id=mission_id, **payload)
    flow.g.add_event(
        "candidate.decided",
        severity="info",
        mission_id=mission_id,
        vehicle_id=mission.vehicle_id,
        payload=payload,
    )
    if body.decision == "reopen" and d.before == _S.CONFIRMED.value:
        flow.g.add_event(
            "candidate.unconfirmed",
            severity="warning",
            mission_id=mission_id,
            vehicle_id=mission.vehicle_id,
            payload=payload | {"note": body.note},
        )
    flow.publish(
        "candidates",
        "candidate.decided",
        out["card"] | {"decision": payload},
        vehicle_id=mission.vehicle_id,
        mission_id=mission_id,
    )
    await chat_service.add_line(
        flow,
        mission,
        msg_id=str(new_id()),
        role="system",
        author_ref="system:ground",
        kind="status",
        text=_chat_text(d, body.decision, who.display_name or who.issuer_ref),
        in_reply_to=None,
        refs={"candidate_id": candidate_id, "decision": body.decision, "note": body.note},
    )
    return out


# --- large views and the position for the ground team ---------------------------------------


async def open_view(flow: MissionFlow, detection_id: str, who: Principal) -> CandidateView:
    """Log that this session opened a detection's picture in large size. The picture is
    the crop the drone sent (the largest the ground station holds)."""
    detection_id = detection_id.lower()
    session_id = _person(who, flow.g.test_rig)
    r = repo(flow)
    found = await r.crop_path(detection_id)
    if found is None:
        raise ApiStatusError(404, "CROP_NOT_FOUND", "Bilden finns inte.")
    mission_id, _ = found
    await flow.mission(mission_id)
    view_id = await r.add_view(mission_id, detection_id, who.issuer_ref, session_id, flow.now())
    return CandidateView.model_validate({"view_id": view_id, "detection_id": detection_id})


async def view_file(flow: MissionFlow, view_id: str) -> Path:
    r = repo(flow)
    found = await r.view(view_id.lower())
    crop = None if found is None else await r.crop_path(found[1])
    path = None if crop is None else files_service.resolve_file(flow.mission_dir(crop[0]), crop[1])
    if path is None:
        raise ApiStatusError(404, "VIEW_NOT_FOUND", "Bilden finns inte.")
    return path


async def gpx(flow: MissionFlow, candidate_id: str) -> str:
    """A confirmed find as GPX 1.1 with one waypoint (409 for any other candidate)."""
    d = await detail(flow, candidate_id)
    card = d["card"]
    if card["state"] != _S.CONFIRMED.value:
        raise ApiStatusError(409, "NOT_CONFIRMED", "Bara ett bekräftat fynd kan exporteras.")
    confirmed = [x for x in d["decisions"] if x["decision"] == "confirm"]
    when = confirmed[-1]["created_at"] if confirmed else flow.now()
    sigma = round(float(card["sigma95_m"]), 1)
    name = escape(f"K{card['seq']}")
    desc = escape(f"Bekräftat fynd K{card['seq']}, osäkerhet ±{sigma} m (95 %)")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" creator="Searchdrone" xmlns="http://www.topografix.com/GPX/1/1">\n'
        f'  <wpt lat="{float(card["lat"]):.7f}" lon="{float(card["lon"]):.7f}">\n'
        f"    <time>{when.isoformat().replace('+00:00', 'Z')}</time>\n"
        f"    <name>{name}</name>\n"
        f"    <desc>{desc}</desc>\n"
        "  </wpt>\n"
        "</gpx>\n"
    )
