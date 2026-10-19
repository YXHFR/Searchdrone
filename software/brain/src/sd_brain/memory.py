"""The mission memory: the plan note, the summary, a waiting start request and the
operator's latest request (the mission, ``UPPDRAG``).

``pending_operator_request`` (a free JSON object in the ground's schema) holds both
requests: ``{"takeoff": {...} | null, "mission": {...} | null}``. An older memory that
holds a start request alone (``{"kind": "takeoff", ...}``) still reads as that request.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any, Literal

from pydantic import AwareDatetime, Field, ValidationError

from sd_schemas.base import Id, SdModel
from sd_schemas.commander import MissionMemory, MissionMemoryUpdate
from sd_schemas.sv_commands import normalize

__all__ = [
    "MISSION_REQUEST_MAX_BYTES",
    "MISSION_REQUEST_MAX_CHARS",
    "MissionRequest",
    "PendingRequest",
    "empty_memory",
    "fit_json",
    "is_acknowledgement",
    "mission_request",
    "next_request",
    "pending_request",
    "to_update",
    "with_changes",
]

# The operator's request as kept between turns. The pending field holds at most 4 KiB of
# JSON with non-ASCII escaped (an "ö" is 6 bytes, an emoji 12), so each line is also cut
# to a byte budget: the latest and the two before it fit with the start request.
MISSION_REQUEST_MAX_CHARS = 600
MISSION_REQUEST_MAX_BYTES = 1500
EARLIER_REQUEST_MAX_BYTES = 700
EARLIER_REQUESTS = 2

# Lines that only answer (a "ja", an "ok"): no new request, so they replace nothing.
_ACK_WORDS = frozenset(
    {
        "ja", "japp", "jo", "jajamen", "ok", "okej", "okay", "kor", "gor", "det", "bra",
        "fint", "toppen", "tack", "perfekt", "absolut", "visst", "precis", "yes", "nu",
        "sa", "kanon", "grymt",
    }
)  # fmt: skip


class PendingRequest(SdModel):
    kind: Literal["takeoff"]
    message_id: Id
    requested_at: AwareDatetime
    alt_m: float | None = None


class MissionRequest(SdModel):
    """The operator's latest request, word for word (data, never an instruction to the
    system), and the two before it (``earlier``, oldest first), so a request is not lost
    to a later question. ``flown``: the drone has been airborne since, so a landing ends
    it."""

    text: str = Field(min_length=1, max_length=MISSION_REQUEST_MAX_CHARS)
    message_id: Id
    requested_at: AwareDatetime
    flown: bool = False
    earlier: list[str] = Field(default_factory=list[str], max_length=EARLIER_REQUESTS)


def is_acknowledgement(text: str) -> bool:
    """A short line of only answer words ("Ja kör", "ok tack"): it asks for nothing."""
    words = normalize(text).replace("å", "a").replace("ä", "a").replace("ö", "o").split()
    return 0 < len(words) <= 4 and all(w in _ACK_WORDS for w in words)


def fit_json(text: str, max_bytes: int) -> str:
    """``text`` cut so that its JSON string (ASCII escaped, as the ground measures it)
    is at most ``max_bytes`` long."""
    text = text[:MISSION_REQUEST_MAX_CHARS]
    while text and len(json.dumps(text)) > max_bytes:
        text = text[: max(len(text) - max(1, (len(json.dumps(text)) - max_bytes) // 12), 0)]
    return text


def next_request(
    prev: MissionRequest | None, text: str, message_id: Any, at: dt.datetime, *, flown: bool
) -> MissionRequest:
    """The new latest request; the one before it moves to ``earlier``."""
    earlier: list[str] = []
    if prev is not None:
        earlier = [*prev.earlier, prev.text][-EARLIER_REQUESTS:]
    return MissionRequest(
        text=fit_json(" ".join(text.split()), MISSION_REQUEST_MAX_BYTES) or "…",
        message_id=message_id,
        requested_at=at,
        flown=flown,
        earlier=[fit_json(e, EARLIER_REQUEST_MAX_BYTES) for e in earlier if e],
    )


class _Requests(SdModel):
    takeoff: PendingRequest | None = None
    mission: MissionRequest | None = None


def empty_memory(mission_id: Any) -> MissionMemory:
    return MissionMemory(
        mission_id=mission_id,
        summary_sv="",
        plan_note="",
        summarized_up_to_seq=0,
        pending_operator_request=None,
        updated_at=None,
    )


def _requests(mem: MissionMemory) -> _Requests:
    raw = mem.pending_operator_request
    if not raw:
        return _Requests()
    try:
        if "kind" in raw:  # an older memory: the start request alone
            return _Requests(takeoff=PendingRequest.model_validate(raw))
        return _Requests.model_validate(raw)
    except ValidationError:
        return _Requests()


def pending_request(mem: MissionMemory) -> PendingRequest | None:
    return _requests(mem).takeoff


def mission_request(mem: MissionMemory) -> MissionRequest | None:
    return _requests(mem).mission


def with_changes(
    mem: MissionMemory,
    *,
    plan_note: str | None = None,
    summary_sv: str | None = None,
    pending: PendingRequest | Literal["keep"] | None = "keep",
    mission: MissionRequest | Literal["keep"] | None = "keep",
) -> MissionMemory:
    update: dict[str, Any] = {}
    if plan_note is not None:
        update["plan_note"] = plan_note[:400]
    if summary_sv is not None:
        update["summary_sv"] = summary_sv[:1200]
    if pending != "keep" or mission != "keep":
        req = _requests(mem)
        if pending != "keep":
            req = req.model_copy(update={"takeoff": pending})
        if mission != "keep":
            req = req.model_copy(update={"mission": mission})
        update["pending_operator_request"] = (
            None if req.takeoff is None and req.mission is None else req.model_dump(mode="json")
        )
    return mem.model_copy(update=update)


def to_update(mem: MissionMemory) -> MissionMemoryUpdate:
    return MissionMemoryUpdate(
        mission_id=mem.mission_id,
        summary_sv=mem.summary_sv,
        plan_note=mem.plan_note,
        summarized_up_to_seq=mem.summarized_up_to_seq,
        pending_operator_request=mem.pending_operator_request,
    )
