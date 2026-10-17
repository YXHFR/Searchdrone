"""The mission memory: the plan note, the summary and a waiting start request."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AwareDatetime, ValidationError

from sd_schemas.base import Id, SdModel
from sd_schemas.commander import MissionMemory, MissionMemoryUpdate

__all__ = ["PendingRequest", "empty_memory", "pending_request", "to_update", "with_changes"]


class PendingRequest(SdModel):
    kind: Literal["takeoff"]
    message_id: Id
    requested_at: AwareDatetime
    alt_m: float | None = None


def empty_memory(mission_id: Any) -> MissionMemory:
    return MissionMemory(
        mission_id=mission_id,
        summary_sv="",
        plan_note="",
        summarized_up_to_seq=0,
        pending_operator_request=None,
        updated_at=None,
    )


def pending_request(mem: MissionMemory) -> PendingRequest | None:
    raw = mem.pending_operator_request
    if not raw:
        return None
    try:
        return PendingRequest.model_validate(raw)
    except ValidationError:
        return None


def with_changes(
    mem: MissionMemory,
    *,
    plan_note: str | None = None,
    summary_sv: str | None = None,
    pending: PendingRequest | Literal["keep"] | None = "keep",
) -> MissionMemory:
    update: dict[str, Any] = {}
    if plan_note is not None:
        update["plan_note"] = plan_note[:400]
    if summary_sv is not None:
        update["summary_sv"] = summary_sv[:1200]
    if pending != "keep":
        update["pending_operator_request"] = (
            None if pending is None else pending.model_dump(mode="json")
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
