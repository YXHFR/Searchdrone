"""The search profile: the AI's proposals, the pilot's approval and the version in force.

The AI commander proposes a profile from the chat (``propose``); the proposal is stored
as a row without ``approved_by`` and changes nothing. Only a pilot's click on the card
approves (``approve``): the card sends the whole profile as the pilot corrected it, and
it becomes the next version (``profile_version`` grows by one). The matching reads only
the approved version in force (``approved``), never a proposal. Every row is kept
(``search_profiles`` is append-only).

The profile's text (free text, visual notes, sources, kinds) may hold health details, so
it is never written to a log, an event or the live stream: those carry the ids and the
version only, and the panel reads the profile over REST.

Events on the bus (topic ``search_profile``): ``search_profile.proposed`` and
``search_profile.approved``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sd_ground.logging import get_logger
from sd_ground.services.missions import FlowError
from sd_schemas.search import (
    SearchProfile,
    SearchProfileApproval,
    SearchProfileDraft,
    SearchProfileState,
)

if TYPE_CHECKING:
    from sd_ground.services.missions import MissionFlow

__all__ = ["approve", "approved", "propose", "state", "versions"]

log = get_logger("sd_ground.search_profile")

TOPIC = "search_profile"


def _profile(row: dict[str, Any]) -> SearchProfile:
    return SearchProfile.model_validate(row["body"])


def _ids(row: dict[str, Any]) -> dict[str, Any]:
    """What the live stream, the event log and the application log may carry."""
    return {
        "row_id": row["row_id"],
        "profile_id": row["profile_id"],
        "profile_version": row["profile_version"],
    }


async def propose(flow: MissionFlow, mission_id: str, draft: SearchProfileDraft, by: str) -> str:
    """Store the AI's proposal; returns its row id. A closed mission takes none (409)."""
    mission = await flow.open_mission_of(mission_id)
    row = await flow.repo.add_search_profile(
        str(mission.id),
        draft.model_dump(mode="json"),
        proposed_by=by,
        approved_by=None,
        approved_at=None,
    )
    ids = _ids(row)
    log.info("search_profile.proposed", mission_id=str(mission.id), **ids)
    flow.g.add_event(
        "search_profile.proposed",
        severity="info",
        mission_id=str(mission.id),
        vehicle_id=mission.vehicle_id,
        payload=ids,
    )
    flow.publish(
        TOPIC, "search_profile.proposed", ids, vehicle_id=mission.vehicle_id, mission_id=mission_id
    )
    return str(row["row_id"])


async def approve(
    flow: MissionFlow, mission_id: str, approval: SearchProfileApproval, by: str
) -> SearchProfile:
    """A pilot's approval: the profile as sent becomes the next version. 409 when the
    mission is closed, or when a version or proposal newer than the one the card showed
    has arrived (the pilot reviews that first). Only the panel route for ``pilot`` calls
    this."""
    mission = await flow.open_mission_of(mission_id)
    if not by.startswith("operator:"):
        raise FlowError(403, "FORBIDDEN_ROLE", "Bara piloten kan godkänna sökprofilen.")
    current, proposal = await flow.repo.search_profile_rows(str(mission.id))
    version = 0 if current is None else int(current["profile_version"])
    pending = None if proposal is None else str(proposal["row_id"])
    shown = None if approval.proposal_id is None else str(approval.proposal_id)
    if approval.base_version != version or pending not in (None, shown):
        raise FlowError(
            409,
            "PROFILE_STALE",
            "Sökprofilen har ändrats sedan du började. Granska den nya versionen först.",
        )
    row = await flow.repo.add_search_profile(
        str(mission.id),
        approval.profile.model_dump(mode="json"),
        proposed_by=by,
        approved_by=by,
        approved_at=flow.now(),
    )
    profile = _profile(row)
    ids = _ids(row)
    log.info("search_profile.approved", mission_id=str(mission.id), **ids)
    flow.g.add_event(
        "search_profile.approved",
        severity="info",
        mission_id=str(mission.id),
        vehicle_id=mission.vehicle_id,
        payload=ids | {"approved_by": by},
    )
    flow.publish(
        TOPIC,
        "search_profile.approved",
        ids | {"approved_by": by},
        vehicle_id=mission.vehicle_id,
        mission_id=mission_id,
    )
    return profile


async def state(flow: MissionFlow, mission_id: str) -> SearchProfileState:
    """The version in force and a newer proposal, as the panel's card shows them."""
    await flow.mission(mission_id)
    current, proposal = await flow.repo.search_profile_rows(mission_id)
    return SearchProfileState(
        approved=None if current is None else _profile(current),
        proposal=None if proposal is None else _profile(proposal),
        proposal_id=None if proposal is None else proposal["row_id"],
        proposed_by=None if proposal is None else proposal["proposed_by"],
    )


async def approved(flow: MissionFlow, mission_id: str) -> SearchProfile | None:
    """The approved version in force, or ``None``: never a proposal (E10)."""
    await flow.mission(mission_id)
    current, _ = await flow.repo.search_profile_rows(mission_id)
    if current is None:
        return None
    profile = _profile(current)
    return profile if profile.approved else None


async def versions(flow: MissionFlow, mission_id: str) -> list[SearchProfile]:
    await flow.mission(mission_id)
    return [_profile(r) for r in await flow.repo.search_profile_versions(mission_id)]
