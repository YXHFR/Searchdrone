"""Cards the pilot answers: charter proposals, orders that need a click, and Resume.

* **Charter** (``propose`` -> ``decide``): a proposal (from the pilot or the AI) is
  checked and stored as ``proposed`` with a card without a time limit; a newer proposal
  makes older ones ``superseded``. Only a pilot's click approves it, choosing the
  autonomy mode (``charters.resolve_autonomy``). The check runs again at the click. Then
  the fence (``install``) and ``CharterInstall``; ``READY`` once both are confirmed.
* **Order** (``commands`` creates the card on ``PENDING_APPROVAL``): bound to exactly one
  order and expiring after ``APPROVAL_TIMEOUT_S``. At the click the gate runs again
  (phase ``approve``); ``subject_version`` must match (else 409 ``SUPERSEDED``).
* STOP, FLY HOME, the battery guard and a takeover make every waiting order card
  ``superseded``. At start every waiting order card is ``expired`` (``SERVER_RESTART``).

The AI has no route here: it can propose a charter, never approve one.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import TYPE_CHECKING, Any

from sd_ground.logging import get_logger
from sd_ground.mav import fence as mav_fence
from sd_ground.services import charters as charter_rules
from sd_ground.services import fence as fence_rules
from sd_ground.services.missions import FlowError
from sd_mav.mission import MissionError
from sd_mav.vehicle import ControlNotAllowedError, ForbiddenMavlinkError, Vehicle
from sd_safety.constants import CLEARANCE_VALID_S
from sd_schemas.api import Approval, ApprovalDecisionRequest, Charter, Mission
from sd_schemas.base import new_id
from sd_schemas.charter import AutonomyMode, CharterProposal, CharterValidationError, MissionCharter
from sd_schemas.link import CharterAck, CharterInstall
from sd_schemas.mission import MissionState, Trigger

if TYPE_CHECKING:
    from sd_ground.services.missions import MissionFlow

__all__ = [
    "decide",
    "expire",
    "install",
    "new_version_with_mode",
    "on_charter_ack",
    "pending",
    "propose",
    "send_install",
]

log = get_logger("sd_ground.approvals")
_INSTALL_TTL_MS = 30_000


async def pending(flow: MissionFlow, mission_id: str) -> list[Approval]:
    return await flow.repo.approvals(mission_id, "pending")


def _publish_approval(flow: MissionFlow, approval: Approval, type_: str) -> None:
    flow.publish(
        "mission",
        type_,
        approval.model_dump(mode="json"),
        vehicle_id=approval.vehicle_id,
        mission_id=str(approval.mission_id),
    )


def _publish_charter(flow: MissionFlow, charter: Charter, vehicle_id: str) -> None:
    flow.publish(
        "mission",
        "charter.updated",
        charter.model_dump(mode="json"),
        vehicle_id=vehicle_id,
        mission_id=str(charter.mission_id),
    )


# --- charter proposals -----------------------------------------------------------------


async def _area_of(
    flow: MissionFlow, mission: Mission, proposal: CharterProposal
) -> dict[str, Any]:
    if proposal.area is not None:
        return proposal.area
    ann = await flow.repo.annotation(str(proposal.area_annotation_id))
    if (
        ann is None
        or ann["deleted_at"] is not None
        or str(ann["mission_id"]) != str(mission.id)
        or ann["kind"] != "area"
    ):
        raise charter_rules.CharterInvalidError(
            [charter_rules.error("AREA_INVALID", field="area_annotation_id")]
        )
    return ann["geometry"]


def _home(flow: MissionFlow, mission: Mission) -> Any:
    snap = flow.mav_snapshot(mission.vehicle_id)
    home = None if snap is None else snap.telemetry.home
    if home is None:
        raise FlowError(409, "HOME_UNKNOWN", "Drönarens hemplats är inte känd än.")
    return home


async def propose(
    flow: MissionFlow,
    mission_id: str,
    proposal: CharterProposal,
    *,
    by: str,
) -> tuple[Charter, Approval]:
    """Check and store a proposal with its card. ``CharterInvalidError`` (422) when it
    cannot be offered; nothing is shown then."""
    mission = await flow.open_mission_of(mission_id)
    if mission.status is MissionState.LANDED:
        raise FlowError(409, "MISSION_STATE_INVALID", "Uppdraget har landat. Öppna ett nytt.")
    active = await flow.active_charter(mission)
    home = active.home if active is not None else _home(flow, mission)
    area = await _area_of(flow, mission, proposal)
    version = await flow.repo.next_version(mission_id)
    mode = active.autonomy_mode if active is not None else AutonomyMode.WITHIN_CHARTER
    charter = charter_rules.build(
        proposal,
        area=area,
        charter_id=str(new_id()),
        mission_id=mission_id,
        vehicle_id=mission.vehicle_id,
        version=version,
        home=home,
        limits=mission.limits,
        registry=flow.registry.values(),
        autonomy_mode=mode,
    )
    errors = _check(flow, charter, mission, active, proposal.base_version)
    if errors:
        raise charter_rules.CharterInvalidError(errors)
    now = flow.now()
    approval: dict[str, Any] = {
        "id": str(new_id()),
        "mission_id": mission_id,
        "vehicle_id": mission.vehicle_id,
        "kind": "charter",
        "subject_id": str(charter.charter_id),
        "subject_version": charter.version,
        "summary_sv": _summary(charter),
        "required_confirmations": [],
        "default_autonomy_mode": str(charter_rules.resolve_autonomy(active, None)),
        "requested_by": by,
        "requested_at": now,
    }
    superseded = await flow.repo.add_proposal(charter, proposed_by=by, approval=approval, now=now)
    for old in superseded:
        row = await flow.repo.charter(old)
        if row is not None:
            _publish_charter(flow, row, mission.vehicle_id)
    stored = await flow.repo.charter(str(charter.charter_id))
    card = await flow.repo.approval(approval["id"])
    if stored is None or card is None:
        raise RuntimeError("row vanished after write")
    _publish_charter(flow, stored, mission.vehicle_id)
    _publish_approval(flow, card, "approval.requested")
    await flow.transition(mission_id, Trigger.CHARTER_PROPOSED)
    return stored, card


def _check(
    flow: MissionFlow,
    charter: MissionCharter,
    mission: Mission,
    active: MissionCharter | None,
    base_version: int | None,
) -> list[CharterValidationError]:
    return charter_rules.validate(
        charter,
        limits=mission.limits,
        registry=flow.registry,
        max_area_km2=flow.g.config.charter.max_area_km2,
        active=active,
        base_version=base_version,
        airborne=flow.in_flight(mission.vehicle_id),
    )


def _summary(c: MissionCharter) -> str:
    return (
        f"Uppdragsram v{c.version}: {len(c.sectors)} sektor(er), höjd "
        f"{c.alt_min_agl_m:g}-{c.alt_max_agl_m:g} m, högst {c.max_speed_ms:g} m/s, "
        f"hem vid {c.battery_rtl_pct:g} %"
    )


# --- decisions -------------------------------------------------------------------------


async def decide(
    flow: MissionFlow, approval_id: str, req: ApprovalDecisionRequest, *, by: str, origin: str
) -> Approval:
    """The pilot's answer on a card (approve or reject)."""
    from sd_ground.services import commands as commands_service

    card = await flow.repo.approval(approval_id)
    if card is None:
        raise FlowError(404, "APPROVAL_NOT_FOUND", "Kortet finns inte.")
    if card.status != "pending":
        raise FlowError(409, "SUPERSEDED", "Kortet gäller inte längre.")
    if req.subject_version != card.subject_version:
        raise FlowError(409, "SUPERSEDED", "Kortet gäller en annan version.")
    mission = await flow.open_mission_of(str(card.mission_id))
    if card.kind == "charter":
        return await _decide_charter(flow, mission, card, req, by=by)
    if card.kind == "command":
        return await commands_service.decide(flow, mission, card, req, by=by, origin=origin)
    raise FlowError(409, "SUPERSEDED", "Kortet kan inte besvaras.")


async def _decide_charter(
    flow: MissionFlow, mission: Mission, card: Approval, req: ApprovalDecisionRequest, *, by: str
) -> Approval:
    row = await flow.repo.charter(str(card.subject_id))
    if row is None or row.status != "proposed":
        raise FlowError(409, "SUPERSEDED", "Förslaget gäller inte längre.")
    now = flow.now()
    if req.decision == "reject":
        await flow.repo.reject_charter(str(row.id))
        await flow.repo.decide_approval(
            str(card.id), status="rejected", by=by, now=now, reason=req.reason or "APPROVAL_DENIED"
        )
        if await flow.repo.waiting_proposals(str(mission.id)) == 0:
            await flow.transition(str(mission.id), Trigger.PROPOSALS_CLEARED)
        return await _after(flow, card, row, mission)
    active_row = await flow.active_charter_row(mission)
    active = None if active_row is None else active_row.body
    mode = charter_rules.resolve_autonomy(active, req.autonomy_mode)
    charter = row.body.model_copy(update={"autonomy_mode": mode})
    # A waiting proposal is always against the active version: a newer approval (also
    # an autonomy change) supersedes every waiting proposal.
    errors = _check(flow, charter, mission, active, None if active is None else active.version)
    if errors:
        raise FlowError(
            409,
            "CHARTER_INVALID",
            "Förslaget håller inte längre: " + ", ".join(e.code for e in errors),
            [e.model_dump() for e in errors],
        )
    keep = None
    if flow.in_flight(mission.vehicle_id) and active_row is not None:
        keep = str(active_row.id)  # a smaller area in the air keeps the verified fence
    await flow.repo.approve_charter(
        charter=charter,
        approval_id=str(card.id),
        by=by,
        now=now,
        confirmations=list(req.confirmations),
        sectors=charter.sectors,
        keep_fence_from=keep,
    )
    vid = mission.vehicle_id
    await flow.end_clearance(vid, "invalidated", "NEW_CHARTER_VERSION", str(mission.id))
    if active_row is not None:
        old = await flow.repo.charter(str(active_row.id))
        if old is not None:
            _publish_charter(flow, old, vid)
    result = await _after(flow, card, row, mission)
    flow.spawn(install(flow, mission, str(row.id)), "install")
    return result


async def _after(flow: MissionFlow, card: Approval, row: Charter, mission: Mission) -> Approval:
    updated = await flow.repo.approval(str(card.id))
    charter = await flow.repo.charter(str(row.id))
    if updated is None or charter is None:
        raise RuntimeError("row vanished after write")
    _publish_approval(flow, updated, "approval.updated")
    _publish_charter(flow, charter, mission.vehicle_id)
    fresh = await flow.mission(str(mission.id))
    flow.publish(
        "mission",
        "mission.updated",
        fresh.model_dump(mode="json"),
        vehicle_id=fresh.vehicle_id,
        mission_id=str(fresh.id),
    )
    return updated


async def new_version_with_mode(
    flow: MissionFlow, mission: Mission, row: Charter, mode: AutonomyMode, by: str
) -> None:
    """``PUT /missions/{id}/autonomy``: the same charter as a new version with the new
    mode, approved by the pilot in this call. The fence carries over."""
    now = flow.now()
    version = await flow.repo.next_version(str(mission.id))
    charter = row.body.model_copy(
        update={
            "charter_id": new_id(),
            "version": version,
            "autonomy_mode": mode,
            "fence_hash": None,
            "approved_by": None,
            "approved_at": None,
        }
    )
    card: dict[str, Any] = {
        "id": str(new_id()),
        "mission_id": str(mission.id),
        "vehicle_id": mission.vehicle_id,
        "kind": "charter",
        "subject_id": str(charter.charter_id),
        "subject_version": version,
        "summary_sv": "Byte av autonomiläge till "
        + ("B" if mode is AutonomyMode.WITHIN_CHARTER else "A"),
        "required_confirmations": [],
        "default_autonomy_mode": str(mode),
        "requested_by": by,
        "requested_at": now,
    }
    await flow.repo.add_proposal(charter, proposed_by=by, approval=card, now=now)
    await flow.repo.approve_charter(
        charter=charter,
        approval_id=card["id"],
        by=by,
        now=now,
        confirmations=[],
        sectors=charter.sectors,
        keep_fence_from=str(row.id),
    )
    await flow.end_clearance(mission.vehicle_id, "invalidated", "AUTONOMY_CHANGED", str(mission.id))
    flow.g.add_event(
        "mission.autonomy_changed",
        mission_id=str(mission.id),
        vehicle_id=mission.vehicle_id,
        payload={"by": by, "mode": str(mode), "charter_version": version},
    )
    stored = await flow.repo.charter(str(charter.charter_id))
    if stored is not None:
        _publish_charter(flow, stored, mission.vehicle_id)
        await send_install(flow, mission, stored)


# --- the fence and the install ---------------------------------------------------------


async def install(flow: MissionFlow, mission: Mission, charter_id: str) -> None:
    row = await flow.repo.charter(charter_id)
    if row is None or row.status != "approved":
        return
    vid, mid = mission.vehicle_id, str(mission.id)
    if row.fence_verified_at is None:
        verified = await _fence(flow, mission, row)
        row = await flow.repo.charter(charter_id)
        if row is None or not verified:
            return
    await send_install(flow, mission, row)
    if row.install_status == "installed":
        await _maybe_ready(flow, mid)
    log.info("charter.fence_ok", charter_id=charter_id, vehicle_id=vid)


async def _fence(flow: MissionFlow, mission: Mission, row: Charter) -> bool:
    """Move the fence's limits if needed, upload the polygon, read everything back."""
    vid, mid = mission.vehicle_id, str(mission.id)

    def failed(detail: str, code: str = "FENCE_VERIFY_FAILED", **extra: Any) -> bool:
        flow.alert(
            code,
            "critical",
            "Staketet i flygkontrollern kunde inte kontrolleras. Drönaren kan inte lyfta.",
            vehicle_id=vid,
            mission_id=mid,
            details={"detail": detail, "charter_id": str(row.id), **extra},
        )
        return False

    if flow.mav is None:
        return failed("no_mavlink")
    if not flow.on_ground_disarmed(vid):
        return failed("not_on_ground")
    try:
        corners = fence_rules.fence_polygon(row.body)
    except fence_rules.FenceSimplifyError:
        return failed("simplify")
    targets = fence_rules.fence_param_targets(mission.limits)

    async def work(vehicle: Vehicle) -> tuple[str, dict[str, float | None]]:
        current = await mav_fence.read_params(vehicle, targets)
        move = {k: v for k, v in targets.items() if current.get(k) != v}
        if move:
            await mav_fence.set_fence_limits(vehicle, move)
        digest = await mav_fence.upload_fence(vehicle, corners)
        return digest, await mav_fence.read_params(vehicle, fence_rules.SAFETY_PARAMS)

    try:
        digest, params = await flow.mav.call(work, 60.0)
    except (MissionError, ForbiddenMavlinkError, ControlNotAllowedError, TimeoutError) as exc:
        return failed(type(exc).__name__)
    violations = fence_rules.check_fc_safety_params(params, mission.limits)
    if violations:
        await flow.repo.set_fence(str(row.id), digest, None)
        flow.g.add_event(
            "fc.safety_params",
            severity="critical",
            mission_id=mid,
            vehicle_id=vid,
            payload={"violations": [dataclasses.asdict(v) for v in violations]},
        )
        return failed(
            "params",
            code="FC_SAFETY_PARAMS",
            violations=[
                {"param": v.param, "value": v.value, "expected": v.expected} for v in violations
            ],
        )
    await flow.repo.set_fence(str(row.id), digest, flow.now())
    flow.g.add_event(
        "charter.fence_verified",
        mission_id=mid,
        vehicle_id=vid,
        payload={"charter_id": str(row.id), "fence_hash": digest, "corners": len(corners)},
    )
    return True


async def send_install(flow: MissionFlow, mission: Mission, row: Charter) -> None:
    """``CharterInstall`` once the fence hash is set (also after every reconnect)."""
    if row.fence_hash is None or flow.link is None:
        return
    charter = row.body.model_copy(update={"fence_hash": row.fence_hash})
    await flow.repo.set_install_status(str(row.id), "pending")
    sent = await flow.link.send_message(
        mission.vehicle_id,
        CharterInstall(charter=charter),
        mission_id=str(mission.id),
        ttl_ms=_INSTALL_TTL_MS,
    )
    if sent is None:
        await flow.repo.set_install_status(str(row.id), "failed")
        flow.alert(
            "CHARTER_INSTALL_FAILED",
            "warning",
            "Uppdragsramen kunde inte skickas till drönaren. Den skickas igen när länken är uppe.",
            vehicle_id=mission.vehicle_id,
            mission_id=str(mission.id),
        )
    stored = await flow.repo.charter(str(row.id))
    if stored is not None:
        _publish_charter(flow, stored, mission.vehicle_id)


async def on_charter_ack(flow: MissionFlow, vehicle_id: str, ack: CharterAck) -> None:
    row = await flow.repo.charter(str(ack.charter_id))
    if row is None or row.version != ack.charter_version:
        return
    mission = await flow.repo.mission(str(row.mission_id))
    if mission is None:
        return
    await flow.repo.set_install_status(str(row.id), "installed" if ack.accepted else "failed")
    flow.g.add_event(
        "charter.installed" if ack.accepted else "charter.install_failed",
        severity="info" if ack.accepted else "critical",
        mission_id=str(mission.id),
        vehicle_id=vehicle_id,
        payload={
            "charter_id": str(row.id),
            "version": row.version,
            "reason_codes": ack.reason_codes,
        },
    )
    if not ack.accepted:
        flow.alert(
            "CHARTER_INSTALL_FAILED",
            "critical",
            "Drönaren tog inte emot uppdragsramen.",
            vehicle_id=vehicle_id,
            mission_id=str(mission.id),
            details={"reason_codes": ack.reason_codes},
        )
    stored = await flow.repo.charter(str(row.id))
    if stored is not None:
        _publish_charter(flow, stored, vehicle_id)
    if ack.accepted and str(mission.active_charter_id) == str(row.id):
        await _maybe_ready(flow, str(mission.id))


async def _maybe_ready(flow: MissionFlow, mission_id: str) -> None:
    mission = await flow.repo.mission(mission_id)
    if mission is None:
        return
    row = await flow.active_charter_row(mission)
    if row is not None and row.install_status == "installed" and row.fence_verified_at is not None:
        await flow.transition(mission_id, Trigger.CHARTER_READY)


# --- expiry ----------------------------------------------------------------------------


async def expire(flow: MissionFlow) -> None:
    from sd_ground.services import commands as commands_service

    now = flow.now()
    for card in await flow.repo.expired_approvals(now):
        if not await flow.repo.decide_approval(
            str(card.id), status="expired", by=None, now=now, reason="APPROVAL_TIMEOUT"
        ):
            continue
        if card.kind == "command":
            await commands_service.set_status(
                flow, str(card.subject_id), "approval_expired", "APPROVAL_TIMEOUT"
            )
        flow.alert(
            "APPROVAL_EXPIRED",
            "warning",
            "Ett kort gick ut utan svar.",
            vehicle_id=card.vehicle_id,
            mission_id=str(card.mission_id),
            details={"approval_id": str(card.id)},
        )
        updated = await flow.repo.approval(str(card.id))
        if updated is not None:
            _publish_approval(flow, updated, "approval.updated")


async def supersede_command_cards(flow: MissionFlow, vehicle_id: str, reason: str) -> None:
    from sd_ground.services import commands as commands_service

    now = flow.now()
    for card in await flow.repo.pending_command_approvals(vehicle_id):
        if await flow.repo.decide_approval(
            str(card.id), status="superseded", by=None, now=now, reason=reason
        ):
            await commands_service.set_status(flow, str(card.subject_id), "superseded", reason)
            updated = await flow.repo.approval(str(card.id))
            if updated is not None:
                _publish_approval(flow, updated, "approval.updated")


def valid_until(now: dt.datetime) -> dt.datetime:
    return now + dt.timedelta(seconds=CLEARANCE_VALID_S)
