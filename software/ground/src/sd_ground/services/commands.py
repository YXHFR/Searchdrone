"""Orders: through the gate, onto SD-link, and their life until done.

``submit`` is the only way an order (from the panel's order buttons or the AI) reaches
the vehicle. It builds the ground's ``GateContext``, runs ``sd_safety.evaluate`` and
stores the order with its decision (``commands``, ``gate_decisions``) in one
transaction. Then:

* ``REJECT`` -> ``rejected_by_gate``;
* ``PENDING_APPROVAL`` -> a card bound to exactly this order, expiring after
  ``APPROVAL_TIMEOUT_S``; the click runs the gate again (``decide``) and only ``ALLOW``
  sends the order, with the card's ``command_hash`` that the onboard computer checks;
* ``ALLOW`` -> skills that run on the ground (``report``, ``mark_location``) are carried
  out here; the rest go to the onboard computer as ``SkillCommand`` (a ``survey_sector``
  by ``sector_id`` gets the sector's polygon first). A take-off consumes the clearance.

``propose_charter_change`` never reaches the gate: it is checked as a charter proposal
and always ends in a card on the charter (never an order card).

The onboard computer's ``Ack`` moves the order (``received`` -> ``accepted`` ->
``running`` -> ``done``/``failed``, or ``rejected_onboard``). A take-off that is accepted
makes the mission ``ACTIVE``; the AI's ``return_home`` accepted makes it ``RETURNING``.
STOP, FLY HOME and the battery guard cancel every open order after the mode change.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, Final, Literal
from uuid import UUID

from sd_ground.logging import get_logger
from sd_ground.services import annotations as annotations_service
from sd_ground.services import charters as charter_rules
from sd_ground.services.missions import FlowError
from sd_safety import evaluate
from sd_safety.constants import APPROVAL_TIMEOUT_S
from sd_schemas.api import Approval, ApprovalDecisionRequest, Command, Mission
from sd_schemas.approval import command_hash
from sd_schemas.base import new_id
from sd_schemas.charter import CharterProposal
from sd_schemas.commander import EstimateResult, SkillSubmitResult
from sd_schemas.gate import GateDecision, GateRequest, StateRef
from sd_schemas.link import Ack, ApprovalRef, CommandCancel, SkillCommand, StateSnapshot
from sd_schemas.mission import Trigger
from sd_schemas.reason_codes import GATE_REASON_CODES

if TYPE_CHECKING:
    from sd_ground.services.missions import MissionFlow
    from sd_link.session import Expired

__all__ = [
    "Submitted",
    "active_command",
    "cancel",
    "cancel_open",
    "decide",
    "estimate",
    "on_ack",
    "on_expired",
    "on_snapshot",
    "set_status",
    "submit",
]

log = get_logger("sd_ground.commands")
IssuerKind = Literal["operator", "brain"]
_RANK: Final = {
    "dispatched": 0,
    "received": 1,
    "accepted": 2,
    "running": 3,
}
_TERMINAL: Final = frozenset(
    {
        "rejected_by_gate",
        "rejected_by_operator",
        "approval_expired",
        "rejected_onboard",
        "done",
        "failed",
        "superseded",
        "expired",
        "cancelled",
    }
)
_ACK_STATUS: Final = {
    "received": "received",
    "accepted": "accepted",
    "running": "running",
    "done": "done",
    "failed": "failed",
    "rejected": "rejected_onboard",
    "superseded": "superseded",
    "expired": "expired",
}
_GROUND_SKILLS: Final = frozenset({"report", "mark_location"})
BRAIN_ISSUER: Final = "brain:ground"


class Submitted:
    """What ``submit`` gives back to the routes."""

    def __init__(self, command: Command, gate: GateDecision, *, created: bool) -> None:
        self.command = command
        self.gate = gate
        self.created = created

    def result(self) -> SkillSubmitResult:
        return SkillSubmitResult(
            command_id=self.command.id,
            decision=self.gate.decision,
            reason_codes=list(self.gate.reason_codes),
            message_sv=self.gate.message_sv,
            details=self.gate.details,
            approval_id=self.command.approval_id,
            status=self.command.status,
        )


# --- publishing ------------------------------------------------------------------------


def _publish(flow: MissionFlow, cmd: Command, gate: GateDecision | None = None) -> None:
    data = cmd.model_dump(mode="json")
    if gate is not None:
        data |= {
            "reason_codes": list(gate.reason_codes),
            "message_sv": gate.message_sv,
            "details": gate.details,
        }
    elif cmd.status_reason is not None:
        data |= {
            "reason_codes": [cmd.status_reason],
            "message_sv": cmd.status_reason,
            "details": {},
        }
    flow.publish(
        "mission",
        "command.updated",
        data,
        vehicle_id=cmd.vehicle_id,
        mission_id=str(cmd.mission_id),
    )


async def set_status(
    flow: MissionFlow,
    command_id: str,
    status: str,
    reason: str | None,
    *,
    only_from: tuple[str, ...] | None = None,
    **extra: Any,
) -> Command | None:
    changed = await flow.repo.set_command_status(
        command_id, status, reason=reason, now=flow.now(), only_from=only_from, **extra
    )
    if changed is None:
        return None
    cmd = await flow.repo.command(command_id)
    if cmd is not None:
        _publish(flow, cmd)
    return cmd


# --- submit ----------------------------------------------------------------------------


def _gate_request(
    mission: Mission,
    *,
    command_id: str,
    skill: str,
    params: dict[str, Any],
    issuer_kind: IssuerKind,
    issuer_ref: str,
    charter_version: int | None,
    based_on: StateRef | None,
    phase: Literal["submit", "approve", "estimate"],
    confirmations: list[str] | None = None,
) -> GateRequest:
    return GateRequest(
        command_id=UUID(command_id),
        mission_id=mission.id,
        vehicle_id=mission.vehicle_id,
        skill=skill,
        params=params,
        issuer_kind=issuer_kind,
        issuer_ref=issuer_ref,
        charter_version=charter_version,
        based_on=based_on,
        phase=phase,
        approval_confirmations=confirmations or [],
    )


def _summary(skill: str, params: dict[str, Any]) -> str:
    alt = params.get("alt_m")
    speed = params.get("speed_ms")
    parts = {
        "takeoff": "Lyft",
        "goto": "Flyg till punkten",
        "survey_sector": f"Sök av sektor {params.get('sector_id')}"
        if params.get("sector_id")
        else "Sök av området",
        "hold": "Stå still",
        "resume": "Fortsätt",
        "return_home": "Flyg hem",
        "report": "Rapportera",
        "mark_location": "Markera på kartan",
    }
    text = parts.get(skill, skill)
    if alt is not None:
        text += f" på {float(alt):g} m"
    if speed is not None:
        text += f", {float(speed):g} m/s"
    return text


async def submit(
    flow: MissionFlow,
    mission_id: str,
    *,
    command_id: str,
    skill: str,
    params: dict[str, Any],
    issuer_kind: IssuerKind,
    issuer_ref: str,
    rationale_sv: str | None,
    in_reply_to: str | None = None,
    based_on: StateRef | None = None,
    charter_version: int | None = None,
    backend: str | None = None,
    ai_config_hash: str | None = None,
) -> Submitted:
    """Check one order with the gate and act on the answer (idempotent on ``command_id``)."""
    old = await flow.repo.command(command_id)
    if old is not None:
        if str(old.mission_id) != mission_id or old.skill != skill:
            raise FlowError(409, "ID_CONFLICT", "Orderns id används redan.")
        return Submitted(old, await _last_gate(flow, old), created=False)
    async with flow._lock:  # one order at a time: the gate's picture must not race
        mission = await flow.open_mission_of(mission_id)
        if skill == "propose_charter_change":
            return await _propose(
                flow,
                mission,
                command_id,
                params,
                issuer_kind,
                issuer_ref,
                rationale_sv,
                in_reply_to,
                backend,
                ai_config_hash,
            )
        row = await flow.active_charter_row(mission)
        if issuer_kind == "operator":
            charter_version = None if row is None else row.version
            based_on = None
        ctx = await flow.gate_context(mission)
        req = _gate_request(
            mission,
            command_id=command_id,
            skill=skill,
            params=params,
            issuer_kind=issuer_kind,
            issuer_ref=issuer_ref,
            charter_version=charter_version,
            based_on=based_on,
            phase="submit",
        )
        gate = evaluate(req, ctx)
        now = flow.now()
        status = {"REJECT": "rejected_by_gate", "PENDING_APPROVAL": "pending_approval"}.get(
            gate.decision, "done" if skill in _GROUND_SKILLS else "dispatched"
        )
        spec = flow.registry.get(skill)
        cmd = {
            "id": command_id,
            "mission_id": mission_id,
            "vehicle_id": mission.vehicle_id,
            "charter_id": None if row is None else str(row.id),
            "charter_version": charter_version,
            "issuer_kind": issuer_kind,
            "issuer_ref": issuer_ref,
            "backend": backend,
            "ai_config_hash": ai_config_hash,
            "skill": skill,
            "params": params,
            "rationale_sv": rationale_sv,
            "in_reply_to": in_reply_to,
            "epoch": None if based_on is None else based_on.state_epoch,
            "seq": None if based_on is None else based_on.state_seq,
            "status": status,
            "status_reason": None if gate.decision != "REJECT" else gate.reason_codes[0],
            "command_hash": None,
            "expires_at": now + dt.timedelta(seconds=spec.ttl_s if spec else 15),
            "target": _target(params),
        }
        approval = None
        if gate.decision == "PENDING_APPROVAL":
            approval = {
                "id": str(new_id()),
                "mission_id": mission_id,
                "vehicle_id": mission.vehicle_id,
                "kind": "command",
                "subject_id": command_id,
                "subject_version": 1,
                "summary_sv": _summary(skill, params),
                "required_confirmations": list(gate.required_confirmations),
                "gate": gate.model_dump(mode="json"),
                "requested_by": issuer_ref,
                "requested_at": now,
                "expires_at": now + dt.timedelta(seconds=APPROVAL_TIMEOUT_S),
            }
        consume = None
        active_clearance = flow.clearances.get(mission.vehicle_id)
        if gate.decision == "ALLOW" and skill == "takeoff" and active_clearance is not None:
            consume = active_clearance.clearance_id
        await flow.repo.add_command(
            cmd,
            gate,
            phase="submit",
            state_seq=mission.state_seq,
            now=now,
            approval=approval,
            consume_clearance=consume,
        )
        if issuer_kind == "brain" and gate.decision != "REJECT" and spec is not None:
            flow.rates.setdefault(mission.vehicle_id, _rate()).note(spec.rate_class, flow.now_ms())
    if consume is not None:
        flow.clearances.consume(mission.vehicle_id)
        flow.publish(
            "mission",
            "clearance.updated",
            {"status": "consumed", "confirmed_by": None, "valid_until": None},
            vehicle_id=mission.vehicle_id,
            mission_id=mission_id,
        )
    stored = await flow.repo.command(command_id)
    if stored is None:
        raise RuntimeError("row vanished after write")
    _publish(flow, stored, gate)
    if approval is not None:
        card = await flow.repo.approval(approval["id"])
        if card is not None:
            flow.publish(
                "mission",
                "approval.requested",
                card.model_dump(mode="json"),
                vehicle_id=mission.vehicle_id,
                mission_id=mission_id,
            )
    if gate.decision == "ALLOW":
        if skill in _GROUND_SKILLS:
            await _run_on_ground(flow, mission, stored)
        else:
            await _dispatch(flow, mission, stored, approval=None)
        stored = await flow.repo.command(command_id) or stored
    return Submitted(stored, gate, created=True)


def _rate() -> Any:
    from sd_ground.services.gate_context import BrainRate

    return BrainRate()


def _target(params: dict[str, Any]) -> dict[str, Any] | None:
    lat, lon = params.get("lat"), params.get("lon")
    if isinstance(lat, int | float) and isinstance(lon, int | float):
        return {"type": "Point", "coordinates": [float(lon), float(lat)]}
    poly = params.get("polygon")
    if isinstance(poly, dict) and poly.get("type") == "Polygon":
        return poly
    return None


async def _last_gate(flow: MissionFlow, cmd: Command) -> GateDecision:
    """The decision stored for a repeated ``command_id`` (same answer as the first time)."""
    gate = await flow.repo.first_gate(str(cmd.id))
    if gate is not None:
        return gate
    return GateDecision(
        decision="REJECT", reason_codes=[], rule_ids=[], message_sv="", details={}, side="ground"
    )


async def _propose(
    flow: MissionFlow,
    mission: Mission,
    command_id: str,
    params: dict[str, Any],
    issuer_kind: IssuerKind,
    issuer_ref: str,
    rationale_sv: str | None,
    in_reply_to: str | None,
    backend: str | None,
    ai_config_hash: str | None,
) -> Submitted:
    """``propose_charter_change``: a charter proposal, never an order card."""
    from sd_ground.services import approvals as approvals_service

    now = flow.now()
    codes: list[str]
    card_id: UUID | None = None
    try:
        proposal = CharterProposal.model_validate(params)
        _charter, card = await approvals_service.propose(
            flow, str(mission.id), proposal, by=issuer_ref
        )
        codes, status, reason, card_id = (
            ["NEEDS_APPROVAL_SKILL"],
            "done",
            "CHARTER_PROPOSED",
            UUID(str(card.id)),
        )
        decision: Literal["PENDING_APPROVAL", "REJECT"] = "PENDING_APPROVAL"
    except charter_rules.CharterInvalidError as exc:
        codes = [e.code for e in exc.errors]
        decision, status, reason = "REJECT", "rejected_by_gate", "CHARTER_INVALID"
    except ValueError:
        codes, decision, status, reason = (
            ["SCHEMA_INVALID"],
            "REJECT",
            "rejected_by_gate",
            "CHARTER_INVALID",
        )
    except FlowError as exc:
        codes, decision, status, reason = (
            [exc.code],
            "REJECT",
            "rejected_by_gate",
            "CHARTER_INVALID",
        )
    gate = GateDecision(
        decision=decision,
        reason_codes=codes,
        rule_ids=["R-CHARTER-PROPOSAL"],
        message_sv=GATE_REASON_CODES.get(codes[0], "Förslaget kunde inte användas.")
        if decision == "REJECT"
        else "Väntar på operatörens klick på ramkortet.",
        details={},
        side="ground",
    )
    cmd = {
        "id": command_id,
        "mission_id": str(mission.id),
        "vehicle_id": mission.vehicle_id,
        "charter_id": None,
        "charter_version": None,
        "issuer_kind": issuer_kind,
        "issuer_ref": issuer_ref,
        "backend": backend,
        "ai_config_hash": ai_config_hash,
        "skill": "propose_charter_change",
        "params": params,
        "rationale_sv": rationale_sv,
        "in_reply_to": in_reply_to,
        "epoch": None,
        "seq": None,
        "status": status,
        "status_reason": reason,
        "command_hash": None,
        "expires_at": None,
        "target": None,
    }
    await flow.repo.add_command(
        cmd,
        gate,
        phase="submit",
        state_seq=mission.state_seq,
        now=now,
        approval=None,
        consume_clearance=None,
    )
    stored = await flow.repo.command(command_id)
    if stored is None:
        raise RuntimeError("row vanished after write")
    stored = stored.model_copy(update={"approval_id": card_id})
    _publish(flow, stored, gate)
    return Submitted(stored, gate, created=True)


async def estimate(
    flow: MissionFlow,
    mission_id: str,
    *,
    skill: str,
    params: dict[str, Any],
    based_on: StateRef | None,
    charter_version: int | None,
) -> EstimateResult:
    """The gate in phase ``estimate`` (no order is created) plus a rough time and battery."""
    mission = await flow.open_mission_of(mission_id)
    ctx = await flow.gate_context(mission)
    req = _gate_request(
        mission,
        command_id=str(new_id()),
        skill=skill,
        params=params,
        issuer_kind="brain",
        issuer_ref=BRAIN_ISSUER,
        charter_version=charter_version,
        based_on=based_on,
        phase="estimate",
    )
    gate = evaluate(req, ctx)
    duration = None
    v = ctx.vehicle
    lat, lon = params.get("lat"), params.get("lon")
    if v is not None and isinstance(lat, int | float) and isinstance(lon, int | float):
        from sd_safety.targets import distance_m

        speed = float(params.get("speed_ms") or 5.0)
        duration = distance_m(v.lat, v.lon, float(lat), float(lon)) / max(speed, 0.5)
    return EstimateResult(gate=gate, duration_s=duration, battery_after_pct=None)


# --- carrying out ----------------------------------------------------------------------


async def _run_on_ground(flow: MissionFlow, mission: Mission, cmd: Command) -> None:
    """``report`` becomes a chat line from the AI, ``mark_location`` a pin ``M<n>``."""
    from sd_ground.services import chat as chat_service

    p = cmd.params
    if cmd.skill == "report":
        await chat_service.add_line(
            flow,
            mission,
            msg_id=str(new_id()),
            role="brain",
            author_ref=cmd.issuer_ref,
            kind="status",
            text=str(p.get("text_sv", "")),
            in_reply_to=None,
            refs={"command_id": str(cmd.id), "priority": p.get("priority", "info")},
        )
    elif cmd.skill == "mark_location":
        await annotations_service.mark(
            flow,
            mission,
            lat=float(p["lat"]),
            lon=float(p["lon"]),
            note=str(p.get("label", "")),
            by=cmd.issuer_ref,
        )


def _wire_params(flow: MissionFlow, cmd: Command, charter: Any) -> dict[str, Any] | None:
    """The parameters as sent: a ``survey_sector`` by id gets the sector's polygon."""
    params = dict(cmd.params)
    if cmd.skill == "survey_sector" and params.get("sector_id") is not None:
        sector = next((s for s in charter.sectors if s.sector_id == params["sector_id"]), None)
        if sector is None:
            return None
        params["polygon"] = sector.area
        params.pop("sector_id")
    return params


async def _dispatch(
    flow: MissionFlow, mission: Mission, cmd: Command, *, approval: Approval | None
) -> None:
    row = await flow.active_charter_row(mission)
    if row is None or flow.link is None:
        await set_status(flow, str(cmd.id), "expired", "LINK_DOWN")
        return
    params = _wire_params(flow, cmd, row.body)
    if params is None:
        await set_status(flow, str(cmd.id), "rejected_by_gate", "SCHEMA_INVALID")
        return
    spec = flow.registry.get(cmd.skill)
    ref = None
    if approval is not None and spec is not None:
        digest = command_hash(cmd.skill, spec.params_model.model_validate(params), row.version)
        ref = ApprovalRef(approval_id=approval.id, command_hash=digest)
        await flow.repo.set_command_status(
            str(cmd.id),
            "dispatched",
            reason=None,
            now=flow.now(),
            approval_id=str(approval.id),
            command_hash=digest,
            charter_version=row.version,
            charter_id=str(row.id),
        )
    body = SkillCommand(
        command_id=cmd.id,
        skill=cmd.skill,
        params=params,
        issuer=cmd.issuer_ref,
        rationale=cmd.rationale_sv,
        charter_version=row.version,
        based_on=cmd.based_on,
        approval=ref,
    )
    ttl_ms = (spec.ttl_s if spec is not None else 15) * 1000
    sent = await flow.link.send_message(
        mission.vehicle_id, body, mission_id=str(mission.id), ttl_ms=ttl_ms
    )
    if sent is None:
        await set_status(flow, str(cmd.id), "expired", "LINK_DOWN")
    elif approval is not None:
        updated = await flow.repo.command(str(cmd.id))
        if updated is not None:
            _publish(flow, updated)


# --- the click on an order card --------------------------------------------------------


async def decide(
    flow: MissionFlow,
    mission: Mission,
    card: Approval,
    req: ApprovalDecisionRequest,
    *,
    by: str,
    origin: str,
) -> Approval:
    from sd_ground.services import clearance as clearance_service

    cmd = await flow.repo.command(str(card.subject_id))
    if cmd is None or str(cmd.status) != "pending_approval":
        raise FlowError(409, "SUPERSEDED", "Ordern väntar inte längre på svar.")
    now = flow.now()
    if req.decision == "reject":
        await flow.repo.decide_approval(
            str(card.id), status="rejected", by=by, now=now, reason=req.reason or "APPROVAL_DENIED"
        )
        await set_status(flow, str(cmd.id), "rejected_by_operator", "APPROVAL_DENIED")
        return await _card(flow, card)
    confirmations: list[str] = list(req.confirmations)
    if "clear_around_drone" in card.required_confirmations:
        if "clear_around_drone" in confirmations:
            await clearance_service.confirm(flow, mission, by=by, origin=origin)
        else:
            active = flow.clearances.get(mission.vehicle_id)
            row = await flow.active_charter_row(mission)
            if (
                active is None
                or row is None
                or str(active.state.charter_id) != str(row.id)
                or active.state.charter_version != row.version
            ):
                raise FlowError(
                    409, "CLEARANCE_REQUIRED", "Bekräfta att det är fritt runt drönaren först."
                )
            confirmations.append("clear_around_drone")
    async with flow._lock:
        mission = await flow.open_mission_of(str(mission.id))
        row = await flow.active_charter_row(mission)
        ctx = await flow.gate_context(mission)
        gate = evaluate(
            _gate_request(
                mission,
                command_id=str(cmd.id),
                skill=cmd.skill,
                params=cmd.params,
                issuer_kind="brain" if cmd.issuer_kind == "brain" else "operator",
                issuer_ref=cmd.issuer_ref,
                charter_version=cmd.charter_version
                if cmd.issuer_kind == "brain"
                else (None if row is None else row.version),
                based_on=cmd.based_on,
                phase="approve",
                confirmations=confirmations,
            ),
            ctx,
        )
        cmd_row = {
            "id": str(cmd.id),
            "mission_id": str(mission.id),
            "vehicle_id": mission.vehicle_id,
            "charter_version": cmd.charter_version,
        }
        await flow.repo.add_gate_decision(
            cmd_row, gate, phase="approve", state_seq=mission.state_seq, now=now
        )
        if gate.decision != "ALLOW":
            await flow.repo.decide_approval(
                str(card.id), status="rejected", by=by, now=now, reason="GATE_ON_APPROVE"
            )
            await set_status(flow, str(cmd.id), "rejected_by_gate", "GATE_ON_APPROVE")
            return await _card(flow, card)
        await flow.repo.decide_approval(
            str(card.id),
            status="approved",
            by=by,
            now=now,
            reason=None,
            confirmations=confirmations,
        )
        consume = flow.clearances.get(mission.vehicle_id) if cmd.skill == "takeoff" else None
        if consume is not None:
            await flow.repo.consume_clearance(consume.clearance_id, str(cmd.id), now)
            flow.clearances.consume(mission.vehicle_id)
    approved = await flow.repo.approval(str(card.id))
    if approved is None:
        raise RuntimeError("row vanished after write")
    if cmd.skill in _GROUND_SKILLS:
        await set_status(flow, str(cmd.id), "done", None)
        await _run_on_ground(flow, mission, cmd)
    else:
        await _dispatch(flow, mission, cmd, approval=approved)
    return await _card(flow, card)


async def _card(flow: MissionFlow, card: Approval) -> Approval:
    updated = await flow.repo.approval(str(card.id))
    if updated is None:
        raise RuntimeError("row vanished after write")
    flow.publish(
        "mission",
        "approval.updated",
        updated.model_dump(mode="json"),
        vehicle_id=updated.vehicle_id,
        mission_id=str(updated.mission_id),
    )
    return updated


# --- cancelling ------------------------------------------------------------------------


async def cancel(flow: MissionFlow, command_id: str, *, brain_only: bool) -> Command:
    cmd = await flow.repo.command(command_id)
    if cmd is None or (brain_only and cmd.issuer_ref != BRAIN_ISSUER):
        raise FlowError(404, "COMMAND_NOT_FOUND", "Ordern finns inte.")
    status = str(cmd.status)
    if status in _TERMINAL:
        raise FlowError(409, "NOT_CANCELLABLE", "Ordern kan inte avbrytas.")
    if status == "pending_approval":
        if cmd.approval_id is not None:
            await flow.repo.decide_approval(
                str(cmd.approval_id),
                status="superseded",
                by=None,
                now=flow.now(),
                reason="CANCELLED",
            )
        updated = await set_status(flow, command_id, "cancelled", "CANCELLED")
        return updated or cmd
    if flow.link is not None:
        await flow.link.send_message(
            cmd.vehicle_id, CommandCancel(command_id=cmd.id), mission_id=str(cmd.mission_id)
        )
    updated = await set_status(flow, command_id, "cancelled", "CANCELLED")
    return updated or cmd


async def cancel_open(flow: MissionFlow, vehicle_id: str, reason: str) -> None:
    """After STOP / FLY HOME / the battery guard: every open order is cancelled and every
    waiting order card is superseded (the mode change has already been sent)."""
    from sd_ground.services import approvals as approvals_service

    await approvals_service.supersede_command_cards(flow, vehicle_id, reason)
    for cmd in await flow.repo.open_commands(vehicle_id):
        await set_status(flow, str(cmd.id), "cancelled", reason)


# --- what the onboard computer says ----------------------------------------------------


async def on_ack(flow: MissionFlow, vehicle_id: str, ack: Ack) -> None:
    row = await flow.repo.command_row(str(ack.ref_id))
    if row is None or row["vehicle_id"] != vehicle_id:
        return
    status = _ACK_STATUS.get(ack.stage)
    current = str(row["status"])
    if status is None or current in _TERMINAL:
        return
    if status in _RANK and _RANK.get(current, -1) >= _RANK[status]:
        return
    reason = ack.reason_codes[0] if ack.reason_codes else None
    cmd = await set_status(flow, str(ack.ref_id), status, reason)
    if cmd is None:
        return
    mid = str(cmd.mission_id)
    if status == "rejected_onboard" and ack.gate is not None:
        flow.g.add_event(
            "gate.mismatch",
            severity="warning",
            mission_id=mid,
            vehicle_id=vehicle_id,
            payload={"command_id": str(cmd.id), "reason_codes": ack.gate.reason_codes},
        )
        flow.alert(
            "GATE_MISMATCH",
            "warning",
            "Drönaren sa nej till en order som marken godkände.",
            vehicle_id=vehicle_id,
            mission_id=mid,
            details={"command_id": str(cmd.id)},
        )
        await flow.repo.add_gate_decision(
            {
                "id": str(cmd.id),
                "mission_id": mid,
                "vehicle_id": vehicle_id,
                "charter_version": cmd.charter_version,
            },
            ack.gate,
            phase="onboard",
            state_seq=None,
            now=flow.now(),
        )
    if status in ("failed", "rejected_onboard", "expired") and cmd.skill == "takeoff":
        mission = await flow.repo.mission(mid)
        if mission is not None and str(mission.status) == "ACTIVE":
            await flow.takeoff_failed(mission)
    if status == "accepted":
        if cmd.skill == "takeoff":
            await flow.transition(mid, Trigger.TAKEOFF_ACCEPTED)
        elif cmd.skill == "return_home" and cmd.issuer_kind == "brain":
            await flow.transition(mid, Trigger.BRAIN_RTL_ACCEPTED)
    if (
        cmd.skill == "survey_sector"
        and cmd.params.get("sector_id")
        and status in ("running", "done")
    ):
        await flow.repo.set_sector_progress(
            mid,
            str(cmd.params["sector_id"]),
            "done" if status == "done" else "active",
            1.0 if status == "done" else 0.0,
            flow.now(),
        )


async def on_expired(flow: MissionFlow, vehicle_id: str, expired: Expired) -> None:
    env = expired.envelope
    if env.kind != "SkillCommand":
        return
    command_id = str(env.body.get("command_id"))
    await set_status(flow, command_id, "expired", expired.reason, only_from=("dispatched",))


async def on_snapshot(flow: MissionFlow, mission: Mission, snap: StateSnapshot) -> None:
    """A survey's progress moves its sector."""
    if snap.active_command_id is None or snap.progress_pct is None:
        return
    row = await flow.repo.command_row(str(snap.active_command_id))
    if row is None or row["skill"] != "survey_sector":
        return
    sector = row["params"].get("sector_id")
    if sector:
        await flow.repo.set_sector_progress(
            str(mission.id), str(sector), "active", snap.progress_pct / 100.0, flow.now()
        )


async def active_command(flow: MissionFlow, mission: Mission) -> Command | None:
    snap, _ = flow.snapshot(mission.vehicle_id)
    if snap is None or snap.active_command_id is None:
        return None
    return await flow.repo.command(str(snap.active_command_id))
