"""The closer look ("Titta närmare", ``inspect_detection`` v2) as the ground follows it.

A look starts when its order is sent (``started``, from ``commands._dispatch``): the
candidate goes ``inspecting`` and the look gets its start row. While it is open the
ground keeps what the drone reports in memory (``Look``): the question "Gå lägre?"
(``inspect.lower_offer``, answered by the pilot's ``lower``) and the wait for the
decision (``inspect.done``). It ends once, with an end row:

* a person's decision on the candidate (``decided``): ``InspectOutcome`` goes onboard;
* the system's: no decision within ``AWAIT_DECISION_S`` (``timeout``), another order
  replaced it (``resumed_by_pilot``), STOP / FLY HOME / a takeover / the battery / the
  link (``aborted``), the drone refused it (``refused_onboard``) or it failed. The
  candidate then goes back to ``ready`` and stays in the queue.

After a restart the ground does not know where an open look was: it ends as ``timeout``
when even the slowest look would be over (``INSPECT_MAX_S`` + ``LOWER_OFFER_S`` +
``AWAIT_DECISION_S`` after the start).

The gate gets the look's facts here (``gate_facts``): the open look, the candidates a
look may start on, the candidate's own cylinder when the look may fly over it, and the
height floor along the way (from the charter's map pack, when the ground has it).
The permit for overflight in ``live`` (``permit``) is the ground's; a charter approved
afterwards carries it as ``overflight_allowed``. Revoking it counts at once: until the
next charter no look starts (``refuse_revoked``), since the drone's charter still allows
overflight.

During a look only the pilot moves the drone (owner decision): the AI's orders that would
take the drone from it are refused (``refuse_during_look``), and an observer's mark on the
candidate is a note for the pilot that neither ends the look nor reaches the AI.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pydantic import ValidationError

from sd_ground.api.deps import ApiStatusError
from sd_ground.db.inspections import InspectionRepo, LookRow
from sd_ground.logging import get_logger
from sd_safety.floor import floor_along
from sd_safety.floor_pack import FloorPack, FloorPackError, load_floor_pack
from sd_schemas.base import LatLon, new_id
from sd_schemas.gate import GateContext, GateDecision, GateFloor
from sd_schemas.inspect import (
    AWAIT_DECISION_S,
    EVENT_ABORTED,
    EVENT_DONE,
    EVENT_LOWER_OFFER,
    EVENT_PERMIT_CHANGED,
    INSPECT_MAX_S,
    LOWER_OFFER_S,
    InspectAborted,
    InspectDone,
    Inspection,
    InspectLowerOffer,
    InspectProgress,
    Permit,
    PermitChanged,
)
from sd_schemas.link import InspectLower, InspectOutcome
from sd_schemas.reason_codes import GATE_REASON_CODES
from sd_schemas.search import priority_class
from sd_schemas.skills import CANDIDATE_NUMBER_PATTERN

if TYPE_CHECKING:
    from sd_ground.services.missions import MissionFlow
    from sd_schemas.api import Command, Mission

__all__ = [
    "PROGRESS_MIN_MS",
    "Look",
    "candidate_ref",
    "decided",
    "gate_facts",
    "lower",
    "on_event",
    "on_order_end",
    "on_progress",
    "overflight_allowed",
    "overflight_revoked",
    "permit",
    "refuse_during_look",
    "refuse_revoked",
    "replaced",
    "set_permit",
    "started",
    "summary",
    "tick",
    "views",
]

log = get_logger("sd_ground.inspections")

SKILL: Final = "inspect_detection"
PROGRESS_MIN_MS: Final = 500  # candidate.inspect_progress at most 2 Hz
# The slowest look: flying and stepping, the pilot's "Gå lägre?", the wait for a decision.
_LONGEST: Final = dt.timedelta(seconds=INSPECT_MAX_S + LOWER_OFFER_S + AWAIT_DECISION_S)
# An order status that ends the look, and how (``done`` does not: the drone waits).
_ORDER_END: Final = {
    "rejected_onboard": "refused_onboard",
    "failed": "failed",
    "expired": "failed",
    "cancelled": "aborted",
    "superseded": "resumed_by_pilot",
}
# The onboard reason of ``inspect.aborted`` -> the outcome (the rest are "aborted").
_ABORT_OUTCOME: Final = {"refused_onboard": "refused_onboard", "failed": "failed"}
# A person's decision on the candidate during the look -> ``InspectOutcome``.
_DECISION_OUTCOME: Final = {
    "confirm": "confirmed",
    "reject": "rejected",
    "unsure": "unsure",
    "resume": "resume",
}
# Orders that do not replace a look while the drone waits: more pictures of the target.
_KEEPS_LOOK: Final = frozenset({"capture_stills", SKILL})
_SYSTEM_TEXT: Final = {
    "timeout": "Inget beslut om {k}. Sökningen fortsätter och {k} ligger kvar i kön.",
    "resumed_by_pilot": "Titten på {k} avbröts av en ny order. {k} ligger kvar i kön.",
    "aborted": "Titten på {k} avbröts. {k} ligger kvar i kön.",
    "refused_onboard": "Drönaren sa nej till titten på {k}. {k} ligger kvar i kön.",
    "failed": "Titten på {k} misslyckades. {k} ligger kvar i kön.",
}
_K = re.compile(CANDIDATE_NUMBER_PATTERN)


@dataclass
class Look:
    """What the ground knows of an open look beyond its start row."""

    row: LookRow
    lower_offer: InspectLowerOffer | None = None
    lower_offer_until: dt.datetime | None = None
    done: InspectDone | None = None
    await_until: dt.datetime | None = None
    progress_ms: int | None = None

    def view(self) -> Inspection:
        r = self.row
        phase = (
            "ended"
            if r.ended_at is not None
            else "lower_offer"
            if self.lower_offer is not None
            else "await"
            if self.done is not None
            else "flying"
        )
        return Inspection.model_validate(
            {
                "command_id": r.command_id,
                "candidate_id": r.candidate_id,
                "mission_id": r.mission_id,
                "seq": r.seq,
                "started_at": r.started_at,
                "started_by": r.started_by,
                "operation_mode": r.operation_mode,
                "overflight": r.overflight,
                "phase": phase,
                "lower_offer": self.lower_offer,
                "lower_offer_until": self.lower_offer_until,
                "done": self.done,
                "await_until": self.await_until,
                "ended_at": r.ended_at,
                "outcome": r.outcome,
                "detail": r.detail,
            }
        )


def _repo(flow: MissionFlow) -> InspectionRepo:
    return InspectionRepo(flow.repo.engine)


# --- the permit and the operation mode ------------------------------------------------------


def permit(flow: MissionFlow) -> Permit:
    """The overflight permit in force: the config's until the pilot changes it here."""
    if flow.permit is None:
        cfg = flow.g.loaded.config.permit
        return Permit(
            specific_category_ref=cfg.specific_category_ref,
            allow_overflight_live=cfg.allow_overflight_live,
        )
    return flow.permit


def set_permit(flow: MissionFlow, new: Permit, *, by: str) -> Permit:
    """The pilot's change (until the ground restarts; the config is the lasting setting).
    Granting it counts from the next charter approval, revoking it at once
    (``overflight_revoked``); logged as ``permit.changed``."""
    flow.permit = new
    changed = PermitChanged(
        changed_by=by,
        has_specific_category_ref=bool(new.specific_category_ref.strip()),
        allow_overflight_live=new.allow_overflight_live,
    )
    flow.g.add_event(
        EVENT_PERMIT_CHANGED,
        severity="info",
        mission_id=None,
        vehicle_id=None,
        payload=changed.model_dump(mode="json"),
    )
    return new


def overflight_allowed(flow: MissionFlow, mode: str | None) -> bool:
    """Whether a look may fly straight over its candidate: in sim and exercise always, in
    live (also when the mode is unset) only with the permit."""
    if mode in ("sim", "exercise"):
        return True
    p = permit(flow)
    return p.allow_overflight_live and bool(p.specific_category_ref.strip())


def overflight_revoked(flow: MissionFlow, charter: Any) -> bool:
    """The charter in force allows overflight, but the permit no longer does: the pilot
    revoked it (``PUT /api/v1/permit``) or the ground restarted without it. A narrowing
    counts at once, not from the next charter. The drone still holds the charter that
    allows it, so no look may start until a new charter is approved (``refuse_revoked``)."""
    return (
        charter is not None
        and charter.overflight_allowed
        and not overflight_allowed(flow, charter.operation_mode)
    )


def refuse_revoked(flow: MissionFlow, ctx: GateContext, gate: GateDecision) -> GateDecision:
    """The gate's answer for a look, refused (``OVERFLIGHT_NOT_ALLOWED``) when the permit
    behind the charter's overflight was revoked."""
    if not overflight_revoked(flow, ctx.charter):
        return gate
    codes = [
        "OVERFLIGHT_NOT_ALLOWED",
        *(c for c in gate.reason_codes if c != "OVERFLIGHT_NOT_ALLOWED"),
    ]
    rules = ["R-OVERFLIGHT", *(r for r in gate.rule_ids if r != "R-OVERFLIGHT")]
    return gate.model_copy(
        update={
            "decision": "REJECT",
            "reason_codes": codes,
            "rule_ids": rules,
            "message_sv": GATE_REASON_CODES["OVERFLIGHT_NOT_ALLOWED"],
            "details": {**gate.details, "permit_revoked": True},
            "required_confirmations": [],
        }
    )


# The AI's orders that leave a look alone: more pictures of the target, a look (one at a
# time, R-INSPECT-ONE), flying home, and the skills carried out on the ground.
_AI_DURING_LOOK: Final = frozenset({*_KEEPS_LOOK, "return_home", "report", "mark_location"})


PILOT_ONLY_SV: Final = "En titt pågår. Under titten styr bara piloten drönaren."


async def open_look(flow: MissionFlow, mission_id: str) -> LookRow | None:
    """The mission's open look, if one is open."""
    return next((x for x in await _repo(flow).open_looks() if x.mission_id == mission_id), None)


async def refuse_during_look(
    flow: MissionFlow, mission_id: str, skill: str, issuer_kind: str, gate: GateDecision
) -> GateDecision:
    """Owner decision: during a look only the pilot moves the drone. The AI's order that
    would take the drone from it onboard (every order the drone takes replaces the look) is
    refused (``INSPECT_BUSY``) while a look of the mission is open. Its flying home is not,
    and STOP / FLY HOME never pass the gate. The pilot's click on an AI card made before
    the look is the pilot's own order (``commands.decide`` does not ask here)."""
    if issuer_kind != "brain" or skill in _AI_DURING_LOOK or gate.decision == "REJECT":
        return gate
    busy = await open_look(flow, mission_id)
    if busy is None:
        return gate
    return GateDecision(
        decision="REJECT",
        reason_codes=["INSPECT_BUSY"],
        rule_ids=["R-INSPECT-PILOT"],
        message_sv=PILOT_ONLY_SV,
        details={"inspect_command_id": busy.command_id},
        side="ground",
    )


# --- the order: its target and its card -----------------------------------------------------


async def candidate_ref(flow: MissionFlow, mission_id: str, value: Any) -> Any:
    """The AI names a candidate by its number ("K3"): the ground looks up its id. An
    unknown number stays as it is (the gate refuses it)."""
    if isinstance(value, str) and _K.match(value):
        found = await _repo(flow).by_seq(mission_id, int(value[1:]))
        return value if found is None else found
    return value


_COMPASS: Final = ("N", "NO", "O", "SO", "S", "SV", "V", "NV")


def _direction(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, str]:
    north = (lat - lat0) * 111_320.0
    east = (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    bearing = math.degrees(math.atan2(east, north)) % 360.0
    return math.hypot(east, north), _COMPASS[round(bearing / 45.0) % 8]


async def summary(flow: MissionFlow, mission: Mission, params: dict[str, Any]) -> str:
    """The card's text: "Titta närmare på K3 (hög prioritet, sedd i 1 bild, 140 m NO)"."""
    from sd_ground.services import candidates as candidates_service

    cid = str(params.get("candidate_id", "")).lower()
    found = await candidates_service.card_list(flow, str(mission.id), only={cid})
    if not found:
        return "Titta närmare"
    card = found[0]
    level = {"high": "hög", "medium": "medel", "low": "låg"}[priority_class(card.priority)]
    parts = [
        f"{level} prioritet",
        f"sedd i {card.n_seen} bild" + ("er" if card.n_seen != 1 else ""),
    ]
    snap, _ = flow.snapshot(mission.vehicle_id)
    if snap is not None:
        dist, way = _direction(snap.position.lat, snap.position.lon, card.lat, card.lon)
        parts.append(f"{dist:.0f} m {way}")
    return f"Titta närmare på K{card.seq} ({', '.join(parts)})"


# --- the gate's facts -----------------------------------------------------------------------


@lru_cache(maxsize=4)
def _load_pack(folder: Path) -> FloorPack:
    return load_floor_pack(folder)


def _pack(flow: MissionFlow, ctx: GateContext) -> FloorPack | None:
    """The charter's map pack from ``<data_dir>/floor/<id>/``, when the ground has it and
    it is the one the charter names; else none (the drone checks the floor itself)."""
    c = ctx.charter
    if c is None or c.floor_pack_id is None:
        return None
    try:
        pack = _load_pack(flow.data_root() / "floor" / c.floor_pack_id)
    except (FloorPackError, OSError) as exc:
        log.warning("inspect.floor_pack_unavailable", pack=c.floor_pack_id, error=str(exc))
        return None
    if pack.sha256 != c.floor_pack_sha256:
        log.warning("inspect.floor_pack_mismatch", pack=c.floor_pack_id)
        return None
    return pack


def _floor(flow: MissionFlow, ctx: GateContext, params: dict[str, Any]) -> GateFloor | None:
    """The automatic floor (no laser on the ground) along the way to the candidate."""
    v, lat, lon = ctx.vehicle, params.get("lat"), params.get("lon")
    if v is None or not isinstance(lat, float) or not isinstance(lon, float):
        return None
    pack = _pack(flow, ctx)
    if pack is None:
        return None
    found = floor_along(
        pack,
        [LatLon(lat=v.lat, lon=v.lon), LatLon(lat=lat, lon=lon)],
        flight_date=flow.now().date(),
        laser=None,
        start_alt_amsl_m=v.alt_amsl_m,
    )
    return GateFloor(floor_amsl_m=found.floor_auto_amsl_m, reason=found.reason)


async def gate_facts(
    flow: MissionFlow, mission: Mission, ctx: GateContext, params: dict[str, Any]
) -> GateContext:
    """The context for an ``inspect_detection`` order (``params`` with its target)."""
    mid = str(mission.id)
    r = _repo(flow)
    active = next((x for x in await r.open_looks() if x.mission_id == mid), None)
    cid = str(params.get("candidate_id", "")).lower()
    suspended: tuple[Any, ...] = ()
    over = ctx.charter is not None and ctx.charter.overflight_allowed
    if over and not overflight_revoked(flow, ctx.charter):
        suspended = tuple(k.keepout_id for k in ctx.keepouts if str(k.source_id).lower() == cid)
    return ctx.model_copy(
        update={
            "inspect_active": None if active is None else active.command_id,
            "inspectable_candidates": tuple(await r.inspectable(mid)),
            "suspended_keepout_ids": suspended,
            "floor": _floor(flow, ctx, params),
        }
    )


# --- the look's life ------------------------------------------------------------------------


async def _publish(flow: MissionFlow, look: Look, *, card: bool) -> None:
    from sd_ground.services import candidates as candidates_service

    r = look.row
    vid = flow.g.config.default_vehicle_id
    flow.publish(
        "candidates",
        "inspection.updated",
        look.view().model_dump(mode="json"),
        vehicle_id=vid,
        mission_id=r.mission_id,
    )
    if card:
        for c in await candidates_service.card_list(flow, r.mission_id, only={r.candidate_id}):
            flow.publish(
                "candidates",
                "candidate.updated",
                c.model_dump(mode="json"),
                vehicle_id=vid,
                mission_id=r.mission_id,
            )


async def _say(flow: MissionFlow, mission_id: str, text: str, refs: dict[str, Any]) -> None:
    from sd_ground.services import chat as chat_service

    mission = await flow.repo.mission(mission_id)
    if mission is None:
        return
    await chat_service.add_line(
        flow,
        mission,
        msg_id=str(new_id()),
        role="system",
        author_ref="system:ground",
        kind="status",
        text=text,
        in_reply_to=None,
        refs=refs,
    )


async def started(flow: MissionFlow, mission: Mission, cmd: Command, *, by: str) -> bool:
    """The order is about to be sent: the look starts (``False``: the candidate is no
    longer open, or another look is, and the order must not go)."""
    row = await flow.active_charter_row(mission)
    charter = None if row is None else row.body
    mode = "live" if charter is None or charter.operation_mode is None else charter.operation_mode
    cid = str(cmd.params.get("candidate_id", "")).lower()
    r = _repo(flow)
    ok = await r.start(
        command_id=str(cmd.id),
        candidate_id=cid,
        mission_id=str(mission.id),
        started_by=by,
        operation_mode=mode,
        overflight=charter is not None
        and charter.overflight_allowed
        and not overflight_revoked(flow, charter),
        now=flow.now(),
    )
    look_row = await r.look(str(cmd.id)) if ok else None
    if look_row is None:
        return False
    look = Look(look_row)
    flow.looks[look_row.command_id] = look
    log.info("inspect.started", command_id=look_row.command_id, candidate_id=cid, by=by)
    await _publish(flow, look, card=True)
    return True


async def _open(flow: MissionFlow, command_id: str) -> Look | None:
    """The open look of the order, from memory or (after a restart) from its start row."""
    look = flow.looks.get(command_id)
    if look is not None:
        return look
    row = await _repo(flow).look(command_id)
    if row is None or row.ended_at is not None:
        return None
    return flow.looks.setdefault(command_id, Look(row))


async def _end(
    flow: MissionFlow,
    look: Look,
    outcome: str,
    *,
    detail: str | None = None,
    to_ready: bool = True,
) -> None:
    done = look.done
    ended = await _repo(flow).end(
        look.row,
        outcome=outcome,
        detail=detail,
        now=flow.now(),
        to_ready=to_ready,
        steps=None if done is None else done.steps,
        lowest_above_ground_m=None if done is None else done.lowest_above_ground_m,
        clear=None if done is None else done.clear,
    )
    flow.looks.pop(look.row.command_id, None)
    if not ended:
        return
    final = await _repo(flow).look(look.row.command_id)
    if final is None:  # pragma: no cover - the row was just written
        return
    look.row = final
    payload = {"command_id": final.command_id, "candidate_id": final.candidate_id}
    log.info("inspect.ended", outcome=outcome, detail=detail, **payload)
    flow.g.add_event(
        "inspection.ended",
        severity="info",
        mission_id=final.mission_id,
        vehicle_id=flow.g.config.default_vehicle_id,
        payload=payload | {"outcome": outcome, "detail": detail},
    )
    await _publish(flow, look, card=to_ready)
    text = _SYSTEM_TEXT.get(outcome)
    if text is not None:
        await _say(flow, final.mission_id, text.format(k=f"K{final.seq}"), payload)


async def decided(flow: MissionFlow, mission: Mission, candidate_id: str, decision: str) -> None:
    """A person decided on the candidate (``candidates.decide``, the state already moved):
    an open look on it ends, and the drone is told (``InspectOutcome``)."""
    outcome = _DECISION_OUTCOME.get(decision)
    if outcome is None:
        return
    mid = str(mission.id)
    look_row = next(
        (
            x
            for x in await _repo(flow).open_looks()
            if x.mission_id == mid and x.candidate_id == candidate_id
        ),
        None,
    )
    if look_row is None:
        return
    look = await _open(flow, look_row.command_id) or Look(look_row)
    if flow.link is not None:
        msg = InspectOutcome.model_validate({"command_id": look_row.command_id, "outcome": outcome})
        await flow.link.send_message(mission.vehicle_id, msg, mission_id=mid)
    await _end(flow, look, outcome, to_ready=False)


async def on_order_end(flow: MissionFlow, cmd: Command, status: str, reason: str | None) -> None:
    """The look's order ended before a decision (``commands.set_status``)."""
    outcome = _ORDER_END.get(status)
    look = None if outcome is None else await _open(flow, str(cmd.id))
    if look is not None and outcome is not None:
        await _end(flow, look, outcome, detail=reason)


async def _sent_at(flow: MissionFlow, cmd: Command) -> dt.datetime:
    """When the order went to the drone: at once when the gate said yes, or on the click
    on its card."""
    if cmd.approval_id is not None:
        card = await flow.repo.approval(str(cmd.approval_id))
        if card is not None and card.decided_at is not None:
            return card.decided_at
    return cmd.created_at


async def replaced(flow: MissionFlow, cmd: Command) -> None:
    """Another order is in force on the drone (it took it: ``accepted``, ``running`` or
    ``done``): that order replaces the look. Not before: an order the link loses or the
    drone refuses leaves the drone looking, and the look stays open so that the pilot's
    decision still reaches it (``decided``). Only an order sent after the look started:
    a late answer to one sent before it (the link's round trip) is about an order the
    look itself replaced onboard."""
    if cmd.skill in _KEEPS_LOOK:
        return
    mid = str(cmd.mission_id)
    sent = await _sent_at(flow, cmd)
    for row in await _repo(flow).open_looks():
        if row.mission_id == mid and sent >= row.started_at:
            look = await _open(flow, row.command_id)
            if look is not None:
                await _end(flow, look, "resumed_by_pilot", detail=cmd.skill)


async def on_event(flow: MissionFlow, kind: str, data: dict[str, Any]) -> None:
    """``inspect.lower_offer``, ``inspect.done`` and ``inspect.aborted`` from the drone
    (already in the event log)."""
    model = {EVENT_LOWER_OFFER: InspectLowerOffer, EVENT_DONE: InspectDone}.get(
        kind, InspectAborted if kind == EVENT_ABORTED else None
    )
    if model is None:
        return
    try:
        body = model.model_validate(data)
    except ValidationError:
        log.warning("inspect.bad_event", type=kind)
        return
    look = await _open(flow, str(body.command_id))
    if look is None:
        return
    now = flow.now()
    if isinstance(body, InspectLowerOffer):
        look.lower_offer = body
        look.lower_offer_until = now + dt.timedelta(seconds=LOWER_OFFER_S)
    elif isinstance(body, InspectDone):
        look.lower_offer = look.lower_offer_until = None
        look.done = body
        look.await_until = now + dt.timedelta(seconds=AWAIT_DECISION_S)
    else:  # InspectAborted
        outcome = _ABORT_OUTCOME.get(body.reason, "aborted")
        await _end(flow, look, outcome, detail=body.reason)
        return
    await _publish(flow, look, card=False)


async def on_progress(flow: MissionFlow, mission: Mission, progress: InspectProgress) -> None:
    """The drone's progress (state snapshot) as ``candidate.inspect_progress``, <= 2 Hz."""
    look = flow.looks.get(str(progress.command_id))
    if look is None:
        return
    now_ms = flow.now_ms()
    if look.progress_ms is not None and now_ms - look.progress_ms < PROGRESS_MIN_MS:
        return
    look.progress_ms = now_ms
    flow.publish(
        "candidates",
        "candidate.inspect_progress",
        progress.model_dump(mode="json") | {"seq": look.row.seq},
        vehicle_id=mission.vehicle_id,
        mission_id=str(mission.id),
    )


async def lower(flow: MissionFlow, command_id: str) -> Inspection:
    """The pilot's yes to "Gå lägre?": ``InspectLower`` goes onboard."""
    look = await _open(flow, command_id.lower())
    if look is None:
        raise ApiStatusError(404, "INSPECTION_NOT_FOUND", "Titten pågår inte.")
    if look.lower_offer is None:
        raise ApiStatusError(409, "NO_LOWER_OFFER", "Drönaren frågar inte om att gå lägre nu.")
    sent = None
    if flow.link is not None:
        sent = await flow.link.send_message(
            flow.g.config.default_vehicle_id,
            InspectLower.model_validate({"command_id": look.row.command_id}),
            mission_id=look.row.mission_id,
        )
    if sent is None:
        raise ApiStatusError(503, "LINK_DOWN", "Ingen länk till drönaren.")
    look.lower_offer = look.lower_offer_until = None
    await _publish(flow, look, card=False)
    return look.view()


async def views(flow: MissionFlow, mission_id: str) -> list[Inspection]:
    """Every look of the mission, oldest first, the open one with what the drone said."""
    out: list[Inspection] = []
    for row in await _repo(flow).looks(mission_id):
        look = flow.looks.get(row.command_id) if row.ended_at is None else None
        out.append((look or Look(row)).view())
    return out


async def tick(flow: MissionFlow, mission: Mission) -> None:
    """Once a second: "Gå lägre?" unanswered, the wait for a decision over, or a look the
    ground lost track of (a restart) that cannot be going on any more."""
    mid = str(mission.id)
    now = flow.now()
    for row in await _repo(flow).open_looks():
        if row.mission_id != mid:
            continue
        look = flow.looks.setdefault(row.command_id, Look(row))
        if look.await_until is not None and now >= look.await_until:
            await _end(flow, look, "timeout")
        elif now >= row.started_at + _LONGEST:
            await _end(flow, look, "timeout", detail="lost")
        elif look.lower_offer_until is not None and now >= look.lower_offer_until:
            look.lower_offer = look.lower_offer_until = None
            await _publish(flow, look, card=False)
