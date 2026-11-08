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
take the drone from it are refused (``refuse_during_look``), with a hint to wait for the
look's end, and an observer's mark on the candidate is a note for the pilot that neither
ends the look nor reaches the AI. The AI's situation says while a look is open and on
which candidate it waits (``situation_line``).

The look's end reaches the AI once (``inspection.closed``, a skill event of the look's
order with status "closed"): the ground's end row says how it ended (the outcome, the
ground's word), the drone's ``inspect.closed`` what it does next (back to the search, a
hover, home ...). The event goes out when both are in, or when the drone's report is not
coming: the look never reached it, the ground lost track of it, or ``CLOSED_WAIT_S`` passed
(a lost link; ``next`` is then unknown). A report that comes after that is only logged.

A search flown on after a look (the "linked event" model): the drone flies the
interrupted search on by its own ``resume``, an order the ground has no row for, while the
search's own order stays ``superseded`` on the ground (a terminal row is never revived).
The ground keeps the look in memory until that resume ends (``flown_on``): the search's
order is then the active order in the situation, its sector's progress and coverage count
(``commands``, ``coverage``), and its end reaches the AI as a skill event of that order
(``inspection.resume_ended``: the search's skill and sector, status done, failed or
superseded, ``details.resumed_after_look`` the look's order). In memory only: after a
restart of the ground the search flies on unreported, as before. When that end does not
come and the drone's state no longer names the resume (``on_snapshot``), the run's link
ends there, and that moment is kept in the event log (``inspection.resume_left``) for
the coverage rebuilt after a restart.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, NamedTuple

from pydantic import ValidationError

from sd_ground.api.deps import ApiStatusError
from sd_ground.db.inspections import InspectionRepo, LookRow
from sd_ground.logging import get_logger
from sd_ground.services import floor_packs
from sd_safety.floor import floor_along
from sd_safety.floor_pack import FloorPack, FloorPackError
from sd_schemas.base import LatLon, new_id
from sd_schemas.gate import GateContext, GateDecision, GateFloor
from sd_schemas.inspect import (
    AWAIT_DECISION_S,
    EVENT_ABORTED,
    EVENT_CLOSED,
    EVENT_DONE,
    EVENT_LOWER_OFFER,
    EVENT_PERMIT_CHANGED,
    EVENT_RESUME_ENDED,
    INSPECT_MAX_S,
    LOWER_OFFER_S,
    InspectAborted,
    InspectAbortReason,
    InspectClosed,
    InspectDone,
    Inspection,
    InspectLowerOffer,
    InspectProgress,
    InspectResumeEnded,
    Permit,
    PermitChanged,
)
from sd_schemas.link import InspectLower, InspectOutcome, InspectStay
from sd_schemas.reason_codes import GATE_REASON_CODES
from sd_schemas.search import priority_class
from sd_schemas.skills import CANDIDATE_NUMBER_PATTERN

if TYPE_CHECKING:
    from sd_ground.services.missions import MissionFlow
    from sd_schemas.api import Command, Mission

__all__ = [
    "CLOSED_WAIT_S",
    "PROGRESS_MIN_MS",
    "FlownOn",
    "Look",
    "candidate_ref",
    "decided",
    "flies_resume",
    "flown_on",
    "gate_facts",
    "lower",
    "on_event",
    "on_order_end",
    "on_progress",
    "overflight_allowed",
    "overflight_revoked",
    "permit",
    "record_resume_left",
    "refuse_during_look",
    "refuse_revoked",
    "replaced",
    "set_permit",
    "situation_line",
    "started",
    "summary",
    "tick",
    "views",
]

log = get_logger("sd_ground.inspections")

SKILL: Final = "inspect_detection"
PROGRESS_MIN_MS: Final = 500  # candidate.inspect_progress at most 2 Hz
# The drone's ``inspect.closed`` after the ground ended the look, at most: the return's
# climb comes first (from low over the target, at CLIMB_MIN_MS). Then the AI hears the end
# without it.
CLOSED_WAIT_S: Final = 60.0
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
# The ground's latch cancelling the open orders (``commands.cancel_open``) -> the reason
# the drone's ``inspect.aborted`` gives for the same mode change.
_LATCH_ABORT: Final[dict[str, InspectAbortReason]] = {
    "OPERATOR_STOP": "stop",
    "WATCHDOG_STOP": "stop",
    "OPERATOR_RTL": "home",
    "WATCHDOG_RTL": "home",
    "BATTERY_RTL": "battery",
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
    """What the ground knows of a look beyond its start row: while it is open, and after
    its end until the AI has heard it and the search flown on after it has ended."""

    row: LookRow
    lower_offer: InspectLowerOffer | None = None
    lower_offer_until: dt.datetime | None = None
    done: InspectDone | None = None
    await_until: dt.datetime | None = None
    progress_ms: int | None = None
    closed: InspectClosed | None = None  # the drone's ``inspect.closed``
    closed_at: dt.datetime | None = None  # when it came
    tell_by: dt.datetime | None = None  # ended here: the drone's report awaited until then
    closing: bool = False  # its end on the ground under way (``_end``)
    told: bool = False  # ``inspection.closed`` sent
    resume_ended: bool = False  # the drone's ``inspect.resume_ended`` came
    resume_seen: bool = False  # the drone's state named the resume as its active order
    resume_left_at: dt.datetime | None = None  # its state named another order from here

    def linked(self) -> bool:
        """The drone said it flies the interrupted search on after this look (its own
        resume), and the end of that run has not come yet."""
        c = self.closed
        return (
            c is not None
            and c.next == "resume"
            and c.resume_command_id is not None
            and c.resumed_command_id is not None
            and not self.resume_ended
        )

    def flying_on(self) -> bool:
        """The drone flies the interrupted search on after this look, by its own resume:
        linked, and its state has not named another order since (``on_snapshot``)."""
        return self.linked() and self.resume_left_at is None

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


class FlownOn(NamedTuple):
    """A search the drone flies on after a look (``Look.flying_on``)."""

    command_id: str  # the search's own order (``superseded`` on the ground)
    resume_command_id: str  # the executor's own resume that flies it
    look_command_id: str
    since: dt.datetime  # its track counts from here (``coverage``)


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
# After the look's end on the ground, until the drone has said what it does next.
CLOSING_SV: Final = (
    "En titt avslutas. Drönaren har ännu inte sagt vad den gör sedan, så bara piloten styr den."
)
# The INSPECT_BUSY refusal's hint to the AI: the plan waits for the look's end, which says
# what the drone does next (on "resume" it flies the search on by itself: no order).
WAIT_FOR_CLOSE_SV: Final = (
    "Vänta tills titten är stängd (skill-händelsen inspect_detection med status closed); "
    "den säger vad drönaren gör sedan och om ordern behövs."
)


async def open_look(flow: MissionFlow, mission_id: str) -> LookRow | None:
    """The mission's open look, if one is open."""
    return next((x for x in await _repo(flow).open_looks() if x.mission_id == mission_id), None)


def _unclosed(flow: MissionFlow, mission_id: str) -> LookRow | None:
    """A look of the mission ended on the ground whose end the AI has not heard yet: the
    drone may still be in it (its ``inspect.closed`` is awaited, ``CLOSED_WAIT_S``)."""
    return next(
        (
            x.row
            for x in flow.looks.values()
            if x.row.mission_id == mission_id
            and (x.closing or x.row.ended_at is not None)
            and not x.told
        ),
        None,
    )


async def busy_look(flow: MissionFlow, mission_id: str) -> LookRow | None:
    """The look that holds the AI's moves: the open one, or one ended on the ground until
    the drone's report of its end came or was given up (``_settle``)."""
    return await open_look(flow, mission_id) or _unclosed(flow, mission_id)


async def refuse_during_look(
    flow: MissionFlow, mission_id: str, skill: str, issuer_kind: str, gate: GateDecision
) -> GateDecision:
    """Owner decision: during a look only the pilot moves the drone. The AI's order that
    would take the drone from it onboard (every order the drone takes replaces the look) is
    refused (``INSPECT_BUSY``) while a look of the mission is open, and after its end on
    the ground until the AI has heard it (``busy_look``: the drone may still be in it).
    Its flying home is not, and STOP / FLY HOME never pass the gate. The pilot's click on
    an AI card made before the look is the pilot's own order (``commands.decide`` does
    not ask here)."""
    if issuer_kind != "brain" or skill in _AI_DURING_LOOK or gate.decision == "REJECT":
        return gate
    busy = await busy_look(flow, mission_id)
    if busy is None:
        return gate
    look = flow.looks.get(busy.command_id)
    closing = busy.ended_at is not None or (look is not None and look.closing)
    return GateDecision(
        decision="REJECT",
        reason_codes=["INSPECT_BUSY"],
        rule_ids=["R-INSPECT-PILOT"],
        message_sv=f"{CLOSING_SV if closing else PILOT_ONLY_SV} {WAIT_FOR_CLOSE_SV}",
        details={
            "inspect_command_id": busy.command_id,
            "candidate": f"K{busy.seq}",
            "phase": _phase(flow, busy),
            # The order is not lost: send it again once the look's end has come.
            "retry_after": "inspect_closed",
        },
        side="ground",
    )


def _phase(flow: MissionFlow, row: LookRow) -> str:
    look = flow.looks.get(row.command_id)
    return (Look(row) if look is None else look).view().phase


async def situation_line(flow: MissionFlow, mission_id: str) -> str | None:
    """The AI's line while a look of the mission holds its moves (``busy_look``): on which
    candidate, and whether the drone still flies it, waits for the operator's decision, or
    has not yet said what it does after it. None without one."""
    row = await busy_look(flow, mission_id)
    if row is None:
        return None
    k = f"K{row.seq}"
    look = flow.looks.get(row.command_id)
    if row.ended_at is not None or (look is not None and look.closing):
        now = f"Titt på {k} avslutas: drönarens besked om vad den gör sedan väntas."
    elif _phase(flow, row) == "await":
        now = f"Titt på {k} pågår: drönaren väntar på operatörens beslut om {k}."
    else:
        now = f"Titt på {k} pågår: drönaren tittar närmare på {k}."
    return (
        f"{now} Order som flyttar drönaren nekas (INSPECT_BUSY) tills titten är stängd; "
        "vänta på skill-händelsen inspect_detection closed, den säger vad drönaren gör sedan."
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


def _pack(flow: MissionFlow, ctx: GateContext) -> FloorPack | None:
    """The charter's map pack from ``<data_dir>/floor/<id>/``, when the ground has it and
    it is the one the charter names; else none (the drone checks the floor itself)."""
    c = ctx.charter
    if c is None or c.floor_pack_id is None:
        return None
    try:
        pack = floor_packs.load(floor_packs.root(flow) / c.floor_pack_id)
    except FloorPackError as exc:
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
        return look if look.row.ended_at is None and not look.closing else None
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
    cid = look.row.command_id
    if look.closing:
        return  # another end of it is under way
    # Busy for the AI from here (``_unclosed``), through every await below to ``_settle``.
    look.closing = True
    flow.looks.setdefault(cid, look)
    done = look.done
    try:
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
        final = await _repo(flow).look(cid) if ended else None
    except BaseException:
        look.closing = False
        raise
    if final is None:
        # Ended elsewhere (it handles the end), or not at all: not this end's to tell.
        look.closing = False
        if flow.looks.get(cid) is look:
            flow.looks.pop(cid, None)
        return
    look.row = final
    # Told by ``tick`` at the latest (``_settle``), even if a step below fails.
    look.tell_by = flow.now() + dt.timedelta(seconds=CLOSED_WAIT_S)
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
    await _settle(flow, look)


# --- the look's end for the AI, and the search flown on after it ---------------------------

# The ground's outcome (``Inspection.outcome``) in the AI's line.
_OUTCOME_SV: Final = {
    "confirmed": "operatören bekräftade",
    "rejected": "operatören sa att det inte är personen",
    "unsure": "operatören är osäker",
    "resume": "operatören bad om att söka vidare",
    "timeout": "inget beslut i tid",
    "resumed_by_pilot": "en ny order tog över",
    "aborted": "avbruten",
    "refused_onboard": "drönaren sa nej",
    "failed": "misslyckades",
}
_NEXT_SV: Final = {
    "hold": "Drönaren står still och väntar på order.",
    "home": "Drönaren flyger hem.",
    "held": "Piloten eller nödstoppet styr drönaren.",
    "new_order": "En ny order styr drönaren.",
    "link_loss": "Länken bröts; drönaren följer reglerna för länkbortfall.",
}
_RESUME_END_SV: Final = {"done": "är klar", "failed": "misslyckades", "superseded": "avbröts"}


def _drone_reports(row: LookRow) -> bool:
    """Whether the drone has a look to report the end of: not when it refused the order,
    nor when the ground lost track of the look (a restart)."""
    return row.outcome != "refused_onboard" and row.detail != "lost"


async def _settle(flow: MissionFlow, look: Look) -> None:
    """The look ended on the ground or the drone reported its end: once both are in, or
    the drone's report is not coming (``_drone_reports``, ``tell_by``), the AI hears it,
    once. The look stays in memory while the report is awaited and while the drone flies
    the search on after it (``flying_on``)."""
    cid = look.row.command_id
    if look.row.ended_at is None:
        return  # still open on the ground: its end comes here again
    if not look.told:
        now = flow.now()
        if look.tell_by is None:
            look.tell_by = now + dt.timedelta(seconds=CLOSED_WAIT_S)
        if look.closed is None and _drone_reports(look.row) and now < look.tell_by:
            flow.looks[cid] = look
            return
        look.told = True
        await _tell_closed(flow, look)
    if look.linked():
        flow.looks[cid] = look
    else:
        flow.looks.pop(cid, None)


def _closed_text(look: Look, sector_id: Any) -> str:
    r, c = look.row, look.closed
    k = f"K{r.seq}"
    head = f"Titten på {k} är stängd ({_OUTCOME_SV.get(str(r.outcome), str(r.outcome))})."
    if c is None:
        return (
            f"{head} Drönaren har inte sagt vad den gör nu. Se efter i läget vad den gör, "
            "eller fråga operatören, innan du ger en ny order."
        )
    if c.next == "resume" and c.breakpoint == "survey":
        where = f"i sektor {sector_id} " if sector_id else ""
        return f"{head} Drönaren söker vidare {where}där sökningen bröts."
    if c.next == "resume":
        return f"{head} Drönaren flyger tillbaka dit titten började och står sedan still."
    if c.cause == "return_refused":
        return f"{head} Drönaren kunde inte flyga tillbaka och står still."
    return f"{head} {_NEXT_SV[c.next]}"


async def _tell_closed(flow: MissionFlow, look: Look) -> None:
    """``inspection.closed``: for the AI a skill event of the look's order with status
    "closed" (every look: the AI's moves were refused until it; ``issuer_kind`` says
    whether the look or the search flown on is the AI's), and the event log."""
    r, c = look.row, look.closed
    look_cmd = await flow.repo.command(r.command_id)
    resumed = None
    if c is not None and c.resumed_command_id is not None:
        resumed = await flow.repo.command(str(c.resumed_command_id))
    sector_id = None if resumed is None else resumed.params.get("sector_id")
    brain = any(x is not None and x.issuer_kind == "brain" for x in (look_cmd, resumed))
    data: dict[str, Any] = {
        "command_id": r.command_id,
        "skill": SKILL,
        "status": "closed",
        "reason_codes": [],
        "message_sv": _closed_text(look, sector_id),
        "details": {
            "candidate_id": r.candidate_id,
            "candidate": f"K{r.seq}",
            "outcome": r.outcome,
            "detail": r.detail,
            "cause": None if c is None else c.cause,
            "next": None if c is None else c.next,
            "breakpoint": None if c is None else c.breakpoint,
            "resumed_command_id": None if resumed is None else str(resumed.id),
            "resumed_skill": None if resumed is None else resumed.skill,
            "resume_command_id": (
                None if c is None or c.resume_command_id is None else str(c.resume_command_id)
            ),
            "sector_id": sector_id,
        },
        "issuer_kind": "brain" if brain else "operator",
    }
    log.info(
        "inspect.closed", command_id=r.command_id, outcome=r.outcome, next=data["details"]["next"]
    )
    vid = flow.g.config.default_vehicle_id
    flow.g.add_event(
        "inspection.closed", severity="info", mission_id=r.mission_id, vehicle_id=vid, payload=data
    )
    flow.publish("mission", "inspection.closed", data, vehicle_id=vid, mission_id=r.mission_id)


def flown_on(flow: MissionFlow, mission_id: str) -> list[FlownOn]:
    """The search the drone flies on after a look of the mission, if one (at most one: the
    drone has one resume at a time)."""
    out: list[FlownOn] = []
    for look in flow.looks.values():
        c = look.closed
        if look.row.mission_id != mission_id or not look.flying_on() or c is None:
            continue
        out.append(
            FlownOn(
                command_id=str(c.resumed_command_id),
                resume_command_id=str(c.resume_command_id),
                look_command_id=look.row.command_id,
                since=look.closed_at or look.row.started_at,
            )
        )
    return out


async def _on_closed(flow: MissionFlow, body: InspectClosed) -> None:
    cid = str(body.command_id)
    look = flow.looks.get(cid) or await _open(flow, cid)
    if look is None or look.closed is not None:
        # Already told without it (CLOSED_WAIT_S), or unknown here (a restart).
        log.info("inspect.closed_late", command_id=cid, next=body.next)
        return
    if body.resume_command_id is not None:
        # One resume at a time onboard: an older search flown on is over.
        for other in [x for x in flow.looks.values() if x is not look and x.linked()]:
            if other.row.mission_id == look.row.mission_id:
                other.resume_ended = True
                flow.looks.pop(other.row.command_id, None)
    look.closed, look.closed_at = body, flow.now()
    await _settle(flow, look)


def _names_the_run(
    look: Look | None, row: LookRow | None, search: Any, resumed: str | None
) -> bool:
    """Whether a run's end (``inspect.resume_ended``) is that of the run the look's end
    named: in memory, the same search order (or none: back to a point) its
    ``inspect.closed`` named. After a ground restart (no end in memory) the search the
    look broke off: an order of the look's mission, superseded, from before the look."""
    if row is None:
        return False
    c = None if look is None else look.closed
    if c is not None:
        named = None if c.resumed_command_id is None else str(c.resumed_command_id)
        if named != resumed:
            return False
    elif resumed is not None and (
        search is None or search.status != "superseded" or search.created_at > row.started_at
    ):
        return False
    return resumed is None or (search is not None and str(search.mission_id) == row.mission_id)


async def _on_resume_ended(flow: MissionFlow, body: InspectResumeEnded) -> None:
    """The drone's own resume after a look ended. For a search flown on: the search's
    sector is done when it was flown to its end, and the AI hears the end as a skill event
    of the search's own order (``inspection.resume_ended``), once. Only for the run the
    look's end named (``_names_the_run``); another report is only logged and changes
    nothing here."""
    cid = str(body.command_id)
    look = flow.looks.get(cid)
    resumed = None if body.resumed_command_id is None else str(body.resumed_command_id)
    search = None if resumed is None else await flow.repo.command(resumed)
    row = look.row if look is not None else await _repo(flow).look(cid)
    if cid in flow.resume_told or not _names_the_run(look, row, search, resumed):
        log.info("inspect.resume_ended_unmatched", command_id=cid, resumed_command_id=resumed)
        return
    if look is not None:
        look.resume_ended = True
        if look.told:
            flow.looks.pop(cid, None)
    if resumed is None or search is None or row is None:
        return  # back to a point only: the look's end said so
    flow.resume_told[cid] = row.mission_id
    sector_id = search.params.get("sector_id")
    if body.status == "done" and search.skill == "survey_sector" and sector_id:
        await flow.repo.set_sector_progress(row.mission_id, str(sector_id), "done", 1.0, flow.now())
    what = f"Sökningen i sektor {sector_id}" if sector_id else "Den fortsatta sökningen"
    data: dict[str, Any] = {
        "command_id": str(search.id),
        "skill": search.skill,
        "status": body.status,
        "reason_codes": list(body.reason_codes),
        "message_sv": (
            f"{what} {_RESUME_END_SV[body.status]} (fortsatt efter titten på K{row.seq})."
        ),
        "details": {"resumed_after_look": row.command_id, "sector_id": sector_id},
        "sector_id": sector_id,
        "issuer_kind": search.issuer_kind,
    }
    log.info("inspect.resume_ended", command_id=str(search.id), status=body.status, look=cid)
    vid = flow.g.config.default_vehicle_id
    flow.g.add_event(
        "inspection.resume_ended",
        severity="info",
        mission_id=row.mission_id,
        vehicle_id=vid,
        payload=data,
    )
    flow.publish(
        "mission", "inspection.resume_ended", data, vehicle_id=vid, mission_id=row.mission_id
    )


def _flies_resume(
    c: InspectClosed, active_command_id: str | None, active_skill: str | None
) -> bool:
    """Whether the drone's state names the look's resume as its running order. The real
    executor never answers for its own resume (no acks), so its state names it by skill
    only: ``active_command_id`` is None and ``active_skill`` "resume" (one resume at a
    time onboard). A resume acked as an order is named by its id."""
    resume = None if c.resume_command_id is None else str(c.resume_command_id)
    return flies_resume(resume, active_command_id, active_skill)


def flies_resume(
    resume_command_id: str | None, active_command_id: str | None, active_skill: str | None
) -> bool:
    """``_flies_resume`` by the resume's id (a look's end rebuilt from the event log)."""
    if active_command_id is not None:
        return active_command_id == resume_command_id
    return active_skill == "resume"


def search_flown_on(
    flow: MissionFlow, mission_id: str, active_command_id: str | None, active_skill: str | None
) -> str | None:
    """The search order the drone flies on after a look of the mission, when its state names
    that look's resume as its running order (``_flies_resume``)."""
    for look in flow.looks.values():
        c = look.closed
        if (
            look.row.mission_id == mission_id
            and look.flying_on()
            and c is not None
            and _flies_resume(c, active_command_id, active_skill)
        ):
            return str(c.resumed_command_id)
    return None


def on_snapshot(
    flow: MissionFlow,
    mission_id: str,
    active_command_id: str | None,
    active_skill: str | None,
) -> None:
    """The drone's running order (state snapshot) ends the link to a search flown on after a
    look once it is no longer that resume (``_flies_resume``): the resume was seen and
    another order (or none) runs, or it was never seen within ``CLOSED_WAIT_S`` of the
    look's end. The look stays in memory a while longer (``tick``) so that a late
    ``inspect.resume_ended`` is still told; the search no longer counts as flown on
    (``flying_on``)."""
    now = flow.now()
    for look in flow.looks.values():
        c = look.closed
        if look.row.mission_id != mission_id or not look.flying_on() or c is None:
            continue
        if _flies_resume(c, active_command_id, active_skill):
            look.resume_seen = True
            continue
        since = look.closed_at or now
        if look.resume_seen or now >= since + dt.timedelta(seconds=CLOSED_WAIT_S):
            look.resume_left_at = now
            log.info(
                "inspect.resume_left",
                command_id=look.row.command_id,
                resume_command_id=str(c.resume_command_id),
                active_command_id=active_command_id,
                active_skill=active_skill,
                seen=look.resume_seen,
            )
            record_resume_left(
                flow, mission_id, str(c.resumed_command_id), look.row.command_id, look.resume_seen
            )


def record_resume_left(
    flow: MissionFlow, mission_id: str, search_id: str, look_id: str | None, seen: bool
) -> None:
    """The drone's state no longer names the resume of the search flown on after a look
    (``on_snapshot``): kept in the event log (``inspection.resume_left``, at the time it was
    seen), so that after a ground restart the run ends there (``coverage._run_end``) even
    when its ``inspection.resume_ended`` never comes."""
    flow.g.add_event(
        "inspection.resume_left",
        severity="info",
        mission_id=mission_id,
        vehicle_id=flow.g.config.default_vehicle_id,
        payload={"command_id": search_id, "look": look_id, "seen": seen},
    )


def forget(flow: MissionFlow, mission_id: str) -> None:
    """The mission ended: no search of it is flown on any more, no look of it is told."""
    for cid in [k for k, x in flow.looks.items() if x.row.mission_id == mission_id]:
        look = flow.looks.pop(cid)
        log.info(
            "inspect.forgotten",
            command_id=cid,
            told=look.told,
            linked=look.linked(),
            ended=look.row.ended_at is not None,
        )
    for cid in [k for k, m in flow.resume_told.items() if m == mission_id]:
        del flow.resume_told[cid]


async def decided(flow: MissionFlow, mission: Mission, candidate_id: str, decision: str) -> None:
    """A person decided on the candidate (``candidates.decide``, the state already moved):
    an open look on it ends, and the drone is told (``InspectOutcome``). A rejection says
    whether the ground releases the candidate's cylinder (``keepout_released``): only a
    candidate known not to be a person (``FlowRepo.keepout_candidates``)."""
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
        released = outcome == "rejected" and await flow.repo.candidate_not_a_person(candidate_id)
        msg = InspectOutcome.model_validate(
            {"command_id": look_row.command_id, "outcome": outcome, "keepout_released": released}
        )
        await flow.link.send_message(mission.vehicle_id, msg, mission_id=mid)
    await _end(flow, look, outcome, to_ready=False)


async def on_order_end(flow: MissionFlow, cmd: Command, status: str, reason: str | None) -> None:
    """The look's order ended before a decision (``commands.set_status``).

    Superseded with a code, the drone dropped the look itself (STOP, a takeover, a limit,
    a cylinder: ``ONB_CONTROL_HELD``, ``L-BATT`` ...), not for another order: its
    ``inspect.aborted`` that follows says why (``aborted`` with the reason), and the tick
    ends a look whose report never comes. Another order's replacement comes uncoded and
    is closed by ``replaced`` as the drone takes the new order.

    Cancelled by the ground's own STOP, FLY HOME or battery guard (``cancel_open``), the
    look ends at once with the reason the drone's ``inspect.aborted`` gives for it
    (``stop``, ``home``, ``battery``): the drone may report later than the cancel (it
    waits for the ground's notice to name a mode change), and the report then finds the
    look already ended."""
    if status == "superseded" and reason is not None:
        return
    outcome = _ORDER_END.get(status)
    look = None if outcome is None else await _open(flow, str(cmd.id))
    if look is not None and outcome is not None:
        if status == "cancelled" and reason is not None:
            reason = _LATCH_ABORT.get(reason, reason)
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
    """``inspect.lower_offer``, ``inspect.done``, ``inspect.aborted``, ``inspect.closed``
    and ``inspect.resume_ended`` from the drone (already in the event log)."""
    if kind in (EVENT_CLOSED, EVENT_RESUME_ENDED):
        try:
            if kind == EVENT_CLOSED:
                await _on_closed(flow, InspectClosed.model_validate(data))
            else:
                await _on_resume_ended(flow, InspectResumeEnded.model_validate(data))
        except ValidationError:
            log.warning("inspect.bad_event", type=kind)
        return
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
    if look is None or look.row.ended_at is not None:
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
    return await _answer_lower(flow, command_id, InspectLower)


async def stay(flow: MissionFlow, command_id: str) -> Inspection:
    """The pilot's no to "Gå lägre?" ("Nej, stanna här"): ``InspectStay`` goes onboard, and
    the drone ends the look where it hovers at once instead of waiting out the question."""
    return await _answer_lower(flow, command_id, InspectStay)


async def _answer_lower(
    flow: MissionFlow, command_id: str, kind: type[InspectLower | InspectStay]
) -> Inspection:
    """The pilot's answer to the question the drone asks now, through the link."""
    look = await _open(flow, command_id.lower())
    if look is None:
        raise ApiStatusError(404, "INSPECTION_NOT_FOUND", "Titten pågår inte.")
    if look.lower_offer is None:
        raise ApiStatusError(409, "NO_LOWER_OFFER", "Drönaren frågar inte om att gå lägre nu.")
    sent = None
    if flow.link is not None:
        sent = await flow.link.send_message(
            flow.g.config.default_vehicle_id,
            kind.model_validate({"command_id": look.row.command_id}),
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
    """Once a second: "Gå lägre?" unanswered, the wait for a decision over, a look the
    ground lost track of (a restart) that cannot be going on any more, or an ended look
    whose report from the drone did not come in time (``CLOSED_WAIT_S``), and a search
    flown on after a look that the drone left (``on_snapshot``) without its end coming."""
    mid = str(mission.id)
    now = flow.now()
    for look in list(flow.looks.values()):
        r, until = look.row, look.tell_by
        if r.mission_id == mid and not look.told and until is not None and now >= until:
            await _settle(flow, look)
        left = look.resume_left_at
        if (
            r.mission_id == mid
            and left is not None
            and now >= left + dt.timedelta(seconds=CLOSED_WAIT_S)
        ):
            log.info("inspect.resume_link_dropped", command_id=r.command_id)
            flow.looks.pop(r.command_id, None)
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
