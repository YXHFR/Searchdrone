"""The commander API on ``127.0.0.1:7700`` (``/commander/v1``), only for ``sd-brain``.

A narrow API on its own loopback-only port. Everything the AI may never do (approve, STOP /
RETURN HOME / Resume, clearance, limits, autonomy, the AI switch, map pins, flight
controller parameters) has no route here at all, so it is a 404, and the commander token is
refused on the panel listeners (403). All text from here is stripped of control characters
(except newline) before it is validated, and too long text is refused (422), never cut.
"""

from __future__ import annotations

import contextlib
import json
import os
import unicodedata
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    Query,
    Request,
    Response,
    WebSocket,
)
from fastapi import Path as PathParam
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from sd_ground.api.deps import ApiStatusError, ground, repo_of
from sd_ground.api.tasks import until_first_done
from sd_ground.bus import BusEvent
from sd_ground.context import Ground
from sd_ground.logging import get_logger
from sd_ground.services import approvals as approvals_service
from sd_ground.services import charters as charter_rules
from sd_ground.services import chat as chat_service
from sd_ground.services import commands as commands_service
from sd_ground.services.missions import MissionFlow
from sd_safety.constants import BRAIN_HEARTBEAT_TIMEOUT_S
from sd_schemas.api import ActionDispatched, ChatMessage
from sd_schemas.charter import CharterProposal, CharterValidationResult
from sd_schemas.commander import (
    PROMPT_JSON_MAX_BYTES,
    BrainMessage,
    CharterProposalResult,
    DecisionLog,
    EstimateResult,
    MissionMemory,
    MissionMemoryUpdate,
    Situation,
    SkillRequest,
    SkillSubmitResult,
)
from sd_schemas.decide import ChatIntentReport
from sd_schemas.skills import SKILL_REGISTRY, SkillSpecPublic

__all__ = [
    "BRAIN_HEARTBEAT_TIMEOUT_S",
    "BRAIN_ISSUER",
    "RAW_BODY_MODELS",
    "CommanderHub",
    "clean_text",
    "router",
    "to_commander",
]

BRAIN_PING_S = 5.0
# Bodies read raw (for our own error codes) and named in ``openapi_extra``: their schemas
# are added to the document's components by ``app.create_commander_app``.
RAW_BODY_MODELS: tuple[type[BaseModel], ...] = (
    BrainMessage,
    CharterProposal,
    DecisionLog,
    MissionMemoryUpdate,
)
BRAIN_ISSUER = "brain:ground"
log = get_logger("sd_ground.commander")

router = APIRouter(prefix="/commander/v1")

_UUID = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
IdPath = Annotated[str, PathParam(pattern=_UUID)]
VehiclePath = Annotated[str, PathParam(pattern=r"^[a-z0-9][a-z0-9-]{2,31}$")]


def _uuid_or_none(value: object) -> str | None:
    """Only a well-formed UUID from the AI reaches the panel's event stream."""
    try:
        return str(UUID(str(value))) if value is not None else None
    except ValueError:
        return None


def clean_text(value: str) -> str:
    """Drop Unicode control characters (category Cc) except newline."""
    return "".join(ch for ch in value if ch == "\n" or unicodedata.category(ch) != "Cc")


def _clean(data: Any, keys: frozenset[str]) -> Any:
    if isinstance(data, dict):
        return {
            k: clean_text(v) if k in keys and isinstance(v, str) else _clean(v, keys)
            for k, v in data.items()  # pyright: ignore[reportUnknownVariableType]
        }
    if isinstance(data, list):
        return [_clean(v, keys) for v in data]  # pyright: ignore[reportUnknownVariableType]
    return data


_TEXT_KEYS = frozenset({"text", "rationale_sv", "reason_sv", "summary_sv", "plan_note"})


async def _body(request: Request) -> Any:
    try:
        return _clean(json.loads(await request.body()), _TEXT_KEYS)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ApiStatusError(422, "VALIDATION", "Fel i indata (inte JSON).") from None


def _validate[M](model: type[M], data: Any) -> M:
    try:
        return model.model_validate(data)  # type: ignore[attr-defined, no-any-return]
    except ValidationError as exc:
        errors = [
            {
                "loc": ["body", *[str(p) for p in e["loc"]]],
                "code": e["type"],
                "detail_sv": e["msg"],
            }
            for e in exc.errors()
        ]
        raise ApiStatusError(422, "VALIDATION", "Fel i indata.", errors) from None


def require_commander(request: Request) -> None:
    g = ground(request)
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not g.brain_token.matches(token.strip()):
        if scheme.lower() == "bearer" and g.search_token.matches(token.strip()):
            raise ApiStatusError(403, "SEARCH_FORBIDDEN", "Söktjänstens nyckel gäller inte här.")
        raise ApiStatusError(401, "UNAUTHENTICATED", "Befälhavarens nyckel saknas eller är fel.")


Commander = Annotated[None, Depends(require_commander)]
G = Annotated[Ground, Depends(ground)]


def _flow(g: Ground) -> MissionFlow:
    repo_of(g)
    if g.flow is None:
        raise ApiStatusError(503, "DB_DOWN", "Uppdragsflödet är inte igång just nu.")
    return g.flow


async def _open_mission(g: Ground, mission_id: IdPath) -> tuple[str, str]:
    mission = await repo_of(g).mission(mission_id)
    if mission is None:
        raise ApiStatusError(404, "MISSION_NOT_FOUND", "Uppdraget finns inte.")
    if mission.status == "CLOSED":
        raise ApiStatusError(409, "MISSION_CLOSED", "Uppdraget är stängt.")
    return str(mission.id), mission.vehicle_id


# --- REST ---------------------------------------------------------------------------


@router.get("/skills", response_model=list[SkillSpecPublic])
async def get_skills(_: Commander) -> list[SkillSpecPublic]:
    return SKILL_REGISTRY.public()


@router.get("/missions/active", responses={204: {"description": "none"}})
async def get_active(g: G, _: Commander, vehicle_id: Annotated[str | None, Query()] = None) -> Any:
    mission = await repo_of(g).open_mission(vehicle_id or g.config.default_vehicle_id)
    return Response(status_code=204) if mission is None else {"mission_id": str(mission.id)}


@router.get("/missions/{mission_id}/situation", response_model=Situation)
async def get_situation(mission_id: IdPath, g: G, _: Commander) -> Situation:
    situation: Situation = await _flow(g).situation(mission_id)
    return situation


@router.get("/missions/{mission_id}/chat", response_model=list[ChatMessage])
async def get_chat(
    mission_id: IdPath,
    g: G,
    _: Commander,
    after_seq: Annotated[int | None, Query(ge=0)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[ChatMessage]:
    if await repo_of(g).mission(mission_id) is None:
        raise ApiStatusError(404, "MISSION_NOT_FOUND", "Uppdraget finns inte.")
    return await repo_of(g).chat(mission_id, after_seq, None, limit)


@router.post(
    "/missions/{mission_id}/messages",
    status_code=201,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/BrainMessage"}}
            },
            "required": True,
        }
    },
)
async def post_message(mission_id: IdPath, request: Request, g: G, _: Commander) -> dict[str, str]:
    body = _validate(BrainMessage, await _body(request))
    mid, vehicle_id = await _open_mission(g, mission_id)
    repo = repo_of(g)
    if body.in_reply_to is not None and not await repo.chat_message_exists(
        mid, str(body.in_reply_to)
    ):
        raise ApiStatusError(
            422, "REPLY_TARGET_UNKNOWN", "Meddelandet som svaret gäller finns inte."
        )
    try:
        msg, created = await repo.add_chat(
            mission_id=mid,
            vehicle_id=vehicle_id,
            msg_id=str(body.client_msg_id),
            role="brain",
            author_ref=BRAIN_ISSUER,
            kind=body.kind,
            text_=body.text,
            in_reply_to=None if body.in_reply_to is None else str(body.in_reply_to),
            refs=body.refs,
        )
    except LookupError:
        raise ApiStatusError(409, "ID_CONFLICT", "Meddelandets id används redan.") from None
    if created:
        g.bus.publish(
            "chat",
            "chat.message",
            msg.model_dump(mode="json"),
            ts=g.now(),
            vehicle_id=vehicle_id,
            mission_id=mid,
        )
    return {"message_id": str(msg.id)}


@router.post(
    "/missions/{mission_id}/commands",
    status_code=201,
    response_model=SkillSubmitResult,
    responses={200: {"description": "the same command_id again: the first answer"}},
)
async def post_command(
    mission_id: IdPath, body: SkillRequest, response: Response, g: G, _: Commander
) -> SkillSubmitResult:
    """Every order of the AI goes through the ground's gate (201 for every gate outcome)."""
    done = await commands_service.submit(
        _flow(g),
        mission_id,
        command_id=str(body.command_id),
        skill=body.skill,
        params=body.params,
        issuer_kind="brain",
        issuer_ref=BRAIN_ISSUER,
        rationale_sv=clean_text(body.rationale_sv),
        in_reply_to=None if body.in_reply_to is None else str(body.in_reply_to),
        based_on=body.based_on,
        charter_version=body.charter_version,
        backend=body.backend,
        ai_config_hash=body.ai_config_hash,
    )
    if not done.created:
        response.status_code = 200
    return done.result()


@router.post("/missions/{mission_id}/estimate", response_model=EstimateResult)
async def post_estimate(
    mission_id: IdPath, body: SkillRequest, g: G, _: Commander
) -> EstimateResult:
    return await commands_service.estimate(
        _flow(g),
        mission_id,
        skill=body.skill,
        params=body.params,
        based_on=body.based_on,
        charter_version=body.charter_version,
    )


@router.post("/missions/{mission_id}/commands/{command_id}/cancel", status_code=202)
async def post_cancel(mission_id: IdPath, command_id: IdPath, g: G, _: Commander) -> dict[str, Any]:
    """Only orders the AI itself gave (404 otherwise)."""
    cmd = await commands_service.cancel(
        _flow(g), command_id, brain_only=True, mission_id=str(UUID(mission_id))
    )
    return {"command_id": str(cmd.id), "status": str(cmd.status)}


_FORBIDDEN_PROPOSAL_FIELDS = {
    "autonomy_mode": ("AUTONOMY_CHANGE_FORBIDDEN", "AI:n kan inte ändra autonomiläget."),
    "limits": ("LIMITS_CHANGE_FORBIDDEN", "AI:n kan inte ändra uppdragets gränser."),
}


@router.post(
    "/missions/{mission_id}/charter-proposals",
    status_code=201,
    response_model=CharterProposalResult,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/CharterProposal"}}
            },
            "required": True,
        }
    },
)
async def post_charter_proposal(mission_id: IdPath, request: Request, g: G, _: Commander) -> Any:
    """A proposal only: the pilot's click on the card is the only way it is approved."""
    raw = await _body(request)
    if isinstance(raw, dict):
        for key, (code, text) in _FORBIDDEN_PROPOSAL_FIELDS.items():
            if key in raw:
                raise ApiStatusError(422, code, text)
    body = _validate(CharterProposal, raw)
    try:
        charter, card = await approvals_service.propose(_flow(g), mission_id, body, by=BRAIN_ISSUER)
    except charter_rules.CharterInvalidError as exc:
        result = CharterProposalResult(
            charter_id=None,
            approval_id=None,
            validation=CharterValidationResult(ok=False, errors=exc.errors),
        )
        return JSONResponse(result.model_dump(mode="json"), status_code=422)
    return CharterProposalResult(
        charter_id=charter.id,
        approval_id=card.id,
        validation=CharterValidationResult(ok=True, errors=[]),
    )


@router.post(
    "/missions/{mission_id}/chat-intents", status_code=202, response_model=ActionDispatched
)
async def post_chat_intent(
    mission_id: IdPath, body: ChatIntentReport, g: G, _: Commander
) -> ActionDispatched:
    """Only for a pilot's own line in this mission, at most 10 s old; the pilot is the
    line's author. A model's reading only ever stops the vehicle where it is; the exact
    words act like the button. Never a release."""
    return await chat_service.report_intent(
        _flow(g),
        mission_id,
        message_id=str(body.message_id),
        intent=body.intent,
        engine=body.engine,
        probability=body.probability,
    )


@router.get("/missions/{mission_id}/memory", response_model=MissionMemory)
async def get_memory(mission_id: IdPath, g: G, _: Commander) -> MissionMemory:
    if await repo_of(g).mission(mission_id) is None:
        raise ApiStatusError(404, "MISSION_NOT_FOUND", "Uppdraget finns inte.")
    stored = await repo_of(g).memory(mission_id)
    if stored is not None:
        return stored
    return MissionMemory(
        mission_id=UUID(mission_id),
        summary_sv="",
        plan_note="",
        summarized_up_to_seq=0,
        updated_at=None,
    )


@router.put(
    "/missions/{mission_id}/memory",
    status_code=204,
    response_class=Response,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/MissionMemoryUpdate"}}
            },
            "required": True,
        }
    },
)
async def put_memory(mission_id: IdPath, request: Request, g: G, _: Commander) -> Response:
    body = _validate(MissionMemoryUpdate, await _body(request))
    mid, _vehicle = await _open_mission(g, mission_id)
    if str(body.mission_id) != mid:
        raise ApiStatusError(422, "MISSION_MISMATCH", "mission_id stämmer inte med adressen.")
    await repo_of(g).put_memory(mid, body.model_dump(mode="json"), g.now())
    return Response(status_code=204)


def _prompt_path(g: Ground, mission_id: IdPath, turn_id: str) -> tuple[Path, str]:
    base = (g.loaded.data_dir / "missions" / mission_id / "brain").resolve()
    rel = f"brain/{turn_id}.json"
    target = (base / f"{turn_id}.json").resolve()
    if not target.is_relative_to(base):
        raise ApiStatusError(422, "VALIDATION", "Ogiltigt turn_id.")
    return target, rel


@router.post(
    "/missions/{mission_id}/decisions",
    status_code=204,
    response_class=Response,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/DecisionLog"}}
            },
            "required": True,
        }
    },
)
async def post_decision(mission_id: IdPath, request: Request, g: G, _: Commander) -> Response:
    raw = await _body(request)
    if isinstance(raw, dict) and raw.get("prompt_json") is not None:  # pyright: ignore[reportUnknownMemberType]
        size = len(json.dumps(raw["prompt_json"], ensure_ascii=False).encode())
        if size > PROMPT_JSON_MAX_BYTES:
            raise ApiStatusError(413, "PROMPT_TOO_LARGE", "Promptfilen är större än 1 MiB.")
    body = _validate(DecisionLog, raw)
    mid, vehicle_id = await _open_mission(g, mission_id)
    stored = body.model_dump(mode="json", exclude={"prompt_json"})
    stored["prompt_file"] = None
    target: Path | None = None
    if body.prompt_json is not None:
        target, stored["prompt_file"] = _prompt_path(g, mid, str(body.turn_id))
        # The audit file of an earlier turn is never rewritten.
        if target.is_symlink() or target.exists():
            raise ApiStatusError(409, "TURN_EXISTS", "Det finns redan en promptfil för turen.")
    created = await repo_of(g).add_decision(
        decision_id=str(body.id),
        mission_id=mid,
        vehicle_id=vehicle_id,
        turn_id=str(body.turn_id),
        body=stored,
    )
    if created and target is not None and body.prompt_json is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            raise ApiStatusError(
                409, "TURN_EXISTS", "Det finns redan en promptfil för turen."
            ) from None
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(body.prompt_json, ensure_ascii=False))
    return Response(status_code=204)


# --- WebSocket ----------------------------------------------------------------------


def to_commander(event: BusEvent) -> dict[str, Any] | None:
    """Map a live event to the commander's stream, or ``None`` when the AI does not get it."""
    d, kind = event.data, None
    body: dict[str, Any] = {}
    if event.type == "chat.message" and d.get("role") == "operator":
        kind = "event.operator_message"
        body = {
            "message_id": d.get("id"),
            "seq": d.get("seq"),
            "text": d.get("text"),
            "author": d.get("author_ref"),
            "annotation_ids": d.get("annotation_ids", []),
            "parsed_intent": d.get("parsed_intent"),
            "handled_by": d.get("handled_by"),
            "delivery": d.get("delivery", "delivered"),
            "created_at": d.get("created_at"),
        }
    elif event.type == "vehicle.action" and d.get("action") == "resume":
        # The pilot's Resume wakes the AI as an override ``resume`` (the latch event only
        # says "no latch", which a landing says too). Each outcome update is the same press.
        if d.get("result") != "dispatched":
            return None
        kind = "event.operator_override"
        body = {"active": False, "action": "resume", "by": d.get("by"), "via": d.get("via")}
    elif event.type == "vehicle.action":
        kind = "event.vehicle_action"
        body = {k: d.get(k) for k in ("action_id", "action", "via", "by", "result")}
    elif event.type in ("approval.requested", "approval.updated"):
        kind = (
            "event.approval_requested" if event.type == "approval.requested" else "event.approval"
        )
        body = {
            "approval_id": d.get("id"),
            "kind": d.get("kind"),
            "subject_id": d.get("subject_id"),
            "summary_sv": d.get("summary_sv"),
            "expires_at": d.get("expires_at"),
            "result": d.get("status"),
        }
    elif event.type == "command.updated" and d.get("issuer_kind") == "brain":
        kind = "event.skill"
        body = {
            k: d.get(k)
            for k in ("command_id", "skill", "status", "reason_codes", "message_sv", "details")
        }
        body["command_id"] = d.get("id")
    elif event.type == "charter.updated":
        kind = "event.charter"
        body = {
            "charter_id": d.get("id"),
            "version": d.get("version"),
            "status": d.get("status"),
            "install_status": d.get("install_status"),
        }
    elif event.type == "clearance.updated":
        kind = "event.takeoff_clearance"
        body = {k: d.get(k) for k in ("status", "confirmed_by", "valid_until")}
    elif event.type == "latch.updated" and not d.get("active", True):
        return None  # a lifted latch: the pilot's Resume (above) or a landing says why
    elif event.type == "latch.updated":
        kind = "event.operator_override"
        body = {
            "active": d.get("active", True),
            "action": d.get("action"),
            "by": d.get("set_by"),
            "via": d.get("via", "button"),
        }
    elif event.type == "alert":
        kind = "event.alert"
        body = {"code": d.get("code"), "level": d.get("severity"), "text_sv": d.get("message_sv")}
    elif event.type == "mission.state_changed":
        kind = "event.mission"
        body = {
            # A new mission (the panel opened one): sd-brain follows it from here on.
            "created": bool(d.get("created", False)),
            "status": d.get("to"),
            "previous": d.get("from"),
            "paused_by": d.get("paused_by"),
            "returning_cause": d.get("returning_cause"),
        }
    elif event.type == "annotation.created":
        geometry = d.get("geometry") or {}
        if d.get("kind") == "area":
            kind = "event.area_drawn"
            body = {"area_id": d.get("id"), "label": d.get("label"), "area_m2": d.get("area_m2")}
        elif d.get("kind") == "path":
            kind = "event.path_drawn"
            body = {"path_id": d.get("label"), "length_m": d.get("length_m")}
        else:
            kind = "event.pin_set"
            coords = geometry.get("coordinates") or [None, None]
            body = {"pin_id": d.get("label"), "lat": coords[1], "lon": coords[0]}
    if kind is None:
        return None
    return {
        "type": kind,
        "mission_id": event.mission_id,
        "seq": event.seq,
        "ts": event.ts.isoformat().replace("+00:00", "Z"),
        "body": body,
    }


class CommanderHub:
    """At most one connected commander. An unauthorised socket never takes the place."""

    def __init__(self, g: Ground) -> None:
        self.g = g
        self.connected: WebSocket | None = None
        self.last_heartbeat: float | None = None

    def _set_state(self, state: str, engine: str | None = None) -> None:
        """``engine`` (cloud model or command mode) comes with the brain's heartbeat;
        it is cleared when the brain goes."""
        new_engine: Literal["cloud", "command"] | None = None
        if state != "down":
            new_engine = self.g.brain_engine
            if engine == "cloud" or engine == "command":
                new_engine = engine
        if state == self.g.brain_state and new_engine == self.g.brain_engine:
            return
        previous = self.g.brain_state
        self.g.brain_state = state
        self.g.brain_engine = new_engine
        self.g.bus.publish(
            "chat",
            "chat.brain_status",
            {"state": state, "typing": False, "engine": new_engine},
            ts=self.g.now(),
        )
        if state == "down" and previous != "down":
            self.g.add_event("brain.disconnected", severity="warning")
            self.g.bus.publish(
                "alerts",
                "alert",
                {
                    "id": None,
                    "severity": "warning",
                    "code": "BRAIN_DISCONNECTED",
                    "message_sv": "AI-befälhavaren svarar inte. STOPP och FLYG HEM fungerar.",
                    "data": {},
                },
                ts=self.g.now(),
            )
        elif previous == "down":
            self.g.add_event("brain.connected")

    async def serve(self, ws: WebSocket) -> None:
        g = self.g
        scheme, _, token = ws.headers.get("authorization", "").partition(" ")
        await ws.accept()
        if scheme.lower() == "bearer" and g.search_token.matches(token.strip()):
            await ws.close(4403)
            return
        if scheme.lower() != "bearer" or not g.brain_token.matches(token.strip()):
            await ws.close(4401)
            return
        if self.connected is not None:
            await ws.close(4409)
            return
        self.connected = ws
        self.last_heartbeat = g.clock.monotonic()
        sub = g.bus.subscribe()
        try:
            await ws.send_json(
                {
                    "type": "hello",
                    "seq": None,
                    "body": {"server_epoch": g.bus.server_epoch, "version": g.version},
                }
            )
            await self._session(ws, sub)
        finally:
            g.bus.unsubscribe(sub)
            self.connected = None
            self._set_state("down")

    async def _session(self, ws: WebSocket, sub: Any) -> None:
        g = self.g

        async def reader() -> None:
            while True:
                msg: dict[str, Any] = await ws.receive_json()
                kind = msg.get("type") or msg.get("op")
                if kind in ("brain.hello", "subscribe"):
                    body: dict[str, Any] = msg.get("body", msg) or {}
                    await ws.send_json(
                        {
                            "type": "server.hello",
                            "seq": g.bus.last_seq,
                            "body": {
                                "server_epoch": g.bus.server_epoch,
                                "seq": g.bus.last_seq,
                                "profile": g.config.profile,
                            },
                        }
                    )
                    replay = g.bus.replay(body.get("server_epoch"), body.get("since_seq"))
                    if replay is None:
                        await ws.send_json(
                            {
                                "type": "resync_required",
                                "seq": None,
                                "body": {"server_epoch": g.bus.server_epoch},
                            }
                        )
                    else:
                        for event in replay:
                            out = to_commander(event)
                            if out is not None:
                                await ws.send_json(out)
                    self.last_heartbeat = g.clock.monotonic()
                    self._set_state("ready")
                elif kind == "brain.heartbeat":
                    self.last_heartbeat = g.clock.monotonic()
                    beat: dict[str, Any] = msg.get("body") or {}
                    status = str(beat.get("status", "ready"))
                    self._set_state(
                        status if status in ("ready", "busy", "degraded") else "ready",
                        engine=beat.get("engine"),
                    )
                elif kind == "brain.typing":
                    body = msg.get("body") or {}
                    g.bus.publish(
                        "chat",
                        "chat.brain_status",
                        {
                            "state": g.brain_state,
                            "typing": bool(body.get("active")),
                            "engine": g.brain_engine,
                        },
                        ts=g.now(),
                        mission_id=_uuid_or_none(body.get("mission_id")),
                    )
                elif kind == "brain.pong":
                    self.last_heartbeat = g.clock.monotonic()
                elif kind == "brain.bye":
                    return

        async def writer() -> None:
            while True:
                event = await sub.queue.get()
                if event is None:
                    await ws.close(1001)
                    return
                out = to_commander(event)
                if out is not None:
                    await ws.send_json(out)

        async def pinger() -> None:
            while True:
                await g.clock.sleep(BRAIN_PING_S)
                await ws.send_json({"type": "brain.ping", "seq": None, "body": {}})

        async def watchdog() -> None:
            while True:
                await g.clock.sleep(0.5)
                last = self.last_heartbeat or 0.0
                if g.clock.monotonic() - last > BRAIN_HEARTBEAT_TIMEOUT_S:
                    self._set_state("down")

        await until_first_done(reader, writer, pinger, watchdog)
        with contextlib.suppress(Exception):
            await ws.close(1000)


@router.websocket("/ws")
async def commander_ws(ws: WebSocket) -> None:
    hub = ws.app.state.commander_hub
    if not isinstance(hub, CommanderHub):
        raise TypeError("app.state.commander_hub is not set")
    await hub.serve(ws)
