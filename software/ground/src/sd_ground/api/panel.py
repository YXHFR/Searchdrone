"""The panel API ``/api/v1`` (listener :8080, and :8443 in LAN mode).

Every route checks the caller's role. Routes whose logic another part of the ground station
owns are already here, with their final paths, bodies and role checks, and answer
501 ``NOT_BUILT`` until that part lands:

- STOP / RETURN HOME run in the ingress thread (``sd_ground.ingress``) and call the
  emergency path (``services.vehicle_actions``); Resume, takeover and hand-back and the
  action outcomes are answered here from the same service.
- The mission flow (open, limits, autonomy, clearance, charters, approvals, orders, chat
  posts, map pins, sectors, track, files) comes with the mission services.
- Video comes with its own part. The offline map is ``sd_ground.api.tiles``.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Path, Query, Request, Response

from sd_ground import emergency
from sd_ground.api import tiles
from sd_ground.api.deps import (
    COOKIE,
    ApiStatusError,
    ground,
    not_built,
    repo_of,
    require,
    require_headroom,
    require_interactive,
    session_of,
)
from sd_ground.auth import LoginError, Principal, hash_token
from sd_ground.config import limit_caps
from sd_ground.context import Ground
from sd_ground.emergency import EmergencyRefusedError
from sd_ground.logging import get_logger
from sd_ground.services import health
from sd_ground.services.vehicle_actions import ActionRefusedError, VehicleActions
from sd_ground.sources import EmergencyUnavailableError
from sd_schemas.api import (
    AbsoluteLimits,
    ActionDispatched,
    ActionDispatchedAll,
    ActionOutcome,
    ActionRequest,
    AiToggleRequest,
    Annotation,
    AnnotationCreate,
    Approval,
    ApprovalDecisionRequest,
    AutonomyRequest,
    Charter,
    ChatMessage,
    ChatPost,
    Clearance,
    Command,
    CommandResult,
    Event,
    Features,
    Health,
    LoginRequest,
    Mission,
    MissionDetail,
    MissionEndRequest,
    MissionFile,
    MissionPatch,
    MissionSummary,
    OperatorCommandRequest,
    PanelConfig,
    Session,
    SystemStatus,
    Vehicle,
    VehicleLive,
    VideoConfig,
)
from sd_schemas.charter import CharterProposal
from sd_schemas.gate import GateDecision
from sd_schemas.safety import SafetyLimits

__all__ = ["OPERATOR_ALIAS", "dispatch_emergency", "router"]

log = get_logger("sd_ground.panel")
OPERATOR_ALIAS = "demo"  # the single operator of v0.1.0 (sim: operator:demo)

router = APIRouter(prefix="/api/v1")

_UUID = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
IdPath = Annotated[str, Path(pattern=_UUID)]
VehiclePath = Annotated[str, Path(pattern=r"^[a-z0-9][a-z0-9-]{2,31}$")]
Observer = Annotated[Principal, Depends(require("observer"))]
Pilot = Annotated[Principal, Depends(require("pilot"))]
Interactive = Annotated[Principal, Depends(require_interactive)]
G = Annotated[Ground, Depends(ground)]

MISSION_FLOW = "B1-12"
EMERGENCY = "B1-08"


async def _mission_or_404(g: Ground, mission_id: IdPath) -> Mission:
    mission = await repo_of(g).mission(mission_id)
    if mission is None:
        raise ApiStatusError(404, "MISSION_NOT_FOUND", "Uppdraget finns inte.")
    return mission


# --- system and session -------------------------------------------------------------


@router.get("/health", response_model=Health)
async def get_health(g: G) -> Health:
    return health.health(g)


@router.get("/system/status", response_model=SystemStatus)
async def get_system_status(g: G, _: Observer) -> SystemStatus:
    return health.system_status(g)


@router.get("/panel/config", response_model=PanelConfig)
async def get_panel_config(g: G, _: Observer) -> PanelConfig:
    vid = g.config.default_vehicle_id
    caps = limit_caps(g.loaded)
    return PanelConfig(
        default_vehicle_id=vid,
        map=tiles.map_config(g),
        video=VideoConfig(whep_url=f"/api/v1/video/{vid}/main/whep", hls_url=None),
        vehicle_kind="sim",
        limits_default=SafetyLimits(),
        limits_absolute=AbsoluteLimits(
            **caps.model_dump(), max_area_km2=g.config.charter.max_area_km2
        ),
        features=Features(),
        remote=False,
    )


def _session_model(g: Ground, token: str) -> Session:
    rec = g.sessions.lookup(token, touch=False)
    if rec is None:
        raise ApiStatusError(401, "UNAUTHENTICATED", "Logga in först.")
    return Session(
        role=rec.role,
        issuer_ref=rec.issuer_ref,
        display_name=rec.issuer_ref.split(":", 1)[-1],
        auth_mode=rec.auth_mode,
        expires_at=g.sessions.expires_at(rec),
    )


@router.post("/session", response_model=Session)
async def post_session(body: LoginRequest, request: Request, response: Response, g: G) -> Session:
    client = request.client.host if request.client else "unknown"
    try:
        mode = g.logins.redeem(client, login_code=body.login_code, pin=body.pin)
    except LoginError as exc:
        raise ApiStatusError(exc.status, exc.code, exc.detail_sv) from None
    token, rec = g.sessions.create_session(
        role="pilot", issuer_ref=f"operator:{OPERATOR_ALIAS}", auth_mode=mode
    )
    if g.repo is not None and g.db_ok:
        try:
            await g.repo.save_session(rec)
            g.sessions.mark_persisted(rec.token_sha256)
        except Exception as exc:  # kept in memory; saved when the database is back
            log.warning("session.persist_deferred", error=type(exc).__name__)
    response.set_cookie(
        COOKIE,
        token,
        # No Max-Age: a browser-session cookie. The server alone decides when the session
        # ends (12 h idle, 7 days, never during a flight), so the browser cannot drop it
        # mid-flight.
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
        path="/",
    )
    return _session_model(g, token)


@router.get("/session", response_model=Session)
async def get_session(request: Request, g: G) -> Session:
    rec = session_of(request)
    if rec is None:
        raise ApiStatusError(401, "UNAUTHENTICATED", "Logga in först.")
    return Session(
        role=rec.role,
        issuer_ref=rec.issuer_ref,
        display_name=rec.issuer_ref.split(":", 1)[-1],
        auth_mode=rec.auth_mode,
        expires_at=g.sessions.expires_at(rec),
    )


@router.post("/session/logout", status_code=204, response_class=Response)
async def post_logout(request: Request, g: G, _: Observer) -> Response:
    token = request.cookies.get(COOKIE)
    if token:
        sha = hash_token(token)
        g.sessions.revoke(sha)
        if g.repo is not None and g.db_ok:
            try:
                await g.repo.revoke_session(sha)
                g.sessions.revocation_saved(sha)
            except Exception as exc:  # saved by the next database check
                log.warning("session.revoke_deferred", error=type(exc).__name__)
    response = Response(status_code=204)
    response.delete_cookie(COOKIE, path="/", httponly=True, samesite="strict")
    return response


# --- vehicles and the emergency path -------------------------------------------------


@router.get("/vehicles", response_model=list[Vehicle])
async def get_vehicles(g: G, _: Observer) -> list[Vehicle]:
    rows = await repo_of(g).vehicles()
    return [g.repo.vehicle_model(r, connected=g.vehicle_link.connected(r["id"])) for r in rows]  # type: ignore[union-attr]


@router.get("/vehicles/{vehicle_id}/state", response_model=VehicleLive)
async def get_vehicle_state(vehicle_id: VehiclePath, g: G, _: Observer) -> VehicleLive:
    return await health.vehicle_live(g, vehicle_id)


def dispatch_emergency(
    g: Ground,
    action: Literal["stop", "return_home"],
    vehicle_id: str | None,
    body: ActionRequest,
    who: Principal,
) -> Any:
    """In production the ingress thread answers these routes (``sd_ground.ingress``).

    The routes here document them in OpenAPI and, if ever reached, use the same rule and
    dispatch (``sd_ground.emergency``), so the two cannot differ.
    """
    try:
        if vehicle_id is None:
            return emergency.dispatch_all(g.emergency, action, str(body.client_action_id), who)
        return emergency.dispatch(g.emergency, action, vehicle_id, str(body.client_action_id), who)
    except (EmergencyRefusedError, EmergencyUnavailableError) as exc:
        raise ApiStatusError(exc.status, exc.code, exc.detail_sv) from None


@router.post(
    "/vehicles/{vehicle_id}/actions/stop", status_code=202, response_model=ActionDispatched
)
async def post_stop(
    vehicle_id: VehiclePath, body: ActionRequest, g: G, who: Observer
) -> ActionDispatched:
    """Every logged-in user may press STOP (observers too); RETURN HOME is for the pilot."""
    return dispatch_emergency(g, "stop", vehicle_id, body, who)


@router.post(
    "/vehicles/{vehicle_id}/actions/return_home", status_code=202, response_model=ActionDispatched
)
async def post_return_home(
    vehicle_id: VehiclePath, body: ActionRequest, g: G, who: Pilot
) -> ActionDispatched:
    return dispatch_emergency(g, "return_home", vehicle_id, body, who)


def _actions(g: Ground) -> VehicleActions:
    if g.vehicle_actions is None:
        raise not_built(EMERGENCY)
    return g.vehicle_actions


def _pilot_action(
    g: Ground,
    action: Literal["resume", "takeover", "handback"],
    vehicle_id: str,
    body: ActionRequest,
    who: Principal,
) -> ActionDispatched:
    actions = _actions(g)
    run = {"resume": actions.resume, "takeover": actions.takeover, "handback": actions.handback}
    try:
        return run[action](vehicle_id, client_action_id=str(body.client_action_id), by=who)
    except ActionRefusedError as exc:
        raise ApiStatusError(exc.status, exc.code, exc.detail_sv) from None


@router.post(
    "/vehicles/{vehicle_id}/actions/resume", status_code=202, response_model=ActionDispatched
)
async def post_resume(
    vehicle_id: VehiclePath, body: ActionRequest, g: G, who: Pilot
) -> ActionDispatched:
    """The pilot's Resume: lifts the latch and tells the onboard computer. Never MAVLink."""
    return _pilot_action(g, "resume", vehicle_id, body, who)


@router.post(
    "/vehicles/{vehicle_id}/actions/takeover", status_code=202, response_model=ActionDispatched
)
async def post_takeover(
    vehicle_id: VehiclePath, body: ActionRequest, g: G, who: Pilot
) -> ActionDispatched:
    """Ta över: LOITER, the AI's orders pause, the sticks go from the ground station."""
    return _pilot_action(g, "takeover", vehicle_id, body, who)


@router.post(
    "/vehicles/{vehicle_id}/actions/handback", status_code=202, response_model=ActionDispatched
)
async def post_handback(
    vehicle_id: VehiclePath, body: ActionRequest, g: G, who: Pilot
) -> ActionDispatched:
    """Lämna tillbaka: no more sticks."""
    return _pilot_action(g, "handback", vehicle_id, body, who)


@router.get("/vehicles/{vehicle_id}/actions/{action_id}", response_model=ActionOutcome)
async def get_action(
    vehicle_id: VehiclePath, action_id: IdPath, g: G, _: Observer
) -> ActionOutcome:
    found = _actions(g).outcome(vehicle_id, action_id)
    if found is None:
        raise ApiStatusError(404, "ACTION_NOT_FOUND", "Åtgärden finns inte (eller är för gammal).")
    return found


@router.post("/actions/stop_all", status_code=202, response_model=ActionDispatchedAll)
async def post_stop_all(body: ActionRequest, g: G, who: Observer) -> ActionDispatchedAll:
    return dispatch_emergency(g, "stop", None, body, who)


@router.post("/actions/return_home_all", status_code=202, response_model=ActionDispatchedAll)
async def post_return_home_all(body: ActionRequest, g: G, who: Pilot) -> ActionDispatchedAll:
    return dispatch_emergency(g, "return_home", None, body, who)


# --- missions -----------------------------------------------------------------------


@router.post("/vehicles/{vehicle_id}/missions/open", response_model=Mission)
async def post_mission_open(
    vehicle_id: VehiclePath, _: Pilot, body: dict[str, Any] | None = None
) -> Mission:
    raise not_built(MISSION_FLOW)


@router.get("/missions/active", response_model=Mission, responses={204: {"description": "none"}})
async def get_mission_active(
    g: G, _: Observer, vehicle_id: Annotated[str | None, Query()] = None
) -> Mission | Response:
    mission = await repo_of(g).open_mission(vehicle_id or g.config.default_vehicle_id)
    return Response(status_code=204) if mission is None else mission


@router.get("/missions", response_model=list[MissionSummary])
async def get_missions(
    g: G,
    _: Observer,
    status: Annotated[str | None, Query(pattern=r"^[A-Z_]{3,20}$")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[MissionSummary]:
    return await repo_of(g).missions(status, limit)


@router.get("/missions/{mission_id}", response_model=MissionDetail)
async def get_mission(mission_id: IdPath, _: Observer) -> MissionDetail:
    raise not_built(MISSION_FLOW)


@router.patch("/missions/{mission_id}", response_model=Mission)
async def patch_mission(mission_id: IdPath, body: MissionPatch, g: G, who: Pilot) -> Mission:
    mission = await _mission_or_404(g, mission_id)
    await repo_of(g).patch_mission(mission_id, body.title, body.notes)
    g.add_event(
        "mission.updated",
        mission_id=mission_id,
        vehicle_id=mission.vehicle_id,
        payload={"by": who.issuer_ref, "fields": sorted(body.model_dump(exclude_none=True))},
    )
    return await _mission_or_404(g, mission_id)


@router.put("/missions/{mission_id}/limits", response_model=Mission)
async def put_limits(mission_id: IdPath, body: SafetyLimits, _: Pilot) -> Mission:
    raise not_built(MISSION_FLOW)


@router.put("/missions/{mission_id}/autonomy", response_model=Mission)
async def put_autonomy(mission_id: IdPath, body: AutonomyRequest, _: Pilot) -> Mission:
    raise not_built(MISSION_FLOW)


@router.put("/missions/{mission_id}/ai", response_model=Mission)
async def put_ai(mission_id: IdPath, body: AiToggleRequest, g: G, who: Pilot) -> Mission:
    mission = await _mission_or_404(g, mission_id)
    if not await repo_of(g).set_ai_enabled(mission_id, body.enabled):
        raise ApiStatusError(409, "MISSION_CLOSED", "Uppdraget är stängt.")
    g.add_event(
        "mission.ai_toggled",
        mission_id=mission_id,
        vehicle_id=mission.vehicle_id,
        payload={"enabled": body.enabled, "by": who.issuer_ref},
    )
    updated = await _mission_or_404(g, mission_id)
    g.bus.publish(
        "mission",
        "mission.updated",
        updated.model_dump(mode="json"),
        ts=g.now(),
        vehicle_id=updated.vehicle_id,
        mission_id=mission_id,
    )
    return updated


@router.post(
    "/missions/{mission_id}/clearance",
    response_model=Clearance,
    dependencies=[Depends(require_headroom)],
)
async def post_clearance(
    mission_id: IdPath, _: Interactive, body: dict[str, Any] | None = None
) -> Clearance:
    raise not_built(MISSION_FLOW)


@router.post("/missions/{mission_id}/end", response_model=Mission)
async def post_end(mission_id: IdPath, body: MissionEndRequest, _: Pilot) -> Mission:
    raise not_built(MISSION_FLOW)


# --- charters, sectors, approvals ---------------------------------------------------


@router.get("/missions/{mission_id}/charters", response_model=list[Charter])
async def get_charters(mission_id: IdPath, _: Observer) -> list[Charter]:
    raise not_built(MISSION_FLOW)


@router.post("/missions/{mission_id}/charters", status_code=201, response_model=Charter)
async def post_charter(mission_id: IdPath, body: CharterProposal, _: Pilot) -> Charter:
    raise not_built(MISSION_FLOW)


@router.get("/missions/{mission_id}/sectors")
async def get_sectors(mission_id: IdPath, _: Observer) -> dict[str, Any]:
    raise not_built(MISSION_FLOW)


@router.get("/missions/{mission_id}/approvals", response_model=list[Approval])
async def get_approvals(
    mission_id: IdPath, _: Observer, status: Annotated[str | None, Query()] = None
) -> list[Approval]:
    raise not_built(MISSION_FLOW)


@router.post(
    "/approvals/{approval_id}/decision",
    response_model=Approval,
    dependencies=[Depends(require_headroom)],
)
async def post_decision(
    approval_id: IdPath, body: ApprovalDecisionRequest, _: Interactive
) -> Approval:
    raise not_built(MISSION_FLOW)


# --- orders -------------------------------------------------------------------------


@router.get("/missions/{mission_id}/commands", response_model=list[Command])
async def get_commands(
    mission_id: IdPath, _: Observer, status: Annotated[str | None, Query()] = None
) -> list[Command]:
    raise not_built(MISSION_FLOW)


@router.post("/missions/{mission_id}/commands", status_code=201, response_model=CommandResult)
async def post_command(
    mission_id: IdPath, body: OperatorCommandRequest, request: Request, _: Pilot
) -> CommandResult:
    if body.skill == "takeoff":
        require_headroom(request)
    raise not_built(MISSION_FLOW)


@router.post("/commands/{command_id}/cancel", status_code=202, response_model=Command)
async def post_cancel(command_id: IdPath, _: Pilot, body: dict[str, Any] | None = None) -> Command:
    raise not_built(MISSION_FLOW)


# --- chat and map -------------------------------------------------------------------


@router.get("/missions/{mission_id}/chat")
async def get_chat(
    mission_id: IdPath,
    g: G,
    _: Observer,
    after_seq: Annotated[int | None, Query(ge=0)] = None,
    before_seq: Annotated[int | None, Query(ge=0)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    await _mission_or_404(g, mission_id)
    items = await repo_of(g).chat(mission_id, after_seq, before_seq, limit)
    next_before = items[0].seq if len(items) == limit else None
    return {"items": [i.model_dump(mode="json") for i in items], "next_before": next_before}


@router.post("/missions/{mission_id}/chat", status_code=201, response_model=ChatMessage)
async def post_chat(mission_id: IdPath, body: ChatPost, _: Pilot) -> ChatMessage:
    raise not_built(MISSION_FLOW)


@router.get("/missions/{mission_id}/annotations", response_model=list[Annotation])
async def get_annotations(mission_id: IdPath, g: G, _: Observer) -> list[Annotation]:
    await _mission_or_404(g, mission_id)
    return await repo_of(g).annotations(mission_id)


@router.post("/missions/{mission_id}/annotations", status_code=201, response_model=Annotation)
async def post_annotation(mission_id: IdPath, body: AnnotationCreate, _: Pilot) -> Annotation:
    raise not_built(MISSION_FLOW)


@router.delete("/annotations/{annotation_id}", status_code=204, response_class=Response)
async def delete_annotation(annotation_id: IdPath, _: Pilot) -> Response:
    raise not_built(MISSION_FLOW)


# --- data, files, video, stubs ------------------------------------------------------


@router.get("/missions/{mission_id}/track")
async def get_track(
    mission_id: IdPath,
    _: Observer,
    since: Annotated[dt.datetime | None, Query()] = None,
    max_points: Annotated[int, Query(ge=2, le=50_000)] = 5000,
) -> dict[str, Any]:
    raise not_built(MISSION_FLOW)


@router.get("/missions/{mission_id}/events", response_model=list[Event])
async def get_events(
    mission_id: IdPath,
    g: G,
    _: Observer,
    after: Annotated[dt.datetime | None, Query()] = None,
    type: Annotated[str | None, Query(max_length=64)] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> list[Event]:
    await _mission_or_404(g, mission_id)
    return await repo_of(g).events(mission_id, after, type, limit)


@router.get("/missions/{mission_id}/gate_decisions", response_model=list[GateDecision])
async def get_gate_decisions(
    mission_id: IdPath, _: Observer, command_id: Annotated[str | None, Query()] = None
) -> list[GateDecision]:
    raise not_built(MISSION_FLOW)


@router.get("/missions/{mission_id}/files", response_model=list[MissionFile])
async def get_files(mission_id: IdPath, g: G, _: Observer) -> list[MissionFile]:
    await _mission_or_404(g, mission_id)
    return await repo_of(g).files(mission_id)


@router.get("/missions/{mission_id}/files/{file_id}")
async def get_file(mission_id: IdPath, file_id: IdPath, _: Observer) -> Response:
    raise not_built(MISSION_FLOW)


@router.get("/missions/{mission_id}/candidates")
async def get_candidates(mission_id: IdPath, _: Observer) -> dict[str, list[Any]]:
    return {"items": []}  # stage 2


@router.get("/missions/{mission_id}/detections")
async def get_detections(mission_id: IdPath, _: Observer) -> list[Any]:
    return []  # stage 2


@router.post("/video/{vehicle_id}/main/whep")
async def whep_post(vehicle_id: VehiclePath, _: Observer) -> Response:
    raise not_built("B1-09")


@router.patch("/video/{vehicle_id}/main/whep")
async def whep_patch(vehicle_id: VehiclePath, _: Observer) -> Response:
    raise not_built("B1-09")


@router.delete("/video/{vehicle_id}/main/whep")
async def whep_delete(vehicle_id: VehiclePath, _: Observer) -> Response:
    raise not_built("B1-09")


@router.get("/video/{vehicle_id}/main/hls/{path:path}")
async def hls(vehicle_id: VehiclePath, path: str, _: Observer) -> Response:
    raise not_built("B1-09")
