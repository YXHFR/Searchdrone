"""The panel API (``/api/v1``): response models, request bodies and live-event payloads."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, Field, model_validator

from sd_schemas.base import (
    PARAMS_MAX_BYTES,
    PARAMS_MAX_KEYS,
    SKILL_NAME_PATTERN,
    GeoJsonLineString,
    GeoJsonPoint,
    GeoJsonPolygon,
    Id,
    LatLon,
    SdModel,
    VehicleId,
    json_bytes_at_most,
)
from sd_schemas.charter import AutonomyMode, MissionCharter
from sd_schemas.gate import GateDecision, StateRef
from sd_schemas.link import StateSnapshot, StickAxes
from sd_schemas.mission import MissionState, OperatorLatch, PausedBy, ReturningCause
from sd_schemas.safety import LimitCaps, SafetyLimits
from sd_schemas.sv_commands import OverrideIntent

__all__ = [
    "CHAT_MAX_CHARS",
    "PATH_LABEL_PATTERN",
    "AbsoluteLimits",
    "ActionDispatched",
    "ActionDispatchedAll",
    "ActionDispatchedVehicle",
    "ActionOutcome",
    "ActionRequest",
    "AiToggleRequest",
    "AlertCode",
    "Annotation",
    "AnnotationCreate",
    "AnnotationKind",
    "ApiError",
    "ApiErrorItem",
    "Approval",
    "ApprovalDecisionRequest",
    "Attitude",
    "AutonomyRequest",
    "Backend",
    "Charter",
    "ChatMessage",
    "ChatPost",
    "Clearance",
    "Command",
    "CommandResult",
    "CommandStatus",
    "Event",
    "Features",
    "Health",
    "LinkStatus",
    "LoginRequest",
    "ManualControlInput",
    "ManualControlState",
    "MapConfig",
    "Mission",
    "MissionDetail",
    "MissionEndRequest",
    "MissionFile",
    "MissionPatch",
    "MissionSummary",
    "OperatorCommandRequest",
    "PanelConfig",
    "PartStatus",
    "Session",
    "SystemStatus",
    "Telemetry",
    "Vehicle",
    "VehicleLive",
    "VideoConfig",
    "WindCheck",
    "WindReason",
]

CHAT_MAX_CHARS = 4000

Backend = Literal["llm", "grammar", "scripted", "rule", "cloud"]

PartState = Literal["ok", "degraded", "down"]


class AlertCode(StrEnum):
    MAVLINK_LOST = "MAVLINK_LOST"
    SDLINK_LOST = "SDLINK_LOST"
    ONBOARD_SILENT = "ONBOARD_SILENT"
    BRAIN_DISCONNECTED = "BRAIN_DISCONNECTED"
    DB_DOWN = "DB_DOWN"
    BATTERY_RTL = "BATTERY_RTL"
    GATE_MISMATCH = "GATE_MISMATCH"
    CHARTER_INSTALL_FAILED = "CHARTER_INSTALL_FAILED"
    FENCE_VERIFY_FAILED = "FENCE_VERIFY_FAILED"
    APPROVAL_EXPIRED = "APPROVAL_EXPIRED"
    STOP_PATH_FAILED = "STOP_PATH_FAILED"
    HOME_PATH_FAILED = "HOME_PATH_FAILED"
    OPERATOR_ABSENT = "OPERATOR_ABSENT"
    MAVLINK_UNKNOWN_SOURCE = "MAVLINK_UNKNOWN_SOURCE"
    MAVLINK_DUPLICATE_SOURCE = "MAVLINK_DUPLICATE_SOURCE"
    FC_SAFETY_PARAMS = "FC_SAFETY_PARAMS"
    LATCH_ADOPTED = "LATCH_ADOPTED"
    LOOP_STALL = "LOOP_STALL"
    PIN_INVALIDATED = "PIN_INVALIDATED"
    MANUAL_CONTROL_LOST = "MANUAL_CONTROL_LOST"


class CommandStatus(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    REJECTED_BY_GATE = "rejected_by_gate"
    REJECTED_BY_OPERATOR = "rejected_by_operator"
    APPROVAL_EXPIRED = "approval_expired"
    DISPATCHED = "dispatched"
    RECEIVED = "received"
    ACCEPTED = "accepted"
    REJECTED_ONBOARD = "rejected_onboard"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


# --- errors --------------------------------------------------------------------------


class ApiErrorItem(SdModel):
    loc: list[str | int] = Field(default_factory=list[str | int])
    code: str
    detail_sv: str
    field: str | None = None
    limit: float | None = None


class ApiError(SdModel):
    type: str = "about:blank"
    title: str
    status: int
    code: str  # stable, e.g. "CHARTER_NOT_APPROVED"
    detail_sv: str
    errors: list[ApiErrorItem] = Field(default_factory=list[ApiErrorItem])


# --- session and configuration -------------------------------------------------------


class Session(SdModel):
    role: Literal["pilot", "observer"]
    issuer_ref: str  # "operator:demo"
    display_name: str
    auth_mode: Literal["local", "pin", "cli", "cloudflare_access"]  # cloudflare_access reserved
    expires_at: AwareDatetime | None  # None while the session cannot expire (airborne vehicle)


class LoginRequest(SdModel):
    login_code: str | None = Field(default=None, min_length=1, max_length=200)
    pin: str | None = Field(default=None, pattern=r"^[0-9]{4,12}$")

    @model_validator(mode="after")
    def _one(self) -> LoginRequest:
        if (self.login_code is None) == (self.pin is None):
            raise ValueError("give login_code or pin")
        return self


MAP_ATTRIBUTION = "© OpenStreetMap contributors, Protomaps"


class MapConfig(SdModel):
    pack_url: str  # e.g. "/tiles/demo.pmtiles"
    center: LatLon
    zoom: float = Field(ge=0, le=22)
    maxzoom: int = Field(default=15, ge=0, le=22)  # highest zoom in the pack; MapLibre overzooms
    # Plain text (never HTML). OpenStreetMap data is ODbL, which requires this line.
    attribution: str = Field(default=MAP_ATTRIBUTION, min_length=1, max_length=200)
    assets_url: str = "/tiles/assets"  # glyphs under fonts/, sprites under sprites/v4/


class VideoConfig(SdModel):
    whep_url: str  # e.g. "/api/v1/video/sim-01/main/whep"
    hls_url: str | None


class AbsoluteLimits(LimitCaps):
    """The outer cap the pilot may raise limits to, plus the largest search area (read-only)."""

    max_area_km2: float = Field(default=4.0, gt=0)


class Features(SdModel):
    candidates: bool = False
    follow_me: bool = False
    voice: bool = False
    manual_control: bool = True
    order_buttons: bool = True


class PanelConfig(SdModel):
    default_vehicle_id: VehicleId
    map: MapConfig | None
    video: VideoConfig | None
    vehicle_kind: Literal["sim", "real"]
    limits_default: SafetyLimits
    limits_absolute: AbsoluteLimits
    features: Features
    remote: bool = False  # always False in v0.1.0


# --- vehicles ------------------------------------------------------------------------


class Vehicle(SdModel):
    id: VehicleId
    name: str
    kind: Literal["sim", "real"]
    mav_sysid: int = Field(ge=1, le=255)
    connected: bool  # flight controller heartbeat within 3 s


class Attitude(SdModel):
    """MAVLink ATTITUDE in degrees. Roll is positive right wing down, pitch positive nose
    up, yaw 0-360 clockwise from north."""

    roll_deg: float = Field(ge=-180, le=180)
    pitch_deg: float = Field(ge=-90, le=90)
    yaw_deg: float = Field(ge=0, lt=360)


WindReason = Literal["TILT_MARGIN", "ENERGY_HOME", "WIND_LIMIT"]


class WindCheck(SdModel):
    """The ground station's "too windy" warning. A warning only: nothing on board or on
    the ground acts on it, and STOP and FLY HOME are unchanged. Its thresholds are not yet
    calibrated against flight logs, so ``calibrated`` is always false for now.

    * ``TILT_MARGIN``: the mean tilt over the last 10 s in the air is within 10 degrees of
      the flight controller's lean limit (ATC_ANGLE_MAX).
    * ``ENERGY_HOME``: flying home now at RTL speed, at the measured current with a 30 %
      margin, would leave less than the flight controller's low-battery reserve.
    * ``WIND_LIMIT``: the flight controller's own wind estimate (MAVLink WIND, only sent
      when EK3_DRAG_* are set) is over the configured wind limit.
    """

    warning: bool
    reasons: list[WindReason]
    calibrated: Literal[False] = False
    tilt_avg_deg: float | None  # None until 10 s of attitude in the air
    tilt_warn_deg: float
    wind_ms: float | None  # mean of WIND over the last 60 s; None without WIND
    wind_gust_ms: float | None  # highest WIND over the last 60 s
    wind_from_deg: float | None
    wind_limit_ms: float
    wind_gust_limit_ms: float | None  # None: no gust limit at this stage
    home_mah: float | None  # estimated cost of flying home now, margin included
    left_mah: float | None
    reserve_mah: float


class Telemetry(SdModel):
    """From the ground station's own MAVLink channel."""

    t: AwareDatetime
    time_boot_ms: int
    lat: float
    lon: float
    alt_amsl_m: float
    alt_rel_home_m: float
    heading_deg: float
    groundspeed_ms: float
    climb_ms: float
    battery_pct: float | None
    battery_v: float | None
    flight_mode: str
    armed: bool
    landed_state: str
    gps_fix: int
    gps_sats: int
    home: LatLon | None
    dist_to_pilot_m: float | None
    # None when the flight controller has sent no ATTITUDE within the last second; the
    # panel then draws no horizon rather than a guessed one.
    attitude: Attitude | None = None
    # The ground station's wind warning; None until the ground has computed one.
    wind: WindCheck | None = None


class LinkStatus(SdModel):
    sdlink: Literal["up", "degraded", "lost"]
    mavlink: Literal["up", "lost"]
    onboard_heartbeat_age_ms: int | None
    rtt_ms: float | None


class ManualControlState(SdModel):
    """Whether the pilot is steering by hand (Ta över) and since when."""

    active: bool
    takeover_id: Id | None
    by: str | None  # "operator:<alias>"
    since: AwareDatetime | None
    input_age_ms: int | None  # age of the last stick value; > 1000 ms = the vehicle stands still


class VehicleLive(SdModel):
    vehicle_id: VehicleId
    telemetry: Telemetry | None
    snapshot: StateSnapshot | None
    link: LinkStatus
    latch: OperatorLatch | None
    manual: ManualControlState | None = None
    airborne: bool  # the gate's strict definition; display
    in_flight: bool  # the wide definition; drives the emergency path


class ActionRequest(SdModel):
    """Body of STOP, FLY HOME, Resume and Take over / Hand back."""

    client_action_id: Id


class ActionDispatched(SdModel):
    action_id: Id
    action_seq: int
    result: Literal["dispatched"] = "dispatched"
    paths: dict[str, Literal["queued", "unavailable"]]  # {"mavlink": ..., "sdlink": ...}


class ActionDispatchedVehicle(ActionDispatched):
    vehicle_id: VehicleId


class ActionDispatchedAll(SdModel):
    actions: list[ActionDispatchedVehicle]


class ActionOutcome(SdModel):
    action_id: Id
    vehicle_id: VehicleId
    action: Literal[
        "stop",
        "return_home",
        "resume",
        "battery_rtl",
        "watchdog_stop",
        "watchdog_rtl",
        "takeover",
        "handback",
    ]
    action_seq: int
    via: Literal["button", "chat", "cli", "system"]
    by: str  # "operator:<alias>" | "system:ground"
    result: Literal["dispatched", "acked", "already_active", "failed"]
    mode_requested: str | None  # last requested mode, e.g. "BRAKE", or "DISARM"
    mode_confirmed: str | None  # the mode HEARTBEAT showed when acked
    paths: dict[str, Literal["queued", "unavailable", "sent", "failed"]]
    detail_sv: str | None
    t: AwareDatetime

    @model_validator(mode="after")
    def _chat_only_stops(self) -> ActionOutcome:
        # Chat may trigger STOP or FLY HOME, never Resume or a takeover.
        if self.via == "chat" and self.action not in ("stop", "return_home"):
            raise ValueError("only stop and return_home can come from chat")
        return self


class ManualControlInput(StickAxes):
    """Panel -> ground (WebSocket op "manual"), 10-20 Hz while the pilot steers."""

    source: Literal["touch", "gamepad"]


# --- missions ------------------------------------------------------------------------


class Mission(SdModel):
    id: Id
    vehicle_id: VehicleId
    title: str
    status: MissionState
    paused_by: PausedBy | None
    returning_cause: ReturningCause | None
    state_seq: int
    limits: SafetyLimits
    active_charter_id: Id | None
    autonomy_mode: AutonomyMode | None  # *server*, copy of the active charter's value
    ai_enabled: bool
    created_by: str
    created_at: AwareDatetime
    activated_at: AwareDatetime | None
    landed_at: AwareDatetime | None
    closed_at: AwareDatetime | None
    closed_reason: Literal["auto", "manual"] | None
    notes: str | None


class MissionSummary(SdModel):
    id: Id
    vehicle_id: VehicleId
    title: str
    status: MissionState
    created_at: AwareDatetime


class MissionPatch(SdModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    notes: str | None = Field(default=None, max_length=2000)


class AutonomyRequest(SdModel):
    mode: AutonomyMode


class AiToggleRequest(SdModel):
    enabled: bool


class MissionEndRequest(SdModel):
    reason: str = Field(min_length=1, max_length=500)


class Charter(SdModel):
    id: Id  # = MissionCharter.charter_id
    mission_id: Id
    version: int
    status: Literal["proposed", "approved", "superseded", "rejected"]
    body: MissionCharter
    proposed_by: str
    approved_by: str | None
    approved_at: AwareDatetime | None
    install_status: Literal["none", "pending", "installed", "failed"]
    fence_hash: str | None
    fence_verified_at: AwareDatetime | None
    approval_id: Id | None  # *server*
    created_at: AwareDatetime


class Approval(SdModel):
    id: Id
    mission_id: Id
    vehicle_id: VehicleId
    kind: Literal["charter", "command", "resume"]
    subject_id: Id  # charter.id, command.id or the latch action's action_id
    subject_version: int  # charter.version; always 1 for a command
    status: Literal["pending", "approved", "rejected", "expired", "superseded"]
    summary_sv: str  # e.g. "Flyg till P1 på 40 m, 8 m/s"
    required_confirmations: list[Literal["clear_around_drone"]]
    default_autonomy_mode: AutonomyMode | None  # *server*, only for kind = charter
    gate: GateDecision | None
    requested_by: str
    requested_at: AwareDatetime
    expires_at: AwareDatetime | None
    decided_by: str | None
    decided_at: AwareDatetime | None
    reason: str | None


class ApprovalDecisionRequest(SdModel):
    decision: Literal["approve", "reject"]
    subject_version: int
    confirmations: list[Literal["clear_around_drone"]] = Field(
        default_factory=list[Literal["clear_around_drone"]], max_length=4
    )
    autonomy_mode: AutonomyMode | None = None
    reason: str | None = Field(default=None, max_length=500)


class Command(SdModel):
    id: Id
    mission_id: Id
    vehicle_id: VehicleId
    skill: str
    params: dict[str, Any]
    issuer_kind: Literal["operator", "brain", "system"]
    issuer_ref: str
    backend: Backend | None
    rationale_sv: str | None
    in_reply_to: Id | None
    charter_version: int | None
    based_on: StateRef | None
    status: CommandStatus
    status_reason: str | None
    approval_id: Id | None
    expires_at: AwareDatetime | None  # display only
    created_at: AwareDatetime
    updated_at: AwareDatetime


class OperatorCommandRequest(SdModel):
    """An order from the pilot, e.g. an order button. Same gate and latch as the AI's orders."""

    client_command_id: Id
    skill: str = Field(pattern=SKILL_NAME_PATTERN)
    params: Annotated[dict[str, Any], json_bytes_at_most(PARAMS_MAX_BYTES)] = Field(
        max_length=PARAMS_MAX_KEYS
    )
    rationale_sv: str | None = Field(default=None, max_length=500)


class CommandResult(SdModel):
    command: Command
    gate: GateDecision


class ChatMessage(SdModel):
    id: Id
    seq: int
    mission_id: Id
    role: Literal["operator", "brain", "system"]
    author_ref: str
    kind: Literal["text", "question", "status", "proposal"]
    text: str = Field(min_length=1, max_length=CHAT_MAX_CHARS)
    in_reply_to: Id | None
    refs: dict[str, Any]
    author_role: Literal["pilot", "system"] | None  # set by the server from the session
    parsed_intent: OverrideIntent | None  # never "resume"
    handled_by: Literal["ground"] | None
    delivery: Literal["n/a", "queued", "delivered", "summarized"]
    created_at: AwareDatetime


class ChatPost(SdModel):
    client_msg_id: Id
    text: str = Field(min_length=1, max_length=CHAT_MAX_CHARS)
    annotation_ids: list[Id] = Field(default_factory=list[Id], max_length=20)
    in_reply_to: Id | None = None


AnnotationKind = Literal["pin", "area", "path"]
PATH_LABEL_PATTERN = r"^pa-[0-9]{1,3}$"  # what the AI refers to, like P1 for pins
_GEOMETRY_OF: dict[str, str] = {"pin": "Point", "area": "Polygon", "path": "LineString"}


class Annotation(SdModel):
    id: Id
    mission_id: Id
    kind: AnnotationKind
    label: str
    geometry: dict[str, Any]  # GeoJSON Point | Polygon | LineString
    created_by: str
    created_at: AwareDatetime
    area_m2: float | None
    length_m: float | None = None  # paths only


class AnnotationCreate(SdModel):
    kind: AnnotationKind
    label: str = Field(min_length=1, max_length=40)
    geometry: GeoJsonPoint | GeoJsonPolygon | GeoJsonLineString

    @model_validator(mode="after")
    def _matches_kind(self) -> AnnotationCreate:
        expected = _GEOMETRY_OF[self.kind]
        if self.geometry.get("type") != expected:
            raise ValueError(f"a {self.kind} needs a GeoJSON {expected}")
        if self.kind == "path" and re.fullmatch(PATH_LABEL_PATTERN, self.label) is None:
            raise ValueError("a path is called pa-<n>")
        return self


class Clearance(SdModel):
    id: Id
    mission_id: Id
    vehicle_id: VehicleId
    charter_id: Id
    charter_version: int
    status: Literal["active", "consumed", "expired", "invalidated"]
    confirmed_by: str
    confirmed_at: AwareDatetime
    valid_until: AwareDatetime  # display only


class MissionDetail(SdModel):
    mission: Mission
    charter: Charter | None
    sectors: dict[str, Any]  # GeoJSON FeatureCollection
    pending_approvals: list[Approval]
    active_command: Command | None
    latch: OperatorLatch | None
    clearance: Clearance | None


class Event(SdModel):
    id: Id
    mission_id: Id | None
    vehicle_id: VehicleId | None
    t: AwareDatetime
    source: str
    type: str
    severity: Literal["debug", "info", "warning", "critical"]
    payload: dict[str, Any]


class MissionFile(SdModel):
    id: Id
    kind: str
    rel_path: str
    bytes: int | None
    sha256: str | None
    started_at: AwareDatetime | None
    ended_at: AwareDatetime | None


# --- health --------------------------------------------------------------------------


class Health(SdModel):
    status: Literal["ok"] = "ok"
    ready: bool
    version: str
    profile: str
    parts: dict[str, PartState]
    test_rig: bool


class PartStatus(SdModel):
    state: PartState
    since: AwareDatetime | None
    detail: str | None = None


class SystemStatus(SdModel):
    parts: dict[str, PartStatus]
    internet: Literal["up", "down", "unknown"]
    operator_present: bool
    brain_engine: Literal["cloud", "command"] | None = None  # cloud model or command mode now
    version: str
    profile: str
