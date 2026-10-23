"""The commander API (``127.0.0.1:7700/commander/v1``): what ``sd-brain`` reads and sends.

Nothing here lets the AI approve, stop, change limits, autonomy mode or the fence.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, Field

from sd_schemas.api import (
    CHAT_MAX_CHARS,
    Annotation,
    Approval,
    Backend,
    Charter,
    Command,
    CommandStatus,
    Event,
)
from sd_schemas.base import (
    PARAMS_MAX_BYTES,
    PARAMS_MAX_KEYS,
    REFS_MAX_BYTES,
    SKILL_NAME_PATTERN,
    Id,
    LatLon,
    SdModel,
    VehicleId,
    json_bytes_at_most,
)
from sd_schemas.charter import (
    AutonomyMode,
    CharterValidationError,
    CharterValidationResult,
    MissionCharter,
)
from sd_schemas.executor import ExecutorState
from sd_schemas.gate import GateDecision, StateRef
from sd_schemas.mission import MissionState, OperatorLatch, PausedBy, ReturningCause
from sd_schemas.safety import SafetyLimits

__all__ = [
    "Backend",
    "BrainMessage",
    "CharterProposalResult",
    "DecisionLog",
    "DecisionLogStored",
    "EstimateResult",
    "MapPin",
    "MissionMemory",
    "MissionMemoryUpdate",
    "Situation",
    "SkillRequest",
    "SkillSubmitResult",
    "VehicleSummary",
]


# Size caps on what the AI process sends; it is the least trusted caller.
PROMPT_JSON_MAX_BYTES = 1024 * 1024
_HASH = r"^[0-9a-zA-Z_.:-]{1,128}$"


class MapPin(SdModel):
    id: str = Field(pattern=r"^P[0-9]{1,3}$")  # the label, e.g. "P1"
    annotation_id: Id
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    note: str | None = Field(default=None, max_length=40)


class VehicleSummary(SdModel):
    connected: bool
    armed: bool
    landed: bool
    airborne: bool
    flight_mode: str
    executor_state: ExecutorState | None
    lat: float | None
    lon: float | None
    alt_rel_home_m: float | None
    groundspeed_ms: float | None
    battery_pct: float | None
    battery_to_reserve_s: int | None
    dist_to_pilot_m: float | None
    gnss_ok: bool  # fix >= 3 and >= 8 satellites
    link_ok: bool  # SD-link up
    manual_control: bool = False  # the pilot is steering by hand; the AI waits


class Situation(SdModel):
    mission_id: Id
    vehicle_id: VehicleId
    state_seq: int
    snapshot_state_epoch: int | None
    snapshot_state_seq: int | None
    mission_status: MissionState
    paused_by: PausedBy | None
    returning_cause: ReturningCause | None
    autonomy_mode: AutonomyMode | None
    ai_enabled: bool
    limits: SafetyLimits
    charter: MissionCharter | None
    charter_installed: bool
    fence_verified: bool
    pending_proposal: Charter | None
    pending_approvals: list[Approval]
    takeoff_clearance_valid_until: AwareDatetime | None
    brain_may_takeoff: bool
    override: OperatorLatch | None
    vehicle: VehicleSummary
    active_command: Command | None
    sectors: list[dict[str, Any]]  # {sector_id, name, area_m2, status, progress}
    pins: list[MapPin]
    areas: list[Annotation]
    paths: list[Annotation] = Field(default_factory=list[Annotation])  # drawn paths pa-<n>
    home: LatLon | None
    pilot: LatLon | None
    alerts: list[dict[str, Any]]
    recent_events: list[Event]  # the latest 20
    # The probability map's sector line (POA left / covered / mean POD / PSR), or None
    # before the first map. Only a ranking: orders still go through the gate.
    probability_line: str | None = None
    # The open search candidates (best first, with field matches, distance and the
    # review counts) and the confirmed finds. Built by the ground from enum-like values
    # only, never from free text. Only a person confirms a find.
    candidates_line: str | None = None
    finds_line: str | None = None


class BrainMessage(SdModel):
    client_msg_id: Id
    text: str = Field(min_length=1, max_length=CHAT_MAX_CHARS)
    kind: Literal["text", "question", "status", "proposal"] = "text"
    in_reply_to: Id | None = None
    refs: Annotated[dict[str, Any], json_bytes_at_most(REFS_MAX_BYTES)] = Field(
        default_factory=dict[str, Any], max_length=16
    )


class SkillRequest(SdModel):
    command_id: Id  # created by sd-brain; idempotent
    skill: str = Field(pattern=SKILL_NAME_PATTERN)
    params: Annotated[dict[str, Any], json_bytes_at_most(PARAMS_MAX_BYTES)] = Field(
        max_length=PARAMS_MAX_KEYS
    )
    rationale_sv: str = Field(max_length=500)
    in_reply_to: Id | None
    based_on: StateRef | None  # the ground station never fills it in
    charter_version: int | None
    backend: Backend
    ai_config_hash: str = Field(pattern=_HASH)


class SkillSubmitResult(SdModel):
    command_id: Id
    decision: Literal["ALLOW", "REJECT", "PENDING_APPROVAL"]
    reason_codes: list[str]
    message_sv: str
    details: dict[str, Any]
    approval_id: Id | None
    status: CommandStatus


class EstimateResult(SdModel):
    gate: GateDecision
    duration_s: float | None
    battery_after_pct: float | None


class CharterProposalResult(SdModel):
    charter_id: Id | None
    approval_id: Id | None
    validation: CharterValidationResult


class MissionMemoryUpdate(SdModel):
    """What sd-brain sends (PUT .../memory). The server adds ``updated_at``."""

    mission_id: Id
    summary_sv: str = Field(max_length=1200)
    plan_note: str = Field(max_length=400)
    summarized_up_to_seq: int = Field(ge=0)
    pending_operator_request: (
        Annotated[dict[str, Any], json_bytes_at_most(REFS_MAX_BYTES)] | None
    ) = None


class MissionMemory(MissionMemoryUpdate):
    """What the server stores and returns."""

    updated_at: AwareDatetime | None  # *server*


class DecisionLog(SdModel):
    id: Id
    turn_id: Id
    trigger: str = Field(pattern=r"^[a-z][a-z0-9_.]{0,63}$")
    backend: Backend
    engine: Literal["cloud", "command"] | None = None  # the cloud model or the command mode
    model_id: str | None = Field(default=None, max_length=128)
    ai_config_hash: str = Field(pattern=_HASH)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    rounds: int = Field(ge=0)
    repairs: int = Field(ge=0)
    tool_calls: Annotated[list[dict[str, Any]], json_bytes_at_most(PARAMS_MAX_BYTES * 4)] = Field(
        max_length=32
    )
    command_id: Id | None
    gate_decision: Literal["ALLOW", "REJECT", "PENDING_APPROVAL"] | None
    reason_codes: list[str] = Field(default_factory=list[str], max_length=64)
    error: Literal["deadline", "timeout", "http_error", "parse_error", "round_limit"] | None
    retry_of: Id | None = None  # the turn whose refused order or charter this one retries
    attempt: int | None = Field(default=None, ge=1, le=10)  # the charter attempt (1 = first)
    proposal_id: Id | None = None
    proposal_ok: bool | None = None
    # Why the charter proposal was refused (by the brain in the turn or by the ground),
    # and the draft the AI sent, for reading a failed run afterwards.
    proposal_errors: list[CharterValidationError] = Field(
        default_factory=list[CharterValidationError], max_length=32
    )
    proposal_draft: Annotated[dict[str, Any], json_bytes_at_most(PARAMS_MAX_BYTES)] | None = None
    prompt_json: Annotated[dict[str, Any], json_bytes_at_most(PROMPT_JSON_MAX_BYTES)] | None
    # stored as a file, never in the database


class DecisionLogStored(DecisionLog):
    prompt_file: str | None  # *server*: always f"brain/{turn_id}.json"
