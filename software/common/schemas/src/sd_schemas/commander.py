"""The commander API (``127.0.0.1:7700/commander/v1``): what ``sd-brain`` reads and sends.

Nothing here lets the AI approve, stop, change limits, autonomy mode or the fence.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AwareDatetime, Field

from sd_schemas.api import (
    CHAT_MAX_CHARS,
    Annotation,
    Approval,
    Charter,
    Command,
    CommandStatus,
    Event,
)
from sd_schemas.base import Id, LatLon, SdModel, VehicleId
from sd_schemas.charter import AutonomyMode, CharterValidationResult, MissionCharter
from sd_schemas.gate import GateDecision, StateRef
from sd_schemas.mission import MissionState, OperatorLatch
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
    "Situation",
    "SkillRequest",
    "SkillSubmitResult",
    "VehicleSummary",
]

Backend = Literal["llm", "grammar", "scripted", "rule", "cloud"]


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
    executor_state: str | None
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
    paused_by: str | None
    returning_cause: str | None
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
    home: LatLon | None
    pilot: LatLon | None
    alerts: list[dict[str, Any]]
    recent_events: list[Event]  # the latest 20


class BrainMessage(SdModel):
    client_msg_id: Id
    text: str = Field(min_length=1, max_length=CHAT_MAX_CHARS)
    kind: Literal["text", "question", "status", "proposal"] = "text"
    in_reply_to: Id | None = None
    refs: dict[str, Any] = Field(default_factory=dict[str, Any])


class SkillRequest(SdModel):
    command_id: Id  # created by sd-brain; idempotent
    skill: str
    params: dict[str, Any]
    rationale_sv: str = Field(max_length=500)
    in_reply_to: Id | None
    based_on: StateRef | None  # the ground station never fills it in
    charter_version: int | None
    backend: Backend
    ai_config_hash: str


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


class MissionMemory(SdModel):
    mission_id: Id
    summary_sv: str = Field(max_length=1200)
    plan_note: str = Field(max_length=400)
    summarized_up_to_seq: int
    pending_operator_request: dict[str, Any] | None
    updated_at: AwareDatetime | None  # *server*


class DecisionLog(SdModel):
    id: Id
    turn_id: Id
    trigger: str
    backend: Backend
    engine: Literal["cloud", "local"] | None = None  # which model answered
    model_id: str | None = None
    ai_config_hash: str
    prompt_sha256: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    rounds: int
    repairs: int
    tool_calls: list[dict[str, Any]]
    command_id: Id | None
    gate_decision: str | None
    reason_codes: list[str] = Field(default_factory=list[str])
    error: Literal["deadline", "timeout", "http_error", "parse_error", "round_limit"] | None
    retry_of: Id | None = None
    proposal_id: Id | None = None
    proposal_ok: bool | None = None
    prompt_json: dict[str, Any] | None  # at most 1 MiB serialised; stored as a file


class DecisionLogStored(DecisionLog):
    prompt_file: str | None  # *server*: always f"brain/{turn_id}.json"
