"""Input and output of the safety gate ``sd_safety.evaluate(req, ctx) -> GateDecision``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from sd_schemas.base import ISSUER_PATTERN, Id, LatLonAlt, SdModel, VehicleId
from sd_schemas.charter import AutonomyMode, MissionCharter
from sd_schemas.executor import ExecutorState
from sd_schemas.mission import MissionState, OperatorLatch, PausedBy, ReturningCause
from sd_schemas.safety import SafetyLimits
from sd_schemas.skills import SkillSpec
from sd_schemas.vehicle import VehicleState

__all__ = [
    "CLEARANCE_VALID_MS",
    "ClearanceState",
    "GateContext",
    "GateDecision",
    "GateRequest",
    "StateRef",
]


# "Clear around the drone" is valid at most this long after the click (or receipt onboard).
CLEARANCE_VALID_MS = 120_000


class StateRef(SdModel):
    """Which state picture an order was based on (onboard StateSnapshot epoch and sequence)."""

    state_epoch: int
    state_seq: int


class GateRequest(SdModel):
    command_id: Id
    mission_id: Id
    vehicle_id: VehicleId
    skill: str
    params: dict[str, Any]  # raw; validated by the rule R-SCHEMA
    issuer_kind: Literal["operator", "brain", "system"]
    issuer_ref: str = Field(pattern=ISSUER_PATTERN)
    charter_version: int | None
    based_on: StateRef | None
    phase: Literal["submit", "approve", "estimate", "onboard"]
    approval_confirmations: list[str] = Field(default_factory=list[str])  # phase approve: ticked
    planned_path: list[LatLonAlt] | None = None  # onboard: the handler's path; ground: None


class ClearanceState(SdModel):
    clearance_id: Id
    origin: Literal["local", "lan", "remote"]  # remote is never valid for a takeoff
    mission_id: Id
    vehicle_id: VehicleId
    charter_id: Id
    charter_version: int
    received_mono_ms: int  # monotonic time of the click (ground) or of receipt (onboard)
    valid_for_ms: int = Field(gt=0, le=CLEARANCE_VALID_MS)
    consumed: bool


class GateContext(SdModel):
    """Everything the gate may look at. Built by the caller; the gate does no I/O."""

    side: Literal["ground", "onboard"]
    now_mono_ms: int
    registry: Mapping[str, SkillSpec]
    limits: SafetyLimits
    charter: MissionCharter | None  # the active approved charter
    charter_installed: bool
    fence_verified: bool
    autonomy_mode: AutonomyMode | None  # = charter.autonomy_mode
    mission_state: MissionState | None  # ground; None onboard
    paused_by: PausedBy | None
    returning_cause: ReturningCause | None
    ai_enabled: bool
    vehicle: VehicleState | None
    vehicle_age_ms: int | None
    snapshot_age_ms: int | None  # ground: age of the latest StateSnapshot; onboard 0
    executor_state: ExecutorState | None
    state_epoch: int | None
    last_transition_state_seq: int | None
    preflight_ok: bool | None
    sdlink_up: bool  # onboard: always True
    stop_latch: bool  # onboard: its stop_latch; ground: latch is not None
    latch: OperatorLatch | None  # ground
    manual_control: bool  # the pilot steers by hand: every AI order that moves is rejected
    clearance: ClearanceState | None
    recent_brain_actions_mono_ms: tuple[int, ...]  # the last 60 s
    recent_brain_reports_mono_ms: tuple[int, ...]
    brain_pending_approvals: int
    operator_present: bool


class GateDecision(SdModel):
    decision: Literal["ALLOW", "REJECT", "PENDING_APPROVAL"]
    reason_codes: list[str]  # in priority order; [] on ALLOW
    rule_ids: list[str]
    message_sv: str  # text for the first code (+ "och N till")
    details: dict[str, Any]  # e.g. {"outside_by_m": 34.2, "alt_m": 200}
    required_confirmations: list[str] = Field(default_factory=list[str])  # needed at the click
    side: Literal["ground", "onboard"]
