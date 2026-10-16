"""SD-link: the messages between the ground station and the onboard computer.

Every message travels in an ``Envelope`` whose ``body`` is validated against
``KIND_MODELS[kind]``. The transport and reliability live in ``sd_link``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, Field, field_validator

from sd_schemas.base import (
    ISSUER_PATTERN,
    OPERATOR_REF_PATTERN,
    PARAMS_MAX_BYTES,
    PARAMS_MAX_KEYS,
    SKILL_NAME_PATTERN,
    VEHICLE_ID_PATTERN,
    Id,
    LatLonAlt,
    SdModel,
    VehicleId,
    json_bytes_at_most,
)
from sd_schemas.charter import MissionCharter
from sd_schemas.executor import ExecutorState, HeldBy, HoldReason
from sd_schemas.gate import CLEARANCE_VALID_MS, GateDecision, StateRef
from sd_schemas.mission import LatchAction

__all__ = [
    "KIND_MODELS",
    "MANUAL_CONTROL_TIMEOUT_MS",
    "RELIABLE_KINDS",
    "SCHEMA_MAJOR",
    "Ack",
    "AckStage",
    "ApprovalRef",
    "CharterAck",
    "CharterInstall",
    "ClearanceConfirmation",
    "CommandCancel",
    "CommanderStatus",
    "Envelope",
    "Heartbeat",
    "Hello",
    "HelloAck",
    "LatchRef",
    "ManualControl",
    "ManualTakeover",
    "OnboardEvent",
    "OperatorOverrideNotice",
    "OperatorPosition",
    "OperatorRelease",
    "PilotPosition",
    "SkillCommand",
    "StateSnapshot",
    "StickAxes",
    "is_reliable",
]

SCHEMA_MAJOR = 1

# Manual stick values older than this make the vehicle stand still (onboard watchdog).
MANUAL_CONTROL_TIMEOUT_MS = 1000

AckStage = Literal[
    "received", "accepted", "rejected", "running", "done", "failed", "superseded", "expired"
]


_SENDER_PATTERN = "^(ground:main|vehicle:" + VEHICLE_ID_PATTERN.strip("^$") + ")$"


class Envelope(SdModel):
    v: Literal[1] = 1
    msg_id: Id
    kind: str  # the body class name, one of KIND_MODELS
    sender: str = Field(pattern=_SENDER_PATTERN)
    vehicle_id: VehicleId
    mission_id: Id | None
    seq: int = Field(ge=0)  # per sender and connection, +1 per message
    sent_utc: AwareDatetime  # logs only; never logic
    ttl_ms: int | None  # counted from receipt with the receiver's monotonic clock
    reliable: bool
    corr_id: Id | None
    sig: str | None = None  # reserved for HMAC; always None in the simulator
    body: dict[str, Any]  # validated against KIND_MODELS[kind]

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, value: str) -> str:
        if value not in KIND_MODELS:
            raise ValueError(f"unknown kind: {value[:64]}")
        return value

    def parse_body(self) -> SdModel:
        """The body as its typed model; ValueError for a bad body."""
        return KIND_MODELS[self.kind].model_validate(self.body)


class LatchRef(SdModel):
    action: LatchAction
    action_seq: int


class Hello(SdModel):
    """Onboard -> ground, the first message on a connection."""

    schema_major: int
    boot_id: int  # grows with every start of sd-onboard (= state_epoch)
    executor_state: ExecutorState
    charter_id: Id | None
    charter_version: int | None
    fc_firmware: str | None  # e.g. "ArduCopter V4.7.1"
    operator_latch: LatchRef | None
    last_action_seq: int  # highest action_seq seen onboard, even without an active latch


class HelloAck(SdModel):
    """Ground -> onboard, within 1 s of Hello."""

    schema_major: int
    server_epoch: Id  # new at every start of sd-ground
    operator_latch: LatchRef | None  # the ground's latch after reconciliation


class Heartbeat(SdModel):
    """Both ways, 1 Hz, not reliable, never acknowledged."""


class CharterInstall(SdModel):
    charter: MissionCharter  # the whole charter, including fence_hash and autonomy_mode


class CharterAck(SdModel):
    charter_id: Id
    charter_version: int
    accepted: bool
    reason_codes: list[str] = Field(default_factory=list[str])


class ApprovalRef(SdModel):
    approval_id: Id
    command_hash: str


class SkillCommand(SdModel):
    """Ground -> onboard, reliable; envelope ttl_ms = the registry's ttl_s x 1000."""

    command_id: Id
    skill: str = Field(pattern=SKILL_NAME_PATTERN)
    params: Annotated[dict[str, Any], json_bytes_at_most(PARAMS_MAX_BYTES)] = Field(
        max_length=PARAMS_MAX_KEYS
    )
    issuer: str = Field(pattern=ISSUER_PATTERN)  # "operator:<alias>" | "brain:ground"
    rationale: str | None
    charter_version: int
    based_on: StateRef | None
    approval: ApprovalRef | None


class CommandCancel(SdModel):
    command_id: Id


class Ack(SdModel):
    """Both ways; terminal stages are sent reliable."""

    ref_id: Id  # the msg_id or command_id being acknowledged
    stage: AckStage
    reason_codes: list[str] = Field(default_factory=list[str])
    detail_sv: str | None = None
    gate: GateDecision | None = None  # the onboard decision (side = "onboard") when rejected


class StateSnapshot(SdModel):
    """Onboard -> ground, 2 Hz, not reliable. What only the onboard computer knows."""

    state_seq: int  # +1 per snapshot sent
    state_epoch: int  # = boot_id
    last_transition_state_seq: int  # state_seq of the first snapshot with this executor_state
    executor_state: ExecutorState
    hold_reason: HoldReason | None
    held_by: HeldBy | None
    control_suspended: bool
    expected_mode: str | None
    flight_mode: str
    armed: bool
    position: LatLonAlt
    alt_rel_home_m: float
    groundspeed_ms: float
    heading_deg: float
    battery_pct: float | None
    active_command_id: Id | None
    active_skill: str | None
    progress_pct: float | None
    resume_available: bool
    charter_id: Id | None
    charter_version: int | None
    clearance_available: bool
    preflight_ok: bool
    preflight: dict[str, bool]  # PF1 ... PF11
    limits_active: list[str]  # e.g. ["L-BATT"]
    stop_latch: bool
    battery_rtl_active: bool
    manual_control: bool = False  # the pilot is steering with sticks (Ta över)
    tick_ms_p95: float
    sector_progress: dict[str, float]


class OnboardEvent(SdModel):
    type: str = Field(pattern=r"^executor\.[a-z_]+$")
    severity: Literal["debug", "info", "warning", "critical"]
    data: dict[str, Any] = Field(default_factory=dict[str, Any])


class OperatorOverrideNotice(SdModel):
    """Ground -> onboard: STOP, FLY HOME, battery or watchdog. Never from the AI."""

    override_id: Id
    kind: LatchAction
    action_seq: int  # strictly increasing per vehicle; lower than the last seen is ignored
    issuer_kind: Literal["operator", "system"]
    operator_id: str  # "operator:<alias>" | "system:ground"


class OperatorRelease(SdModel):
    """Ground -> onboard; only from the pilot's Resume click."""

    action_seq: int
    operator_id: str


class ClearanceConfirmation(SdModel):
    """ "Clear around the drone". Never resent after a reconnect; envelope ttl_ms <= 5000."""

    clearance_id: Id
    mission_id: Id
    vehicle_id: VehicleId
    charter_id: Id
    charter_version: int
    valid_for_ms: int = Field(gt=0, le=CLEARANCE_VALID_MS)  # from receipt, onboard clock
    confirmed_by: str
    origin: Literal[
        "local", "lan", "remote"
    ]  # set by sd-ground from the listener, never the client


class PilotPosition(SdModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    source: Literal["charter", "phone"]  # always "charter" in v1
    accuracy_m: float | None


class OperatorPosition(SdModel):
    """Schema only in v1 (stage 2)."""

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    accuracy_m: float | None
    t: AwareDatetime


class CommanderStatus(SdModel):
    connected: bool
    state: Literal["ready", "busy", "degraded", "down"]


class ManualTakeover(SdModel):
    """Ground -> onboard, reliable. The pilot takes over (Ta över) or hands back (Lämna tillbaka).

    On ``active = True`` the AI's orders pause and the vehicle goes to LOITER; stick
    values (``ManualControl``) are then forwarded as MAVLink MANUAL_CONTROL. Limits and
    the geofence still apply. A physical RC transmitter always wins.
    """

    takeover_id: Id
    active: bool
    action_seq: int = Field(ge=0)  # shared with latches; lower than the last seen is ignored
    operator_id: str = Field(pattern=OPERATOR_REF_PATTERN)  # only the pilot role steers


class StickAxes(SdModel):
    """Stick values. Axes are -1..1; ``throttle`` 0..1 with 0.5 = hold height (LOITER)."""

    takeover_id: Id
    input_seq: int = Field(ge=0)  # +1 per value from the panel; older values are dropped
    pitch: float = Field(ge=-1, le=1)  # forward +
    roll: float = Field(ge=-1, le=1)  # right +
    throttle: float = Field(ge=0, le=1)  # 0.5 = keep height
    yaw: float = Field(ge=-1, le=1)  # clockwise +


class ManualControl(StickAxes):
    """Ground -> onboard, 10-20 Hz, not reliable. Without a new value for
    ``MANUAL_CONTROL_TIMEOUT_MS`` the vehicle stands still."""


KIND_MODELS: dict[str, type[SdModel]] = {
    m.__name__: m
    for m in (
        Hello,
        HelloAck,
        Heartbeat,
        CharterInstall,
        CharterAck,
        SkillCommand,
        CommandCancel,
        Ack,
        StateSnapshot,
        OnboardEvent,
        OperatorOverrideNotice,
        OperatorRelease,
        ClearanceConfirmation,
        PilotPosition,
        OperatorPosition,
        CommanderStatus,
        ManualTakeover,
        ManualControl,
    )
}

RELIABLE_KINDS = frozenset(
    {
        "CharterInstall",
        "CharterAck",
        "SkillCommand",
        "CommandCancel",
        "OnboardEvent",
        "OperatorOverrideNotice",
        "OperatorRelease",
        "ClearanceConfirmation",
        "ManualTakeover",
    }
)

_TERMINAL_ACK_STAGES = frozenset({"done", "failed", "superseded", "expired", "rejected"})


def is_reliable(kind: str, body: SdModel) -> bool:
    """Whether a message of this kind must be acknowledged and resent."""
    if isinstance(body, Ack):
        return body.stage in _TERMINAL_ACK_STAGES
    return kind in RELIABLE_KINDS
