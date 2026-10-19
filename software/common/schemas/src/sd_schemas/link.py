"""SD-link: the messages between the ground station and the onboard computer.

Every message travels in an ``Envelope`` whose ``body`` is validated against
``KIND_MODELS[kind]``. The transport and reliability live in ``sd_link``.
"""

from __future__ import annotations

import base64
import binascii
from typing import Annotated, Any, Final, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BeforeValidator,
    Field,
    PlainSerializer,
    WithJsonSchema,
    field_validator,
)

from sd_schemas.base import (
    ISSUER_PATTERN,
    OPERATOR_REF_PATTERN,
    PARAMS_MAX_BYTES,
    PARAMS_MAX_KEYS,
    SKILL_NAME_PATTERN,
    VEHICLE_ID_PATTERN,
    Id,
    LatLon,
    LatLonAlt,
    SdModel,
    VehicleId,
    json_bytes_at_most,
)
from sd_schemas.charter import MissionCharter
from sd_schemas.executor import ExecutorState, HeldBy, HoldReason, LinkLossPhase
from sd_schemas.gate import CLEARANCE_VALID_MS, GateDecision, StateRef
from sd_schemas.mission import LatchAction

__all__ = [
    "CROP_JPEG_MAX_BYTES",
    "CROP_MAX_PX",
    "KEEPOUTS_MAX",
    "KIND_MODELS",
    "MANUAL_CONTROL_TIMEOUT_MS",
    "PERCEPTION_DETECTIONS_MAX",
    "RELIABLE_KINDS",
    "SCHEMA_MAJOR",
    "TTL_REQUIRED_KINDS",
    "VIEW_SIDE_MAX_PX",
    "Ack",
    "AckStage",
    "ApprovalRef",
    "BBox",
    "CharterAck",
    "CharterInstall",
    "ClearanceConfirmation",
    "CommandCancel",
    "CommanderStatus",
    "CropData",
    "CropRequest",
    "DetectorInfo",
    "Envelope",
    "GeoDetectionMsg",
    "Heartbeat",
    "Hello",
    "HelloAck",
    "Keepout",
    "KeepoutInstall",
    "LatchRef",
    "ManualControl",
    "ManualTakeover",
    "OnboardEvent",
    "OperatorOverrideNotice",
    "OperatorPosition",
    "OperatorRelease",
    "PerceptionReport",
    "PilotPosition",
    "Pose",
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
    preflight: dict[str, bool]  # PF1 ... PF11, PF12 (RC link) when required
    limits_active: list[str]  # e.g. ["L-BATT"]
    stop_latch: bool
    battery_rtl_active: bool
    manual_control: bool = False  # the pilot is steering with sticks (Ta över)
    tick_ms_p95: float
    sector_progress: dict[str, float]
    link_loss_phase: LinkLossPhase = LinkLossPhase.NONE  # the onboard link-loss rule
    find_hold_keepout_id: Id | None = None  # the find it hovers beside without a link


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

    On ``active = True`` the AI's orders pause onboard (HELD). The ground station puts
    the vehicle in LOITER and sends the pilot's sticks itself as MAVLink MANUAL_CONTROL
    from the GCS id (250/190), the way STOP goes; the onboard computer forwards no
    sticks. The flight controller's limits and geofence still apply. A physical RC
    transmitter always wins.
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
    """Kept in the catalogue for compatibility; v0.1.0 never sends it over SD-link.

    The ground station sends the sticks straight to the flight controller and stops
    after ``MANUAL_CONTROL_TIMEOUT_MS`` without a new value, so the vehicle stands
    still in LOITER."""


# --- stage 2: perception reports, crops and keep-out cylinders ------------------------
# Only reports and small crops travel: never whole stills or video. Every message below
# is reliable and needs ``ttl_ms`` (``TTL_REQUIRED_KINDS``). A frame stays under the
# link's 64 KiB message limit, so a crop's JPEG is capped at ``CROP_JPEG_MAX_BYTES``
# (base64 in JSON adds a third).

PERCEPTION_DETECTIONS_MAX: Final = 20  # per still; the rest are only counted
CROP_MAX_PX: Final = 512  # a crop's longest side
CROP_JPEG_MAX_BYTES: Final = 45_000
VIEW_SIDE_MAX_PX: Final = 1536  # the square a view is cut from, in still pixels
KEEPOUTS_MAX: Final = 200

_JPEG_MAGIC = b"\xff\xd8\xff"


def _jpeg_from_base64(value: object) -> object:
    """JSON carries the JPEG as base64; Python callers may pass the bytes themselves."""
    if isinstance(value, str):
        try:
            return base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("jpeg must be base64") from None
    return value


def _check_jpeg(value: bytes) -> bytes:
    if len(value) > CROP_JPEG_MAX_BYTES:
        raise ValueError(f"jpeg too large: {len(value)} bytes, at most {CROP_JPEG_MAX_BYTES}")
    if not value.startswith(_JPEG_MAGIC):
        raise ValueError("not a JPEG")
    return value


def _jpeg_to_base64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


JpegBytes = Annotated[
    bytes,
    BeforeValidator(_jpeg_from_base64),
    AfterValidator(_check_jpeg),
    PlainSerializer(_jpeg_to_base64, return_type=str, when_used="json"),
    WithJsonSchema(
        {
            "type": "string",
            "contentEncoding": "base64",
            "contentMediaType": "image/jpeg",
            "maxLength": (CROP_JPEG_MAX_BYTES + 2) // 3 * 4,
        }
    ),
]
"""A JPEG of at most ``CROP_JPEG_MAX_BYTES``; base64 in JSON."""


class BBox(SdModel):
    """A box in image pixels: top-left corner, width and height (also ``sd_detect.BBox``)."""

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    w: float = Field(gt=0)
    h: float = Field(gt=0)


class Pose(SdModel):
    """Where the camera was at exposure, interpolated from telemetry."""

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    alt_amsl_m: float = Field(ge=-500, le=9000)
    agl_m: float = Field(ge=-100, le=1000)
    roll_deg: float = Field(ge=-180, le=180)
    pitch_deg: float = Field(ge=-180, le=180)
    yaw_deg: float = Field(ge=-360, le=360)
    source_ts_ms: int = Field(ge=0)  # telemetry time the pose was interpolated to


class DetectorInfo(SdModel):
    """Which detector looked at a still, and how it went."""

    name: str = Field(min_length=1, max_length=64)
    license: str = Field(min_length=1, max_length=64)  # SPDX id
    version: str = Field(min_length=1, max_length=64)
    profile_id: str | None = Field(default=None, max_length=64)  # det_<model>_vN
    placement: Literal["onboard", "ground"]
    ok: bool  # False: the detector failed on this still (no detections is not a failure)
    tiles: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    rate_hz: float = Field(ge=0)  # the measured still rate, always sent


class GeoDetectionMsg(SdModel):
    """One detection, placed on the map. A hint for a person to look at, never a find."""

    detection_id: Id
    bbox: BBox
    score: float = Field(ge=0, le=1)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    sigma95_m: float = Field(ge=0, le=1000)  # 95 % position uncertainty
    px_size: float = Field(gt=0)  # the box's longest side in still pixels
    crop_inline: bool  # True: a CropData for this detection follows
    # The class from the detector's class register (``sd_detect``) and the kind of source
    # that saw it ("oracle" = the simulator's own labels, simulator only). Empty from a
    # detector that has no classes.
    label: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{1,31}$")
    source: Literal["model", "color", "oracle"] | None = None


class PerceptionReport(SdModel):
    """Onboard -> ground: what the detector found in one still."""

    still_id: Id
    command_id: Id | None  # the order the still was taken during
    captured_utc: AwareDatetime
    sensor: Literal["rgb"]  # a thermal camera comes later, with its own migration
    pose: Pose
    footprint: list[LatLon] = Field(min_length=4, max_length=4)  # corners on the ground
    px_per_m: float = Field(gt=0)
    synthetic: bool = False  # a placeholder image (CI), not a camera picture
    detector: DetectorInfo
    detections: list[GeoDetectionMsg] = Field(
        default_factory=list[GeoDetectionMsg], max_length=PERCEPTION_DETECTIONS_MAX
    )
    detections_dropped: int = Field(default=0, ge=0)


class CropData(SdModel):
    """Onboard -> ground: a small JPEG cut from a still, for a detection or a view."""

    ref_id: Id  # the detection_id or the CropRequest's request_id
    purpose: Literal["detection", "view"]
    still_id: Id
    bbox_in_still: BBox
    width: int = Field(ge=1, le=CROP_MAX_PX)
    height: int = Field(ge=1, le=CROP_MAX_PX)
    jpeg: JpegBytes


class CropRequest(SdModel):
    """Ground -> onboard: cut a view from a stored still (a person asked to see more)."""

    request_id: Id
    still_id: Id
    center_px: tuple[Annotated[int, Field(ge=0)], Annotated[int, Field(ge=0)]]
    side_px: int = Field(ge=1, le=VIEW_SIDE_MAX_PX)
    max_px: int = Field(ge=1, le=CROP_MAX_PX)


class Keepout(SdModel):
    """A no-fly cylinder around a possible person."""

    keepout_id: Id
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    radius_m: float = Field(gt=0, le=500)  # person_keepout_m + sigma95_m
    source: Literal["detection", "candidate", "confirmed"]
    source_id: Id  # the detection_id or candidate_id


class KeepoutInstall(SdModel):
    """Ground -> onboard: the whole list of ground cylinders; it replaces the previous one.

    ``install_seq`` grows with every list; a lower one than the last seen is ignored, so a
    late resend never brings back a removed cylinder or drops a new one. The installed
    list stays in force until a list with a higher ``install_seq`` arrives: the envelope's
    ``ttl_ms`` only limits how late the message may be delivered, never how long the
    cylinders apply.
    """

    install_seq: int = Field(ge=0)
    keepouts: list[Keepout] = Field(default_factory=list[Keepout], max_length=KEEPOUTS_MAX)


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
        PerceptionReport,
        CropData,
        CropRequest,
        KeepoutInstall,
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
        "PerceptionReport",
        "CropData",
        "CropRequest",
        "KeepoutInstall",
    }
)

TTL_REQUIRED_KINDS = frozenset({"PerceptionReport", "CropData", "CropRequest", "KeepoutInstall"})
"""Stage 2 messages that are never sent without ``ttl_ms`` (checked by ``sd_link``)."""

_TERMINAL_ACK_STAGES = frozenset({"done", "failed", "superseded", "expired", "rejected"})


def is_reliable(kind: str, body: SdModel) -> bool:
    """Whether a message of this kind must be acknowledged and resent."""
    if isinstance(body, Ack):
        return body.stage in _TERMINAL_ACK_STAGES
    return kind in RELIABLE_KINDS
