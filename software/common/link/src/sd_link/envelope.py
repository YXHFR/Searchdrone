"""Building, encoding and decoding SD-link envelopes.

Every message is an ``sd_schemas.link.Envelope`` serialised as one JSON text frame.
This module also knows which side may send which kind and the per-kind rules on
``ttl_ms`` that the link enforces before anything leaves the process.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final, Literal

from pydantic import ValidationError

from sd_link.codes import MSG_TOO_LARGE, SCHEMA_INVALID
from sd_link.constants import MAX_MESSAGE_BYTES
from sd_schemas.base import SdModel, new_id
from sd_schemas.link import KIND_MODELS, MANUAL_CONTROL_TIMEOUT_MS, Envelope, is_reliable

__all__ = [
    "CLEARANCE_TTL_MAX_MS",
    "GROUND_SENDER",
    "GROUND_TO_VEHICLE",
    "VEHICLE_TO_GROUND",
    "DecodeError",
    "MessageTooLargeError",
    "Side",
    "build_envelope",
    "check_ttl",
    "decode",
    "encode",
    "kind_of",
    "may_send",
    "peer_sender",
    "vehicle_sender",
]

Side = Literal["ground", "vehicle"]

GROUND_SENDER: Final = "ground:main"
CLEARANCE_TTL_MAX_MS: Final = 5000

VEHICLE_TO_GROUND: Final = frozenset(
    {"Hello", "CharterAck", "StateSnapshot", "OnboardEvent", "Ack", "Heartbeat"}
)
GROUND_TO_VEHICLE: Final = frozenset(
    {
        "HelloAck",
        "CharterInstall",
        "SkillCommand",
        "CommandCancel",
        "OperatorOverrideNotice",
        "OperatorRelease",
        "ClearanceConfirmation",
        "PilotPosition",
        "OperatorPosition",
        "CommanderStatus",
        "ManualTakeover",
        "ManualControl",
        "Ack",
        "Heartbeat",
    }
)


def vehicle_sender(vehicle_id: str) -> str:
    return f"vehicle:{vehicle_id}"


def peer_sender(side: Side, vehicle_id: str) -> str:
    """The sender the other side must use: ground expects the vehicle and vice versa."""
    return vehicle_sender(vehicle_id) if side == "ground" else GROUND_SENDER


def kind_of(body: SdModel) -> str:
    kind = type(body).__name__
    if kind not in KIND_MODELS or not isinstance(body, KIND_MODELS[kind]):
        raise ValueError(f"not an SD-link message: {kind}")
    return kind


def may_send(side: Side, kind: str) -> bool:
    return kind in (GROUND_TO_VEHICLE if side == "ground" else VEHICLE_TO_GROUND)


def check_ttl(kind: str, ttl_ms: int | None) -> None:
    """Per-kind ttl rules. Raises ValueError so that a bad message is never sent."""
    if ttl_ms is not None and ttl_ms <= 0:
        raise ValueError("ttl_ms must be positive")
    if kind == "SkillCommand" and ttl_ms is None:
        raise ValueError("a SkillCommand needs ttl_ms (registry ttl_s x 1000)")
    if kind == "ClearanceConfirmation" and (ttl_ms is None or ttl_ms > CLEARANCE_TTL_MAX_MS):
        raise ValueError(f"a ClearanceConfirmation needs ttl_ms <= {CLEARANCE_TTL_MAX_MS}")
    if kind == "ManualControl" and (ttl_ms is None or ttl_ms > MANUAL_CONTROL_TIMEOUT_MS):
        raise ValueError(f"ManualControl needs ttl_ms <= {MANUAL_CONTROL_TIMEOUT_MS}")


def build_envelope(
    body: SdModel,
    *,
    sender: str,
    vehicle_id: str,
    seq: int,
    sent_utc: datetime,
    mission_id: uuid.UUID | None = None,
    ttl_ms: int | None = None,
    corr_id: uuid.UUID | None = None,
    msg_id: uuid.UUID | None = None,
) -> Envelope:
    kind = kind_of(body)
    check_ttl(kind, ttl_ms)
    return Envelope(
        msg_id=msg_id or new_id(),
        kind=kind,
        sender=sender,
        vehicle_id=vehicle_id,
        mission_id=mission_id,
        seq=seq,
        sent_utc=sent_utc,
        ttl_ms=ttl_ms,
        reliable=is_reliable(kind, body),
        corr_id=corr_id,
        body=body.model_dump(mode="json"),
    )


class MessageTooLargeError(ValueError):
    pass


def encode(envelope: Envelope) -> str:
    text = envelope.model_dump_json()
    if len(text.encode()) > MAX_MESSAGE_BYTES:
        raise MessageTooLargeError(f"message over {MAX_MESSAGE_BYTES} bytes ({envelope.kind})")
    return text


@dataclass(frozen=True, slots=True)
class DecodeError(Exception):
    """A frame that is not a valid message. ``msg_id``/``reliable`` are set when they
    could be read, so that the receiver can answer with ``Ack(rejected, reason)``."""

    reason: str
    detail: str
    msg_id: uuid.UUID | None = None
    reliable: bool = False

    def __str__(self) -> str:
        return f"{self.reason}: {self.detail}"


def _peek(text: str | bytes) -> tuple[uuid.UUID | None, bool]:
    """Best-effort msg_id and reliable flag from a frame that failed validation."""
    try:
        raw: Any = json.loads(text)
    except ValueError:
        return None, False
    if not isinstance(raw, dict):
        return None, False
    try:
        msg_id = uuid.UUID(str(raw.get("msg_id")))  # pyright: ignore[reportUnknownMemberType]
    except ValueError:
        return None, False
    if msg_id.version != 7:
        return None, False
    return msg_id, raw.get("reliable") is True  # pyright: ignore[reportUnknownMemberType]


def decode(text: str | bytes) -> tuple[Envelope, SdModel]:
    """Parse and validate one frame: envelope and typed body. Raises DecodeError."""
    if isinstance(text, bytes):
        raise DecodeError(SCHEMA_INVALID, "binary frames are not allowed")
    if len(text.encode()) > MAX_MESSAGE_BYTES:
        msg_id, reliable = _peek(text)
        raise DecodeError(MSG_TOO_LARGE, "message too large", msg_id, reliable)
    try:
        envelope = Envelope.model_validate_json(text)
        body = envelope.parse_body()
    except (ValidationError, ValueError) as exc:
        msg_id, reliable = _peek(text)
        detail = exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
        raise DecodeError(SCHEMA_INVALID, str(detail)[:200], msg_id, reliable) from None
    if envelope.reliable != is_reliable(envelope.kind, body):
        raise DecodeError(
            SCHEMA_INVALID, "reliable flag does not match the kind", envelope.msg_id, True
        )
    try:
        check_ttl(envelope.kind, envelope.ttl_ms)  # the receiver enforces the rules too
    except ValueError as exc:
        raise DecodeError(SCHEMA_INVALID, str(exc), envelope.msg_id, envelope.reliable) from None
    return envelope, body
