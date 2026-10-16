"""Message builders shared by the sd_link tests. Coordinates are around CMAC."""

from __future__ import annotations

from datetime import UTC, datetime

from sd_link.envelope import GROUND_SENDER, build_envelope, encode, vehicle_sender
from sd_schemas.base import SdModel, new_id
from sd_schemas.executor import ExecutorState
from sd_schemas.gate import StateRef
from sd_schemas.link import (
    SCHEMA_MAJOR,
    ClearanceConfirmation,
    Envelope,
    Hello,
    ManualControl,
    ManualTakeover,
    OperatorOverrideNotice,
    SkillCommand,
)

VEHICLE = "sim-01"
T0 = datetime(2026, 10, 15, 12, 30, tzinfo=UTC)


def hello(boot_id: int = 1, schema_major: int = SCHEMA_MAJOR) -> Hello:
    return Hello(
        schema_major=schema_major,
        boot_id=boot_id,
        executor_state=ExecutorState.IDLE,
        charter_id=None,
        charter_version=None,
        fc_firmware="ArduCopter V4.7.1",
        operator_latch=None,
        last_action_seq=0,
    )


def goto(n: int = 0) -> SkillCommand:
    return SkillCommand(
        command_id=new_id(),
        skill="goto",
        params={"lat": -35.36146 + n * 1e-5, "lon": 149.16523, "alt_m": 40, "speed_ms": 8},
        issuer="brain:ground",
        rationale=None,
        charter_version=1,
        based_on=StateRef(state_epoch=1, state_seq=n),
        approval=None,
    )


def stop_notice(action_seq: int = 1) -> OperatorOverrideNotice:
    return OperatorOverrideNotice(
        override_id=new_id(),
        kind="stop",
        action_seq=action_seq,
        issuer_kind="operator",
        operator_id="operator:demo",
    )


def clearance() -> ClearanceConfirmation:
    return ClearanceConfirmation(
        clearance_id=new_id(),
        mission_id=new_id(),
        vehicle_id=VEHICLE,
        charter_id=new_id(),
        charter_version=1,
        valid_for_ms=120_000,
        confirmed_by="operator:demo",
        origin="local",
    )


def takeover(active: bool = True) -> ManualTakeover:
    return ManualTakeover(
        takeover_id=new_id(), active=active, action_seq=3, operator_id="operator:demo"
    )


def sticks(input_seq: int = 0) -> ManualControl:
    return ManualControl(
        takeover_id=new_id(), input_seq=input_seq, pitch=0.4, roll=0.0, throttle=0.5, yaw=0.0
    )


def envelope_from(
    body: SdModel, *, side: str = "vehicle", seq: int = 0, ttl_ms: int | None = None
) -> Envelope:
    sender = vehicle_sender(VEHICLE) if side == "vehicle" else GROUND_SENDER
    return build_envelope(
        body, sender=sender, vehicle_id=VEHICLE, seq=seq, sent_utc=T0, ttl_ms=ttl_ms
    )


def frame(body: SdModel, **kwargs: object) -> str:
    return encode(envelope_from(body, **kwargs))  # type: ignore[arg-type]
