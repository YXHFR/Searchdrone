"""Mission states, the triggers that move between them, and the operator latch.

The transition table itself is code in ``sd_ground.services.mission_state``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, NamedTuple

from pydantic import AwareDatetime

from sd_schemas.base import SdModel, VehicleId

__all__ = [
    "LatchAction",
    "MissionState",
    "MissionStatusFull",
    "OperatorLatch",
    "PausedBy",
    "ReturningCause",
    "TransitionCtx",
    "Trigger",
]


class MissionState(StrEnum):
    DRAFT = "DRAFT"  # mission created, no charter
    CHARTER_PROPOSED = "CHARTER_PROPOSED"  # waiting for the operator's click
    READY = "READY"  # charter approved and fence verified
    ACTIVE = "ACTIVE"  # airborne, the AI leads
    PAUSED = "PAUSED"  # standing still, see paused_by
    HELD = "HELD"  # a mode change our system did not request; we send nothing that steers
    FAILSAFE = "FAILSAFE"  # the flight controller is in failsafe; we send nothing that steers
    RETURNING = "RETURNING"  # flying home, see returning_cause
    LANDED = "LANDED"
    CLOSED = "CLOSED"


PausedBy = Literal["operator", "brain", "system"]
ReturningCause = Literal["brain", "operator", "battery", "failsafe", "system"]


class MissionStatusFull(NamedTuple):
    """The whole mission state; the sub-fields take part in comparisons."""

    state: MissionState
    paused_by: PausedBy | None = None  # set only in PAUSED
    returning_cause: ReturningCause | None = None  # set only in RETURNING


class Trigger(StrEnum):
    MISSION_OPENED = "MISSION_OPENED"
    CHARTER_PROPOSED = "CHARTER_PROPOSED"
    PROPOSALS_CLEARED = "PROPOSALS_CLEARED"
    CHARTER_READY = "CHARTER_READY"
    TAKEOFF_ACCEPTED = "TAKEOFF_ACCEPTED"
    OPERATOR_STOP = "OPERATOR_STOP"
    OPERATOR_RTL = "OPERATOR_RTL"
    OPERATOR_RESUME_CONFIRMED = "OPERATOR_RESUME_CONFIRMED"
    WATCHDOG_STOP = "WATCHDOG_STOP"
    WATCHDOG_RTL = "WATCHDOG_RTL"
    OPERATOR_ABSENT_HOLD = "OPERATOR_ABSENT_HOLD"
    OPERATOR_ABSENT_RTL = "OPERATOR_ABSENT_RTL"
    BRAIN_HOLD = "BRAIN_HOLD"
    BRAIN_HOLD_END = "BRAIN_HOLD_END"
    BRAIN_RTL_ACCEPTED = "BRAIN_RTL_ACCEPTED"
    BATTERY_RTL = "BATTERY_RTL"
    CONTROL_TAKEN = "CONTROL_TAKEN"
    FC_FAILSAFE = "FC_FAILSAFE"
    LATCH_ADOPTED_STOP = "LATCH_ADOPTED_STOP"
    LATCH_ADOPTED_RTL = "LATCH_ADOPTED_RTL"
    RESTART_FC_STOPPED = "RESTART_FC_STOPPED"
    RESTART_FC_RETURNING = "RESTART_FC_RETURNING"
    LANDED_DISARMED = "LANDED_DISARMED"
    CLOSE_AUTO = "CLOSE_AUTO"
    CLOSE_MANUAL = "CLOSE_MANUAL"


class TransitionCtx(NamedTuple):
    in_flight: bool  # wide definition: landed_state in the air, or armed and above 0.5 m
    fc_home_mode: bool  # flight controller in RTL or SMART_RTL per the latest HEARTBEAT


LatchAction = Literal["stop", "return_home", "battery_rtl", "watchdog_stop", "watchdog_rtl"]


class OperatorLatch(SdModel):
    """Set by STOP, FLY HOME, the battery guard or the watchdog. Only Resume lifts it."""

    vehicle_id: VehicleId
    action: LatchAction
    underlying: Literal["battery_rtl"] | None = None  # the battery latch stays under a STOP
    action_seq: int  # strictly increasing per vehicle
    set_by: str  # "operator:<alias>" | "system:ground"
    set_at: AwareDatetime  # display only
    set_mono_ms: int
