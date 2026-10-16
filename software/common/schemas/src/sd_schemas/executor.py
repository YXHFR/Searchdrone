"""The onboard mission executor: its states and the points it can resume from."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from sd_schemas.base import Id, LatLon, SdModel

__all__ = [
    "ExecutorState",
    "GotoResumePoint",
    "HeldBy",
    "HoldReason",
    "ResumePoint",
    "SurveyResumePoint",
]


class ExecutorState(StrEnum):
    IDLE = "IDLE"  # on the ground, disarmed, no charter
    PREFLIGHT = "PREFLIGHT"  # charter installed, preflight checks running
    TAKEOFF = "TAKEOFF"
    TRANSIT = "TRANSIT"  # goto in progress
    SURVEY = "SURVEY"  # survey_sector in progress
    HOLD = "HOLD"  # standing still, waiting
    RTL = "RTL"  # flying home and landing
    LAND = "LAND"  # reserved for land_at (stage 2); no transitions in v1
    LANDED = "LANDED"
    HELD = "HELD"  # a person or the ground's emergency path has taken over
    FAILSAFE = "FAILSAFE"  # the flight controller is in failsafe


HoldReason = Literal[
    "ai_hold", "skill_done", "operator_released", "limit", "link_loss", "ai_silent", "startup"
]
HeldBy = Literal[
    "operator_stop",
    "operator_home",
    "system_battery",
    "system_watchdog",
    "external",
    "onboard_restart",
    "fc_silent",
]


class SurveyResumePoint(SdModel):
    kind: Literal["survey"] = "survey"
    command_id: Id
    waypoints: list[LatLon]  # every search point in the original order
    alt_m: float
    speed_ms: float
    base_index: int  # index in waypoints of the first search point of the current mission
    seq_offset: int  # 0 = original mission; 1 = resumed
    next_index: int  # the point the vehicle was heading for
    interrupted_at: LatLon


class GotoResumePoint(SdModel):
    kind: Literal["goto"] = "goto"
    command_id: Id
    lat: float
    lon: float
    alt_m: float
    speed_ms: float


ResumePoint = Annotated[SurveyResumePoint | GotoResumePoint, Field(discriminator="kind")]
