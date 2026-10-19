"""The onboard mission executor: its states and the points it can resume from."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from sd_schemas.base import Id, LatLon, SdModel

__all__ = [
    "ExecutorState",
    "GotoResumePoint",
    "HeldBy",
    "HoldReason",
    "LinkLossPhase",
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
    "ai_hold",
    "skill_done",
    "operator_released",
    "limit",
    "link_loss",
    "find_hold",  # SD-link lost at a confirmed find: hovering beside it (link-loss rule)
    "ai_silent",
    "startup",
    "manual_input_lost",  # no stick value for MANUAL_CONTROL_TIMEOUT_MS while steering by hand
]


class LinkLossPhase(StrEnum):
    """Where the onboard link-loss rule is (sd_onboard.executor.link_loss).

    Times count from the last inbound SD-link message."""

    NONE = "none"  # the link is up
    GRACE = "grace"  # 3-23 s: the running step goes on
    HOLD = "hold"  # 23-53 s: hovering
    FIND_MOVE = "find_move"  # 3 to at most 15 s: flying to the hover point at a find
    FIND_HOVER = "find_hover"  # until 123 s: hovering beside the find
    HOME = "home"  # return home sent by the link-loss rule
    YIELDED = "yielded"  # someone else took over: a person, RC, the FC, another limit


HeldBy = Literal[
    "operator_stop",
    "operator_home",
    "operator_manual",  # the pilot steers by hand (Ta över); the AI's orders wait
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
    alt_m: float = Field(gt=0)
    speed_ms: float = Field(gt=0)
    base_index: int = Field(ge=0)  # index of the first search point of the current mission
    seq_offset: int = Field(ge=0, le=1)  # 0 = original mission; 1 = resumed
    next_index: int = Field(ge=0)  # the point the vehicle was heading for
    interrupted_at: LatLon
    path: bool = False  # the points are a drawn path (follow_path): resumes in TRANSIT

    @model_validator(mode="after")
    def _indices(self) -> SurveyResumePoint:
        if not self.base_index <= self.next_index < len(self.waypoints):
            raise ValueError("need base_index <= next_index < len(waypoints)")
        return self


class GotoResumePoint(SdModel):
    kind: Literal["goto"] = "goto"
    command_id: Id
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    alt_m: float = Field(gt=0)
    speed_ms: float = Field(gt=0)


ResumePoint = Annotated[SurveyResumePoint | GotoResumePoint, Field(discriminator="kind")]
