"""The closer look (``inspect_detection`` v2, "Titta närmare"): its numbers and messages.

The drone flies over a candidate and comes down in steps, both cameras looking straight
down, until the picture is clear or the height floor is reached; then it hovers and
waits for the pilot's decision (``AWAIT_DECISION_S``) and flies back to where the search
broke off. In ``live`` without the overflight permit it looks from the cylinder's edge
instead, from one height with the main camera only (the stage-2 side look).

Every number of the step-wise approach lives here, so the executor, the gate and the
simulator use the same values. They are a first guess, to be tuned in the simulator.
The events and the progress below are the onboard computer's reports of a look; the
ground's answers are ``sd_schemas.link.InspectOutcome`` and ``InspectLower``.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import Field

from sd_schemas.base import Id, SdModel
from sd_schemas.keepout import INSPECT_SPEED_MS

__all__ = [
    "AUTO_FLOOR_ABOVE_GROUND_M",
    "AWAIT_DECISION_S",
    "CLEAR_PX",
    "CLEAR_SCORE",
    "CLIMB_MIN_MS",
    "DESCENT_MAX_MS",
    "DESCENT_SLOW_MS",
    "EVENT_ABORTED",
    "EVENT_DONE",
    "EVENT_LASER_LOST",
    "EVENT_LOWER_OFFER",
    "EVENT_PERMIT_CHANGED",
    "INSPECT_MAX_S",
    "INSPECT_SPEED_MS",
    "LOWER_OFFER_S",
    "MATCH_RADIUS_MIN_M",
    "STEPS_MAX",
    "STEP_DOWN_M",
    "STEP_MIN_M",
    "STEP_ROUNDS",
    "STEP_ROUND_INTERVAL_S",
    "STEP_SETTLE_S",
    "FloorReason",
    "InspectAbortReason",
    "InspectAborted",
    "InspectDone",
    "InspectEnd",
    "InspectLaserLost",
    "InspectLowerOffer",
    "InspectOutcomeKind",
    "InspectPhase",
    "InspectProgress",
    "PermitChanged",
    "match_radius_m",
]

STEP_DOWN_M: Final = 10.0  # each step down, never below the floor in force
STEP_MIN_M: Final = 2.0  # a shorter step than this counts as "floor reached"
STEP_SETTLE_S: Final = 1.5  # hover before the stills, so they are not blurred
STEP_ROUNDS: Final = 2  # rounds of stills per step: one from each camera per round
STEP_ROUND_INTERVAL_S: Final = 1.0
STEPS_MAX: Final = 8  # steps down after step 0 (over the target)
MATCH_RADIUS_MIN_M: Final = 15.0  # a detection this near the target's estimate is the target
CLEAR_SCORE: Final = 0.5  # a clear picture: the target at least this sure ...
CLEAR_PX: Final = 96  # ... and this big (full sensor pixels), in two stills of one step
INSPECT_MAX_S: Final = 180.0  # from the start; the pilot's LOWER_OFFER_S does not count
LOWER_OFFER_S: Final = 60  # = sd_safety.constants.APPROVAL_TIMEOUT_S (a test checks)
AWAIT_DECISION_S: Final = 120.0  # hovering for the pilot's decision, then back to the search
AUTO_FLOOR_ABOVE_GROUND_M: Final = 30.0  # lower than this above ground only on the pilot's yes
DESCENT_MAX_MS: Final = 1.5
DESCENT_SLOW_MS: Final = 1.0  # below the map's floor without the laser, so the laser can see
CLIMB_MIN_MS: Final = 2.5  # climbing away when the link or the laser is lost


def match_radius_m(sigma95_m: float) -> float:
    """How near the target's estimate a person detection must be to count as the target."""
    return max(MATCH_RADIUS_MIN_M, 2.0 * sigma95_m)


# The pilot's decision while the drone waits (``link.InspectOutcome``): "rejected" (not
# them), "unsure", "resume" (search on now) fly back to where the search broke off;
# "confirmed" hovers over the find for the pilot's next order.
InspectOutcomeKind = Literal["rejected", "unsure", "resume", "confirmed"]

# What the look is doing (``InspectProgress.phase``).
InspectPhase = Literal["approach", "step", "lower_offer", "await", "return"]

# Why the floor is where it is (``sd_safety.floor``).
FloorReason = Literal["map", "unknown", "powerline", "laser", "no_map", "laser_lost"]

# Why the steps ended and the drone went to wait (``InspectDone.end``).
InspectEnd = Literal[
    "clear",  # a clear picture of the target
    "floor",  # the lowest allowed height reached
    "lower_declined",  # the pilot said no to going lower, or did not answer in time
    "steps_max",
    "timeout",  # INSPECT_MAX_S
    "keepout",  # a new cylinder blocks the next step and there is no way around
    "laser_lost",
    "side_look",  # live without the permit: the look from the cylinder's edge is done
]

# Why a look ended without the wait (``InspectAborted.reason``).
InspectAbortReason = Literal[
    "stop", "home", "override", "battery", "gnss", "link_loss", "refused_onboard", "failed"
]

# Onboard event types (``link.OnboardEvent.type``) and the ground's own event log.
EVENT_LOWER_OFFER: Final = "inspect.lower_offer"
EVENT_DONE: Final = "inspect.done"
EVENT_ABORTED: Final = "inspect.aborted"
EVENT_LASER_LOST: Final = "inspect.laser_lost"
EVENT_PERMIT_CHANGED: Final = "permit.changed"

_AboveGround = Field(ge=0, le=1000)  # metres above the ground under the drone
_AboveGroundOrNone = Field(default=None, ge=0, le=1000)


class InspectProgress(SdModel):
    """How a look is going, in the onboard computer's state snapshot (at most 2 Hz)."""

    command_id: Id
    candidate_id: Id
    step: int = Field(ge=0)  # 0 = over the target (or at the cylinder's edge in live)
    phase: InspectPhase
    above_ground_m: float | None = _AboveGroundOrNone
    floor_auto_above_ground_m: float | None = _AboveGroundOrNone
    floor_lowest_above_ground_m: float | None = _AboveGroundOrNone
    floor_reason: FloorReason | None = None
    laser_confirmed: bool = False
    await_left_s: float | None = Field(default=None, ge=0)  # while waiting for the decision


class InspectLowerOffer(SdModel):
    """``inspect.lower_offer``: the next step would go below the automatic floor (30 m
    above ground); the drone hovers and asks the pilot (``link.InspectLower``)."""

    command_id: Id
    now_above_ground_m: float = _AboveGround
    lowest_above_ground_m: float = _AboveGround
    reason: FloorReason


class InspectDone(SdModel):
    """``inspect.done``: the steps are over and the drone hovers for the pilot's decision.
    The candidate stays ``inspecting`` until it comes, or ``AWAIT_DECISION_S`` passes."""

    command_id: Id
    candidate_id: Id
    end: InspectEnd
    steps: int = Field(ge=0)
    lowest_above_ground_m: float | None = _AboveGroundOrNone
    still_ids: list[Id] = Field(default_factory=list[Id], max_length=200)
    clear: bool


class InspectAborted(SdModel):
    """``inspect.aborted``: the look ended without the wait (STOP, FLY HOME, a takeover,
    the battery, the link ...); the candidate goes back to ``ready``. Never resumed."""

    command_id: Id
    candidate_id: Id
    reason: InspectAbortReason


class InspectLaserLost(SdModel):
    """``inspect.laser_lost``: the laser stopped confirming below the map's floor without
    it; the drone climbs to that floor and goes to wait."""

    command_id: Id
    above_ground_m: float | None = _AboveGroundOrNone


class PermitChanged(SdModel):
    """``permit.changed`` on the ground: the overflight permit setting for ``live``."""

    changed_by: str = Field(min_length=1, max_length=64)  # "operator:<alias>" or "config"
    has_specific_category_ref: bool
    allow_overflight_live: bool
