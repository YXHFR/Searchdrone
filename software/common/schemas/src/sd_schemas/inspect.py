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

``InspectClosed`` and ``InspectResumeEnded`` (how a look ended onboard and what the drone
does next; the end of the search it flies on afterwards) pass between the drone and the
ground only: the AI hears them as skill events, so they stay out of the package's exports
and the generated JSON Schema and TypeScript.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import AwareDatetime, Field, model_validator

from sd_schemas.base import Id, SdModel
from sd_schemas.charter import OperationMode
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
    "EVENT_CLOSED",
    "EVENT_DONE",
    "EVENT_LASER_LOST",
    "EVENT_LOWER_OFFER",
    "EVENT_PERMIT_CHANGED",
    "EVENT_RESUME_ENDED",
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
    "InspectCloseCause",
    "InspectClosed",
    "InspectDone",
    "InspectEnd",
    "InspectLaserLost",
    "InspectLowerOffer",
    "InspectNext",
    "InspectOutcomeKind",
    "InspectPhase",
    "InspectProgress",
    "InspectResumeEnded",
    "Inspection",
    "InspectionOutcome",
    "InspectionPhase",
    "Permit",
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
EVENT_CLOSED: Final = "inspect.closed"
EVENT_RESUME_ENDED: Final = "inspect.resume_ended"
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


# How the look ended onboard (``InspectClosed.cause``): flown back to the breakpoint
# (``resume`` issued), the way back refused, hovering over a confirmed find, another
# order took the drone, the look's own order failed, the link lost, or the executor left
# the look's state (STOP, FLY HOME, a takeover, a limit ...).
InspectCloseCause = Literal[
    "returned", "return_refused", "confirmed", "superseded", "failed", "link_loss", "left"
]

# What the drone does once the look is over (``InspectClosed.next``): flies back to where
# the search broke off (and, from a search, flies it on), hovers waiting for an order,
# flies home, is held by a person or the emergency path, flies another order, or follows
# the link-loss rule.
InspectNext = Literal["resume", "hold", "home", "held", "new_order", "link_loss"]


class InspectClosed(SdModel):
    """``inspect.closed``: the look is over onboard, once per look, whatever ended it.

    With ``next`` "resume" the executor flies back by its own ``resume`` order
    (``resume_command_id``, issued onboard: the ground has no row for it); from a search
    (``breakpoint`` "survey") that order flies the interrupted search on, whose command id
    is ``resumed_command_id``. From anywhere else (``breakpoint`` "goto") it only flies
    back to where the look began and hovers there."""

    command_id: Id  # the look's order
    candidate_id: Id
    cause: InspectCloseCause
    next: InspectNext
    breakpoint: Literal["survey", "goto"] | None = None  # with next "resume"
    resume_command_id: Id | None = None  # the executor's own resume (next "resume")
    resumed_command_id: Id | None = None  # the search flown on (breakpoint "survey")


class InspectResumeEnded(SdModel):
    """``inspect.resume_ended``: the executor's own ``resume`` after a look ended (done,
    failed, or replaced by anything else: another order, STOP, FLY HOME, a limit ...).
    For a search flown on, ``resumed_command_id`` is that search's order."""

    command_id: Id  # the look's order
    resumed_command_id: Id | None = None
    status: Literal["done", "failed", "superseded"]
    reason_codes: list[str] = Field(default_factory=list[str], max_length=8)


class PermitChanged(SdModel):
    """``permit.changed`` on the ground: the overflight permit setting for ``live``."""

    changed_by: str = Field(min_length=1, max_length=64)  # "operator:<alias>" or "config"
    has_specific_category_ref: bool
    allow_overflight_live: bool


class Permit(SdModel):
    """The ground's overflight permit for ``live`` (``GET``/``PUT /api/v1/permit``): the
    reference of the operator's permit in the specific category (Transportstyrelsen) and
    whether a closer look may then fly straight over its candidate. A charter approved
    in ``live`` carries ``overflight_allowed`` from it; sim and exercise always may."""

    specific_category_ref: str = Field(default="", max_length=64)
    allow_overflight_live: bool = False

    @model_validator(mode="after")
    def _needs_the_permit(self) -> Permit:
        if self.allow_overflight_live and not self.specific_category_ref.strip():
            raise ValueError("allow_overflight_live needs specific_category_ref")
        return self


# How a look ended on the ground (``Inspection.outcome``).
InspectionOutcome = Literal[
    # A person's decision while the look was open, sent onboard as ``link.InspectOutcome``:
    "confirmed",
    "rejected",
    "unsure",
    "resume",
    # The system's; the candidate goes back to ``ready`` (it stays in the queue):
    "timeout",  # no decision within AWAIT_DECISION_S after ``inspect.done``
    "resumed_by_pilot",  # another order replaced the look
    "aborted",  # STOP, FLY HOME, a takeover, the battery, the link ... (``detail``)
    "refused_onboard",
    "failed",
]

# Where a look is, as the ground sees it: the drone flies it; it asks to go lower; it
# hovers for the pilot's decision; it is over.
InspectionPhase = Literal["flying", "lower_offer", "await", "ended"]


class Inspection(SdModel):
    """A closer look as the ground keeps it (``GET /api/v1/missions/{id}/inspections``
    and ``inspection.updated`` on the live topic ``candidates``). At most one is open per
    mission (``ended_at`` None)."""

    command_id: Id  # the inspect_detection order
    candidate_id: Id
    mission_id: Id
    seq: int = Field(ge=1)  # the candidate's display number (K3)
    started_at: AwareDatetime
    started_by: str  # "operator:<alias>": the pilot's click, directly or on the AI's card
    operation_mode: OperationMode  # the charter's when the look started (unset: live)
    overflight: bool  # it flies straight over the candidate (else the side look)
    phase: InspectionPhase
    # The drone's question "Gå lägre?" while it waits for the pilot's yes, and until when.
    lower_offer: InspectLowerOffer | None = None
    lower_offer_until: AwareDatetime | None = None
    # The steps are over; the drone hovers for the decision until ``await_until``.
    done: InspectDone | None = None
    await_until: AwareDatetime | None = None
    ended_at: AwareDatetime | None = None
    outcome: InspectionOutcome | None = None
    detail: str | None = Field(default=None, max_length=64)  # e.g. the abort's reason
