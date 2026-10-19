"""The executor's states, the triggers that move it and the transition table.

``TRANSITIONS`` maps (from-state, trigger) to the allowed target states and the name of
the guard in ``executor/guards.py`` that picks the target or refuses. Anything not in the
table is refused with ``ONB_INVALID_TRANSITION``. Skills join the table as triggers
``skill:<name>``. A handler with its own states adds rows through ``TRANSITIONS_EXTRA``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, NamedTuple

from sd_schemas.executor import ExecutorState

__all__ = [
    "AIRBORNE_CONTROLLED",
    "MOVING",
    "ON_GROUND",
    "SAFETY_TRIGGERS",
    "TRANSITIONS",
    "Transition",
    "Trigger",
    "TriggerKey",
    "skill_key",
]

S = ExecutorState


class Trigger(StrEnum):
    CHARTER_INSTALL = "charter_install"  # a valid charter (an invalid one moves nothing)
    SKILL_DONE = "skill_done"
    SKILL_FAILED = "skill_failed"
    ORDER_CANCEL = "order_cancel"  # CommandCancel from the ground
    TAKEOFF_TIMEOUT = "takeoff_timeout"
    LIMIT_HOLD = "limit_hold"  # a limit monitor -> hold
    LIMIT_RTL = "limit_rtl"  # a limit monitor -> return_home
    LIMIT_ALT = "limit_alt"  # too high -> internal goto down
    LIMIT_FIND = "limit_find"  # SD-link lost at a confirmed find -> internal goto beside it
    SPOOF_DETECTED = "spoof_detected"  # L-SPOOF switched the EKF off GPS: hands off
    UNEXPECTED_MODE = "unexpected_mode"  # a mode change we did not ask for
    FAILSAFE_DETECTED = "failsafe_detected"
    OPERATOR_OVERRIDE = "operator_override"  # OperatorOverrideNotice (all kinds), takeover
    OPERATOR_RELEASE = "operator_release"  # OperatorRelease, handback
    FC_SILENT = "fc_silent"
    LANDED_DISARMED = "landed_disarmed"
    LANDED_TIMEOUT_5S = "landed_timeout_5s"


TriggerKey = Trigger | str  # str = "skill:<name>" for an accepted skill order


def skill_key(name: str) -> str:
    return f"skill:{name}"


class Transition(NamedTuple):
    targets: frozenset[ExecutorState]
    guard: str  # the name of a function in executor/guards.py


AIRBORNE_CONTROLLED: Final = frozenset({S.TAKEOFF, S.TRANSIT, S.SURVEY, S.HOLD, S.RTL})
MOVING: Final = frozenset({S.TAKEOFF, S.TRANSIT, S.SURVEY, S.HOLD})  # limit monitors run here
ON_GROUND: Final = frozenset({S.IDLE, S.PREFLIGHT, S.LANDED})
# Every state we steer in must answer these (test_safety_triggers_cover_all_states).
SAFETY_TRIGGERS: Final = (
    Trigger.OPERATOR_OVERRIDE,
    Trigger.UNEXPECTED_MODE,
    Trigger.FAILSAFE_DETECTED,
    Trigger.FC_SILENT,
    Trigger.LANDED_DISARMED,
)

_TAKEN_OVER: Final = frozenset({S.HELD, S.FAILSAFE})
_STEERING: Final = frozenset({S.HOLD, S.TRANSIT, S.SURVEY})
_LIMITED: Final = frozenset({S.TAKEOFF, S.TRANSIT, S.SURVEY, S.HOLD})
_RELEASE_TARGETS: Final = frozenset({S.HOLD, S.RTL, S.LANDED, S.PREFLIGHT})


def _rows() -> dict[tuple[ExecutorState, TriggerKey], Transition]:
    rows: dict[tuple[ExecutorState, TriggerKey], Transition] = {}

    def add(
        sources: frozenset[ExecutorState] | set[ExecutorState],
        key: TriggerKey,
        targets: set[ExecutorState] | None,
        guard: str,
    ) -> None:
        for state in sources:
            to = frozenset(targets) if targets is not None else frozenset({state})
            rows[(state, key)] = Transition(to, guard)

    tc = Trigger
    add({S.IDLE, S.PREFLIGHT, S.LANDED}, tc.CHARTER_INSTALL, {S.PREFLIGHT}, "always")  # T1
    add(
        AIRBORNE_CONTROLLED | _TAKEN_OVER, tc.CHARTER_INSTALL, None, "charter_inflight_compatible"
    )  # T2
    add({S.PREFLIGHT}, skill_key("takeoff"), {S.TAKEOFF}, "always")  # T3
    add({S.TAKEOFF}, tc.SKILL_DONE, {S.HOLD}, "always")  # T4
    for key in (tc.TAKEOFF_TIMEOUT, tc.SKILL_FAILED, tc.ORDER_CANCEL):  # T5
        add({S.TAKEOFF}, key, {S.HOLD, S.PREFLIGHT}, "airborne_or_ground")
    add(_STEERING, skill_key("goto"), {S.TRANSIT}, "always")  # T6
    add(_STEERING, skill_key("survey_sector"), {S.SURVEY}, "always")  # T7
    add(_STEERING, skill_key("follow_path"), {S.TRANSIT}, "always")  # T7b: a drawn path
    add(_STEERING, skill_key("hold"), {S.HOLD}, "always")  # T8
    add({S.HOLD}, skill_key("resume"), {S.TRANSIT, S.SURVEY}, "resume_target")  # T9
    add(_LIMITED, skill_key("return_home"), {S.RTL}, "always")  # T10
    add({S.RTL}, skill_key("return_home"), {S.RTL}, "always")  # T11
    add({S.TRANSIT, S.SURVEY}, tc.SKILL_DONE, {S.HOLD}, "always")  # T12
    for key in (tc.SKILL_FAILED, tc.ORDER_CANCEL):  # T13
        add({S.TRANSIT, S.SURVEY}, key, {S.HOLD}, "always")
    add(_LIMITED, tc.LIMIT_HOLD, {S.HOLD}, "always")  # T14
    add(_LIMITED, tc.LIMIT_RTL, {S.RTL}, "always")  # T15
    add(_LIMITED, tc.LIMIT_ALT, {S.TRANSIT}, "always")  # T16
    add(AIRBORNE_CONTROLLED, tc.UNEXPECTED_MODE, {S.HELD}, "always")  # T17
    add(AIRBORNE_CONTROLLED, tc.FAILSAFE_DETECTED, {S.FAILSAFE}, "always")  # T18
    add({S.HELD}, tc.FAILSAFE_DETECTED, {S.FAILSAFE}, "within_classification_window")  # T19
    add(AIRBORNE_CONTROLLED | _TAKEN_OVER, tc.OPERATOR_OVERRIDE, {S.HELD}, "always")  # T20
    add({S.HELD}, tc.OPERATOR_RELEASE, set(_RELEASE_TARGETS), "release_target")  # T21
    add({S.FAILSAFE}, tc.OPERATOR_RELEASE, set(_RELEASE_TARGETS), "failsafe_release")  # T22
    add(AIRBORNE_CONTROLLED, tc.FC_SILENT, {S.HELD}, "always")  # T23
    add(
        AIRBORNE_CONTROLLED | _TAKEN_OVER, tc.LANDED_DISARMED, {S.LANDED}, "landed_on_ground"
    )  # T24
    add({S.LANDED}, tc.LANDED_TIMEOUT_5S, {S.PREFLIGHT}, "always")  # T25
    add(_STEERING, tc.LIMIT_FIND, {S.TRANSIT, S.HOLD}, "find_target")  # T26
    # T27: SKILL_DONE of the internal find goto is T12 with hold_reason "find_hold".
    add(_LIMITED | {S.RTL, S.FAILSAFE}, tc.SPOOF_DETECTED, {S.FAILSAFE}, "always")  # T28
    return rows


TRANSITIONS: Final[dict[tuple[ExecutorState, TriggerKey], Transition]] = _rows()
