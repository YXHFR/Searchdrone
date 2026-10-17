"""The mission's state machine: one table, one pure function.

``next_state(current, trigger, ctx)`` walks ``TRANSITIONS`` in order and returns the
``to`` of the first row whose ``from_`` holds ``current``, whose trigger matches and
whose condition holds. ``None`` means "unchanged": no transition and no event. A latch
the caller already set still applies.

The pilot's takeover (``MANUAL_TAKEOVER`` / ``MANUAL_HANDBACK``) has no row: steering by
hand is a flag on the vehicle (``GateContext.manual_control``), not a mission state, so a
hand-back can never lift a STOP.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final, Literal, NamedTuple

from sd_schemas.mission import MissionState, MissionStatusFull, TransitionCtx, Trigger

__all__ = [
    "AIR",
    "ALL_STATES",
    "GROUND",
    "TRANSITIONS",
    "Cond",
    "TransitionRow",
    "next_state",
]

S = MissionState
Cond = Literal["in_flight", "not in_flight", "fc_home_mode", "not fc_home_mode", "-"]


def _f(state: MissionState) -> MissionStatusFull:
    return MissionStatusFull(state)


def _p(by: Literal["operator", "brain", "system"]) -> MissionStatusFull:
    return MissionStatusFull(S.PAUSED, paused_by=by)


def _r(
    cause: Literal["brain", "operator", "battery", "failsafe", "system"],
) -> MissionStatusFull:
    return MissionStatusFull(S.RETURNING, returning_cause=cause)


P_ALL: Final = frozenset({_p("operator"), _p("brain"), _p("system")})
R_ALL: Final = frozenset({_r("brain"), _r("operator"), _r("battery"), _r("failsafe"), _r("system")})
AIR: Final = frozenset({_f(S.ACTIVE), *P_ALL, *R_ALL, _f(S.HELD), _f(S.FAILSAFE)})
GROUND: Final = frozenset({_f(S.DRAFT), _f(S.CHARTER_PROPOSED), _f(S.READY), _f(S.LANDED)})
ALL_STATES: Final = frozenset({*AIR, *GROUND, _f(S.CLOSED)})


class TransitionRow(NamedTuple):
    no: int
    from_: frozenset[MissionStatusFull | None]
    trigger: Trigger
    cond: Cond
    to: MissionStatusFull


def _row(
    no: int,
    from_: Iterable[MissionStatusFull | None],
    trigger: Trigger,
    cond: Cond,
    to: MissionStatusFull,
) -> TransitionRow:
    return TransitionRow(no, frozenset(from_), trigger, cond, to)


T = Trigger
ACTIVE, HELD, FAILSAFE = _f(S.ACTIVE), _f(S.HELD), _f(S.FAILSAFE)

TRANSITIONS: Final[tuple[TransitionRow, ...]] = (
    _row(1, [None], T.MISSION_OPENED, "-", _f(S.DRAFT)),
    _row(2, [_f(S.DRAFT)], T.CHARTER_PROPOSED, "-", _f(S.CHARTER_PROPOSED)),
    _row(3, [_f(S.CHARTER_PROPOSED)], T.PROPOSALS_CLEARED, "-", _f(S.DRAFT)),
    _row(4, [_f(S.DRAFT), _f(S.CHARTER_PROPOSED)], T.CHARTER_READY, "-", _f(S.READY)),
    _row(5, [_f(S.READY)], T.TAKEOFF_ACCEPTED, "-", ACTIVE),
    _row(6, AIR, T.OPERATOR_STOP, "in_flight", _p("operator")),
    _row(
        7,
        AIR - {_r("battery"), _r("failsafe"), _r("operator")},
        T.OPERATOR_RTL,
        "in_flight",
        _r("operator"),
    ),
    _row(
        8,
        [_p("operator"), _p("system")],
        T.OPERATOR_RESUME_CONFIRMED,
        "not fc_home_mode",
        ACTIVE,
    ),
    _row(9, [HELD, FAILSAFE], T.OPERATOR_RESUME_CONFIRMED, "not fc_home_mode", ACTIVE),
    _row(10, [HELD, FAILSAFE], T.OPERATOR_RESUME_CONFIRMED, "fc_home_mode", _r("operator")),
    _row(11, [ACTIVE, _p("brain")], T.WATCHDOG_STOP, "in_flight", _p("system")),
    _row(12, [ACTIVE, _p("brain"), _p("system")], T.WATCHDOG_RTL, "in_flight", _r("system")),
    _row(13, [ACTIVE, _p("brain")], T.OPERATOR_ABSENT_HOLD, "in_flight", _p("system")),
    _row(14, [ACTIVE, _p("brain")], T.OPERATOR_ABSENT_RTL, "in_flight", _r("system")),
    _row(15, [ACTIVE], T.BRAIN_HOLD, "-", _p("brain")),
    _row(16, [_p("brain")], T.BRAIN_HOLD_END, "-", ACTIVE),
    _row(17, [ACTIVE, _p("brain")], T.BRAIN_RTL_ACCEPTED, "-", _r("brain")),
    _row(
        18,
        [ACTIVE, *P_ALL, _r("brain"), _r("system")],
        T.BATTERY_RTL,
        "in_flight",
        _r("battery"),
    ),
    _row(19, [ACTIVE, *P_ALL, *R_ALL], T.CONTROL_TAKEN, "-", HELD),
    _row(20, [ACTIVE, *P_ALL, *R_ALL, HELD], T.FC_FAILSAFE, "-", FAILSAFE),
    _row(21, [ACTIVE, _p("brain"), _r("brain")], T.LATCH_ADOPTED_STOP, "in_flight", _p("system")),
    _row(
        22,
        [ACTIVE, _p("brain"), _p("system"), _r("brain")],
        T.LATCH_ADOPTED_RTL,
        "in_flight",
        _r("system"),
    ),
    _row(23, [ACTIVE, _p("brain"), _r("brain")], T.RESTART_FC_STOPPED, "in_flight", _p("system")),
    _row(
        24,
        [ACTIVE, _p("brain"), _p("system"), _r("brain")],
        T.RESTART_FC_RETURNING,
        "in_flight",
        _r("system"),
    ),
    _row(25, AIR, T.LANDED_DISARMED, "not in_flight", _f(S.LANDED)),
    _row(26, [_f(S.LANDED)], T.CLOSE_AUTO, "-", _f(S.CLOSED)),
    _row(27, GROUND | AIR, T.CLOSE_MANUAL, "not in_flight", _f(S.CLOSED)),
)


def _holds(cond: Cond, ctx: TransitionCtx) -> bool:
    if cond == "-":
        return True
    if cond == "in_flight":
        return ctx.in_flight
    if cond == "not in_flight":
        return not ctx.in_flight
    if cond == "fc_home_mode":
        return ctx.fc_home_mode
    return not ctx.fc_home_mode


def next_state(
    current: MissionStatusFull | None, trigger: Trigger, ctx: TransitionCtx
) -> MissionStatusFull | None:
    """The state after ``trigger``, or ``None`` when nothing changes."""
    for row in TRANSITIONS:
        if row.trigger is trigger and current in row.from_ and _holds(row.cond, ctx):
            return None if row.to == current else row.to
    return None
