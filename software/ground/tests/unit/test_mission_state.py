"""T28 and T19: the mission's transition table and the normal flight through it."""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from sd_ground.services.mission_state import ALL_STATES, TRANSITIONS, next_state
from sd_schemas.mission import MissionState, MissionStatusFull, TransitionCtx, Trigger

FIXTURE = Path(__file__).parent / "fixtures" / "transitions_8_2.yaml"
S = MissionState
_PAUSED = ("operator", "brain", "system")
_RETURNING = ("brain", "operator", "battery", "failsafe", "system")


def _expand(token: str) -> set[MissionStatusFull | None]:
    if token == "-":
        return {None}
    if token == "AIR":
        return (
            {MissionStatusFull(S.ACTIVE), MissionStatusFull(S.HELD), MissionStatusFull(S.FAILSAFE)}
            | _expand("P(*)")
            | _expand("R(*)")
        )
    if token == "GROUND":
        return {MissionStatusFull(s) for s in (S.DRAFT, S.CHARTER_PROPOSED, S.READY, S.LANDED)}
    match = re.fullmatch(r"([PR])\((\*|[a-z]+)\)", token)
    if match is None:
        return {MissionStatusFull(S(token))}
    kind, sub = match.groups()
    if kind == "P":
        subs = _PAUSED if sub == "*" else (sub,)
        return {MissionStatusFull(S.PAUSED, paused_by=x) for x in subs}  # type: ignore[arg-type]
    subs = _RETURNING if sub == "*" else (sub,)
    return {MissionStatusFull(S.RETURNING, returning_cause=x) for x in subs}  # type: ignore[arg-type]


def _one(token: str) -> MissionStatusFull:
    (value,) = _expand(token)
    assert value is not None
    return value


def _rows() -> list[dict[str, Any]]:
    data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    out = []
    for raw in data["rows"]:
        states: set[MissionStatusFull | None] = set()
        for token in raw["from"]:
            states |= _expand(token)
        out.append(
            {
                "no": raw["row"],
                "from": frozenset(states),
                "trigger": Trigger(raw["trigger"]),
                "cond": raw["cond"],
                "to": _one(raw["to"]),
            }
        )
    return out


ROWS = _rows()
CTXS = [TransitionCtx(a, b) for a, b in itertools.product([False, True], repeat=2)]


def _holds(cond: str, ctx: TransitionCtx) -> bool:
    return {
        "-": True,
        "in_flight": ctx.in_flight,
        "not in_flight": not ctx.in_flight,
        "fc_home_mode": ctx.fc_home_mode,
        "not fc_home_mode": not ctx.fc_home_mode,
    }[cond]


def _expected(
    current: MissionStatusFull | None, trigger: Trigger, ctx: TransitionCtx
) -> MissionStatusFull | None:
    for row in ROWS:
        if row["trigger"] is trigger and current in row["from"] and _holds(row["cond"], ctx):
            return None if row["to"] == current else row["to"]
    return None


def test_sixteen_states() -> None:
    assert len(ALL_STATES) == 16


def test_table_matches_fixture_row_by_row() -> None:
    assert len(TRANSITIONS) == len(ROWS) == 27
    for code, row in zip(TRANSITIONS, ROWS, strict=True):
        assert code.no == row["no"]
        assert code.trigger is row["trigger"], code.no
        assert code.cond == row["cond"], code.no
        assert code.to == row["to"], code.no
        assert code.from_ == row["from"], code.no


@pytest.mark.parametrize("trigger", list(Trigger))
def test_every_combination(trigger: Trigger) -> None:
    for current in [None, *sorted(ALL_STATES, key=repr)]:
        for ctx in CTXS:
            assert next_state(current, trigger, ctx) == _expected(current, trigger, ctx), (
                current,
                trigger,
                ctx,
            )


def test_takeover_never_changes_the_state() -> None:
    for current, ctx in itertools.product(ALL_STATES, CTXS):
        assert next_state(current, Trigger.MANUAL_TAKEOVER, ctx) is None
        assert next_state(current, Trigger.MANUAL_HANDBACK, ctx) is None


def test_t19_normal_flight() -> None:
    """DRAFT -> CHARTER_PROPOSED -> READY -> ACTIVE -> PAUSED -> ACTIVE -> RETURNING ->
    LANDED -> CLOSED."""
    air = TransitionCtx(in_flight=True, fc_home_mode=False)
    home = TransitionCtx(in_flight=True, fc_home_mode=True)
    ground = TransitionCtx(in_flight=False, fc_home_mode=False)
    steps = [
        (None, Trigger.MISSION_OPENED, ground, S.DRAFT),
        (S.DRAFT, Trigger.CHARTER_PROPOSED, ground, S.CHARTER_PROPOSED),
        (S.CHARTER_PROPOSED, Trigger.CHARTER_READY, ground, S.READY),
        (S.READY, Trigger.TAKEOFF_ACCEPTED, air, S.ACTIVE),
        (S.ACTIVE, Trigger.OPERATOR_STOP, air, S.PAUSED),
        (S.PAUSED, Trigger.OPERATOR_RESUME_CONFIRMED, air, S.ACTIVE),
        (S.ACTIVE, Trigger.BRAIN_RTL_ACCEPTED, home, S.RETURNING),
        (S.RETURNING, Trigger.LANDED_DISARMED, ground, S.LANDED),
        (S.LANDED, Trigger.CLOSE_AUTO, ground, S.CLOSED),
    ]
    current: MissionStatusFull | None = None
    for start, trigger, ctx, want in steps:
        assert (None if current is None else current.state) == start
        nxt = next_state(current, trigger, ctx)
        assert nxt is not None and nxt.state is want, (start, trigger)
        current = nxt


def test_resume_refused_while_flying_home() -> None:
    ctx = TransitionCtx(in_flight=True, fc_home_mode=True)
    for cause in ("operator", "system"):
        cur = MissionStatusFull(S.RETURNING, returning_cause=cause)  # type: ignore[arg-type]
        assert next_state(cur, Trigger.OPERATOR_RESUME_CONFIRMED, ctx) is None


def test_stop_on_the_ground_keeps_the_state() -> None:
    ground = TransitionCtx(in_flight=False, fc_home_mode=False)
    for s in (S.DRAFT, S.READY, S.LANDED):
        assert next_state(MissionStatusFull(s), Trigger.OPERATOR_STOP, ground) is None
