"""The transition table is the expected one, and every steered state answers the
safety triggers."""

from __future__ import annotations

from pathlib import Path

import yaml

from sd_onboard.executor.guards import GUARDS
from sd_onboard.executor.state import (
    AIRBORNE_CONTROLLED,
    MOVING,
    ON_GROUND,
    SAFETY_TRIGGERS,
    TRANSITIONS,
    Transition,
)
from sd_onboard.skills import HANDLERS, extra_transitions
from sd_schemas.executor import ExecutorState as S

SPEC = Path(__file__).with_name("transitions_spec.yaml")


def _expected() -> dict[tuple[S, str], Transition]:
    rows: dict[tuple[S, str], Transition] = {}
    for row in yaml.safe_load(SPEC.read_text()):
        for state in row["from"]:
            for trigger in row["trigger"]:
                to = [state] if row["to"] == "same" else row["to"]
                key = (S(state), trigger)
                assert key not in rows, f"{row['id']}: {key} twice"
                rows[key] = Transition(frozenset(S(t) for t in to), row["guard"])
    return rows


def test_transitions_table_matches_spec() -> None:
    actual = {(s, str(k)): row for (s, k), row in TRANSITIONS.items()}
    assert actual == _expected()


def test_every_guard_exists() -> None:
    assert {row.guard for row in TRANSITIONS.values()} <= set(GUARDS)


def test_safety_triggers_cover_all_states() -> None:
    for state in AIRBORNE_CONTROLLED:
        for trigger in SAFETY_TRIGGERS:
            assert (state, trigger) in TRANSITIONS, (state, trigger)


def test_no_transition_to_land_v1() -> None:
    assert all(S.LAND not in row.targets for row in TRANSITIONS.values())
    assert all(state != S.LAND for state, _ in TRANSITIONS)


def test_state_sets() -> None:
    assert AIRBORNE_CONTROLLED - {S.RTL} == MOVING
    assert not ON_GROUND & AIRBORNE_CONTROLLED


def test_no_handler_adds_rows_yet() -> None:
    assert extra_transitions() == {}
    assert set(HANDLERS) == {
        "takeoff",
        "goto",
        "survey_sector",
        "follow_path",
        "hold",
        "resume",
        "return_home",
        "orbit",
        "inspect_detection",
        "capture_stills",
    }
