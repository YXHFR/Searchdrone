"""Reason codes, the SD-link catalogue, mission enums and the chat stop skeleton."""

from __future__ import annotations

import pytest

from sd_schemas import link
from sd_schemas.mission import MissionState, Trigger
from sd_schemas.reason_codes import (
    APPROVAL_OUTCOME_CODES,
    CHARTER_VALIDATION_CODES,
    GATE_REASON_CODES,
    ONBOARD_REASON_CODES,
    approval_codes,
    is_known,
    text_sv,
)
from sd_schemas.sv_commands import classify_override


def test_gate_codes_priority_order() -> None:
    codes = list(GATE_REASON_CODES)
    assert codes[0] == "SKILL_UNKNOWN"
    assert codes.index("AREA_BEYOND_FC_FENCE") == codes.index("OUTSIDE_CHARTER") + 1
    assert codes[-3:] == ["NEEDS_CLEARANCE", "NEEDS_APPROVAL_SKILL", "NEEDS_APPROVAL_MODE_A"]
    assert approval_codes() == {"NEEDS_CLEARANCE", "NEEDS_APPROVAL_SKILL", "NEEDS_APPROVAL_MODE_A"}


def test_codes_are_upper_snake_with_text() -> None:
    for table in (GATE_REASON_CODES, APPROVAL_OUTCOME_CODES, CHARTER_VALIDATION_CODES):
        for code, text in table.items():
            assert code.isupper() and " " not in code
            assert text.strip()
    for code in ONBOARD_REASON_CODES:
        assert code.startswith("ONB_")
    assert len(ONBOARD_REASON_CODES) == 33


def test_text_lookup() -> None:
    assert text_sv("ALT_ABOVE_LIMIT").startswith("Höjden")
    assert is_known("ONB_STOP_LATCHED")
    assert not is_known("G4_ALTITUDE")
    with pytest.raises(KeyError):
        text_sv("G4_ALTITUDE")


def test_mission_states_and_triggers() -> None:
    assert [s.value for s in MissionState] == [
        "DRAFT",
        "CHARTER_PROPOSED",
        "READY",
        "ACTIVE",
        "PAUSED",
        "HELD",
        "FAILSAFE",
        "RETURNING",
        "LANDED",
        "CLOSED",
    ]
    assert len(Trigger) == 27
    assert {"MANUAL_TAKEOVER", "MANUAL_HANDBACK"} <= {t.value for t in Trigger}


def test_kind_models_cover_the_catalogue() -> None:
    assert set(link.KIND_MODELS) == {
        "Hello",
        "HelloAck",
        "Heartbeat",
        "CharterInstall",
        "CharterAck",
        "SkillCommand",
        "CommandCancel",
        "Ack",
        "StateSnapshot",
        "OnboardEvent",
        "OperatorOverrideNotice",
        "OperatorRelease",
        "ClearanceConfirmation",
        "PilotPosition",
        "OperatorPosition",
        "CommanderStatus",
        "ManualTakeover",
        "ManualControl",
    }
    assert set(link.KIND_MODELS) >= link.RELIABLE_KINDS


def test_reliability() -> None:
    ack_done = link.Ack.model_validate(
        {"ref_id": "0192f4a1-7b90-7a22-b1c3-77aa00bb11cc", "stage": "done"}
    )
    ack_recv = ack_done.model_copy(update={"stage": "received"})
    assert link.is_reliable("Ack", ack_done)
    assert not link.is_reliable("Ack", ack_recv)
    assert link.is_reliable(
        "SkillCommand",
        link.CommandCancel.model_validate({"command_id": "0192f4a1-7b90-7a22-b1c3-77aa00bb11cc"}),
    )
    assert not link.is_reliable("ManualControl", link.Heartbeat())


def test_manual_control_bounds() -> None:
    base = {
        "takeover_id": "0192f4a1-7b90-7a22-b1c3-77aa00bb11cc",
        "input_seq": 0,
        "pitch": 0,
        "roll": 0,
        "throttle": 0.5,
        "yaw": 0,
    }
    link.ManualControl.model_validate(base)
    for key, value in (("pitch", 1.01), ("roll", -1.01), ("throttle", -0.1), ("yaw", 2)):
        with pytest.raises(ValueError):
            link.ManualControl.model_validate({**base, key: value})
    assert link.MANUAL_CONTROL_TIMEOUT_MS == 1000


@pytest.mark.parametrize("text", ["stopp", "flyg hem", "fortsätt", "", "STOPP!"])
def test_classify_override_is_a_skeleton_that_never_resumes(text: str) -> None:
    assert classify_override(text) in (None, "stop", "return_home")
    assert classify_override(text) != "resume"
