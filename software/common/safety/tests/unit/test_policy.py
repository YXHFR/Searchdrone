"""The hard-coded policy, the priority of the reason codes and the constants."""

from __future__ import annotations

from sd_safety import constants, policy
from sd_safety.priority import REASON_PRIORITY, message_sv, sort_codes
from sd_schemas import skills
from sd_schemas.gate import CLEARANCE_VALID_MS
from sd_schemas.reason_codes import GATE_REASON_CODES
from sd_schemas.skills import SKILL_REGISTRY, SkillRegistry

# The owner's priority list, plus the two codes added since (manual control, FC fence).
SPEC_PRIORITY = [
    "SKILL_UNKNOWN",
    "SCHEMA_INVALID",
    "SKILL_NOT_IMPLEMENTED",
    "FORBIDDEN_ACTION",
    "AUTONOMY_CHANGE_FORBIDDEN",
    "SKILL_NOT_ALLOWED",
    "AI_DISABLED",
    "MISSION_STATE_INVALID",
    "RELEASE_REQUIRED",
    "RESUME_NEEDS_OPERATOR",
    "OPERATOR_OVERRIDE_ACTIVE",
    "FAILSAFE_ACTIVE",
    "CONTROL_HELD",
    "MANUAL_CONTROL_ACTIVE",
    "CHARTER_NOT_APPROVED",
    "CHARTER_NOT_INSTALLED",
    "CHARTER_VERSION_MISMATCH",
    "FENCE_NOT_VERIFIED",
    "LINK_DOWN",
    "STATE_STALE",
    "STATE_CHANGED",
    "INVALID_TRANSITION",
    "CLEARANCE_REQUIRED",
    "CLEARANCE_EXPIRED",
    "RC_LINK_DOWN",
    "TAKEOFF_PRECONDITION",
    "BATTERY_RESERVE",
    "BATTERY_UNKNOWN",
    "ALT_ABOVE_LIMIT",
    "ALT_ABOVE_CHARTER",
    "ALT_BELOW_CHARTER",
    "OUTSIDE_CHARTER",
    "AREA_BEYOND_FC_FENCE",
    "PILOT_DISTANCE",
    "SPEED_ABOVE_LIMIT",
    "SPEED_ABOVE_CHARTER",
    "OPERATOR_ABSENT",
    "PENDING_LIMIT",
    "RATE_LIMIT",
    "NEEDS_CLEARANCE",
    "NEEDS_APPROVAL_SKILL",
    "NEEDS_APPROVAL_MODE_A",
]


def test_reason_priority() -> None:
    assert list(REASON_PRIORITY) == SPEC_PRIORITY
    assert set(REASON_PRIORITY) == set(GATE_REASON_CODES)


def test_sort_and_message() -> None:
    assert sort_codes(["RATE_LIMIT", "SKILL_UNKNOWN", "RATE_LIMIT"]) == [
        "SKILL_UNKNOWN",
        "RATE_LIMIT",
    ]
    assert message_sv([]) == "Ordern är tillåten."
    assert message_sv(["LINK_DOWN"]) == GATE_REASON_CODES["LINK_DOWN"]
    assert message_sv(["LINK_DOWN", "RATE_LIMIT", "PENDING_LIMIT"]).endswith("(och 2 till)")


def test_forbidden_skill_names_are_equal() -> None:
    assert policy.FORBIDDEN_SKILL_NAMES == skills.FORBIDDEN_SKILL_NAMES


def test_policy_matches_registry() -> None:
    names = SKILL_REGISTRY.names()
    assert len(SKILL_REGISTRY) == 18
    assert not names & policy.FORBIDDEN_SKILL_NAMES
    for name in policy.ALWAYS_REQUIRE_APPROVAL:
        assert SKILL_REGISTRY.get(name).requires_approval_in_b, name
    for group in (
        policy.ALWAYS_REQUIRE_APPROVAL,
        policy.CLEARANCE_EXEMPT,
        policy.ALLOWED_OUTSIDE_CHARTER_SKILLS,
        policy.UNDER_OVERRIDE_ALLOWED,
        policy.MODE_A_EXEMPT,
        policy.OPERATOR_ALLOWED_UNDER_LATCH,
        policy.GROUND_UNCHECKED_TARGETS,
    ):
        assert group <= names
    for name in policy.UNDER_OVERRIDE_ALLOWED:
        spec = SKILL_REGISTRY.get(name)
        assert spec.allowed_under_latch and not spec.moves_vehicle
    assert not SKILL_REGISTRY.get("return_home").allowed_under_latch


def test_owner_policy_sets_are_complete() -> None:
    assert {"follow_track", "follow_me", "find_me", "land_at", "propose_charter_change"} <= (
        policy.ALWAYS_REQUIRE_APPROVAL
    )
    assert {
        "arm",
        "disarm",
        "set_param",
        "set_mode",
        "disable_fence",
        "set_fence",
        "set_autonomy",
        "set_autonomy_mode",
        "change_autonomy",
        "confirm_detection",
        "confirm_candidate",
        "set_velocity",
        "set_attitude",
    } <= policy.FORBIDDEN_SKILL_NAMES


def test_registry_refuses_every_forbidden_name() -> None:
    hold = SKILL_REGISTRY.get("hold")
    for name in policy.FORBIDDEN_SKILL_NAMES:
        try:
            SkillRegistry([hold.model_copy(update={"name": name})])
        except ValueError:
            continue
        raise AssertionError(name)


def test_constants() -> None:
    assert constants.CLEARANCE_VALID_S <= 120
    assert constants.CLEARANCE_VALID_S * 1000 == CLEARANCE_VALID_MS
    assert constants.APPROVAL_TIMEOUT_S == 60
    assert constants.BRAIN_BATTERY_MARGIN_PCT == 5
    assert constants.TAKEOFF_BATTERY_MARGIN_PCT == 20
    assert constants.FENCE_POLYGON_MARGIN_M == 10
    assert constants.SDLINK_GRACE_S == constants.FS_GCS_TIMEOUT_S == 20
