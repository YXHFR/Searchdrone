"""Hard-coded protection the gate applies whatever the skill registry says.

The registry can be wrong. These sets are the minimum the gate always enforces; a test
checks that the registry agrees with them.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "ALLOWED_OUTSIDE_CHARTER_SKILLS",
    "ALWAYS_REQUIRE_APPROVAL",
    "CLEARANCE_EXEMPT",
    "FORBIDDEN_SKILL_NAMES",
    "GROUND_UNCHECKED_TARGETS",
    "MODE_A_EXEMPT",
    "OPERATOR_ALLOWED_UNDER_LATCH",
    "UNDER_OVERRIDE_ALLOWED",
]

# Always an approval card, in every autonomy mode.
ALWAYS_REQUIRE_APPROVAL: Final = frozenset(
    {"follow_track", "follow_me", "find_me", "land_at", "propose_charter_change", "follow_path"}
)

# Never skills: they would let an order bypass the gate's model of the vehicle (arming, modes,
# parameters, the fence, the autonomy mode) or confirm a find for a person. The gate answers
# SKILL_UNKNOWN even if a registry has one.
FORBIDDEN_SKILL_NAMES: Final = frozenset(
    {
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
    }
)

# Need no "clear around the drone" confirmation on the ground.
CLEARANCE_EXEMPT: Final = frozenset(
    {
        "report",
        "mark_location",
        "hold",
        "return_home",
        "set_camera_tilt",
        "capture_stills",
        "propose_charter_change",
    }
)

# The AI may use these even when the charter's allowed_skills leaves them out.
ALLOWED_OUTSIDE_CHARTER_SKILLS: Final = frozenset(
    {"report", "mark_location", "hold", "return_home", "propose_charter_change"}
)

# The only AI orders that pass while the operator or the flight controller is in charge.
UNDER_OVERRIDE_ALLOWED: Final = frozenset({"report", "mark_location"})

# Mode A (approve each step): these alone need no click.
MODE_A_EXEMPT: Final = frozenset({"report", "mark_location"})

# The operator's own orders allowed while a latch is set (Resume lifts the latch).
OPERATOR_ALLOWED_UNDER_LATCH: Final = frozenset({"return_home", "report", "mark_location"})

# Their route exists only onboard; the onboard gate checks it from the handler's path.
GROUND_UNCHECKED_TARGETS: Final = frozenset({"resume"})
