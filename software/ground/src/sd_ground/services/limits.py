"""Changing a mission's limits (``PUT /missions/{id}/limits``, pilot only).

The pilot may make the limits stricter, or raise them by click up to the outer cap in
the owner's local settings file (``config/local/limits.toml``). The AI has no route to
this. Raising them above the standard values moves the flight controller's fence too
(``services.fence.fence_param_targets``), and the ground station only writes those
parameters while the vehicle stands disarmed on the ground, so such a change waits for
landing. A change that the active charter no longer fits in is refused: the pilot first
approves a stricter charter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sd_ground.services.fence import fence_param_targets
from sd_safety.limits import charter_violations, check_limits
from sd_schemas.api import ApiErrorItem
from sd_schemas.charter import MissionCharter
from sd_schemas.reason_codes import CHARTER_VALIDATION_CODES, LIMITS_CODES
from sd_schemas.safety import LimitCaps, SafetyLimits

__all__ = ["LimitsRefusal", "check_change", "moves_fence"]


@dataclass(frozen=True, slots=True)
class LimitsRefusal:
    status: int
    code: str
    detail_sv: str
    errors: list[ApiErrorItem] = field(default_factory=list[ApiErrorItem])


def moves_fence(old: SafetyLimits, new: SafetyLimits) -> bool:
    """Whether the flight controller's fence parameters differ for the new limits."""
    return fence_param_targets(old) != fence_param_targets(new)


def check_change(
    new: SafetyLimits,
    *,
    old: SafetyLimits,
    caps: LimitCaps,
    active: MissionCharter | None,
    in_flight: bool,
) -> LimitsRefusal | None:
    """Why ``new`` cannot replace ``old`` now, or ``None`` when it can."""
    problems = check_limits(new, caps)
    if problems:
        items = [
            ApiErrorItem(
                loc=["body", p.field],
                code=p.code,
                detail_sv=LIMITS_CODES.get(p.code, CHARTER_VALIDATION_CODES["VALIDATION"]),
                field=p.field,
                limit=p.limit,
            )
            for p in problems
        ]
        code = items[0].code
        status = 422
        return LimitsRefusal(status, code, items[0].detail_sv, items)
    if active is not None:
        loose = charter_violations(active, new)
        if loose:
            items = [
                ApiErrorItem(
                    loc=["charter", name],
                    code="LIMITS_BELOW_ACTIVE_CHARTER",
                    detail_sv=LIMITS_CODES["LIMITS_BELOW_ACTIVE_CHARTER"],
                    field=name,
                )
                for name in loose
            ]
            return LimitsRefusal(
                409,
                "LIMITS_BELOW_ACTIVE_CHARTER",
                LIMITS_CODES["LIMITS_BELOW_ACTIVE_CHARTER"],
                items,
            )
    if in_flight and moves_fence(old, new):
        return LimitsRefusal(
            409,
            "LIMITS_CHANGE_REQUIRES_LANDING",
            LIMITS_CODES["LIMITS_CHANGE_REQUIRES_LANDING"],
        )
    return None
