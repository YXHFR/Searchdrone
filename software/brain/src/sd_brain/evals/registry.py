"""The skill registry as it is once the foundation's nine skills are implemented.

The registry in ``sd_schemas`` marks a skill ``implemented`` only when its executor and
tests exist (the onboard and mission-flow tickets). The evaluations already test the
commander against the nine foundation skills, so they use this copy.
"""

from __future__ import annotations

from sd_schemas.skills import SKILL_REGISTRY, SkillRegistry

__all__ = ["FOUNDATION_SKILLS", "foundation_registry"]

FOUNDATION_SKILLS = frozenset(
    {
        "takeoff",
        "goto",
        "survey_sector",
        "hold",
        "resume",
        "return_home",
        "report",
        "mark_location",
        "propose_charter_change",
    }
)


def foundation_registry() -> SkillRegistry:
    return SkillRegistry(
        spec.model_copy(update={"status": "implemented"})
        if spec.name in FOUNDATION_SKILLS
        else spec
        for spec in SKILL_REGISTRY
    )
