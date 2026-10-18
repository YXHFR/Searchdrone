"""``ai_config_hash``: which model, prompt, tools and sampling an answer came from.

The sd-brain version is deliberately not part of it, so a version bump does not void a
model evaluation. The per-turn bounds on height and speed are not part of it either.
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources

from sd_brain.config import CloudConfig
from sd_brain.runtime.base import Sampling
from sd_brain.tools import registry_tool_schemas
from sd_schemas.skills import SkillRegistry

__all__ = ["ai_config_hash", "cloud_identity", "prompt_text"]


def prompt_text(name: str) -> str:
    return resources.files("sd_brain.prompts").joinpath(name).read_text(encoding="utf-8")


def cloud_identity(cloud: CloudConfig | None) -> str:
    """The cloud settings that change an answer (models, reasoning, providers)."""
    if cloud is None:
        return ""
    return json.dumps(
        {
            "model": cloud.model,
            "fallback_model": cloud.fallback_model,
            **cloud.extra_body(),
        },
        sort_keys=True,
    )


def ai_config_hash(
    reg: SkillRegistry, sampling: Sampling, *, cloud: CloudConfig | None = None
) -> str:
    h = hashlib.sha256()
    for part in (
        prompt_text("system_v1.md"),
        prompt_text("examples_sv_v1.yaml"),
        json.dumps(registry_tool_schemas(reg), sort_keys=True, ensure_ascii=False),
        json.dumps(sampling.model_dump(), sort_keys=True),
        cloud_identity(cloud),
    ):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()
