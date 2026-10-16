"""``ai_config_hash``: which model, prompt, tools and sampling an answer came from.

The sd-brain version is deliberately not part of it, so a version bump does not void a
model evaluation. The per-turn bounds on height and speed are not part of it either.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from importlib import resources
from pathlib import Path
from typing import Any

from sd_brain.runtime.base import Sampling
from sd_brain.tools import registry_tool_schemas
from sd_schemas.skills import SkillRegistry

__all__ = ["RUNTIME_LOCK", "ai_config_hash", "model_sha256", "prompt_text"]

RUNTIME_LOCK = Path(__file__).resolve().parents[2] / "runtime.lock.toml"


def prompt_text(name: str) -> str:
    return resources.files("sd_brain.prompts").joinpath(name).read_text(encoding="utf-8")


def _lock_text() -> str:
    return RUNTIME_LOCK.read_text(encoding="utf-8") if RUNTIME_LOCK.is_file() else ""


def model_sha256() -> str:
    text = _lock_text()
    if not text:
        return ""
    data: dict[str, Any] = tomllib.loads(text)
    return str(data.get("model", {}).get("sha256", ""))


def ai_config_hash(
    reg: SkillRegistry, sampling: Sampling, *, cloud_model: str | None = None
) -> str:
    h = hashlib.sha256()
    for part in (
        model_sha256(),
        _lock_text(),
        prompt_text("system_v1.md"),
        prompt_text("examples_sv_v1.yaml"),
        json.dumps(registry_tool_schemas(reg), sort_keys=True, ensure_ascii=False),
        json.dumps(sampling.model_dump(), sort_keys=True),
        cloud_model or "",
    ):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()
