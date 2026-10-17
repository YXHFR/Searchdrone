"""``command_hash``: what an approval click binds to. Ground and onboard compute it the
same way, so an approval can only ever release exactly the order that was shown."""

from __future__ import annotations

import hashlib
import json

from sd_schemas.base import SdModel

__all__ = ["command_hash"]


def command_hash(skill: str, params: SdModel, charter_version: int) -> str:
    """SHA-256 of the skill, its *validated* parameters (defaults filled in) and the
    charter version, as compact sorted JSON."""
    payload = {
        "skill": skill,
        "params": params.model_dump(mode="json"),
        "charter_version": charter_version,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
