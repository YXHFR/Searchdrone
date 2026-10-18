"""The onboard skill handlers. ``HANDLERS`` maps each implemented onboard skill to its
handler class; ``register_handler`` adds one. ``TRANSITIONS_EXTRA`` of every handler is
collected into ``extra_transitions()`` and joined with the executor's table at start."""

from __future__ import annotations

from typing import Any

__all__ = ["HANDLERS", "extra_transitions", "register_handler"]

HANDLERS: dict[str, type[Any]] = {}


def register_handler[H: type[Any]](cls: H) -> H:
    name = cls.name
    if name in HANDLERS:
        raise ValueError(f"two handlers for {name}")
    HANDLERS[name] = cls
    return cls


def extra_transitions() -> dict[Any, Any]:
    out: dict[Any, Any] = {}
    for cls in HANDLERS.values():
        for key, row in getattr(cls, "TRANSITIONS_EXTRA", {}).items():
            if key in out:
                raise ValueError(f"two handlers add the transition {key}")
            out[key] = row
    return out


# Importing the modules registers the handlers.
from sd_onboard.skills import (  # noqa: E402
    follow_path,
    goto,
    hold,
    resume,
    return_home,
    survey_sector,
    takeoff,
)

__all__ += ["follow_path", "goto", "hold", "resume", "return_home", "survey_sector", "takeoff"]
