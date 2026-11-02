"""``CommanderBackend``: turns a ``TurnInput`` into a ``TurnOutput``."""

from __future__ import annotations

from typing import Any, Protocol

from sd_brain.turn import TurnInput, TurnOutput

__all__ = ["PARSE_FAIL_TEXT", "BackendCrashError", "CommanderBackend", "ToolExecutor"]

# The operator's line when the AI's answer could not be read (``turn_failed: parse``).
PARSE_FAIL_TEXT = "Jag fick inte ihop ordern. Kan du säga det på ett annat sätt?"


class BackendCrashError(RuntimeError):
    """The backend failed in a way the turn cannot recover from (counted as a turn error)."""


class ToolExecutor(Protocol):
    """Runs read tools during a turn. Actions are never run here."""

    async def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]: ...


class CommanderBackend(Protocol):
    name: str  # "cloud", "grammar", "scripted"

    async def decide(self, turn: TurnInput, tools: ToolExecutor) -> TurnOutput: ...
