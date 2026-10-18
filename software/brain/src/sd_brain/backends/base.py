"""``CommanderBackend``: turns a ``TurnInput`` into a ``TurnOutput``."""

from __future__ import annotations

from typing import Any, Protocol

from sd_brain.turn import TurnInput, TurnOutput

__all__ = ["BackendCrashError", "CommanderBackend", "ToolExecutor"]


class BackendCrashError(RuntimeError):
    """The backend failed in a way the turn cannot recover from (counted as a turn error)."""


class ToolExecutor(Protocol):
    """Runs read tools during a turn. Actions are never run here."""

    async def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]: ...


class CommanderBackend(Protocol):
    name: str  # "cloud", "grammar", "scripted"

    async def decide(self, turn: TurnInput, tools: ToolExecutor) -> TurnOutput: ...
