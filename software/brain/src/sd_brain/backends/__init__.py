"""Commander backends. A new one implements ``CommanderBackend`` and is listed here."""

from sd_brain.backends.base import BackendCrashError, CommanderBackend, ToolExecutor
from sd_brain.backends.grammar import GrammarBackend
from sd_brain.backends.llm import LlmCommander
from sd_brain.backends.scripted import ScriptedBackend

__all__ = [
    "BACKENDS",
    "BackendCrashError",
    "CommanderBackend",
    "GrammarBackend",
    "LlmCommander",
    "ScriptedBackend",
    "ToolExecutor",
]

BACKENDS: dict[str, type] = {
    "llm_local": LlmCommander,
    "scripted": ScriptedBackend,
    "grammar": GrammarBackend,
}
