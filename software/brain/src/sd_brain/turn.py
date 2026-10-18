"""What one turn gets (``TurnInput``) and gives back (``TurnOutput``)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from sd_brain.charter_draft import CharterDraft, CharterError
from sd_schemas.api import ChatMessage
from sd_schemas.base import Id
from sd_schemas.charter import CharterProposal
from sd_schemas.commander import MissionMemory, Situation, SkillRequest
from sd_schemas.gate import GateDecision

__all__ = [
    "TRIGGER_PRIORITY",
    "AiAction",
    "BrainModel",
    "OutMessage",
    "ToolSpec",
    "Trigger",
    "TurnDiagnostics",
    "TurnInput",
    "TurnOutput",
]

Trigger = Literal[
    "operator_override",
    "operator_message",
    "alert_critical",
    "skill_event",
    "detection",
    "approval_event",
    "charter_event",
    "takeoff_clearance",
    "alert_warning",
    "mission_created",
    "idle_tick",
    "gate_reject_retry",
    "recovered",
]

# Highest first; used when several events wait for the same turn.
TRIGGER_PRIORITY: tuple[Trigger, ...] = (
    "operator_override",
    "operator_message",
    "alert_critical",
    "skill_event",
    "detection",
    "approval_event",
    "charter_event",
    "takeoff_clearance",
    "alert_warning",
    "mission_created",
    "idle_tick",
)

BackendName = Literal["llm", "grammar", "scripted", "rule", "cloud"]
TurnError = Literal["deadline", "timeout", "http_error", "parse_error", "round_limit"]


class BrainModel(BaseModel):
    """Internal models: unknown fields refused, but mutable while a turn is built."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class ToolSpec(BrainModel):
    name: str
    kind: Literal["action", "proposal", "read"]
    openai: dict[str, Any]


class OutMessage(BrainModel):
    text: str = Field(min_length=1, max_length=600)
    kind: Literal["text", "question", "status", "proposal"] = "text"
    refs: dict[str, Any] = Field(default_factory=dict[str, Any])

    @model_validator(mode="after")
    def _question(self) -> OutMessage:
        if self.kind == "text" and self.text.rstrip().endswith("?"):
            self.kind = "question"
        return self


class TurnDiagnostics(BrainModel):
    backend: BackendName
    engine: Literal["cloud", "local"] | None = None
    model_id: str | None = None
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    rounds: int = 0
    repairs: int = 0
    error: TurnError | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    prompt_json: dict[str, Any] | None = None


class TurnInput(BrainModel):
    turn_id: Id
    mission_id: Id
    trigger: Trigger
    events: list[dict[str, Any]]
    situation: Situation
    chat_tail: list[ChatMessage]
    backlog: list[ChatMessage] = Field(default_factory=list[ChatMessage])
    memory: MissionMemory
    tools: list[ToolSpec]
    retry_of: Id | None = None
    reject: GateDecision | None = None
    charter_errors: list[CharterError] = Field(default_factory=list[CharterError])
    in_reply_to: Id | None = None
    now: AwareDatetime

    def tool(self, name: str) -> ToolSpec | None:
        return next((t for t in self.tools if t.name == name), None)


class AiAction(BrainModel):
    """An order as the backend chose it, before the loop adapts it for the gate.

    ``params`` are the AI parameters (e.g. ``{"pin": "P1"}``); ``adapted`` says they are
    already the registry's parameters (the scripted commander, the rule start).
    """

    skill: str
    params: dict[str, Any] = Field(default_factory=dict[str, Any])
    rationale_sv: str = Field(default="", max_length=500)
    adapted: bool = False
    raw: bool = False  # scripted "bad commander" test only: sent past the tool registry


class TurnOutput(BrainModel):
    messages: list[OutMessage] = Field(default_factory=list[OutMessage], max_length=2)
    action: AiAction | None = None
    request: SkillRequest | None = None  # filled by the loop when the order is sent
    charter_proposal: CharterDraft | None = None
    ready_proposal: CharterProposal | None = None  # scripted: already the wire form
    note_update: str | None = Field(default=None, max_length=400)
    diagnostics: TurnDiagnostics

    @model_validator(mode="after")
    def _one_action(self) -> TurnOutput:
        proposals = (self.charter_proposal is not None) + (self.ready_proposal is not None)
        if (self.action is not None) + proposals > 1:
            raise ValueError("a turn has at most one action: an order or a charter proposal")
        return self

    def has_action(self) -> bool:
        return (
            self.action is not None
            or self.charter_proposal is not None
            or self.ready_proposal is not None
        )
