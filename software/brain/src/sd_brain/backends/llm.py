"""``LlmCommander``: one turn with the cloud language model.

When the model does not answer, the turn ends with the error and the loop lets the
command mode (``GrammarBackend``) answer the same turn. A refused call is sent back for
repair (``max_repair_rounds``; a charter draft ``max_proposal_repair_rounds``). A draft
still refused when its rounds run out, when the model cannot fix it, when the model stops
answering or when another call would not end before the turn's deadline ends the turn as
a failed proposal: the loop handles it like a proposal the ground station refused, not as
a parse error.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from sd_brain.backends.base import ToolExecutor
from sd_brain.charter_draft import CharterError
from sd_brain.config import BehaviourConfig
from sd_brain.constants import (
    LLM_CALL_TIMEOUT_S,
    MESSAGE_MAX_CHARS,
    MIN_CALL_TIMEOUT_S,
    TOOL_RESULT_MAX_CHARS,
    TURN_MARGIN_S,
)
from sd_brain.context import ContextBuilder
from sd_brain.runtime.base import ModelError, ModelReply, ModelRuntime, Sampling
from sd_brain.state_view import current_limits
from sd_brain.toolcall_parse import (
    CallError,
    ParsedCall,
    parse_tool_calls,
    strip_think,
    truncate,
)
from sd_brain.turn import (
    AiAction,
    FailedProposal,
    OutMessage,
    TurnDiagnostics,
    TurnInput,
    TurnOutput,
)
from sd_schemas.skills import SkillRegistry

__all__ = ["LlmCommander"]

PARSE_FAIL_TEXT = "Jag fick inte ihop ordern. Kan du säga det på ett annat sätt?"


def _assistant_message(reply: ModelReply) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": reply.content or ""}
    if reply.tool_calls:
        msg["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments_json},
            }
            for c in reply.tool_calls
        ]
    return msg


def _tool_message(call_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _failed(err: CallError) -> FailedProposal:
    errors = err.errors or [
        CharterError(code="DRAFT_INVALID", detail_sv="Ramförslaget gick inte att läsa.")
    ]
    return FailedProposal(errors=errors, args=err.args, final=not err.repairable)


class LlmCommander:
    name = "cloud"

    def __init__(
        self,
        runtime: ModelRuntime,
        context: ContextBuilder,
        reg: SkillRegistry,
        behaviour: BehaviourConfig,
        sampling: Sampling,
        *,
        context_tokens: int = 8192,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runtime = runtime
        self.context = context
        self.reg = reg
        self.cfg = behaviour
        self.sampling = sampling
        self.context_tokens = context_tokens
        self._clock = clock

    async def decide(self, turn: TurnInput, tools: ToolExecutor) -> TurnOutput:
        start = self._clock()
        # The turn's own end, a little before the loop's deadline: a refused draft is
        # then still handed back as a failed proposal instead of being cut off.
        end = start + turn.budget_s - TURN_MARGIN_S
        messages = self.context.build(turn)
        diag = TurnDiagnostics(backend="llm", prompt_json={"messages": list(messages)})
        out = TurnOutput(diagnostics=diag)
        lim = current_limits(turn.situation)
        offered = [t.openai for t in turn.tools]
        rounds = repairs = 0
        slowest = 0.0  # the longest model call in this turn, in seconds
        draft_err: CallError | None = None  # the last refused draft, until a reply parses
        while True:
            if rounds >= self.cfg.max_read_tool_rounds + repairs + 1:
                diag.error = "round_limit"
                break
            used = self.context.count_messages(messages)
            if used + self.sampling.max_tokens > self.context_tokens:
                diag.error = "round_limit"
                break
            timeout = min(LLM_CALL_TIMEOUT_S, max(end - self._clock(), MIN_CALL_TIMEOUT_S))
            called = self._clock()
            try:
                reply = await self.runtime.chat(messages, offered, timeout, self.sampling)
            except ModelError as exc:
                diag.error = exc.kind
                break
            slowest = max(slowest, self._clock() - called)
            rounds += 1
            diag.rounds = rounds
            diag.engine = reply.engine
            diag.model_id = reply.model_id
            diag.prompt_tokens += reply.prompt_tokens
            diag.completion_tokens += reply.completion_tokens
            if reply.engine == "cloud":
                diag.backend = "cloud"
            text = strip_think(reply.content)
            calls, err = parse_tool_calls(reply, turn.tools, self.reg, turn.situation, lim)
            diag.tool_calls.extend(
                {"name": c.name, "arguments": c.arguments_json[:TOOL_RESULT_MAX_CHARS]}
                for c in reply.tool_calls
            )
            draft_err = err if err is not None and err.kind == "proposal" else None
            budget = (
                self.cfg.max_proposal_repair_rounds
                if draft_err is not None
                else self.cfg.max_repair_rounds
            )
            if draft_err is not None and (
                not draft_err.repairable or end - self._clock() < slowest
            ):
                # The model cannot fix it, or another call would not end before the turn
                # does: the draft goes back to the loop as refused.
                out.failed_proposal = _failed(draft_err)
                break
            if err is not None and repairs < budget:
                repairs += 1
                diag.repairs = repairs
                messages.append(_assistant_message(reply))
                if reply.tool_calls:
                    for c in reply.tool_calls:
                        messages.append(_tool_message(c.id, err))
                else:
                    again = err if err.endswith(".") else f"{err}. Try again."
                    messages.append({"role": "user", "content": again})
                continue
            if draft_err is not None:
                out.failed_proposal = _failed(draft_err)
                break
            if err is not None:
                diag.error = "parse_error"
                out.messages = [OutMessage(text=PARSE_FAIL_TEXT, refs={"turn_failed": "parse"})]
                break
            if text:
                out.messages.append(OutMessage(text=truncate(text, MESSAGE_MAX_CHARS)))
            if not calls:
                break
            first = next((c for c in calls if c.kind in ("action", "proposal")), None)
            if first is not None:
                self._set_action(out, first)
                break
            messages.append(_assistant_message(reply))
            for c in calls:
                result = await tools.call(c.name, c.args)
                if c.name == "brain_note":
                    out.note_update = str(c.args.get("text", ""))[:400]
                    result = {"ok": True}
                content = truncate(json.dumps(result, ensure_ascii=False), TOOL_RESULT_MAX_CHARS)
                messages.append(_tool_message(c.id, content))
        if draft_err is not None and out.failed_proposal is None and not out.has_action():
            out.failed_proposal = _failed(draft_err)  # the rounds or the context ran out
        diag.latency_ms = int((self._clock() - start) * 1000)
        diag.prompt_json = {"messages": messages}
        out.messages = out.messages[:2]
        return out

    @staticmethod
    def _set_action(out: TurnOutput, call: ParsedCall) -> None:
        if call.kind == "proposal" and call.draft is not None:
            out.charter_proposal = call.draft
        else:
            out.action = AiAction(skill=call.name, params=call.args, rationale_sv=call.rationale_sv)
