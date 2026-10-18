"""``LlmCommander``: one turn with the cloud language model.

When the model does not answer, the turn ends with the error and the loop lets the
command mode (``GrammarBackend``) answer the same turn.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from sd_brain.backends.base import ToolExecutor
from sd_brain.config import BehaviourConfig
from sd_brain.constants import LLM_CALL_TIMEOUT_S, MESSAGE_MAX_CHARS, TOOL_RESULT_MAX_CHARS
from sd_brain.context import ContextBuilder
from sd_brain.runtime.base import ModelError, ModelReply, ModelRuntime, Sampling
from sd_brain.state_view import current_limits
from sd_brain.toolcall_parse import ParsedCall, parse_tool_calls, strip_think, truncate
from sd_brain.turn import AiAction, OutMessage, TurnDiagnostics, TurnInput, TurnOutput
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
        remaining: Callable[[], float] | None = None,
    ) -> None:
        self.runtime = runtime
        self.context = context
        self.reg = reg
        self.cfg = behaviour
        self.sampling = sampling
        self.context_tokens = context_tokens
        self._remaining = remaining or (lambda: LLM_CALL_TIMEOUT_S)

    async def decide(self, turn: TurnInput, tools: ToolExecutor) -> TurnOutput:
        start = time.monotonic()
        messages = self.context.build(turn)
        diag = TurnDiagnostics(backend="llm", prompt_json={"messages": list(messages)})
        out = TurnOutput(diagnostics=diag)
        lim = current_limits(turn.situation)
        offered = [t.openai for t in turn.tools]
        rounds = repairs = 0
        while True:
            if rounds >= self.cfg.max_read_tool_rounds + repairs + 1:
                diag.error = "round_limit"
                break
            used = self.context.count_messages(messages)
            if used + self.sampling.max_tokens > self.context_tokens:
                diag.error = "round_limit"
                break
            timeout = min(LLM_CALL_TIMEOUT_S, self._remaining())
            try:
                reply = await self.runtime.chat(messages, offered, timeout, self.sampling)
            except ModelError as exc:
                diag.error = exc.kind
                break
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
            if err is not None and repairs < self.cfg.max_repair_rounds:
                repairs += 1
                diag.repairs = repairs
                messages.append(_assistant_message(reply))
                if reply.tool_calls:
                    for c in reply.tool_calls:
                        messages.append(_tool_message(c.id, err))
                else:
                    messages.append({"role": "user", "content": f"{err}. Try again."})
                continue
            if err is not None:
                diag.error = "parse_error"
                out.messages = [OutMessage(text=PARSE_FAIL_TEXT)]
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
        diag.latency_ms = int((time.monotonic() - start) * 1000)
        diag.prompt_json = {"messages": messages}
        out.messages = out.messages[:2]
        return out

    @staticmethod
    def _set_action(out: TurnOutput, call: ParsedCall) -> None:
        if call.kind == "proposal" and call.draft is not None:
            out.charter_proposal = call.draft
        else:
            out.action = AiAction(skill=call.name, params=call.args, rationale_sv=call.rationale_sv)
