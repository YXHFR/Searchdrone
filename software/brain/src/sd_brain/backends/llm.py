"""``LlmCommander``: one turn with the cloud language model.

When the model does not answer, the turn ends with the error and the loop lets the
command mode (``GrammarBackend``) answer the same turn. A refused call is sent back for
repair (``max_repair_rounds``; a charter draft ``max_proposal_repair_rounds``). A draft
still refused when its rounds run out, when the model cannot fix it, when the model stops
answering, when the model sends the same refused draft again or when another call would
not end before the turn's deadline ends the turn as a failed proposal: the loop handles it
like a proposal the ground station refused, not as a parse error. So does a turn whose
draft was refused and that then ends without a proposal (a text reply, a read tool, an
error), and a retry turn of a refused proposal that proposes nothing: the attempt counts
and the retry or the final line follows.

Every model call is bounded as a whole (``asyncio.timeout``: a server that sends its body
slowly counts too), and no call starts when the slowest call so far would not end before
the turn does. The diagnostics are filled while the turn runs (:attr:`partial`), so a turn
the loop cuts off at its deadline still logs its prompt, tools and calls.
"""

from __future__ import annotations

import asyncio
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
from sd_brain.runtime.base import (
    ModelError,
    ModelReply,
    ModelRuntime,
    ModelTimeoutError,
    Sampling,
)
from sd_brain.state_view import current_limits
from sd_brain.sv import Said, read
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

CALL_GUARD_SLACK_S = 0.5
PARSE_FAIL_TEXT = "Jag fick inte ihop ordern. Kan du säga det på ett annat sätt?"
# The model sent the same refused order twice: another repair text will not help.
REPEAT_FAIL_TEXT = (
    "Jag försökte samma felaktiga order två gånger och slutar här. Säg vad jag ska göra, "
    "t.ex. 'flyg till P1' eller 'flyg hem'."
)
# Faults that come from the request itself (more than the mission allows): the model may
# answer them with a text that explains the limit instead of another draft.
REQUEST_LIMIT_CODES = frozenset({"CHARTER_EXCEEDS_LIMITS", "AREA_BEYOND_RANGE"})


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


def _said(turn: TurnInput) -> Said | None:
    """What the operator's request names, for a refused draft's suggestion: the latest
    operator line, in its own turn or in the retry of a refused proposal."""
    if turn.trigger not in ("operator_message", "gate_reject_retry"):
        return None
    line = next((m for m in reversed(turn.chat_tail) if m.role == "operator"), None)
    return read(line.text) if line is not None else None


def _explains_limit(errors: list[CharterError], texts: list[OutMessage]) -> bool:
    """A text reply (``texts``: those written after the refusal) to a draft refused only
    for the mission's limits: the answer the prompt asks for ("Högsta tillåtna höjd är
    120 m."), not a given-up attempt."""
    return (
        bool(errors)
        and all(e.code in REQUEST_LIMIT_CODES for e in errors)
        and any("turn_failed" not in m.refs for m in texts)
    )


def _draft_key(err: CallError) -> str | None:
    return None if err.args is None else json.dumps(err.args, sort_keys=True, default=str)


def _calls_key(reply: ModelReply) -> str | None:
    """The reply's tool calls (names and arguments) as one comparable value."""
    if not reply.tool_calls:
        return None
    calls: list[list[Any]] = []
    for c in reply.tool_calls:
        try:
            args: Any = json.loads(c.arguments_json) if c.arguments_json.strip() else {}
        except json.JSONDecodeError:
            args = c.arguments_json
        calls.append([c.name, args])
    return json.dumps(calls, sort_keys=True, default=str)


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
        # The running (or last) turn's diagnostics as they are filled: what the loop logs
        # when it cuts the turn off at its deadline.
        self.partial: tuple[Any, TurnDiagnostics] | None = None

    def partial_diagnostics(self, turn_id: Any) -> TurnDiagnostics | None:
        """The diagnostics of turn ``turn_id`` so far (``None``: another turn's)."""
        if self.partial is None or self.partial[0] != turn_id:
            return None
        return self.partial[1]

    async def _chat(
        self, messages: list[dict[str, Any]], offered: list[dict[str, Any]], limit_s: float
    ) -> ModelReply:
        """One model call, bounded as a whole: the HTTP client's timeout is per read, so a
        server that trickles its answer could otherwise outlast it. The runtime's own
        bound (``BestAvailableRuntime``) comes first, by ``CALL_GUARD_SLACK_S``, so it
        sees the failure and pauses the cloud; this one covers any runtime."""
        try:
            async with asyncio.timeout(limit_s + CALL_GUARD_SLACK_S):
                return await self.runtime.chat(messages, offered, limit_s, self.sampling)
        except TimeoutError:
            raise ModelTimeoutError("the model did not answer in time") from None

    async def decide(self, turn: TurnInput, tools: ToolExecutor) -> TurnOutput:
        start = self._clock()
        # The turn's own end, a little before the loop's deadline: a refused draft is
        # then still handed back as a failed proposal instead of being cut off.
        end = start + turn.budget_s - TURN_MARGIN_S
        messages = self.context.build(turn)
        offered = [t.openai for t in turn.tools]
        diag = TurnDiagnostics(
            backend="llm", prompt_json={"messages": list(messages), "tools": offered}
        )
        self.partial = (turn.turn_id, diag)
        out = TurnOutput(diagnostics=diag)
        lim = current_limits(turn.situation)
        rounds = repairs = 0
        slowest = 0.0  # the longest model call in this turn, in seconds
        said = _said(turn)
        draft_err: CallError | None = None  # the last refused draft, until a reply parses
        refused: CallError | None = None  # the turn's last refused draft, kept to the end
        refused_at = 0  # how many texts the turn had when that draft was refused
        seen: set[str] = set()  # the refused drafts' arguments in this turn
        refused_calls: set[str] = set()  # the refused orders' calls in this turn
        while True:
            diag.prompt_json = {"messages": list(messages), "tools": offered}
            if rounds >= self.cfg.max_read_tool_rounds + repairs + 1:
                diag.error = "round_limit"
                break
            used = self.context.count_messages(messages)
            if used + self.sampling.max_tokens > self.context_tokens:
                diag.error = "round_limit"
                break
            if rounds > 0 and end - self._clock() < slowest:
                # Another call would not end before the turn does (a refused draft has
                # its own exit below, right after its call). Every call answered, so the
                # loop counts the turn as healthy.
                diag.error = "deadline"
                diag.guard_stop = True
                break
            timeout = min(LLM_CALL_TIMEOUT_S, max(end - self._clock(), MIN_CALL_TIMEOUT_S))
            called = self._clock()
            try:
                reply = await self._chat(messages, offered, timeout)
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
            calls, err = parse_tool_calls(reply, turn.tools, self.reg, turn.situation, lim, said)
            diag.tool_calls.extend(
                {"name": c.name, "arguments": c.arguments_json[:TOOL_RESULT_MAX_CHARS]}
                for c in reply.tool_calls
            )
            draft_err = err if err is not None and err.kind == "proposal" else None
            kinds = {t.name: t.kind for t in turn.tools}
            if (
                err is not None
                and draft_err is None
                and any(kinds.get(c.name) == "action" for c in reply.tool_calls)
            ):
                # Only orders: a read tool asked again with bad arguments gets its repair.
                key = _calls_key(reply)
                if key is not None and key in refused_calls:
                    # The same refused order again: end the turn with a clear line.
                    diag.error = "parse_error"
                    out.messages = [
                        OutMessage(text=REPEAT_FAIL_TEXT, refs={"turn_failed": "repeat"})
                    ]
                    break
                if key is not None:
                    refused_calls.add(key)
            repeat = False
            if draft_err is not None:
                refused, refused_at = draft_err, len(out.messages)
                key = _draft_key(draft_err)
                repeat = key is not None and key in seen
                if key is not None:
                    seen.add(key)
            budget = (
                self.cfg.max_proposal_repair_rounds
                if draft_err is not None
                else self.cfg.max_repair_rounds
            )
            if draft_err is not None and (
                not draft_err.repairable or repeat or end - self._clock() < slowest
            ):
                # The model cannot fix it, sent the same refused draft again (another
                # repair text will not help), or another call would not end before the
                # turn does: the draft goes back to the loop as refused.
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
                # A plan note sent with the action is kept; the turn ends at the action.
                note = next((c for c in calls if c.name == "brain_note"), None)
                if note is not None:
                    out.note_update = str(note.args.get("text", ""))[:400]
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
        given_up: FailedProposal | None = None
        if refused is not None:
            if not _explains_limit(refused.errors, out.messages[refused_at:]):
                given_up = _failed(refused)
        elif (
            turn.trigger == "gate_reject_retry"
            and turn.charter_errors
            and diag.error not in ("timeout", "http_error")
            and not _explains_limit(turn.charter_errors, out.messages)
        ):
            # The retry of a refused proposal that proposes nothing (a text, a read tool,
            # the rounds ran out) gives the attempt up the same way: counted, and retried
            # or answered with the final line, never left as a silent end of the request.
            given_up = FailedProposal(errors=list(turn.charter_errors))
        if given_up is not None and out.failed_proposal is None and not out.has_action():
            # A draft was refused and none went through: the rounds or the context ran
            # out, or the model gave up with a text reply ("rita området …"). It is a
            # failed attempt all the same, so the loop counts it and the retry (or, after
            # the last attempt, the charter_failed question) follows. The model's own text
            # stays in ``messages`` for the record, but the loop never posts the texts of
            # a refused attempt (the follow-up turn or the final line speaks instead), as
            # with a proposal the ground refused; a parse failure's line is dropped too.
            # Only a text that explains a mission limit the request exceeds is the turn's
            # answer and is posted.
            out.failed_proposal = given_up
            out.messages = [m for m in out.messages if "turn_failed" not in m.refs]
        diag.latency_ms = int((self._clock() - start) * 1000)
        diag.prompt_json = {"messages": messages, "tools": offered}
        out.messages = out.messages[:2]
        return out

    @staticmethod
    def _set_action(out: TurnOutput, call: ParsedCall) -> None:
        if call.kind == "proposal" and call.draft is not None:
            out.charter_proposal = call.draft
        else:
            out.action = AiAction(skill=call.name, params=call.args, rationale_sv=call.rationale_sv)
