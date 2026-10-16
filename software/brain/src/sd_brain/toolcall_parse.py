"""Reading tool calls out of a model reply, and checking them before anything is sent.

A call is refused (and the model gets one repair round) when the tool was not offered
in this turn, the arguments do not validate, a height or speed is outside this turn's
bounds, or a pin or sector is unknown.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ValidationError

from sd_brain.charter_draft import CharterDraft
from sd_brain.runtime.base import ModelReply
from sd_brain.state_view import Limits
from sd_brain.tools import AdaptError, adapt
from sd_brain.turn import ToolSpec
from sd_schemas.commander import Situation
from sd_schemas.skills import SkillRegistry

__all__ = ["ParsedCall", "parse_tool_calls", "strip_think", "truncate"]

_THINK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)
_RAW_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def strip_think(text: str | None) -> str:
    """The reply text without ``<think>`` blocks and stray tool-call markup."""
    if not text:
        return ""
    out = _THINK.sub("", text)
    out = _RAW_CALL.sub("", out)
    return out.strip()


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


@dataclass
class ParsedCall:
    id: str
    name: str
    kind: Literal["action", "proposal", "read"]
    args: dict[str, Any]
    draft: CharterDraft | None = None
    rationale_sv: str = ""
    dropped: list[str] = field(default_factory=list[str])


def _raw_calls(reply: ModelReply) -> list[tuple[str, str, str]]:
    calls = [(c.id, c.name, c.arguments_json) for c in reply.tool_calls]
    if calls or not reply.content:
        return calls
    for i, m in enumerate(_RAW_CALL.finditer(reply.content)):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        args = obj.get("arguments", {})
        calls.append((f"raw_{i}", str(obj.get("name", "")), json.dumps(args)))
    return calls


def _check_bounds(args: dict[str, Any], lim: Limits) -> str | None:
    alt = args.get("alt_m")
    if isinstance(alt, int | float):
        if alt > lim.alt_max_agl_m:
            return f"invalid arguments: alt_m must be <= {lim.alt_max_agl_m:g}"
        if alt < lim.alt_min_agl_m:
            return f"invalid arguments: alt_m must be >= {lim.alt_min_agl_m:g}"
    speed = args.get("speed_ms")
    if isinstance(speed, int | float) and speed > lim.max_speed_ms:
        return f"invalid arguments: speed_ms must be <= {lim.max_speed_ms:g}"
    return None


def _validate(
    name: str,
    args: dict[str, Any],
    tool: ToolSpec,
    reg: SkillRegistry,
    s: Situation,
    lim: Limits,
) -> tuple[ParsedCall | None, str | None]:
    if tool.kind == "read":
        if name == "brain_note" and not isinstance(args.get("text"), str):
            return None, "invalid arguments: brain_note needs text"
        if name == "get_log" and not isinstance(args.get("n"), int):
            return None, "invalid arguments: get_log needs an integer n"
        if name == "estimate" and not isinstance(args.get("skill"), str):
            return None, "invalid arguments: estimate needs skill and params"
        return ParsedCall(id="", name=name, kind="read", args=args), None
    if tool.kind == "proposal":
        try:
            draft = CharterDraft.model_validate(args)
        except ValidationError as exc:
            return None, f"invalid arguments: {_first_error(exc)}"
        return ParsedCall(id="", name=name, kind="proposal", args=args, draft=draft), None
    spec = reg.get(name)
    rationale = args.pop("rationale_sv", "")
    model = spec.ai_params_model or spec.params_model
    try:
        model.model_validate(args)
    except ValidationError as exc:
        return None, f"invalid arguments: {_first_error(exc)}"
    bound = _check_bounds(args, lim)
    if bound is not None:
        return None, bound
    try:
        adapt(name, args, s)
    except (AdaptError, KeyError, ValueError) as exc:
        return None, str(exc)
    call = ParsedCall(id="", name=name, kind="action", args=args, rationale_sv=str(rationale)[:200])
    return call, None


def _first_error(exc: ValidationError) -> str:
    err = exc.errors()[0]
    loc = ".".join(str(p) for p in err.get("loc", ())) or "arguments"
    return f"{loc}: {err.get('msg')}"


def parse_tool_calls(
    reply: ModelReply,
    tools: list[ToolSpec],
    reg: SkillRegistry,
    s: Situation,
    lim: Limits,
) -> tuple[list[ParsedCall], str | None]:
    """The valid calls in order, or the first error (for the repair round)."""
    offered = {t.name: t for t in tools}
    out: list[ParsedCall] = []
    for call_id, name, raw in _raw_calls(reply):
        tool = offered.get(name)
        if tool is None:
            return [], f"unknown tool {name!r}: it is not offered now"
        try:
            args = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return [], f"invalid arguments for {name}: not JSON"
        if not isinstance(args, dict):
            return [], f"invalid arguments for {name}: not an object"
        parsed, err = _validate(name, dict(args), tool, reg, s, lim)
        if err is not None or parsed is None:
            return [], err
        parsed.id = call_id
        out.append(parsed)
    return out, None
