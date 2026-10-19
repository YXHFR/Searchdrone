"""Reading tool calls out of a model reply, and checking them before anything is sent.

A call is refused (and the model gets a repair round) when the tool was not offered
in this turn, the arguments do not validate, a height or speed is outside this turn's
bounds, or a pin or sector is unknown. A charter draft is also checked against the
mission's limits and the height band and turned into the ground station's proposal here
(:func:`~sd_brain.charter_draft.to_proposal`), so an unknown pin, a missing size, a skill
name the registry does not have or a flight area without home is answered in the same
turn, before any POST, with every fault of the draft in one repair text. That text also
says what the situation offers (the map pins, the drawn areas or "none", a rect's shape)
and, when the operator's line names a box and a pin, the search area it describes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ValidationError

from sd_brain.charter_draft import CharterDraft, CharterError, DraftError, error_path, to_proposal
from sd_brain.runtime.base import ModelReply
from sd_brain.state_view import Limits
from sd_brain.sv import Said
from sd_brain.tools import GOTO_COORDS, AdaptError, adapt, charter_skills
from sd_brain.turn import ToolSpec
from sd_schemas.charter import CharterProposal
from sd_schemas.commander import Situation
from sd_schemas.safety import MIN_ALT_AGL_M
from sd_schemas.skills import SkillRegistry

__all__ = [
    "CallError",
    "ParsedCall",
    "charter_options",
    "parse_tool_calls",
    "strip_think",
    "truncate",
]

_THINK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)
_RAW_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# The heights the ground station uses for a proposal that leaves them out.
_PROPOSAL_ALT_MIN = float(CharterProposal.model_fields["alt_min_agl_m"].default)
_PROPOSAL_ALT = float(CharterProposal.model_fields["default_alt_agl_m"].default)

_DRAFT_AGAIN = "Call propose_charter_change again with the whole corrected draft."
# Short hints after a draft's validation error, by the field it names.
_DRAFT_HINTS = (
    ("operating_area", "Leave operating_area out: the system computes the flight area."),
    (
        "search_areas",
        "A rect needs center_pin_id, width_m and height_m; a circle center_pin_id and "
        "radius_m; a drawn area area_id.",
    ),
)


class CallError(str):
    """A refused call's text for the model (it is a ``str``), with what the loop needs:
    ``kind`` of the refused tool, and for a charter draft every fault as a
    ``CharterError`` (``errors``: the ground's code, a Swedish text for the operator, the
    field), the first fault's ``detail_sv`` and ``field``, the draft's ``args`` and
    whether the model can fix it at all (``repairable``)."""

    kind: Literal["action", "proposal", "read", "unknown"]
    detail_sv: str
    field: str | None
    args: dict[str, Any] | None
    errors: list[CharterError]
    repairable: bool

    def __new__(
        cls,
        text: str,
        kind: Literal["action", "proposal", "read", "unknown"] = "unknown",
        *,
        errors: list[CharterError] | None = None,
        args: dict[str, Any] | None = None,
        repairable: bool = True,
    ) -> CallError:
        out = super().__new__(cls, text)
        out.kind = kind
        out.errors = list(errors or [])
        out.detail_sv = " ".join(e.detail_sv for e in out.errors)
        out.field = out.errors[0].field if out.errors else None
        out.args = args
        out.repairable = repairable
        return out


@dataclass(frozen=True)
class _Fault:
    """One thing wrong with a draft: ``text`` and ``hint`` for the model (English),
    ``error`` for the operator and the log."""

    text: str
    error: CharterError
    hint: str = ""
    repairable: bool = True


def _fault(
    text: str,
    detail_sv: str,
    field: str | None,
    *,
    code: str = "DRAFT_INVALID",
    limit: float | None = None,
    hint: str = "",
) -> _Fault:
    error = CharterError(code=code, detail_sv=detail_sv, field=field, limit=limit)
    return _Fault(text=text, error=error, hint=hint)


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


def _n(value: float) -> float | int:
    return int(value) if float(value).is_integer() else value


def _fits(draft: dict[str, Any], s: Situation, lim: Limits) -> bool:
    """Whether a suggested draft would pass the checks it is meant to fix."""
    try:
        parsed = CharterDraft.model_validate({**draft, "reason_sv": "-"})
        if _limit_faults(parsed, s, lim):
            return False
        to_proposal(parsed, s)
    except (ValidationError, DraftError):
        return False
    return True


def _suggestion(s: Situation, said: Said | None, lim: Limits) -> str:
    """The search area the operator's line describes ("ruta 100 x 100 m runt P1"), as
    the draft's fields; empty without a box size or a known centre, or when that area
    itself would be refused. A height the mission does not allow is left out, so the
    suggestion never repeats the fault it is attached to."""
    if said is None or said.box_m is None:
        return ""
    known = {p.id for p in s.pins}
    pin = next((p for p in said.pins if p in known), None)
    area: dict[str, Any] = {"kind": "rect"}
    if pin is not None:
        area["center_pin_id"] = pin
    elif said.coord is not None:
        area["center"] = {"lat": said.coord[0], "lon": said.coord[1]}
    else:
        return ""
    area["width_m"], area["height_m"] = _n(said.box_m[0]), _n(said.box_m[1])
    draft: dict[str, Any] = {"search_areas": [area]}
    if said.alt_m is not None and _fits({**draft, "default_alt_agl_m": said.alt_m}, s, lim):
        draft["default_alt_agl_m"] = _n(said.alt_m)
    elif not _fits(draft, s, lim):
        return ""
    fields = ", ".join(f'"{k}": {json.dumps(v)}' for k, v in draft.items())
    # Fields, not a whole draft: the model adds reason_sv (and keeps what else it needs).
    return f"From the operator's line: {fields}."


def charter_options(s: Situation, said: Said | None, lim: Limits) -> str:
    """What this situation lets a charter draft name (map pins, drawn areas, a rect's
    shape) and, from the operator's line (``said``), the search area it describes: for
    a refused draft's repair text and the retry turn's context."""
    pins = ", ".join(dict.fromkeys(p.id for p in s.pins))
    areas = ", ".join(sorted({a.label for a in s.areas if a.kind == "area"}))
    pin = s.pins[0].id if s.pins else None
    if pin is not None:
        shape = (
            f'A rect on a pin: {{"kind": "rect", "center_pin_id": "{pin}", '
            '"width_m": <metres>, "height_m": <metres>}.'
        )
    else:
        shape = "Without a pin, ask the operator to place one (or use center for coordinates)."
    text = f"Map pins: {pins or 'none'}. Drawn areas: {areas or 'none'}. {shape}"
    return " ".join(x for x in (text, _suggestion(s, said, lim)) if x)


def _draft_error(faults: list[_Fault], args: dict[str, Any] | None, options: str = "") -> CallError:
    """Every fault of a draft in one repair text, so one round can fix them all, with
    what the situation offers (``options``, :func:`charter_options`)."""
    hints: list[str] = []
    for f in faults:
        at = f.error.field or ""
        own = [f.hint] if f.hint else [h for prefix, h in _DRAFT_HINTS if at.startswith(prefix)]
        for h in own:
            if h not in hints:
                hints.append(h)
    what = "; ".join(f.text.rstrip(".") for f in faults)
    full = " ".join(
        x for x in (f"invalid charter draft: {what}.", *hints, options, _DRAFT_AGAIN) if x
    )
    return CallError(
        full,
        "proposal",
        errors=[f.error for f in faults],
        args=args,
        repairable=all(f.repairable for f in faults),
    )


def _skill_faults(args: dict[str, Any], reg: SkillRegistry) -> list[_Fault]:
    names = charter_skills(reg)
    given = args.get("allowed_skills")
    if not isinstance(given, list):
        return []
    unknown = [str(x) for x in given if x not in names]  # pyright: ignore[reportUnknownVariableType]
    if not unknown:
        return []
    return [
        _fault(
            f"allowed_skills: unknown skill names {', '.join(unknown)}; use only "
            f"{', '.join(names)}, or leave allowed_skills out for the default",
            f"Okända färdigheter: {', '.join(unknown)}.",
            "allowed_skills",
            code="SKILL_UNKNOWN",
        )
    ]


def _limit_faults(draft: CharterDraft, s: Situation, lim: Limits) -> list[_Fault]:
    """The mission's limits (a charter may never exceed them) and the height band, as the
    ground station checks them, with its defaults for a field left out."""
    out: list[_Fault] = []
    for name in ("alt_min_agl_m", "alt_max_agl_m", "default_alt_agl_m"):
        value = getattr(draft, name)
        if value is None:
            continue
        if value > lim.mission_alt_max_agl_m:
            out.append(
                _fault(
                    f"{name} must be <= {lim.mission_alt_max_agl_m:g} (the mission limit)",
                    f"Högsta tillåtna höjd är {lim.mission_alt_max_agl_m:g} m.",
                    name,
                    code="CHARTER_EXCEEDS_LIMITS",
                    limit=lim.mission_alt_max_agl_m,
                )
            )
        elif value < MIN_ALT_AGL_M:
            out.append(
                _fault(
                    f"{name} must be >= {MIN_ALT_AGL_M:g}",
                    f"Lägsta tillåtna höjd är {MIN_ALT_AGL_M:g} m.",
                    name,
                    code="ALT_BAND_INVALID",
                    limit=MIN_ALT_AGL_M,
                )
            )
    if draft.max_speed_ms and draft.max_speed_ms > lim.mission_max_speed_ms:
        out.append(
            _fault(
                f"max_speed_ms must be <= {lim.mission_max_speed_ms:g} (the mission limit)",
                f"Högsta tillåtna fart är {lim.mission_max_speed_ms:g} m/s.",
                "max_speed_ms",
                code="CHARTER_EXCEEDS_LIMITS",
                limit=lim.mission_max_speed_ms,
            )
        )
    reach = s.limits.max_range_from_pilot_m
    if draft.max_range_from_pilot_m and draft.max_range_from_pilot_m > reach:
        out.append(
            _fault(
                f"max_range_from_pilot_m must be <= {reach:g} (the mission limit)",
                f"Längsta tillåtna avstånd från piloten är {reach:g} m.",
                "max_range_from_pilot_m",
                code="CHARTER_EXCEEDS_LIMITS",
                limit=reach,
            )
        )
    rtl = s.limits.battery_rtl_pct
    if draft.battery_rtl_pct and draft.battery_rtl_pct < rtl:
        out.append(
            _fault(
                f"battery_rtl_pct must be >= {rtl:g} (the mission limit)",
                f"Hemflygning vid batteri får inte vara lägre än {rtl:g} %.",
                "battery_rtl_pct",
                code="CHARTER_EXCEEDS_LIMITS",
                limit=rtl,
            )
        )
    if out:
        return out
    lo = draft.alt_min_agl_m if draft.alt_min_agl_m is not None else _PROPOSAL_ALT_MIN
    hi = draft.alt_max_agl_m if draft.alt_max_agl_m is not None else s.limits.max_alt_agl_m
    mid = draft.default_alt_agl_m if draft.default_alt_agl_m is not None else _PROPOSAL_ALT
    if lo >= hi or not lo <= mid <= hi:
        out.append(
            _fault(
                f"height band invalid: alt_min_agl_m {lo:g} must be below alt_max_agl_m "
                f"{hi:g}, and default_alt_agl_m {mid:g} between them",
                "Höjderna går inte ihop: lägsta höjd måste vara under högsta, och "
                "sökhöjden mellan dem.",
                "alt_min_agl_m",
                code="ALT_BAND_INVALID",
                hint="When the operator names one height, set only default_alt_agl_m.",
            )
        )
    return out


def _version_faults(draft: CharterDraft, s: Situation) -> list[_Fault]:
    current = s.charter.version if s.charter is not None else None
    if draft.base_version is None or draft.base_version == current:
        return []  # left out: the loop fills in the version in force
    return [
        _fault(
            f"base_version {draft.base_version} is not the charter in force ({current})",
            "Förslaget bygger inte på ramen som gäller nu.",
            "base_version",
            code="BASE_VERSION_STALE",
            hint="Leave base_version out; the system fills in the version in force.",
        )
    ]


def _check_draft(
    draft: CharterDraft,
    args: dict[str, Any],
    reg: SkillRegistry,
    s: Situation,
    lim: Limits,
    options: str = "",
) -> CallError | None:
    """The draft as the ground station would see it: skill names, the mission's limits,
    the height band, the version, and the proposal itself (every fault at once)."""
    faults = [
        *_skill_faults(args, reg),
        *_limit_faults(draft, s, lim),
        *_version_faults(draft, s),
    ]
    try:
        to_proposal(draft, s)
    except DraftError as exc:
        if exc.field is None or exc.field not in {f.error.field for f in faults}:
            error = CharterError(code=exc.code, detail_sv=str(exc), field=exc.field)
            faults.append(_Fault(str(exc), error, exc.hint, exc.repairable))
    return _draft_error(faults, args, options) if faults else None


def _validation_faults(exc: ValidationError) -> list[_Fault]:
    out: list[_Fault] = []
    for err in exc.errors()[:6]:
        loc = error_path(err.get("loc", ()))
        detail = f"{loc} saknas." if err.get("type") == "missing" else f"Ogiltigt värde i {loc}."
        out.append(_fault(f"{loc}: {err.get('msg')}", detail, loc))
    return out


def _unreadable(
    name: str,
    why: str,
    kind: Literal["action", "proposal", "read"],
    options: str,
) -> CallError:
    text = f"invalid arguments for {name}: {why}"
    if kind != "proposal":
        return CallError(text, kind)
    fault = _fault(why, "Ramförslaget gick inte att läsa.", None)
    return _draft_error([fault], None, options)


def _validate(
    name: str,
    args: dict[str, Any],
    tool: ToolSpec,
    reg: SkillRegistry,
    s: Situation,
    lim: Limits,
    options: str = "",
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
            faults = [*_validation_faults(exc), *_skill_faults(args, reg)]
            return None, _draft_error(faults, args, options)
        err = _check_draft(draft, args, reg, s, lim, options)
        if err is not None:
            return None, err
        return ParsedCall(id="", name=name, kind="proposal", args=args, draft=draft), None
    spec = reg.get(name)
    rationale = args.pop("rationale_sv", "")
    if name == "goto" and args.get("pin") == GOTO_COORDS:
        args.pop("pin")  # the schema's "fly to lat/lon" choice, not a pin
    model = spec.ai_params_model or spec.params_model
    try:
        model.model_validate(args)
    except ValidationError as exc:
        known = ", ".join(p.id for p in s.pins) or "none"
        return None, (
            f"invalid arguments: {_first_error(exc)}. Map pins: {known}. "
            f"Call {name} again with corrected arguments."
        )
    bound = _check_bounds(args, lim)
    if bound is not None:
        return None, bound
    try:
        adapt(name, args, s)
    except (AdaptError, KeyError, ValueError) as exc:
        return None, str(exc)
    call = ParsedCall(id="", name=name, kind="action", args=args, rationale_sv=str(rationale)[:200])
    return call, None


def _loc_msg(exc: ValidationError) -> tuple[str, str]:
    err = exc.errors()[0]
    loc = ".".join(str(p) for p in err.get("loc", ())) or "arguments"
    return loc, str(err.get("msg"))


def _first_error(exc: ValidationError) -> str:
    loc, msg = _loc_msg(exc)
    return f"{loc}: {msg}"


def parse_tool_calls(
    reply: ModelReply,
    tools: list[ToolSpec],
    reg: SkillRegistry,
    s: Situation,
    lim: Limits,
    said: Said | None = None,
) -> tuple[list[ParsedCall], CallError | None]:
    """The valid calls in order, or the first error (for the repair round). The error's
    ``kind`` says whether a charter draft was refused. ``said``: what the operator's line
    names, for a concrete suggestion in a refused draft's repair text."""
    offered = {t.name: t for t in tools}
    out: list[ParsedCall] = []
    for call_id, name, raw in _raw_calls(reply):
        tool = offered.get(name)
        if tool is None:
            return [], CallError(f"unknown tool {name!r}: it is not offered now")
        options = charter_options(s, said, lim) if tool.kind == "proposal" else ""
        try:
            args = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return [], _unreadable(name, "not JSON", tool.kind, options)
        if not isinstance(args, dict):
            return [], _unreadable(name, "not an object", tool.kind, options)
        parsed, err = _validate(name, dict(args), tool, reg, s, lim, options)
        if err is not None or parsed is None:
            if isinstance(err, CallError):
                return [], err
            return [], CallError(err or "invalid call", tool.kind)
        parsed.id = call_id
        out.append(parsed)
    return out, None
