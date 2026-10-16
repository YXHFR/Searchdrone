"""The tools the model sees, built from the skill registry, and the AI-parameter adapter.

Which tools a turn gets is a pure function of the registry and the situation
(:func:`exposed_tools`): stubs, stage-2 skills and anything the gate would always refuse
in this state are left out. While a STOP, FLY HOME, a battery return, a failsafe or a
manual takeover holds, only the registry's ``allowed_under_latch`` skills (``report``,
``mark_location``) are offered. There is no tool for parameters, the fence, the autonomy
mode, the limits, approvals or releasing a stop.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
from typing import Any, Literal

from sd_brain.charter_draft import CharterDraft
from sd_brain.constants import EXPOSED_STAGE, TAKEOFF_REQUEST_TTL_S
from sd_brain.state_view import Limits, current_limits, motion_block, takeoff_ok
from sd_brain.turn import ToolSpec
from sd_schemas.commander import MissionMemory, Situation
from sd_schemas.skills import SkillRegistry, SkillSpec

__all__ = [
    "ALWAYS_ALLOWED",
    "PROPOSAL_TOOL",
    "READ_TOOL_NAMES",
    "STATE_ALLOWS",
    "AdaptError",
    "adapt",
    "clean_schema",
    "exposed_tools",
    "read_tools",
    "registry_tool_schemas",
    "skill_tool",
]

PROPOSAL_TOOL = "propose_charter_change"
READ_TOOL_NAMES = ("get_situation", "estimate", "get_log", "brain_note")

# Skills that need not be in charter.allowed_skills (the gate's same exception).
ALWAYS_ALLOWED = frozenset({"report", "mark_location", "hold", "return_home", PROPOSAL_TOOL})

# Mirror of the gate's mission-state rule; other states allow every skill.
STATE_ALLOWS: dict[str, frozenset[str]] = {
    "DRAFT": frozenset({"report", "mark_location", PROPOSAL_TOOL}),
    "CHARTER_PROPOSED": frozenset({"report", "mark_location", PROPOSAL_TOOL}),
    "LANDED": frozenset({"report", "mark_location", PROPOSAL_TOOL}),
    "READY": frozenset({"report", "mark_location", PROPOSAL_TOOL, "takeoff"}),
    "CLOSED": frozenset(),
}


class AdaptError(ValueError):
    """The AI parameters do not resolve (unknown pin or sector); sent back to the model."""


# --- JSON Schema helpers ----------------------------------------------------------------


def clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$ref``, drop titles and ``null`` alternatives: easier for a small model."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(x) for x in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            return walk(defs[name])
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in ("title", "$defs") or (key == "default" and value is None):
                continue
            out[key] = walk(value)
        alts = out.get("anyOf")
        if isinstance(alts, list):
            non_null = [a for a in alts if a != {"type": "null"}]
            if len(non_null) == 1:
                del out["anyOf"]
                out.update(non_null[0])
        return out

    return walk(schema)


def _bounded(schema: dict[str, Any], lim: Limits | None) -> dict[str, Any]:
    out = copy.deepcopy(schema)
    props = out.get("properties", {})
    if lim is not None:
        if "alt_m" in props:
            props["alt_m"]["minimum"] = lim.alt_min_agl_m
            props["alt_m"]["maximum"] = lim.alt_max_agl_m
        if "speed_ms" in props:
            props["speed_ms"]["minimum"] = 1
            props["speed_ms"]["maximum"] = lim.max_speed_ms
    return out


def _description(spec: SkillSpec) -> str:
    return f"{spec.description_en} Swedish: {', '.join(spec.synonyms_sv)}."


def skill_tool(spec: SkillSpec, lim: Limits | None) -> ToolSpec:
    """The OpenAI tool for a skill, with this turn's height and speed bounds."""
    if spec.name == PROPOSAL_TOOL:
        params = clean_schema(CharterDraft.model_json_schema())
        params["additionalProperties"] = False
        kind: Literal["action", "proposal"] = "proposal"
    else:
        model = spec.ai_params_model or spec.params_model
        params = _bounded(clean_schema(model.model_json_schema()), lim)
        props = params.setdefault("properties", {})
        props["rationale_sv"] = {"type": "string", "maxLength": 200}
        params["required"] = sorted({*params.get("required", []), "rationale_sv"})
        params["type"] = "object"
        params["additionalProperties"] = False
        kind = "action"
    return ToolSpec(
        name=spec.name,
        kind=kind,
        openai={
            "type": "function",
            "function": {
                "name": spec.name,
                "description": _description(spec),
                "parameters": params,
            },
        },
    )


def _read(name: str, description: str, params: dict[str, Any]) -> ToolSpec:
    return ToolSpec(
        name=name,
        kind="read",
        openai={
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "additionalProperties": False, **params},
            },
        },
    )


def read_tools() -> list[ToolSpec]:
    return [
        _read(
            "get_situation",
            "Fetch a fresh situation picture (LÄGE). Swedish: läget, status.",
            {"properties": {}},
        ),
        _read(
            "estimate",
            "Dry-run an order through the safety gate: verdict, time and battery after. "
            "Nothing is sent to the drone.",
            {
                "properties": {"skill": {"type": "string"}, "params": {"type": "object"}},
                "required": ["skill", "params"],
            },
        ),
        _read(
            "get_log",
            "The latest mission events.",
            {
                "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 20}},
                "required": ["n"],
            },
        ),
        _read(
            "brain_note",
            "Replace your plan note (kept between turns).",
            {
                "properties": {"text": {"type": "string", "maxLength": 400}},
                "required": ["text"],
            },
        ),
    ]


def exposed_tools(
    reg: SkillRegistry,
    s: Situation,
    mem: MissionMemory,
    now: dt.datetime,
    *,
    ttl_s: float = TAKEOFF_REQUEST_TTL_S,
    actions: bool = True,
    pilot: bool = True,
) -> list[ToolSpec]:
    """The tools for one turn (read tools first). ``actions = False``: read tools only."""
    out = read_tools()
    if not actions:
        return out
    lim = current_limits(s)
    blocked = motion_block(s)
    for k in reg:
        if k.status != "implemented" or k.stage > EXPOSED_STAGE or not k.ai_allowed:
            continue
        if not pilot and k.name != "report":
            continue
        if not s.ai_enabled and k.name != "report":
            continue
        if blocked is not None:
            if k.allowed_under_latch:
                out.append(skill_tool(k, lim))
            continue
        allowed_here = STATE_ALLOWS.get(s.mission_status)
        if allowed_here is not None and k.name not in allowed_here:
            continue
        if k.name not in ALWAYS_ALLOWED and (
            s.charter is None or k.name not in s.charter.allowed_skills
        ):
            continue
        if k.allowed_when == "on_ground" and s.vehicle.airborne:
            continue
        if k.allowed_when == "airborne" and not s.vehicle.airborne:
            continue
        if k.starts_flight and not takeoff_ok(s, mem, now, ttl_s=ttl_s):
            continue
        if k.name == "resume" and s.vehicle.executor_state != "HOLD":
            continue
        out.append(skill_tool(k, lim))
    return out


def registry_tool_schemas(reg: SkillRegistry) -> list[dict[str, Any]]:
    """Every exposable tool with the registry's fixed bounds (for ``ai_config_hash``)."""
    tools = [t.openai for t in read_tools()]
    for k in reg:
        if k.status == "implemented" and k.stage <= EXPOSED_STAGE and k.ai_allowed:
            tools.append(skill_tool(k, None).openai)
    return json.loads(json.dumps(tools, sort_keys=True))


# --- adapter: AI parameters -> registry parameters ---------------------------------------


def _pin(s: Situation, pin_id: str) -> tuple[float, float]:
    pin = next((p for p in s.pins if p.id == pin_id), None)
    if pin is None:
        known = ", ".join(p.id for p in s.pins) or "none"
        raise AdaptError(f"unknown pin {pin_id}; known: {known}")
    return pin.lat, pin.lon


def adapt(skill: str, ai: dict[str, Any], s: Situation) -> dict[str, Any]:
    """Turn the model's parameters into the registry's ``params_model`` form."""
    lim = current_limits(s)
    if skill == "goto":
        if ai.get("pin") is not None:
            lat, lon = _pin(s, str(ai["pin"]))
        else:
            lat, lon = float(ai["lat"]), float(ai["lon"])
        out: dict[str, Any] = {
            "lat": lat,
            "lon": lon,
            "alt_m": ai.get("alt_m") if ai.get("alt_m") is not None else lim.default_alt_agl_m,
        }
        if ai.get("speed_ms") is not None:
            out["speed_ms"] = ai["speed_ms"]
        return out
    if skill == "survey_sector":
        sector = str(ai["sector"])
        known = [str(x.get("sector_id")) for x in s.sectors]
        if s.charter is not None:
            known += [x.sector_id for x in s.charter.sectors if x.sector_id not in known]
        if sector not in known:
            raise AdaptError(f"unknown sector {sector}; known: {', '.join(known) or 'none'}")
        out = {
            "sector_id": sector,
            "pattern": "lawnmower",
            "alt_m": ai.get("alt_m") if ai.get("alt_m") is not None else lim.default_alt_agl_m,
        }
        for key in ("spacing_m", "speed_ms"):
            if ai.get(key) is not None:
                out[key] = ai[key]
        return out
    if skill == "mark_location":
        if ai.get("pin") is not None:
            lat, lon = _pin(s, str(ai["pin"]))
        elif ai.get("lat") is not None:
            lat, lon = float(ai["lat"]), float(ai["lon"])
        elif s.vehicle.lat is not None and s.vehicle.lon is not None:
            lat, lon = s.vehicle.lat, s.vehicle.lon
        else:
            raise AdaptError("the drone's position is unknown; give a pin or lat/lon")
        return {"lat": lat, "lon": lon, "label": ai["label"]}
    return {k: v for k, v in ai.items() if k != "rationale_sv"}
