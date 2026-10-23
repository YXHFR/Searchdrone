"""The tools the model sees, built from the skill registry, and the AI-parameter adapter.

Which tools a turn gets is a pure function of the registry and the situation
(:func:`exposed_tools`): stubs, stage-2 skills and anything the gate would always refuse
in this state are left out. While a STOP, FLY HOME, a battery return, a failsafe or a
manual takeover holds, only the registry's ``allowed_under_latch`` skills (``report``,
``mark_location``) are offered. An onboard skill is offered only in the executor states
it may start from (``allowed_from``). There is no tool for parameters, the fence, the
autonomy mode, the limits, approvals or releasing a stop.
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
from sd_schemas.safety import MIN_ALT_AGL_M
from sd_schemas.search import SearchProfileDraft
from sd_schemas.skills import SkillRegistry, SkillSpec

__all__ = [
    "ALWAYS_ALLOWED",
    "MAX_SEEN_HOURS_AGO",
    "PROFILE_TOOL",
    "PROPOSAL_TOOL",
    "READ_TOOL_NAMES",
    "STATE_ALLOWS",
    "AdaptError",
    "adapt",
    "charter_skills",
    "clean_schema",
    "exposed_tools",
    "pin_point",
    "profile_tool",
    "read_tools",
    "registry_tool_schemas",
    "skill_tool",
]

PROPOSAL_TOOL = "propose_charter_change"
READ_TOOL_NAMES = ("get_situation", "estimate", "get_log", "brain_note")
# Not a skill: it never moves the drone. The ground stores a proposal that a pilot must
# approve before anything uses it, and the model gets the answer as a tool result.
PROFILE_TOOL = "propose_search_profile"
MAX_SEEN_HOURS_AGO = 168.0

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
    """The AI parameters do not resolve (unknown pin, sector or path); sent back to the model."""


# --- JSON Schema helpers ----------------------------------------------------------------


def clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$ref``, drop titles and ``null`` alternatives: easier for a small model.

    A tagged union (pydantic's ``oneOf`` + ``discriminator``) becomes a plain ``anyOf``
    and ``const`` a one-value ``enum``: what every OpenAI-compatible tool format reads.
    The validation stays pydantic's, tag first."""
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
            if key in ("title", "$defs", "discriminator") or (key == "default" and value is None):
                continue
            if key == "oneOf":
                key = "anyOf"
            if key == "const":
                key, value = "enum", [value]
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


def charter_skills(reg: SkillRegistry) -> list[str]:
    """The skill names a charter may allow: every implemented skill of this stage (the
    ground station's default set). ``allowed_skills`` in a draft is a subset of these."""
    return sorted(k.name for k in reg if k.status == "implemented" and k.stage <= EXPOSED_STAGE)


def _situated_areas(params: dict[str, Any], s: Situation) -> None:
    """Offer only the area options this situation can fill: ``drawn`` (and
    ``operating_area``) only when the operator drew an area, ``area_id`` and
    ``center_pin_id`` as the known labels, no pin field without pins (``center`` is then
    required). What remains follows ``CharterDraft`` (only "exactly one of center_pin_id
    and center" is left to the turn's check, which is unchanged)."""
    props = params["properties"]
    items = props["search_areas"]["items"]
    shapes = {a["properties"]["kind"]["enum"][0]: a for a in items["anyOf"]}
    # Unique and in a fixed order: an enum must not repeat a value, and labels may.
    areas = sorted({a.label for a in s.areas if a.kind == "area"})
    pins = list(dict.fromkeys(p.id for p in s.pins))
    for kind in ("rect", "circle"):
        shape = shapes[kind]
        fields = shape["properties"]
        if pins:
            fields["center_pin_id"].pop("pattern", None)
            fields["center_pin_id"]["enum"] = pins
            fields["center_pin_id"]["description"] = "A map pin listed under Kartnålar in LÄGE."
        else:
            # Without pins the centre must be coordinates (CharterDraft wants one of the two).
            del fields["center_pin_id"]
            fields["center"]["description"] = "Centre as coordinates the operator wrote."
            fields["kind"]["description"] = f"A {'rectangle' if kind == 'rect' else 'circle'}."
            shape["required"] = [*shape.get("required", []), "center"]
    if areas:
        shapes["drawn"]["properties"]["area_id"]["enum"] = areas
        props["operating_area"]["properties"]["area_id"]["enum"] = areas
    else:
        del shapes["drawn"]
        del props["operating_area"]
        props["search_areas"]["description"] = "The areas to search: a rect or circle."
    items["anyOf"] = list(shapes.values())


# goto's pin value for "the coordinates given" when map pins exist (never a pin id).
GOTO_COORDS = "KOORDINATER"


def _situated_goto(params: dict[str, Any], s: Situation) -> None:
    """A goto target this situation can fill, required: with map pins the pin (one of
    them, or ``GOTO_COORDS`` for the coordinates the operator wrote), without pins
    coordinates. ``GotoAiParams`` takes either; this schema makes the model name its
    target. The parser drops ``GOTO_COORDS`` before the params are checked."""
    props = params["properties"]
    pins = list(dict.fromkeys(p.id for p in s.pins))
    if pins:
        props["pin"].pop("pattern", None)
        props["pin"]["enum"] = [*pins, GOTO_COORDS]
        props["pin"]["description"] = (
            f"The map pin to fly to (Kartnålar in LÄGE), or {GOTO_COORDS} to fly to the "
            "lat and lon the operator wrote. Required."
        )
        props["lat"]["description"] = f"Only with pin {GOTO_COORDS}: latitude the operator wrote."
        props["lon"]["description"] = f"Only with pin {GOTO_COORDS}: longitude."
        required = ["pin"]
    else:
        props.pop("pin", None)
        props["lat"]["description"] = "Latitude the operator wrote (no map pins exist)."
        props["lon"]["description"] = "Longitude the operator wrote."
        required = ["lat", "lon"]
    params["required"] = sorted({*params.get("required", []), *required})


def _proposal_schema(
    lim: Limits | None, skills: list[str] | None, s: Situation | None = None
) -> dict[str, Any]:
    params = clean_schema(CharterDraft.model_json_schema())
    params["additionalProperties"] = False
    props = params.get("properties", {})
    if s is not None:
        _situated_areas(params, s)
    if skills is not None:
        props["allowed_skills"]["items"] = {"type": "string", "enum": list(skills)}
    if lim is not None:
        # A charter is bounded by the mission's own limits, not by the charter in force.
        for name in ("alt_min_agl_m", "alt_max_agl_m", "default_alt_agl_m"):
            props[name]["minimum"] = MIN_ALT_AGL_M
            props[name]["maximum"] = lim.mission_alt_max_agl_m
        props["max_speed_ms"]["maximum"] = lim.mission_max_speed_ms
        props["max_range_from_pilot_m"]["maximum"] = lim.mission_max_range_from_pilot_m
        props["battery_rtl_pct"]["minimum"] = lim.mission_battery_rtl_pct
    return params


def skill_tool(
    spec: SkillSpec,
    lim: Limits | None,
    skills: list[str] | None = None,
    s: Situation | None = None,
) -> ToolSpec:
    """The OpenAI tool for a skill, with this turn's height and speed bounds. ``skills``:
    the names a charter proposal may allow (:func:`charter_skills`); ``s``: the situation
    whose pins and drawn areas a proposal may name (``None``: every option)."""
    if spec.name == PROPOSAL_TOOL:
        params = _proposal_schema(lim, skills, s)
        kind: Literal["action", "proposal"] = "proposal"
    else:
        model = spec.ai_params_model or spec.params_model
        params = _bounded(clean_schema(model.model_json_schema()), lim)
        props = params.setdefault("properties", {})
        props["rationale_sv"] = {"type": "string", "maxLength": 200}
        params["required"] = sorted({*params.get("required", []), "rationale_sv"})
        params["type"] = "object"
        params["additionalProperties"] = False
        if spec.name == "goto" and s is not None:
            _situated_goto(params, s)
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


def profile_tool(s: Situation | None = None) -> ToolSpec:
    """The search-profile proposal: ``SearchProfileDraft`` with the last-seen point as a
    map pin or the coordinates the operator wrote, and the time as hours ago (the model
    never sees the clock or coordinates it was not given)."""
    params = clean_schema(SearchProfileDraft.model_json_schema())
    params["additionalProperties"] = False
    props = params["properties"]
    seen: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "description": "Where and when the person was last seen. Without a place: ask.",
        "properties": {
            "pin": {"type": "string", "description": "A map pin listed under Kartnålar."},
            "lat": {"type": "number", "description": "Only a latitude the operator wrote."},
            "lon": {"type": "number", "description": "Only a longitude the operator wrote."},
            "hours_ago": {
                "type": "number",
                "minimum": 0,
                "maximum": MAX_SEEN_HOURS_AGO,
                "description": "How long ago, in hours (0.5 = half an hour).",
            },
            "heading_deg": props["last_seen"]["properties"]["heading_deg"],
            "kind": props["last_seen"]["properties"]["kind"],
            "source": props["last_seen"]["properties"]["source"],
        },
        "required": ["hours_ago"],
    }
    pins = list(dict.fromkeys(p.id for p in s.pins)) if s is not None else []
    if s is not None and not pins:
        # No pin to name: only coordinates the operator wrote.
        del seen["properties"]["pin"]
        seen["required"] = ["lat", "lon", "hours_ago"]
    elif pins:
        # A place is a pin (the pilot can drop one): no coordinates from the model.
        seen["properties"]["pin"]["enum"] = pins
        del seen["properties"]["lat"], seen["properties"]["lon"]
        seen["required"] = ["pin", "hours_ago"]
    props["last_seen"] = seen
    props["garments"]["description"] = (
        "Every piece of clothing the operator mentioned, one entry each, with its own "
        "colour (never the colour of a bag or another garment), also the ones that are "
        "only a guess (certainty unsure). "
        "part: outer = jacka, kappa, rock, väst; upper = tröja, skjorta, t-shirt; "
        "lower = byxor, jeans, shorts, kjol; headwear = mössa, keps, hatt; "
        "footwear = skor, stövlar; full_body = overall."
    )
    props["items"]["description"] = (
        "What the person carries or has with them (ryggsäck, väska, käpp, rollator, "
        "paraply, cykel, hund; anything else is other with a note)."
    )
    props["visual_notes"]["description"] = (
        "Only visible details that are no garment, item or hair (glasögon, skägg, gångstil); "
        "never a name. Clothes always go in garments."
    )
    props["subject_category"]["description"] = (
        "demens = dementia, vandrare = hiker, nedstämd = despondent, bär-/svampplockare "
        "= gatherer, jägare = hunter, barn 1-6 / 7-12 / 13-15 år, autism, annat = other."
    )
    params.pop("description", None)
    params.pop("title", None)
    return ToolSpec(
        name=PROFILE_TOOL,
        kind="read",
        openai={
            "type": "function",
            "function": {
                "name": PROFILE_TOOL,
                "description": (
                    "Propose the search profile of the missing person from what the operator "
                    "said. Nothing is used until the pilot approves it on the card. "
                    "Swedish: sökprofil, beskrivning, försvunnen person."
                ),
                "parameters": params,
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
    proposals: bool = True,
) -> list[ToolSpec]:
    """The tools for one turn (read tools first). ``actions = False``: read tools only;
    ``proposals = False``: no charter proposal (the loop offers it only in a turn an
    operator line started, or in the retry of a refused proposal)."""
    out = read_tools()
    if not actions:
        return out
    lim = current_limits(s)
    blocked = motion_block(s)
    skills = charter_skills(reg)
    for k in reg:
        if k.status != "implemented" or k.stage > EXPOSED_STAGE or not k.ai_allowed:
            continue
        if not pilot and k.name != "report":
            continue
        if not proposals and k.name == PROPOSAL_TOOL:
            continue
        if not s.ai_enabled and k.name != "report":
            continue
        if blocked is not None:
            if k.allowed_under_latch:
                out.append(skill_tool(k, lim, skills, s))
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
        if k.allowed_from and s.vehicle.executor_state not in k.allowed_from:
            continue  # the onboard executor would refuse it in this state
        out.append(skill_tool(k, lim, skills, s))
    if any(t.name == PROPOSAL_TOOL for t in out):
        # Offered with the charter proposal: a pilot's line, the AI on, no latch.
        out.append(profile_tool(s))
    return out


def registry_tool_schemas(reg: SkillRegistry) -> list[dict[str, Any]]:
    """Every exposable tool with the registry's fixed bounds (for ``ai_config_hash``)."""
    tools = [t.openai for t in read_tools()]
    tools.append(profile_tool().openai)
    skills = charter_skills(reg)
    for k in reg:
        if k.status == "implemented" and k.stage <= EXPOSED_STAGE and k.ai_allowed:
            tools.append(skill_tool(k, None, skills).openai)
    return json.loads(json.dumps(tools, sort_keys=True))


# --- adapter: AI parameters -> registry parameters ---------------------------------------


def pin_point(s: Situation, pin_id: str) -> tuple[float, float]:
    pin = next((p for p in s.pins if p.id == pin_id), None)
    if pin is None:
        known = ", ".join(p.id for p in s.pins) or "none"
        raise AdaptError(f"unknown pin {pin_id}; known: {known}")
    return pin.lat, pin.lon


def _path(s: Situation, name: str) -> list[dict[str, float]]:
    """The points of a path the pilot drew, by its name: the model never writes them."""
    path = next((a for a in s.paths if a.label == name), None)
    if path is None:
        known = ", ".join(a.label for a in s.paths) or "none"
        raise AdaptError(f"unknown path {name}; known: {known}")
    return [{"lat": float(lat), "lon": float(lon)} for lon, lat in path.geometry["coordinates"]]


def adapt(skill: str, ai: dict[str, Any], s: Situation) -> dict[str, Any]:
    """Turn the model's parameters into the registry's ``params_model`` form."""
    lim = current_limits(s)
    if skill == "goto":
        if ai.get("pin") is not None:
            lat, lon = pin_point(s, str(ai["pin"]))
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
    if skill == "follow_path":
        name = str(ai["path"])
        out = {
            "points": _path(s, name),
            "alt_m": ai.get("alt_m") if ai.get("alt_m") is not None else lim.default_alt_agl_m,
            "label": name,
        }
        if ai.get("speed_ms") is not None:
            out["speed_ms"] = ai["speed_ms"]
        return out
    if skill == "mark_location":
        if ai.get("pin") is not None:
            lat, lon = pin_point(s, str(ai["pin"]))
        elif ai.get("lat") is not None:
            lat, lon = float(ai["lat"]), float(ai["lon"])
        elif s.vehicle.lat is not None and s.vehicle.lon is not None:
            lat, lon = s.vehicle.lat, s.vehicle.lon
        else:
            raise AdaptError("the drone's position is unknown; give a pin or lat/lon")
        return {"lat": lat, "lon": lon, "label": ai["label"]}
    return {k: v for k, v in ai.items() if k != "rationale_sv"}
