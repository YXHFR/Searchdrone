"""``ScriptedBackend``: a fixed script per trigger (tests and ``just e2e``), only in ``sim``.

Script format (``simulation/scenarios/scripts/SCxx.yaml``)::

    scenario: SC01
    fixture: ../demo_fixture.yaml        # relative to the script
    turns:                               # tried top-down, the first match runs
      - on: operator_message             # a Trigger value
        match: '^Starta\\.$'              # fullmatch on the operator's text
        say: ["…"]
        action: {skill: takeoff, params: {alt_m: 40}}
      - on: skill_event
        when: {skill: goto, status: done}
        action: {skill: survey_sector, params: {polygon: "fixture:survey_box", alt_m: 40}}

``ai: true`` on an action sends the parameters in the model's form through the adapter
(``{path: pa-1}`` becomes the drawn path's points). ``fixture:<pin>`` under ``at``
becomes ``lat``/``lon`` (``at: here`` is the drone's position in the turn's situation,
``at: "event:find"`` the find that started a ``detection`` turn; ``say`` may use the
event's fields, e.g. ``{name_sv}``); ``fixture:<area>`` under ``polygon`` becomes a
GeoJSON polygon; ``charter_proposal: {from_fixture: <name>}`` sends that charter from the
fixture. ``delay_s`` delays the turn and ``fail: timeout | crash`` simulates a broken
backend. Unknown keys refuse to load.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from sd_brain.backends.base import BackendCrashError, ToolExecutor
from sd_brain.geo import from_enu
from sd_brain.turn import AiAction, OutMessage, TurnDiagnostics, TurnInput, TurnOutput
from sd_schemas.charter import CharterProposal

__all__ = ["ScriptError", "ScriptedBackend", "load_script"]

WHEN_FIELDS: dict[str, frozenset[str]] = {
    "mission_created": frozenset(),
    "operator_message": frozenset(),
    "operator_override": frozenset({"action", "via"}),
    "skill_event": frozenset({"skill", "status", "status_reason", "issuer_kind"}),
    "detection": frozenset({"label", "kind", "source"}),
    "approval_event": frozenset({"kind", "decision"}),
    "charter_event": frozenset({"status", "install_status"}),
    "takeoff_clearance": frozenset(),
    "alert_critical": frozenset({"code"}),
    "alert_warning": frozenset({"code"}),
    "idle_tick": frozenset(),
    "gate_reject_retry": frozenset(),
    "recovered": frozenset(),
}


class _Loader(yaml.SafeLoader):
    """YAML 1.2 booleans only: ``on:`` (the trigger key) stays a string."""


_Loader.yaml_implicit_resolvers = {
    k: [(tag, rx) for tag, rx in v if tag != "tag:yaml.org,2002:bool"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
for _first in "tTfF":
    _Loader.yaml_implicit_resolvers.setdefault(_first, []).append(
        ("tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"))
    )


class ScriptError(ValueError):
    """The script cannot be used (the process exits with code 2)."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScriptAction(_Strict):
    skill: str
    params: dict[str, Any] = Field(default_factory=dict[str, Any])
    # The parameters are in the model's form (e.g. ``{path: pa-1}``) and go through the
    # same adapter as a language model's, against the live situation.
    ai: bool = False


class ScriptTurn(_Strict):
    on: Literal[
        "mission_created",
        "operator_message",
        "operator_override",
        "skill_event",
        "detection",
        "approval_event",
        "charter_event",
        "takeoff_clearance",
        "alert_critical",
        "alert_warning",
        "idle_tick",
        "gate_reject_retry",
        "recovered",
    ]
    match: str | None = None
    when: dict[str, Any] = Field(default_factory=dict[str, Any])
    say: list[str] = Field(default_factory=list[str], max_length=2)
    action: ScriptAction | None = None
    raw: ScriptAction | None = None
    charter_proposal: dict[str, str] | None = None
    once: bool = True
    delay_s: float = Field(default=0, ge=0)
    fail: Literal["timeout", "crash"] | None = None


class Script(_Strict):
    scenario: str
    timeout_s: float | None = None
    expected_runtime_s: float | None = None
    fixture: str | None = None
    turns: list[ScriptTurn]


def load_script(path: str | Path) -> tuple[Script, dict[str, Any] | None]:
    p = Path(path)
    try:
        data = yaml.load(p.read_text(encoding="utf-8"), Loader=_Loader)  # noqa: S506 - a SafeLoader subclass
        script = Script.model_validate(data)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ScriptError(f"manuset {p} går inte att läsa: {exc}") from None
    for t in script.turns:
        bad = set(t.when) - WHEN_FIELDS[t.on]
        if bad:
            raise ScriptError(f"manuset {p}: fälten {sorted(bad)} får inte stå i when för {t.on}")
        if t.match is not None and t.on != "operator_message":
            raise ScriptError(f"manuset {p}: match gäller bara operator_message")
        if t.match is not None:
            re.compile(t.match)
    fixture = None
    if script.fixture:
        fpath = (p.parent / script.fixture).resolve()
        try:
            fixture = yaml.load(fpath.read_text(encoding="utf-8"), Loader=_Loader)  # noqa: S506 - a SafeLoader subclass
        except (OSError, yaml.YAMLError) as exc:
            raise ScriptError(f"fixturen {fpath} går inte att läsa: {exc}") from None
    return script, fixture


class _Fixture:
    def __init__(self, data: dict[str, Any] | None) -> None:
        self.data = data or {}
        home = self.data.get("home") or {}
        self.lat0 = float(home.get("lat", 0.0))
        self.lon0 = float(home.get("lon", 0.0))

    def _need(self) -> None:
        if not self.data:
            raise ScriptError("manuset använder fixture: men har ingen fixtur")

    def latlon(self, east: float, north: float) -> list[float]:
        lat, lon = from_enu(east, north, self.lat0, self.lon0)
        return [round(lon, 7), round(lat, 7)]

    def pin(self, name: str) -> tuple[float, float]:
        self._need()
        pin = next((p for p in self.data.get("pins", []) if p.get("id") == name), None)
        if pin is None:
            raise ScriptError(f"okänd nål i fixturen: {name}")
        lon, lat = self.latlon(float(pin["east_m"]), float(pin["north_m"]))
        return lat, lon

    def polygon(self, name: str) -> dict[str, Any]:
        self._need()
        if name == "survey_box":
            box = self.data["survey_box"]
            self.pin(box["center_pin"])  # raises for an unknown pin
            pin = next(p for p in self.data["pins"] if p["id"] == box["center_pin"])
            cx, cy = float(pin["east_m"]), float(pin["north_m"])
            w, h = (float(x) / 2 for x in box["size_m"])
            ring = [(cx - w, cy - h), (cx + w, cy - h), (cx + w, cy + h), (cx - w, cy + h)]
        else:
            area = self.data.get("areas", {}).get(name) or self.data.get("charters", {}).get(
                name, {}
            )
            if "area_enu" not in area:
                raise ScriptError(f"okänd yta i fixturen: {name}")
            ring = [(float(e), float(n)) for e, n in area["area_enu"]]
        coords = [self.latlon(e, n) for e, n in ring]
        return {"type": "Polygon", "coordinates": [[*coords, coords[0]]]}

    def charter(self, name: str, reason: str) -> CharterProposal:
        self._need()
        c = self.data.get("charters", {}).get(name)
        if c is None:
            raise ScriptError(f"okänd ram i fixturen: {name}")
        body: dict[str, Any] = {"area": self.polygon(name), "reason_sv": reason}
        for key in (
            "geofence_buffer_m",
            "alt_min_agl_m",
            "alt_max_agl_m",
            "default_alt_agl_m",
            "max_speed_ms",
            "max_range_from_pilot_m",
            "battery_rtl_pct",
            "allowed_skills",
        ):
            if key in c:
                body[key] = c[key]
        # The fixture's own sectors (an area by name); without them the server cuts the area.
        sectors = [
            {
                "sector_id": s["sector_id"],
                "name": s.get("name", s["sector_id"]),
                "area": self.polygon(s["from"]),
                "order_idx": i,
            }
            for i, s in enumerate(c.get("sectors") or [])
        ]
        if sectors:
            body["sectors"] = sectors
        return CharterProposal.model_validate(body)


def _resolve(
    params: dict[str, Any],
    fx: _Fixture,
    turn: TurnInput,
    event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``fixture:<name>`` takes a pin or an area from the fixture, ``at: here`` the drone's
    position, and ``event:find`` (only for ``at``) the position of the event that started
    the turn (a find)."""
    out: dict[str, Any] = {}
    for key, value in params.items():
        if key == "at" and value == "here":
            v = turn.situation.vehicle
            if v.lat is None or v.lon is None:
                raise ScriptError("at: here, men drönarens position är okänd")
            out["lat"], out["lon"] = v.lat, v.lon
            continue
        if isinstance(value, str) and value == "event:find":
            if key != "at" or not event or "lat" not in event or "lon" not in event:
                raise ScriptError("event:find går bara att använda för at i en detection-tur")
            out["lat"], out["lon"] = float(event["lat"]), float(event["lon"])
            continue
        if isinstance(value, str) and value.startswith("fixture:"):
            name = value.removeprefix("fixture:")
            if key == "at":
                out["lat"], out["lon"] = fx.pin(name)
                continue
            if key == "polygon":
                out["polygon"] = fx.polygon(name)
                continue
            raise ScriptError(f"fixture: går inte att använda för {key}")
        out[key] = value
    return out


def _event_bodies(turn: TurnInput) -> list[dict[str, Any]]:
    """The turn's event bodies, newest first. A turn can carry more than one event (an
    onboard refusal and its alert arrive together), and ``when`` matches any of them."""
    return [dict(e.get("body") or {}) for e in reversed(turn.events)] or [{}]


def _matches_when(when: dict[str, Any], body: dict[str, Any]) -> bool:
    for key, want in when.items():
        have = body.get(key)
        if isinstance(want, list):
            if have not in want:
                return False
        elif have != want:
            return False
    return True


class ScriptedBackend:
    name = "scripted"

    def __init__(self, script: Script, fixture: dict[str, Any] | None = None) -> None:
        self.script = script
        self.fixture = _Fixture(fixture)
        self._used: set[int] = set()

    @classmethod
    def from_file(cls, path: str | Path) -> ScriptedBackend:
        script, fixture = load_script(path)
        return cls(script, fixture)

    def _pick(self, turn: TurnInput) -> ScriptTurn | None:
        bodies = _event_bodies(turn)
        text = next((m.text for m in reversed(turn.chat_tail) if m.role == "operator"), "")
        for i, t in enumerate(self.script.turns):
            if t.on != turn.trigger or (t.once and i in self._used):
                continue
            if t.match is not None and re.fullmatch(t.match, text) is None:
                continue
            if not any(_matches_when(t.when, body) for body in bodies):
                continue
            if t.once:
                self._used.add(i)
            return t
        return None

    async def decide(self, turn: TurnInput, tools: ToolExecutor) -> TurnOutput:
        start = time.monotonic()
        out = TurnOutput(diagnostics=TurnDiagnostics(backend="scripted"))
        t = self._pick(turn)
        if t is None:
            return out
        if t.delay_s:
            await asyncio.sleep(t.delay_s)
        if t.fail == "timeout":
            await asyncio.sleep(3600)
        if t.fail == "crash":
            raise BackendCrashError("scripted crash")
        groups: dict[str, str] = {}
        if t.match is not None:
            text = next((m.text for m in reversed(turn.chat_tail) if m.role == "operator"), "")
            m = re.fullmatch(t.match, text)
            groups = {k: v for k, v in (m.groupdict() if m else {}).items() if v is not None}
        event = next((b for b in _event_bodies(turn) if _matches_when(t.when, b)), {})
        fields = {k: str(v) for k, v in event.items() if isinstance(v, str | int | float)}
        out.messages = [OutMessage(text=line.format(**{**fields, **groups})) for line in t.say]
        if t.charter_proposal is not None:
            name = t.charter_proposal.get("from_fixture", "").format(**groups)
            out.ready_proposal = self.fixture.charter(name, f"Ram {name} ur manuset.")
        elif t.raw is not None:
            out.action = AiAction(skill=t.raw.skill, params=t.raw.params, adapted=True, raw=True)
        elif t.action is not None:
            out.action = AiAction(
                skill=t.action.skill,
                params=_resolve(t.action.params, self.fixture, turn, event),
                rationale_sv="Steg i manuset.",
                adapted=not t.action.ai,
            )
        out.diagnostics.latency_ms = int((time.monotonic() - start) * 1000)
        return out
