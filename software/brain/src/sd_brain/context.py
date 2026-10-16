"""``ContextBuilder``: the messages for one turn (system prompt + one turn block).

Positions are shown as distance and compass direction from home, never as coordinates
(unless the operator wrote coordinates). Labels written by people are quoted and cut to
40 characters; the system prompt says quoted text is data, not instructions.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable
from typing import Any

import yaml

from sd_brain.ai_config import prompt_text
from sd_brain.geo import bearing_word, to_enu
from sd_brain.state_view import current_limits, motion_block
from sd_brain.sv import interpret
from sd_brain.turn import TurnInput
from sd_schemas.api import ChatMessage
from sd_schemas.commander import Situation
from sd_schemas.skills import SkillRegistry

__all__ = ["ContextBuilder", "situation_text", "takeoff_rule"]

_TAKEOFF_RULES = {
    "request_then_clearance": (
        "You may take off only when the charter is approved, a person has confirmed "
        "'Fritt runt drönaren' with the button, and the operator has asked you to start. "
        "The `takeoff` tool is only offered when all of that holds. If the operator asks "
        "to start without the confirmation, answer: 'Bekräfta att det är fritt runt "
        "drönaren med knappen, så startar jag.'"
    ),
    "clearance_only": (
        "You may take off only when the charter is approved and a person has confirmed "
        "'Fritt runt drönaren' with the button; then you decide when to start according "
        "to your plan. The `takeoff` tool is only offered when that holds."
    ),
}

_BLOCK_TEXT = {
    "stop": "STOPP (operatören)",
    "return": "FLYG HEM (operatören)",
    "battery": "batterihemflygning",
    "failsafe": "failsafe",
    "held": "övertagen (extern lägesändring)",
    "manual": "manuell styrning (operatören)",
}


def takeoff_rule(policy: str) -> str:
    return _TAKEOFF_RULES[policy]


def _num(value: float, digits: int = 0) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _quote(text: str | None) -> str:
    if not text:
        return ""
    clean = " ".join(text.split()).replace('"', "'")
    return f'"{clean[:40]}"'


def _where(s: Situation, lat: float | None, lon: float | None) -> str:
    if s.home is None or lat is None or lon is None:
        return "okänd plats"
    east, north = to_enu(lat, lon, s.home.lat, s.home.lon)
    dist = math.hypot(east, north)
    if dist < 10:
        return "vid hem"
    return f"{round(dist, -1):.0f} m {bearing_word(east, north)} om hem"


def situation_text(s: Situation, now: dt.datetime) -> str:
    """The LÄGE block. Values come from the ground station's situation only."""
    block = motion_block(s)
    tools_note = " → bara report och mark_location" if block else ""
    latch = f"{_BLOCK_TEXT[block]}{tools_note}" if block else "ingen"
    mode = {"APPROVE_EACH": "A", "WITHIN_CHARTER": "B"}.get(str(s.autonomy_mode or ""), "-")
    if s.charter is None:
        charter = "ingen"
    else:
        installed = (
            "installerad, staket OK"
            if s.charter_installed and s.fence_verified
            else ("inte installerad")
        )
        charter = f"v{s.charter.version} ({installed})"
    v = s.vehicle
    lines = [
        f"LÄGE seq={s.state_seq}  uppdrag={s.mission_status}  läge={mode}  "
        f"AI-styrning={'på' if s.ai_enabled else 'av'}  ram={charter}  spärr={latch}"
    ]
    if v.airborne:
        alt = _num(v.alt_rel_home_m or 0)
        speed = _num(v.groundspeed_ms or 0, 1)
        lines.append(
            f"Drönare: i luften, {v.executor_state or '-'}, {v.flight_mode}, {alt} m över hem, "
            f"{speed} m/s"
        )
    else:
        armed = "armerad" if v.armed else "desarmerad"
        conn = "ansluten" if v.connected else "inte ansluten"
        lines.append(f"Drönare: på marken, {armed}, {conn}")
    lines.append(f"Position: {_where(s, v.lat, v.lon)}")
    if v.battery_pct is not None:
        reserve = (
            f"  (till hemgräns {_num(s.limits.battery_rtl_pct)} %: ca "
            f"{round(v.battery_to_reserve_s / 60)} min)"
            if v.battery_to_reserve_s is not None
            else ""
        )
        lines.append(f"Batteri: {_num(v.battery_pct)} %{reserve}")
    pilot = f"{_num(v.dist_to_pilot_m)} m bort" if v.dist_to_pilot_m is not None else "okänt"
    lines.append(
        f"Pilot: {pilot}  |  Länk {'OK' if v.link_ok else 'NERE'}  |  "
        f"GNSS {'OK' if v.gnss_ok else 'svag'}"
    )
    ac = s.active_command
    if ac is not None:
        target = ac.params.get("sector_id") or ""
        lines.append(f"Aktiv: {ac.skill} {target}, status {ac.status}".replace("  ", " "))
    else:
        lines.append("Aktiv: inget")
    if s.sectors:
        parts = []
        for sec in s.sectors:
            pct = round(100 * float(sec.get("progress") or 0))
            parts.append(f"{sec.get('sector_id')} {pct} %")
        lines.append("Sektorer: " + ", ".join(parts))
    elif s.charter is not None and s.charter.sectors:
        lines.append("Sektorer: " + ", ".join(f"{x.sector_id} 0 %" for x in s.charter.sectors))
    if s.pins:
        pins = [
            f"{p.id} {_quote(p.note)} {_where(s, p.lat, p.lon)}".replace("  ", " ") for p in s.pins
        ]
        lines.append("Kartnålar: " + "; ".join(pins))
    areas = [a for a in s.areas if a.kind == "area"]
    if areas:
        parts = [f"{_quote(a.label)} {_num((a.area_m2 or 0) / 10_000, 1)} ha" for a in areas]
        lines.append("Ritade områden: " + "; ".join(parts))
    waiting = [a.summary_sv for a in s.pending_approvals if getattr(a, "summary_sv", None)]
    if s.pending_proposal is not None:
        waiting.append(f"ramförslag v{s.pending_proposal.version}")
    lines.append("Väntar på klick: " + (", ".join(waiting) if waiting else "inget"))
    until = s.takeoff_clearance_valid_until
    clear = "bekräftad" if until is not None and until > now else "ej bekräftad"
    lines.append(f"Fritt runt drönaren: {clear}")
    alerts = [str(a.get("code")) for a in s.alerts if a.get("code")]
    lines.append("Varningar: " + (", ".join(alerts) if alerts else "inga"))
    lim = current_limits(s)
    limit_line = (
        f"Gränser: {_num(lim.alt_min_agl_m)}-{_num(lim.alt_max_agl_m)} m över hem, förval "
        f"{_num(lim.default_alt_agl_m)} m, max {_num(lim.max_speed_ms)} m/s, max "
        f"{_num(lim.max_range_from_pilot_m)} m från pilot, hem vid {_num(lim.battery_rtl_pct)} %"
    )
    if (
        lim.alt_max_agl_m != lim.mission_alt_max_agl_m
        or lim.max_speed_ms != lim.mission_max_speed_ms
    ):
        limit_line += (
            f"  |  Högsta i uppdraget: {_num(lim.mission_alt_max_agl_m)} m, "
            f"{_num(lim.mission_max_speed_ms)} m/s"
        )
    lines.append(limit_line)
    return "\n".join(lines)


def _charter_text(s: Situation) -> str:
    c = s.charter
    if c is None:
        return "INGEN GODKÄND RAM."
    secs = ", ".join(x.sector_id for x in c.sectors) or "inga"
    return (
        f"v{c.version}, sektorer {secs}, höjd {_num(c.alt_min_agl_m)}-{_num(c.alt_max_agl_m)} m "
        f"(sökhöjd {_num(c.default_alt_agl_m)} m), max {_num(c.max_speed_ms)} m/s, "
        f"tillåtna färdigheter: {', '.join(c.allowed_skills)}"
    )


def _chat_line(m: ChatMessage) -> str:
    who = {"operator": "OPERATÖR", "brain": "AI", "system": "SYSTEM"}[m.role]
    clock = m.created_at.strftime("%H:%M")
    return f"{who} {clock}: {m.text}"


class ContextBuilder:
    def __init__(
        self,
        reg: SkillRegistry,
        *,
        takeoff_policy: str,
        prompt_budget_tokens: int,
        count_tokens: Callable[[str], int] | None = None,
    ) -> None:
        latch_tools = ", ".join(k.name for k in reg if k.allowed_under_latch) or "nothing"
        examples = yaml.safe_load(prompt_text("examples_sv_v1.yaml")) or []
        rendered = []
        for ex in examples:
            lines = []
            if ex.get("context"):
                lines.append(f"({ex['context']})")
            lines.append(f"OPERATÖR: {ex['operator']}")
            if "tool" in ex:
                lines.append(f"→ {ex['tool']['name']} {ex['tool']['arguments']}")
            else:
                lines.append(f"AI: {ex['answer']}")
            rendered.append("\n".join(lines))
        self.system = (
            prompt_text("system_v1.md")
            .replace("{takeoff_rule}", takeoff_rule(takeoff_policy))
            .replace("{latch_tools}", latch_tools)
            + "\n\nEXEMPEL\n"
            + "\n\n".join(rendered)
        )
        self.budget = prompt_budget_tokens
        self._count = count_tokens

    def count_tokens(self, text: str) -> int:
        if self._count is not None:
            return self._count(text)
        return math.ceil(len(text) / 3.5)

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for m in messages:
            total += self.count_tokens(str(m.get("content") or ""))
            for c in m.get("tool_calls") or []:
                total += self.count_tokens(str(c))
        return total

    def turn_block(self, turn: TurnInput, chat_rows: int | None = None, memory: bool = True) -> str:
        s, mem = turn.situation, turn.memory
        tail = turn.chat_tail if chat_rows is None else turn.chat_tail[-chat_rows:]
        parts = [f"RAM: {_charter_text(s)}"]
        if memory and mem.summary_sv:
            parts.append(f"MINNE: {mem.summary_sv}")
        parts.append(f"PLAN: {mem.plan_note or 'ingen'}")
        parts.append(situation_text(s, turn.now))
        parts.append("CHATT:")
        parts.extend(_chat_line(m) for m in tail)
        if turn.backlog:
            parts.append("TIDIGARE (skrivet medan AI:n var nere, utför inte):")
            parts.extend(_chat_line(m) for m in turn.backlog)
        parts.append(f"HÄNDELSE: {self.event_summary(turn)}")
        if turn.reject is not None:
            r = turn.reject
            parts.append(f"GRINDSVAR: {', '.join(r.reason_codes)}; {r.message_sv}; {r.details}")
        if turn.charter_errors:
            errs = "; ".join(
                f"{e.code} {e.detail_sv}" + (f" (gräns {e.limit:g})" if e.limit is not None else "")
                for e in turn.charter_errors
            )
            parts.append(f"GRINDSVAR (ramförslag): {errs}")
        operator_lines = [m for m in tail if m.role == "operator"]
        if turn.trigger == "operator_message" and operator_lines:
            hint = interpret(operator_lines[-1].text)
            if hint:
                parts.append(f"Tolkat: {hint}")
        return "\n".join(parts)

    def event_summary(self, turn: TurnInput) -> str:
        if not turn.events:
            return turn.trigger
        bits = []
        for e in turn.events[-4:]:
            body = e.get("body") or {}
            kind = str(e.get("type", turn.trigger)).removeprefix("event.")
            detail = ", ".join(
                f"{k}={body[k]}"
                for k in ("skill", "status", "action", "via", "code", "kind", "result", "version")
                if body.get(k) is not None
            )
            bits.append(f"{kind} ({detail})" if detail else kind)
        return f"{turn.trigger}: " + "; ".join(bits)

    def build(self, turn: TurnInput) -> list[dict[str, Any]]:
        """System + turn block, cut to the prompt budget (chat first, then memory)."""
        system = {"role": "system", "content": self.system}
        for rows, memory in ((None, True), (8, True), (4, True), (4, False)):
            user = {"role": "user", "content": self.turn_block(turn, rows, memory)}
            if self.count_messages([system, user]) <= self.budget:
                return [system, user]
        return [system, user]
