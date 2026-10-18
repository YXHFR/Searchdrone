"""``GrammarBackend``: the fixed Swedish command mode used while the cloud model does not answer.

Pure code, no AI. It also answers when there is no network or no cloud key.
"""

from __future__ import annotations

import time

from sd_brain.backends.base import ToolExecutor
from sd_brain.context import situation_text
from sd_brain.state_view import motion_block
from sd_brain.turn import AiAction, OutMessage, TurnDiagnostics, TurnInput, TurnOutput
from sd_schemas.sv_commands import classify_override, is_resume_request, parse_command

__all__ = ["HELP_TEXT", "GrammarBackend"]

HELP_TEXT = (
    "Kommandon: starta [på N meter], flyg till P1 [på N meter], flyg till <lat> <lon>, "
    "sök sektor A [på N meter], håll position, fortsätt, status. STOPP och FLYG HEM: knapparna."
)
NOT_UNDERSTOOD = "Förstod inte. Skriv 'hjälp'."
RESUME_TEXT = (
    "Tryck Fortsätt i panelen när det är säkert. Att skriva i chatten släpper inte stoppet."
)


class GrammarBackend:
    name = "grammar"

    async def decide(self, turn: TurnInput, tools: ToolExecutor) -> TurnOutput:
        start = time.monotonic()
        out = TurnOutput(diagnostics=TurnDiagnostics(backend="grammar"))
        if turn.trigger != "operator_message":
            return out
        lines = [m for m in turn.chat_tail if m.role == "operator"]
        if not lines:
            return out
        text = lines[-1].text
        s = turn.situation
        if classify_override(text) is not None:
            return out  # the ground station already handled STOP / FLY HOME
        block = motion_block(s)
        if block is not None and is_resume_request(text):
            out.messages = [
                OutMessage(text=RESUME_TEXT, kind="proposal", refs={"proposed_action": "resume"})
            ]
            return out
        cmd = parse_command(
            text,
            [p.id for p in s.pins],
            [str(x.get("sector_id")) for x in s.sectors]
            + ([x.sector_id for x in s.charter.sectors] if s.charter else []),
        )
        if cmd is None:
            out.messages = [OutMessage(text=NOT_UNDERSTOOD)]
        elif cmd.kind == "help":
            out.messages = [OutMessage(text=HELP_TEXT)]
        elif cmd.kind == "status":
            short = "\n".join(situation_text(s, turn.now).splitlines()[:4])
            out.messages = [OutMessage(text=short[:600], kind="status")]
        elif cmd.skill is not None:
            offered = turn.tool(cmd.skill)
            if offered is None:
                out.messages = [OutMessage(text=f"Det går inte just nu ({cmd.skill}).")]
            else:
                out.action = AiAction(
                    skill=cmd.skill, params=cmd.params, rationale_sv="Kommando från operatören."
                )
        out.diagnostics.latency_ms = int((time.monotonic() - start) * 1000)
        return out
