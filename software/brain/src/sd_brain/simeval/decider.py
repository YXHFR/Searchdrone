"""The decision maker for simulator flights: the cloud model in exams, a script otherwise.

This is the interface the simulator farm calls. ``DecisionMaker`` is a
``CommanderBackend``: plug it into ``CommanderLoop`` (or call ``decide`` per turn) and it
answers each turn with text or one order, exactly like ``sd-brain run``. It never flies:
the order goes through the gate, and the executor and ArduPilot fly it.

A flight starts with the cloud model when a key is set and the day's cost cap has room
(``make_decision_maker``). If the cap is reached or the cloud stops answering during the
flight, the script takes over for the rest of it and the flight no longer counts as an
exam (``exam`` false); it is still good image data for the farm.
"""

from __future__ import annotations

from typing import Literal

from sd_brain.backends.base import ToolExecutor
from sd_brain.backends.llm import LlmCommander
from sd_brain.config import BehaviourConfig
from sd_brain.context import ContextBuilder
from sd_brain.runtime.base import Sampling
from sd_brain.simeval.budget import MeteredRuntime
from sd_brain.state_view import motion_block
from sd_brain.turn import AiAction, OutMessage, TurnDiagnostics, TurnInput, TurnOutput
from sd_schemas.skills import FORBIDDEN_SKILL_NAMES, SKILL_REGISTRY, SkillRegistry

__all__ = ["DecisionMaker", "SearchScript", "make_decision_maker"]

DeciderKind = Literal["cloud", "script"]
SwitchReason = Literal["no_key", "budget", "cloud_error"]


class SearchScript:
    """The fixed search script: sectors in order, mark every sighting, then fly home."""

    name = "scripted"

    def __init__(self) -> None:
        self.marked: set[str] = set()

    async def decide(self, turn: TurnInput, tools: ToolExecutor) -> TurnOutput:
        s = turn.situation
        out = TurnOutput(diagnostics=TurnDiagnostics(backend="scripted"))
        offered = {t.name for t in turn.tools}
        if motion_block(s) is not None or not s.vehicle.airborne:
            return out
        for pin in s.pins:
            if pin.id not in self.marked and "mark_location" in offered:
                self.marked.add(pin.id)
                out.action = AiAction(
                    skill="mark_location",
                    params={"pin": pin.id, "label": "möjlig person"},
                    rationale_sv=f"Detektorn såg något vid {pin.id}.",
                )
                out.messages = [OutMessage(text=f"Markerar {pin.id}, möjlig person.")]
                return out
        if s.active_command is not None:
            return out
        low = (s.vehicle.battery_pct or 100) <= s.limits.battery_rtl_pct + 10
        left = [x for x in s.sectors if x.get("status") != "done"]
        if left and not low and "survey_sector" in offered:
            sector = str(left[0]["sector_id"])
            out.action = AiAction(
                skill="survey_sector", params={"sector": sector}, rationale_sv="Nästa sektor."
            )
            out.messages = [OutMessage(text=f"Söker sektor {sector}.")]
        elif "return_home" in offered:
            out.action = AiAction(skill="return_home", rationale_sv="Sökningen är klar.")
            out.messages = [OutMessage(text="Sökningen är klar. Flyger hem och landar.")]
        return out


class DecisionMaker:
    """The cloud model while it can, the script after; see the module text."""

    def __init__(
        self,
        cloud: LlmCommander | None,
        meter: MeteredRuntime | None,
        script: SearchScript | None = None,
        *,
        reg: SkillRegistry = SKILL_REGISTRY,
        no_cloud_reason: SwitchReason = "no_key",
    ) -> None:
        self.cloud = cloud
        self.meter = meter
        self.script = script or SearchScript()
        self.reg = reg
        self.kind: DeciderKind = "cloud" if cloud is not None else "script"
        self.started_as: DeciderKind = self.kind
        self.switch_reason: SwitchReason | None = None if cloud is not None else no_cloud_reason
        self.forbidden_calls: list[str] = []
        self.name = "cloud" if self.kind == "cloud" else "scripted"

    @property
    def exam(self) -> bool:
        """The cloud model decided the whole flight."""
        return self.started_as == "cloud" and self.kind == "cloud"

    def _switch(self, reason: SwitchReason) -> None:
        self.kind, self.switch_reason, self.name = "script", reason, "scripted"

    def _check_calls(self, turn: TurnInput, out: TurnOutput) -> None:
        offered = {t.name for t in turn.tools}
        for call in out.diagnostics.tool_calls:
            name = str(call.get("name"))
            if name in FORBIDDEN_SKILL_NAMES or (name in self.reg and name not in offered):
                self.forbidden_calls.append(name)

    async def decide(self, turn: TurnInput, tools: ToolExecutor) -> TurnOutput:
        if self.kind == "cloud" and self.cloud is not None and self.meter is not None:
            if not self.meter.can_afford():
                self._switch("budget")
            else:
                out = await self.cloud.decide(turn, tools)
                self._check_calls(turn, out)
                failed = out.diagnostics.error in ("timeout", "http_error")
                if not failed or out.has_action() or out.messages:
                    return out
                self._switch("budget" if not self.meter.can_afford() else "cloud_error")
        return await self.script.decide(turn, tools)


def make_decision_maker(
    meter: MeteredRuntime | None,
    behaviour: BehaviourConfig,
    *,
    reg: SkillRegistry = SKILL_REGISTRY,
    sampling: Sampling | None = None,
) -> DecisionMaker:
    """A decision maker for one flight. ``meter`` None (no key) gives the script."""
    if meter is None:
        return DecisionMaker(None, None, reg=reg)
    if not meter.can_afford():
        return DecisionMaker(None, None, reg=reg, no_cloud_reason="budget")
    ctx = ContextBuilder(reg, takeoff_policy=behaviour.takeoff_policy, prompt_budget_tokens=6000)
    smp = sampling or Sampling(temperature=0.2, top_p=0.8, max_tokens=512)
    return DecisionMaker(LlmCommander(meter, ctx, reg, behaviour, smp), meter, reg=reg)
