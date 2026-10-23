"""``CommanderLoop``: events become turns, turns become chat lines and orders.

At most one turn runs at a time. Events that arrive during a turn are merged into the
next one, which takes the trigger with the highest priority. Every order goes to the
ground station and through the safety gate; the AI says what it does only after the
gate's answer. Some steps are rules without a model: the answer to "fortsätt" during a
stop (press Resume), the start after "clear around the drone" when the operator asked
for a start, the quick stop check on every chat line, and the finish backstop: airborne
with nothing to do and nothing said or ordered for ``behaviour.finish_idle_s``, the
system says so and orders ``return_home`` through the gate like any AI order.

The operator's latest request (``UPPDRAG``) is kept in the mission memory from its turn
until the drone has flown and landed, or a newer request replaces it, so it outlives the
chat tail. A charter is proposed only in a turn an operator line started, or in the
retry of a refused proposal.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import hashlib
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ValidationError

from sd_brain.backends.base import BackendCrashError, CommanderBackend
from sd_brain.charter_draft import (
    CharterDraft,
    CharterError,
    CharterProposalOutcome,
    DraftError,
    to_proposal,
)
from sd_brain.clock import utc_now
from sd_brain.config import BrainConfig
from sd_brain.constants import (
    BRAIN_TURN_DEADLINE_S,
    CHAT_TAIL,
    FRESH_SNAPSHOT_WAIT_S,
    MAX_ACTIONS_PER_MINUTE,
    MAX_CHARTER_ATTEMPTS,
    MESSAGE_MAX_CHARS,
    MIN_ACTION_INTERVAL_S,
    SYNC_REJECT_MEMORY_S,
    TAKEOFF_REQUEST_TTL_S,
    TOOL_RESULT_MAX_CHARS,
)
from sd_brain.context import situation_text
from sd_brain.decide.service import DecisionService, EngineOutcome
from sd_brain.decide.stop_check import StopCheckResult, stop_check
from sd_brain.ground_client import GroundApi, GroundError
from sd_brain.health import HealthState
from sd_brain.memory import (
    PendingRequest,
    empty_memory,
    is_acknowledgement,
    mission_request,
    next_request,
    to_update,
    with_changes,
)
from sd_brain.profile_draft import profile_proposal
from sd_brain.runtime.fallback import EngineName
from sd_brain.state_view import current_limits, fresh_request, motion_block, takeoff_ok
from sd_brain.sv import height_m
from sd_brain.toolcall_parse import truncate
from sd_brain.tools import PROFILE_TOOL, PROPOSAL_TOOL, AdaptError, adapt, exposed_tools
from sd_brain.turn import (
    TRIGGER_PRIORITY,
    AiAction,
    OutMessage,
    Trigger,
    TurnDiagnostics,
    TurnError,
    TurnInput,
    TurnOutput,
)
from sd_schemas.api import ChatMessage
from sd_schemas.base import PARAMS_MAX_BYTES, new_id
from sd_schemas.charter import CharterValidationError
from sd_schemas.commander import (
    BrainMessage,
    DecisionLog,
    MissionMemory,
    Situation,
    SkillRequest,
    SkillSubmitResult,
)
from sd_schemas.decide import ChatIntentReport, DecideRequest
from sd_schemas.gate import GateDecision, StateRef
from sd_schemas.mission import MissionState
from sd_schemas.skills import SkillRegistry
from sd_schemas.sv_commands import is_resume_request, is_takeoff_request

__all__ = ["CommanderLoop", "LoopClock", "TurnRecord"]

log = logging.getLogger("sd_brain.loop")

DEADLINE_TEXT = (
    "Jag hann inte svara. Försök igen, eller skriv ett kort kommando "
    "(t.ex. 'håll position', 'flyg hem')."
)
COMMAND_MODE_TEXT = "AI:n svarar inte. Kommandoläge: skriv 'hjälp' eller använd knapparna."
RESUME_TEXT = (
    "Tryck Fortsätt i panelen när det är säkert. Att skriva i chatten släpper inte stoppet."
)
BATTERY_TEXT = (
    "Drönaren flyger hem på grund av batteriet. Det går inte att fortsätta under "
    "batterihemflygningen."
)
CLEARED_NO_REQUEST = "Fritt runt är bekräftat. Säg till när jag ska starta."
CHARTER_WITH_REQUEST = "Ramen gäller. Jag startar direkt när du trycker Fritt runt drönaren."
RATE_TEXT = "Jag skickade för många order. Jag pausar AI:n; skriv 'hjälp' för kommandon."
# The finish backstop's line, posted before its return_home order.
# It is posted before the gate answers, so it says what is asked for, not that it flies.
FINISH_TEXT = (
    "Inget har hänt på {minutes} min och inget väntar på dig, så jag ber om att få flyga "
    "hem. Tryck STOPP om drönaren ska stanna."
)
FINISH_REASON = "Uppdraget är klart och inget har hänt på en stund."
FINISH_DONE_TEXT = "Jag flyger hem."
# A finish check (not a model turn) at most this often while one is due.
FINISH_CHECK_S = 5.0
# Operator input and decisions that restart the backstop's idle time (a target find
# counts too, see handle_event; a charter decision when it wakes a turn).
ACTIVITY_EVENTS = frozenset(
    {
        "event.operator_message",
        "event.operator_override",
        "event.vehicle_action",  # e.g. the hand-back after a manual takeover
        "event.skill",
        "event.approval",
        "event.approval_requested",
        "event.takeoff_clearance",
        "event.pin_set",
        "event.area_drawn",
        "event.path_drawn",
    }
)
# A charter proposal the ground took, when the model wrote nothing with it.
PROPOSED_TEXT = "Jag har lagt fram ett ramförslag. Tryck Godkänn på kortet om det stämmer."
# The brain gives up a charter after MAX_CHARTER_ATTEMPTS refused proposals. The line is a
# question to the pilot (the panel offers Svara) with refs.charter_failed and the codes:
# "Hur vill du gå vidare?" asks for a new instruction, not for an approval.
CHARTER_FAILED_TEXT = "Ramförslaget gick inte igenom efter {n} försök: "
BACK_TEXT = "AI:n är tillbaka."
# The ground station could not be reached with the stop check's report: ask for the
# button and stop where it is (a model's reading never flies home, owner decision).
STOP_FALLBACK = {
    "stop": "Jag tolkar det som att du vill stoppa, men markstationen svarar inte. "
    "Tryck STOPP i panelen om drönaren inte står still.",
    "return_home": "Jag tolkar det som att du vill flyga hem, men markstationen svarar "
    "inte. Jag ber drönaren stanna; tryck FLYG HEM i panelen.",
}

# Skill events that wake a turn (done, failed, …); "running" etc. only update the picture.
SKILL_WAKE = frozenset(
    {
        "done",
        "failed",
        "rejected_onboard",
        "superseded",
        "cancelled",
        "expired",
        "rejected_by_gate",
        "rejected_by_operator",
        "approval_expired",
    }
)


class LoopClock:
    """Real time. Tests pass a clock whose ``sleep`` moves time on at once."""

    def monotonic(self) -> float:
        return time.monotonic()

    def utc_now(self) -> dt.datetime:
        return utc_now()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))


@dataclass
class TurnRecord:
    """What happened in one turn (for evaluations and tests)."""

    turn_id: str
    trigger: Trigger
    backend: str
    output: TurnOutput | None
    texts: list[str] = field(default_factory=list[str])
    messages: list[OutMessage] = field(default_factory=list[OutMessage])
    request: SkillRequest | None = None
    ai_action: AiAction | None = None
    result: SkillSubmitResult | None = None
    proposal: CharterProposalOutcome | None = None
    proposal_draft: Any = None
    error: str | None = None
    retry_of: str | None = None
    attempt: int | None = None  # the charter attempt this turn's proposal was
    refused_draft: dict[str, Any] | None = None  # a draft the brain refused in the turn
    tools: list[str] = field(default_factory=list[str])


@dataclass
class _Pending:
    trigger: Trigger
    event: dict[str, Any]


class CommanderLoop:
    def __init__(
        self,
        ground: GroundApi,
        backend: CommanderBackend,
        fallback: CommanderBackend,
        reg: SkillRegistry,
        cfg: BrainConfig,
        ai_config_hash: str,
        *,
        health: HealthState | None = None,
        decide: DecisionService | None = None,
        clock: LoopClock | None = None,
        engine: Callable[[], EngineName | None] | None = None,
    ) -> None:
        self.ground = ground
        self.backend = backend
        self.fallback = fallback
        self.reg = reg
        self.cfg = cfg
        self.ai_hash = ai_config_hash
        self.clock = clock or LoopClock()
        self.health = health or HealthState(self.clock.monotonic)
        self.decide = decide
        self._engine = engine or (lambda: None)
        self.mission_id: str | None = None
        self.ready_since: dt.datetime = self.clock.utc_now()
        self._queue: list[_Pending] = []
        self._followups: list[tuple[Trigger, dict[str, Any]]] = []
        self._backlog: list[ChatMessage] = []
        self._turn_task: asyncio.Task[None] | None = None
        self._turn_events: list[dict[str, Any]] = []  # what the running turn answers
        self._side_tasks: set[asyncio.Task[Any]] = set()
        self._worker: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._sync_rejects: dict[str, float] = {}
        self._action_times: list[float] = []
        self._rate_rejects = 0
        self._last_turn_at = self.clock.monotonic()
        self._busy_since: float | None = None
        self._last_turn_ms: int | None = None
        self._alert_seen: dict[str, float] = {}
        self._own_approvals: set[str] = set()
        # the AI's own action cards: only these make it wait (its own charter proposals do not)
        self._own_action_approvals: set[str] = set()
        self._backlog_ids: set[str] = set()
        # Refused charter proposals since the last approved/rejected charter (or mission).
        self._charter_attempts = 0
        # A charter (or gate) retry that an operator line overtook: merged into that turn.
        self._carry: dict[str, Any] | None = None
        self.last_stop_check: StopCheckResult | None = None
        self.records: list[TurnRecord] = []
        self.turn_deadline_s = BRAIN_TURN_DEADLINE_S
        # The last operator input, AI order, skill end or card decision (the finish
        # backstop counts its idle time from here), and the last finish check.
        self._activity_at = self.clock.monotonic()
        self._finish_checked_at = -1e9
        # The charter decisions that already woke a turn: (charter_id, version, status).
        self._charter_woken: set[tuple[str, Any, Any]] = set()

    # --- status for the heartbeat -----------------------------------------------------

    def heartbeat(self) -> dict[str, Any]:
        engine = self._current_engine()
        status = (
            "degraded"
            if self.health.degraded or engine == "command"
            else ("thinking" if self._busy_since is not None else "idle")
        )
        body: dict[str, Any] = {"status": status, "backend": self._active_backend().name}
        if engine is not None:
            body["engine"] = engine
        if self._busy_since is not None:
            body["busy_since"] = round(self.clock.monotonic() - self._busy_since, 1)
        if self._last_turn_ms is not None:
            body["last_turn_ms"] = self._last_turn_ms
        return body

    def _current_engine(self) -> EngineName | None:
        """``cloud`` or ``command`` for a backend with a model; ``None`` for the others."""
        engine = self._engine()
        if engine is not None and self.health.degraded:
            return "command"
        return engine

    def _active_backend(self) -> CommanderBackend:
        if self.health.degraded or self._engine() == "command":
            return self.fallback
        return self.backend

    # --- mission ------------------------------------------------------------------------

    async def set_mission(self, mission_id: str | None) -> None:
        if mission_id != self.mission_id:
            self.mission_id = mission_id
            self._queue.clear()
            self._followups.clear()
            self._backlog.clear()
            self._charter_attempts = 0
            self._carry = None
            self._activity_at = self.clock.monotonic()

    async def connected(self, reconnect: bool) -> None:
        """The stream is up. After a reconnect the AI says it is back."""
        self.ready_since = self.clock.utc_now()
        if reconnect and self.mission_id is not None:
            await self._say(OutMessage(text=BACK_TEXT, kind="status"))
            self._followups.append(("recovered", {}))
            self._wake.set()

    async def health_recovered(self) -> None:
        self.health.recovered()
        await self.connected(True)

    # --- events -------------------------------------------------------------------------

    def _author_role(self, body: dict[str, Any]) -> str | None:
        role = body.get("author_role")
        return None if role is None else str(role)

    def handle_event(self, evt: dict[str, Any]) -> None:
        """Turn a commander event into queued work. Never blocks."""
        kind = evt.get("type")
        body: dict[str, Any] = evt.get("body") or {}
        other = evt.get("mission_id") and self.mission_id and evt["mission_id"] != self.mission_id
        if other and not (kind == "event.mission" and body.get("created")):
            return
        if kind in ACTIVITY_EVENTS or (kind == "event.detection" and body.get("kind") == "target"):
            self._activity_at = self.clock.monotonic()  # not idle: the backstop waits
        trigger: Trigger | None = None
        if kind == "event.operator_message":
            if body.get("handled_by") == "ground":
                trigger = "operator_override"
            elif body.get("delivery") in ("queued", "summarized") or self._before_ready(body):
                self._schedule_recovered(body)
                return
            else:
                trigger = "operator_message"
                self._start_stop_check(evt)
        elif kind == "event.skill":
            cid = str(body.get("command_id"))
            if body.get("status") == "rejected_by_gate" and cid in self._sync_rejects:
                return  # already handled by the retry turn
            if body.get("status") in SKILL_WAKE:
                trigger = "skill_event"
        elif kind == "event.approval":
            if body.get("result") in ("rejected", "expired", "superseded") and (
                str(body.get("approval_id")) in self._own_approvals
            ):
                trigger = "approval_event"
        elif kind == "event.charter":
            # An approved charter wakes the AI once the drone has it (install_status
            # "installed"); the earlier updates of the same approval ("none", "pending")
            # would show a charter not yet in force.
            # A resend of the same install (after a reconnect) is no news: it wakes the AI
            # once per charter version.
            key = (str(body.get("charter_id")), body.get("version"), body.get("status"))
            if (
                body.get("status") == "rejected"
                or body.get("install_status") == "failed"
                or (body.get("status") == "approved" and body.get("install_status") == "installed")
            ) and (body.get("install_status") == "failed" or key not in self._charter_woken):
                self._charter_woken.add(key)
                self._activity_at = self.clock.monotonic()
                trigger = "charter_event"
            if body.get("status") in ("approved", "rejected"):
                # The pilot decided: a new request starts over, and an older refusal's
                # errors (carried or queued) no longer apply.
                self._charter_attempts = 0
                self._carry = None
                self._followups = [(t, x) for t, x in self._followups if "charter_errors" not in x]
        elif kind == "event.takeoff_clearance":
            if body.get("status") in ("active", "confirmed", None):
                trigger = "takeoff_clearance"
        elif kind == "event.operator_override":
            trigger = "operator_override"
        elif kind == "event.alert":
            level = body.get("level")
            if level in ("warn", "warning", "critical"):
                code = str(body.get("code"))
                now = self.clock.monotonic()
                if now - self._alert_seen.get(code, -1e9) >= 60:
                    self._alert_seen[code] = now
                    trigger = "alert_critical" if level == "critical" else "alert_warning"
        elif kind == "event.detection" and body.get("kind") == "target":
            # A new find of a search target wakes the AI; context (roads, buildings) only
            # tells it what it flies over. The AI may look closer and ask the pilot; it
            # never confirms a find.
            trigger = "detection"
        elif kind == "event.mission" and body.get("created"):
            trigger = "mission_created"
            mid = evt.get("mission_id") or body.get("mission_id")
            if mid:
                if str(mid) != self.mission_id:
                    self._charter_attempts = 0
                    self._carry = None
                self.mission_id = str(mid)
        if trigger is None:
            return
        if trigger in ("operator_override", "operator_message") and self._turn_task is not None:
            if trigger == "operator_override":
                self._turn_task.cancel()  # the human decision wins; the turn is dropped
            self._drop_followups(keep_retry=trigger == "operator_message")
        self._queue.append(_Pending(trigger, evt))
        self._wake.set()

    def _drop_followups(self, *, keep_retry: bool) -> None:
        """An operator line overtakes the queued follow-ups. A retry's errors (a refused
        charter or order) are kept for that line's turn, so the AI still knows why its
        last try failed; ``keep_retry = False`` (an override) drops them too."""
        if keep_retry:
            for trigger, extra in self._followups:
                if trigger == "gate_reject_retry":
                    self._carry = dict(extra)
        else:
            self._carry = None
            self._charter_attempts = 0  # the operator's decision ends the request
        self._followups.clear()

    def _before_ready(self, body: dict[str, Any]) -> bool:
        created = body.get("created_at")
        if not isinstance(created, str):
            return False
        try:
            when = dt.datetime.fromisoformat(created.replace("Z", "+00:00"))
        except ValueError:
            return False
        return when < self.ready_since

    def _schedule_recovered(self, body: dict[str, Any]) -> None:
        self._backlog_ids.add(str(body.get("message_id")))
        if not any(t == "recovered" for t, _ in self._followups):
            self._followups.append(("recovered", {"type": "recovered", "body": {}}))
        self._wake.set()

    def idle_tick(self, *, force: bool = False) -> bool:
        """Wake a turn when airborne, idle and nothing waits (checked by the next turn).
        Once the finish backstop is due, a finish check (no model) runs instead, at most
        every ``FINISH_CHECK_S``."""
        if self._turn_task is not None or self._queue or self._followups:
            return False
        now = self.clock.monotonic()
        if self._finish_due() and now - self._finish_checked_at >= FINISH_CHECK_S:
            self._finish_checked_at = now
            body = {"finish": True}
            self._queue.append(_Pending("idle_tick", {"type": "idle_tick", "body": body}))
            self._wake.set()
            return True
        if not force and now - self._last_turn_at < self.cfg.behaviour.idle_airborne_tick_s:
            return False
        self._queue.append(_Pending("idle_tick", {"type": "idle_tick", "body": {}}))
        self._wake.set()
        return True

    def _finish_due(self) -> bool:
        return self.clock.monotonic() - self._activity_at >= self.cfg.behaviour.finish_idle_s

    # --- running ------------------------------------------------------------------------

    async def run_forever(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            await self.drain_queue()

    async def drain_queue(self) -> None:
        """Run turns until nothing waits."""
        while self.mission_id is not None and (self._queue or self._followups):
            if self._queue and any(
                p.trigger in ("operator_override", "operator_message") for p in self._queue
            ):
                override = any(p.trigger == "operator_override" for p in self._queue)
                self._drop_followups(keep_retry=not override)
            if self._followups and not self._queue:
                trigger, extra = self._followups.pop(0)
                events = [extra] if extra.get("type") else []
                self._turn_events = events
                task = asyncio.create_task(self._turn(trigger, events, extra))
            else:
                pending, self._queue = self._queue, []
                order = {t: i for i, t in enumerate(TRIGGER_PRIORITY)}
                trigger: Trigger = min((p.trigger for p in pending), key=lambda t: order.get(t, 99))
                self._turn_events = [p.event for p in pending]
                task = asyncio.create_task(self._turn(trigger, self._turn_events, {}))
            self._turn_task = task
            try:
                await task
            except asyncio.CancelledError:
                me = asyncio.current_task()
                if not task.cancelled() or (me is not None and me.cancelling()):
                    raise  # the loop itself is being stopped (shutdown), not just the turn
                log.info("turen avbröts av operatörens beslut")
            except GroundError as exc:
                # The ground station did not answer this turn (a timeout, a restart): the
                # turn is lost, not the loop. Without this the turn loop ended here and the
                # AI never woke again (no idle tick, no resume after a hand-back). A
                # refused token still ends it: that is no passing trouble.
                if exc.status == 401:
                    raise
                log.warning("turen (%s) avbröts, markstationen svarade inte: %s", trigger, exc)
            finally:
                self._turn_task = None
                self._turn_events = []
                self._busy_since = None

    async def settle(self) -> None:
        """Wait for queued turns and side tasks (stop checks, logs). For tests."""
        for _ in range(50):
            await self.drain_queue()
            if self._side_tasks:
                await asyncio.gather(*list(self._side_tasks), return_exceptions=True)
                continue
            if self.decide is not None:
                await self.decide.drain()
            if not (self._queue or self._followups or self._side_tasks):
                return

    # --- the turn -----------------------------------------------------------------------

    async def _turn(
        self, trigger: Trigger, events: list[dict[str, Any]], extra: dict[str, Any]
    ) -> None:
        mid = self.mission_id
        if mid is None:
            return
        # A finish check is no model turn: it leaves the idle tick's own timing alone.
        finish = trigger == "idle_tick" and any((e.get("body") or {}).get("finish") for e in events)
        self._busy_since = self.clock.monotonic()
        if not finish:
            self._last_turn_at = self._busy_since
        elif not self._finish_ok(await self.ground.situation(mid)):
            # Not now (on the ground, held, a card waits, ...): the idle time starts over,
            # so the backstop waits a whole ``finish_idle_s`` after this and the check runs
            # at most once per that time instead of every few seconds.
            self._activity_at = self.clock.monotonic()
            return
        now = self.clock.utc_now()
        # The chat first, then the situation: the picture is at least as new as the chat,
        # so a pin the operator placed before writing is in it.
        chat = await self.ground.chat(mid, 0, 200)
        s = await self.ground.situation(mid)
        if trigger in ("skill_event", "operator_override"):
            s = await self._fresh_situation(mid, s, trigger, events)
        try:
            mem = await self.ground.memory(mid)
        except GroundError:
            mem = empty_memory(s.mission_id)
        backlog = [m for m in chat if str(m.id) in self._backlog_ids]
        tail = [m for m in chat if str(m.id) not in self._backlog_ids][-CHAT_TAIL:]
        if trigger == "recovered":
            self._backlog_ids = set()
        op_msg = self._operator_message(events, chat) if trigger == "operator_message" else None
        pilot = op_msg is None or op_msg.author_role == "pilot"
        if trigger == "operator_message" and pilot and self._carry is not None:
            # The retry this line overtook: its errors and attempt go with this turn (an
            # observer's line cannot propose, so the carry waits for the pilot's next).
            extra = {**self._carry, "in_reply_to": None, **extra}
            self._carry = None
        if trigger == "idle_tick" and not finish and not self._idle_ok(s):
            return
        if finish and not (self._finish_due() and self._finish_ok(s)):
            self._activity_at = self.clock.monotonic()
            return  # a finish check only: no model turn
        if finish:
            self._last_turn_at = self._busy_since
            # Whatever the order's outcome (refused, not sent), the backstop speaks at most
            # once per ``finish_idle_s``.
            self._activity_at = self.clock.monotonic()
        mem_before = mem
        mem = self._mission_memory(mem, s, trigger, op_msg if pilot else None)
        # Rule: a start request is remembered before any model call.
        if op_msg is not None and pilot and s.vehicle.landed and is_takeoff_request(op_msg.text):
            alt = height_m(op_msg.text)
            mem = with_changes(
                mem,
                pending=PendingRequest(
                    kind="takeoff", message_id=op_msg.id, requested_at=now, alt_m=alt
                ),
            )
        if trigger == "operator_override":
            body = (events[-1].get("body") or {}) if events else {}
            if body.get("active", True) and (
                body.get("action") in ("stop", "return_home") or body.get("parsed_intent")
            ):
                mem = with_changes(mem, pending=None)
        record_base: dict[str, Any] = {"trigger": trigger, "retry_of": extra.get("retry_of")}
        rule_out = (
            await self._finish_step() if finish else self._rule_step(trigger, s, mem, now, op_msg)
        )
        if rule_out is not None:
            out = rule_out
        else:
            # A charter is proposed only for an operator's line (its own turn, where the
            # picture is read after the line) or as the retry of a refused proposal; never
            # on an idle tick, an event or a charter's own approval.
            proposals = trigger == "operator_message" or (
                trigger == "gate_reject_retry" and bool(extra.get("charter_errors"))
            )
            tools = exposed_tools(
                self.reg,
                s,
                mem,
                now,
                actions=trigger != "recovered",
                pilot=pilot,
                proposals=proposals,
            )
            turn = TurnInput(
                turn_id=new_id(),
                mission_id=s.mission_id,
                trigger=trigger,
                events=events,
                situation=s,
                chat_tail=tail,
                backlog=backlog if trigger == "recovered" else [],
                memory=mem,
                tools=tools,
                retry_of=extra.get("retry_of"),
                reject=extra.get("reject"),
                charter_errors=extra.get("charter_errors", []),
                attempt=extra.get("attempt"),
                in_reply_to=op_msg.id if op_msg is not None else extra.get("in_reply_to"),
                now=now,
                budget_s=self.turn_deadline_s,
            )
            out = await self._decide(turn)
            order = out.action is not None and (pilot or out.action.skill != "report")
            if (trigger == "recovered" or not pilot) and (
                order or out.charter_proposal is not None
            ):
                out.note_update = None  # a plan note sent with an order that is not sent
            if trigger == "recovered":
                out.action = None
                out.charter_proposal = None
                out.ready_proposal = None
                out.failed_proposal = None
            if not pilot:
                out.action = out.action if out.action and out.action.skill == "report" else None
                out.charter_proposal = None
                out.failed_proposal = None
        if out.note_update is not None:
            mem = with_changes(mem, plan_note=out.note_update)
        record = TurnRecord(
            turn_id=str(new_id()),
            backend=out.diagnostics.backend,
            output=out,
            error=out.diagnostics.error,
            **record_base,
        )
        self.records.append(record)
        mem = await self._execute(mid, s, mem, out, record, trigger, op_msg, extra)
        if mem != mem_before:
            try:
                await self.ground.put_memory(mid, to_update(mem))
            except GroundError:
                pass
            except ValidationError as exc:  # too large for the schema: log, keep the turn
                log.warning("kunde inte spara minnet: %s", exc.errors()[:1])
        await self._log_decision(mid, record, out)
        self._last_turn_ms = out.diagnostics.latency_ms

    def _mission_memory(
        self, mem: MissionMemory, s: Situation, trigger: Trigger, op_msg: ChatMessage | None
    ) -> MissionMemory:
        """Keep the operator's latest request (``UPPDRAG``): a pilot's new line replaces
        it and the one before moves to ``earlier`` (a "fortsätt" to a stop, or a line that
        only answers, "ja kör", "ok", is no new request); it ends once the drone has
        flown and is landed and disarmed again."""
        req = mission_request(mem)
        v = s.vehicle
        if req is not None and req.flown and v.landed and not v.armed:
            mem = with_changes(mem, mission=None)
        elif req is not None and not req.flown and v.airborne:
            mem = with_changes(mem, mission=req.model_copy(update={"flown": True}))
        if (
            trigger == "operator_message"
            and op_msg is not None
            and op_msg.text.strip()
            and not is_resume_request(op_msg.text)
            and not is_acknowledgement(op_msg.text)
        ):
            new = next_request(
                mission_request(mem),
                op_msg.text,
                op_msg.id,
                op_msg.created_at,
                flown=v.airborne,
            )
            mem = with_changes(mem, mission=new)
        return mem

    def _finish_ok(self, s: Situation) -> bool:
        """The finish backstop may act: airborne in an active mission, no skill running,
        nothing waiting for a click, no stop, pause, hold or latch, and the AI enabled."""
        return (
            s.vehicle.airborne
            and s.mission_status == MissionState.ACTIVE
            and s.active_command is None
            and s.pending_proposal is None
            and not any(a.status == "pending" for a in s.pending_approvals)
            and motion_block(s) is None
            and s.ai_enabled
        )

    async def _finish_step(self) -> TurnOutput:
        """The backstop's line first, then the same ``return_home`` an AI order is (it
        goes through the gate; the operator's STOP stops it as always)."""
        minutes = _fmt(round(self.cfg.behaviour.finish_idle_s / 60, 1))
        await self._say(OutMessage(text=FINISH_TEXT.format(minutes=minutes), kind="status"))
        return TurnOutput(
            action=AiAction(
                skill="return_home",
                params={"reason": FINISH_REASON},
                rationale_sv=FINISH_REASON,
                adapted=True,
            ),
            diagnostics=TurnDiagnostics(backend="rule"),
        )

    def _idle_ok(self, s: Situation) -> bool:
        return (
            s.vehicle.airborne
            and s.active_command is None
            and not any(
                a.status == "pending" and str(a.id) in self._own_action_approvals
                for a in s.pending_approvals
            )
            and motion_block(s) is None
            and s.ai_enabled
        )

    def _operator_message(
        self, events: list[dict[str, Any]], chat: list[ChatMessage]
    ) -> ChatMessage | None:
        ids = [
            str((e.get("body") or {}).get("message_id"))
            for e in events
            if e.get("type") == "event.operator_message"
        ]
        for mid in reversed(ids):
            msg = next((m for m in chat if str(m.id) == mid), None)
            if msg is not None:
                return msg
        return next((m for m in reversed(chat) if m.role == "operator"), None)

    def _rule_step(
        self,
        trigger: Trigger,
        s: Situation,
        mem: MissionMemory,
        now: dt.datetime,
        op_msg: ChatMessage | None,
    ) -> TurnOutput | None:
        diag = TurnDiagnostics(backend="rule")
        b = self.cfg.behaviour
        block = motion_block(s)
        resume = op_msg is not None and is_resume_request(op_msg.text)
        if trigger == "operator_message" and block is not None and resume:
            if block == "battery":
                return TurnOutput(messages=[OutMessage(text=BATTERY_TEXT)], diagnostics=diag)
            return TurnOutput(
                messages=[
                    OutMessage(
                        text=RESUME_TEXT, kind="proposal", refs={"proposed_action": "resume"}
                    )
                ],
                diagnostics=diag,
            )
        if trigger == "takeoff_clearance":
            req = fresh_request(s, mem, now, TAKEOFF_REQUEST_TTL_S)
            if req is None:
                return TurnOutput(messages=[OutMessage(text=CLEARED_NO_REQUEST)], diagnostics=diag)
            if b.rule_takeoff and takeoff_ok(s, mem, now, ttl_s=TAKEOFF_REQUEST_TTL_S):
                alt = req.alt_m if req.alt_m is not None else current_limits(s).default_alt_agl_m
                return TurnOutput(
                    action=AiAction(
                        skill="takeoff",
                        params={"alt_m": alt},
                        rationale_sv="Operatören bad om start och fritt runt är bekräftat.",
                        adapted=True,
                    ),
                    diagnostics=diag,
                )
        if (
            trigger == "charter_event"
            and s.charter is not None
            and s.vehicle.landed
            # The ground publishes "installed" just before it moves the mission to READY.
            and (s.mission_status == "READY" or (s.charter_installed and s.fence_verified))
            and fresh_request(s, mem, now, TAKEOFF_REQUEST_TTL_S) is not None
        ):
            return TurnOutput(messages=[OutMessage(text=CHARTER_WITH_REQUEST)], diagnostics=diag)
        return None

    async def _decide(self, turn: TurnInput) -> TurnOutput:
        backend = self._active_backend()
        executor = _ReadTools(self, turn)
        start = self.clock.monotonic()
        try:
            async with asyncio.timeout(self.turn_deadline_s):
                out = await backend.decide(turn, executor)
        except TimeoutError:
            # Keep what the turn had (prompt, tools, calls) for the decision log.
            partial = getattr(backend, "partial_diagnostics", None)
            diag = partial(turn.turn_id) if partial is not None else None
            diag = (
                diag.model_copy(deep=True)
                if diag is not None
                else TurnDiagnostics(backend=_wire_backend(backend.name))
            )
            diag.error = "deadline"
            diag.latency_ms = 0
            out = TurnOutput(messages=[_deadline("deadline")], diagnostics=diag)
        except BackendCrashError:
            out = TurnOutput(
                messages=[_deadline("http_error")],
                diagnostics=TurnDiagnostics(
                    backend=_wire_backend(backend.name), error="http_error"
                ),
            )
        if out.diagnostics.latency_ms == 0:
            out.diagnostics.latency_ms = int((self.clock.monotonic() - start) * 1000)
        err = out.diagnostics.error
        if out.failed_proposal is not None:
            # A refused draft, even when a later call failed: the charter path answers
            # (a retry, or the final line), and the model did answer.
            if backend is self.backend:
                self.health.turn(True)
            return out
        if backend is self.backend:
            # A turn the model's own time guard ended (every call answered) is healthy.
            self.health.turn(
                out.diagnostics.guard_stop
                or err not in ("deadline", "timeout", "http_error", "parse_error")
            )
        if (
            backend is self.backend
            and backend is not self.fallback
            and err in ("timeout", "http_error")
            and out.action is None
            and not out.messages
        ):
            # The model did not answer: the command mode answers the same turn.
            out = await self._command_mode(turn, executor, err)
        elif err in ("timeout", "http_error") and not out.messages:
            out.messages = [_deadline(err)]
        elif (
            err == "deadline"
            and turn.trigger == "operator_message"
            and not out.messages
            and not out.has_action()
        ):
            out.messages = [_deadline(err)]  # the model's own turn ran out of time
        if turn.trigger not in ("operator_message", "gate_reject_retry", "recovered"):
            # "Säg det på ett annat sätt" answers nobody on an event's or a tick's turn.
            out.messages = [
                m for m in out.messages if m.refs.get("turn_failed") not in ("parse", "repeat")
            ]
        return out

    async def _command_mode(
        self, turn: TurnInput, executor: _ReadTools, err: TurnError | None
    ) -> TurnOutput:
        start = self.clock.monotonic()
        # Runs after the turn deadline, so the fallback must stay pure code with no I/O.
        out = await self.fallback.decide(turn, executor)
        if not out.messages and out.action is None and turn.trigger == "operator_message":
            out.messages = [OutMessage(text=COMMAND_MODE_TEXT, refs={"turn_failed": "command"})]
        out.diagnostics.error = err
        out.diagnostics.engine = "command"
        if out.diagnostics.latency_ms == 0:
            out.diagnostics.latency_ms = int((self.clock.monotonic() - start) * 1000)
        return out

    # --- carrying out the turn's output -------------------------------------------------

    async def _say(self, msg: OutMessage, in_reply_to: Any = None) -> None:
        if self.mission_id is None:
            return
        body = BrainMessage(
            client_msg_id=new_id(),
            text=truncate(msg.text, MESSAGE_MAX_CHARS),
            kind=msg.kind,
            in_reply_to=in_reply_to,
            refs=msg.refs,
        )
        if msg.kind in ("question", "proposal"):
            self._activity_at = self.clock.monotonic()  # the operator has something to answer
        try:
            await self.ground.post_message(self.mission_id, body)
        except GroundError as exc:
            log.warning("kunde inte skriva i chatten: %s", exc)

    async def _execute(
        self,
        mid: str,
        s: Situation,
        mem: MissionMemory,
        out: TurnOutput,
        record: TurnRecord,
        trigger: Trigger,
        op_msg: ChatMessage | None,
        extra: dict[str, Any],
    ) -> MissionMemory:
        # A follow-up turn answers the operator line its first try answered.
        reply_to = op_msg.id if op_msg is not None else extra.get("in_reply_to")
        record.texts = [m.text for m in out.messages]
        record.messages = list(out.messages)
        if (
            out.charter_proposal is not None
            or out.ready_proposal is not None
            or out.failed_proposal is not None
        ):
            outcome = await self._propose(mid, s, out)
            attempt = self._charter_attempts + 1
            record.proposal = outcome
            record.attempt = attempt
            record.proposal_draft = out.charter_proposal
            if out.failed_proposal is not None:
                record.refused_draft = out.failed_proposal.args
            if outcome.ok:
                self._charter_attempts = 0
                self._activity_at = self.clock.monotonic()
                if outcome.approval_id is not None:
                    self._own_approvals.add(str(outcome.approval_id))
                said = list(out.messages) or [OutMessage(text=PROPOSED_TEXT)]
                record.texts = [m.text for m in said]
                for m in said:
                    await self._say(m, reply_to)
            elif attempt < MAX_CHARTER_ATTEMPTS and not (
                out.failed_proposal is not None and out.failed_proposal.final
            ):
                self._charter_attempts = attempt
                record.texts = []  # nothing was said: the follow-up turn explains
                self._followups.append(
                    (
                        "gate_reject_retry",
                        {
                            "retry_of": record.turn_id,
                            "charter_errors": outcome.errors,
                            "attempt": attempt + 1,
                            "in_reply_to": reply_to,
                        },
                    )
                )
            else:
                self._charter_attempts = 0  # given up: the next request starts over
                text = CHARTER_FAILED_TEXT.format(n=attempt) + "; ".join(
                    e.detail_sv for e in outcome.errors
                )
                text = truncate(text, 400) + " Hur vill du gå vidare?"
                record.texts = [text]
                refs = {
                    "charter_failed": True,
                    "codes": sorted({e.code for e in outcome.errors})[:16],
                    "attempts": attempt,
                }
                await self._say(OutMessage(text=text, kind="question", refs=refs), reply_to)
            return mem
        if trigger == "gate_reject_retry" and extra.get("charter_errors"):
            # The retry of a refused charter proposal ended without one (a text that
            # explains a limit, an order, a model error): the request is over, so the next
            # one starts with all its attempts.
            self._charter_attempts = 0
        if out.action is None:
            for m in out.messages:
                await self._say(m, reply_to)
            return mem
        record.ai_action = out.action
        try:
            req = self._skill_request(out.action, s, reply_to, out)
        except (AdaptError, KeyError, ValueError) as exc:
            text = "Jag fick inte ihop ordern. Kan du säga det på ett annat sätt?"
            record.error = f"adapt: {exc}"
            record.texts.append(text)
            await self._say(OutMessage(text=text), reply_to)
            return mem
        await self._respect_rate()
        try:
            result = await self.ground.submit(mid, req)
        except GroundError as exc:
            log.warning("ordern gick inte fram: %s", exc)
            await self._say(OutMessage(text="Ordern gick inte fram till markstationen."), reply_to)
            return mem
        self._action_times.append(self.clock.monotonic())
        self._activity_at = self.clock.monotonic()
        record.request = req
        record.result = result
        if result.decision == "ALLOW":
            self._rate_rejects = 0
            if out.action.skill == "takeoff":
                mem = with_changes(mem, pending=None)
            texts = list(out.messages)
            if not texts and out.diagnostics.backend == "rule" and out.action.skill == "takeoff":
                texts = [OutMessage(text=f"Startar till {_fmt(req.params.get('alt_m'))} m.")]
            elif (
                not texts
                and out.diagnostics.backend == "rule"
                and out.action.skill == "return_home"
            ):
                texts = [OutMessage(text=FINISH_DONE_TEXT)]  # the backstop's order went through
            for m in texts:
                await self._say(m, reply_to)
            record.texts = [m.text for m in texts]
        elif result.decision == "PENDING_APPROVAL":
            self._rate_rejects = 0
            if result.approval_id is not None:
                self._own_approvals.add(str(result.approval_id))
                self._own_action_approvals.add(str(result.approval_id))
            what = _what(out.action, req)
            text = f"Väntar på ditt klick: {what}."
            record.texts = [text]
            await self._say(OutMessage(text=text), reply_to)
        else:
            self._sync_rejects[str(req.command_id)] = self.clock.monotonic()
            self._prune_rejects()
            record.texts = []
            if "RATE_LIMIT" in result.reason_codes:
                self._rate_rejects += 1
                if self._rate_rejects >= 3:
                    self.health.force_degrade()
                    record.texts = [RATE_TEXT]
                    await self._say(OutMessage(text=RATE_TEXT))
                    return mem
            if out.diagnostics.backend == "rule":
                record.texts = [result.message_sv]
                await self._say(OutMessage(text=result.message_sv), reply_to)
            elif extra.get("reject") is None:  # this turn was not already the order's retry
                decision = GateDecision(
                    decision="REJECT",
                    reason_codes=result.reason_codes,
                    rule_ids=[],
                    message_sv=result.message_sv,
                    details=result.details,
                    side="ground",
                )
                self._followups.append(
                    (
                        "gate_reject_retry",
                        {"retry_of": record.turn_id, "reject": decision, "in_reply_to": reply_to},
                    )
                )
            else:
                text = (
                    truncate(f"Grinden sa nej igen: {result.message_sv}", 400)
                    + " Hur vill du gå vidare?"
                )
                record.texts = [text]
                await self._say(OutMessage(text=text, refs={"turn_failed": "gate"}), reply_to)
        return mem

    async def _fresh_situation(
        self, mid: str, s: Situation, trigger: Trigger, events: list[dict[str, Any]]
    ) -> Situation:
        """The situation once it shows the vehicle after the event (at most
        ``FRESH_SNAPSHOT_WAIT_S``): after a skill's end, a newer onboard snapshot than the
        one at hand and without that order as active (the ground can mark the order done
        before the snapshot that shows the onboard state change; an order based on the
        older one is refused onboard as ``STATE_CHANGED``); after the pilot's Resume, the
        mission no longer paused by a person nor HELD by the STOP (until the onboard
        computer confirms the release, the gate still says the pilot must resume)."""
        kind = "event.skill" if trigger == "skill_event" else "event.operator_override"
        body = next((e.get("body") or {} for e in reversed(events) if e.get("type") == kind), {})
        first = (s.snapshot_state_epoch, s.snapshot_state_seq)

        def stale(s: Situation) -> bool:
            if trigger == "skill_event":
                ended = body.get("command_id")
                if ended is None:
                    return False
                still_active = s.active_command is not None and str(s.active_command.id) == str(
                    ended
                )
                same_snapshot = (
                    first[1] is not None and (s.snapshot_state_epoch, s.snapshot_state_seq) == first
                )
                return still_active or same_snapshot
            if body.get("action") != "resume":
                return False
            if s.mission_status == MissionState.HELD:  # STOP's BRAKE, not yet released
                return True
            return s.mission_status == MissionState.PAUSED and s.paused_by != "brain"

        end = self.clock.monotonic() + FRESH_SNAPSHOT_WAIT_S
        while stale(s) and self.clock.monotonic() < end:
            await self.clock.sleep(0.1)
            s = await self.ground.situation(mid)
        return s

    def _skill_request(
        self, action: AiAction, s: Situation, reply_to: Any, out: TurnOutput
    ) -> SkillRequest:
        params = action.params if action.adapted else adapt(action.skill, action.params, s)
        if not action.raw and action.skill not in self.reg:
            raise KeyError(action.skill)
        based_on = None
        if s.snapshot_state_epoch is not None and s.snapshot_state_seq is not None:
            based_on = StateRef(state_epoch=s.snapshot_state_epoch, state_seq=s.snapshot_state_seq)
        backend = out.diagnostics.backend
        return SkillRequest(
            command_id=new_id(),
            skill=action.skill,
            params=params,
            rationale_sv=truncate(action.rationale_sv or "-", 200),
            in_reply_to=reply_to,
            based_on=based_on,
            charter_version=s.charter.version if s.charter is not None else None,
            backend=backend,
            ai_config_hash=self.ai_hash,
        )

    async def _respect_rate(self) -> None:
        now = self.clock.monotonic()
        self._action_times = [t for t in self._action_times if now - t < 60]
        wait = 0.0
        if self._action_times:
            wait = max(wait, MIN_ACTION_INTERVAL_S - (now - self._action_times[-1]))
        if len(self._action_times) >= MAX_ACTIONS_PER_MINUTE:
            wait = max(wait, 60 - (now - self._action_times[0]))
        if wait > 0:
            await self.clock.sleep(wait)

    def _prune_rejects(self) -> None:
        now = self.clock.monotonic()
        self._sync_rejects = {
            k: v for k, v in self._sync_rejects.items() if now - v < SYNC_REJECT_MEMORY_S
        }

    async def _propose(self, mid: str, s: Situation, out: TurnOutput) -> CharterProposalOutcome:
        if out.failed_proposal is not None:  # refused in the turn: nothing is sent
            return CharterProposalOutcome(ok=False, errors=out.failed_proposal.errors)
        proposal = out.ready_proposal
        if proposal is None and out.charter_proposal is not None:
            try:
                proposal = to_proposal(out.charter_proposal, s)
            except DraftError as exc:
                error = CharterError(code=exc.code, detail_sv=str(exc), field=exc.field)
                return CharterProposalOutcome(ok=False, errors=[error])
        if proposal is None:
            return CharterProposalOutcome(ok=False)
        if proposal.base_version is None and s.charter is not None:
            # A change builds on the charter in force (a ready-made proposal, e.g. from
            # the scripted commander, does not know its version).
            proposal = proposal.model_copy(update={"base_version": s.charter.version})
        try:
            return await self.ground.propose_charter(mid, proposal)
        except GroundError as exc:
            return CharterProposalOutcome(
                ok=False, errors=[CharterError(code="GROUND_ERROR", detail_sv=str(exc))]
            )

    async def _log_decision(self, mid: str, record: TurnRecord, out: TurnOutput) -> None:
        d = out.diagnostics
        prompt = d.prompt_json or {"trigger": record.trigger, "texts": record.texts}
        prompt_bytes = json.dumps(prompt, sort_keys=True, ensure_ascii=False, default=str)
        result = record.result
        entry = DecisionLog(
            id=new_id(),
            turn_id=uuid.UUID(record.turn_id),
            trigger=record.trigger,
            backend=d.backend,
            engine=d.engine,
            model_id=d.model_id,
            ai_config_hash=self.ai_hash,
            prompt_sha256=hashlib.sha256(prompt_bytes.encode()).hexdigest(),
            prompt_tokens=d.prompt_tokens,
            completion_tokens=d.completion_tokens,
            latency_ms=d.latency_ms,
            rounds=d.rounds,
            repairs=d.repairs,
            tool_calls=d.tool_calls[:32],
            command_id=record.request.command_id if record.request else None,
            gate_decision=result.decision if result else None,
            reason_codes=result.reason_codes if result else [],
            error=d.error,
            retry_of=uuid.UUID(record.retry_of) if record.retry_of else None,
            attempt=record.attempt,
            proposal_id=record.proposal.proposal_id if record.proposal else None,
            proposal_ok=record.proposal.ok if record.proposal else None,
            proposal_errors=_logged_errors(record.proposal),
            proposal_draft=_logged_draft(record.proposal_draft or record.refused_draft),
            prompt_json=json.loads(prompt_bytes) if len(prompt_bytes) < 900_000 else None,
        )
        try:
            await self.ground.post_decision(mid, entry)
        except (GroundError, ValueError) as exc:
            log.warning("kunde inte spara beslutsloggen: %s", exc)

    # --- the quick stop check -------------------------------------------------------------

    def _start_stop_check(self, evt: dict[str, Any]) -> None:
        if self.decide is None or not self.cfg.decide.stop_check:
            return
        body: dict[str, Any] = evt.get("body") or {}
        if body.get("author_role") not in (None, "pilot"):
            return
        if body.get("handled_by") == "ground":
            return  # the exact stop words: the ground already acted like the button
        text = str(body.get("text") or "")
        message_id = body.get("message_id")
        if not text or message_id is None:
            return
        task = asyncio.create_task(self._stop_check(str(message_id), text, evt))
        self._side_tasks.add(task)
        task.add_done_callback(self._side_tasks.discard)

    async def _stop_check(self, message_id: str, text: str, evt: dict[str, Any]) -> None:
        mid = self.mission_id
        if mid is None or self.decide is None:
            return
        result = await stop_check(
            self.decide,
            text,
            timeout_s=self.cfg.decide.stop_check_timeout_s,
            threshold=self.cfg.decide.stop_threshold,
        )
        self.last_stop_check = result
        if result.intent is None:
            return
        report = ChatIntentReport(
            request_id=result.request_id,
            message_id=uuid.UUID(message_id),
            intent=result.intent,
            probability=round(result.probability, 4),
            engine=result.engine or "rule",
            location=result.location,
            latency_ms=result.latency_ms,
        )
        # Was this line still waiting for its answer (queued, or in the running turn)?
        dropped = any(p.event is evt for p in self._queue) or (
            self._turn_task is not None and any(e is evt for e in self._turn_events)
        )
        if self._turn_task is not None:
            self._turn_task.cancel()  # the operator wants to stop: drop the running turn
        self._queue = [p for p in self._queue if p.trigger != "operator_message"]
        self._followups.clear()
        try:
            outcome = await self.ground.report_chat_intent(mid, report)
        except GroundError as exc:
            log.warning("stoppkontrollen nådde inte markstationen: %s", exc)
            await self._stop_fallback(mid, result)
            return
        if outcome == "refused":  # not a pilot's own recent line: the ground decides
            log.info("markstationen tog inte emot stoppkontrollen för rad %s", message_id)
            # The stop check took the line from the queue: answer it as usual, once, and
            # only in the mission it was written in.
            if dropped and self.mission_id == mid:
                self._queue.append(_Pending("operator_message", evt))
                self._wake.set()

    async def _stop_fallback(self, mid: str, result: StopCheckResult) -> None:
        """The ground could not take the report: ask for the button and send a gated
        ``hold`` (a model's reading only ever stops the vehicle where it is)."""
        if result.intent is None:
            return
        await self._say(OutMessage(text=STOP_FALLBACK[result.intent]))
        try:
            s = await self.ground.situation(mid)
        except GroundError:
            return
        if not s.vehicle.airborne or motion_block(s) is not None:
            return
        reason = "Operatören bad om stopp i chatten."
        out = TurnOutput(
            action=AiAction(
                skill="hold", params={"reason": reason}, rationale_sv=reason, adapted=True
            ),
            diagnostics=TurnDiagnostics(backend="rule"),
        )
        req = self._skill_request(out.action, s, None, out)  # type: ignore[arg-type]
        with contextlib.suppress(GroundError):
            await self.ground.submit(mid, req)

    async def log_decide(self, req: DecideRequest, outcomes: list[EngineOutcome]) -> None:
        """Every engine's answer to a decision request, for comparing the engines."""
        mid = self.mission_id
        if mid is None:
            return
        used = next((o for o in outcomes if o.used), None)
        calls = [
            {
                "name": f"decide:{o.engine}",
                "used": o.used,
                "error": o.error,
                "latency_ms": o.answer.latency_ms if o.answer else None,
                "model": o.answer.model_id if o.answer else None,
                "answers": {
                    k: v.model_dump(mode="json")
                    for k, v in (o.answer.answers.items() if o.answer else [])
                },
            }
            for o in outcomes
        ]
        prompt = req.model_dump(mode="json")
        raw = json.dumps(prompt, sort_keys=True, ensure_ascii=False)
        entry = DecisionLog(
            id=new_id(),
            turn_id=req.request_id,
            trigger=req.purpose,
            backend="rule",
            engine="cloud" if used and used.answer and used.answer.location == "cloud" else None,
            model_id=used.answer.model_id if used and used.answer else None,
            ai_config_hash=self.ai_hash,
            prompt_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=used.answer.latency_ms if used and used.answer else 0,
            rounds=len(outcomes),
            repairs=0,
            tool_calls=[dict(c) for c in calls][:32],
            command_id=None,
            gate_decision=None,
            error=None,
            prompt_json=prompt,
        )
        with contextlib.suppress(GroundError, ValueError):
            await self.ground.post_decision(mid, entry)


class _ReadTools:
    """Read tools during a turn (``get_situation``, ``estimate``, ``get_log``)."""

    def __init__(self, loop: CommanderLoop, turn: TurnInput) -> None:
        self.loop = loop
        self.turn = turn
        self.profile_proposed = False  # one search-profile proposal per turn

    async def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        mid = str(self.turn.mission_id)
        g = self.loop.ground
        try:
            if name == "get_situation":
                s = await g.situation(mid)
                return {"situation": situation_text(s, self.loop.clock.utc_now())}
            if name == "get_log":
                n = max(1, min(20, int(args.get("n", 5))))
                s = await g.situation(mid)
                return {
                    "events": [
                        {"t": e.t.isoformat(), "type": e.type, "summary": str(e.payload)[:120]}
                        for e in s.recent_events[-n:]
                    ]
                }
            if name == "estimate":
                skill = str(args.get("skill"))
                if skill not in self.loop.reg or skill == PROPOSAL_TOOL:
                    return {"error": f"unknown skill {skill}"}
                params = adapt(skill, dict(args.get("params") or {}), self.turn.situation)
                req = SkillRequest(
                    command_id=new_id(),
                    skill=skill,
                    params=params,
                    rationale_sv="estimate",
                    in_reply_to=None,
                    based_on=None,
                    charter_version=None,
                    backend="llm",
                    ai_config_hash=self.loop.ai_hash,
                )
                est = await g.estimate(mid, req)
                return json.loads(est.model_dump_json())
            if name == "brain_note":
                return {"ok": True}
            if name == PROFILE_TOOL:
                if self.profile_proposed:
                    # A second call would replace the first on the pilot's card.
                    return {"error": "already proposed in this turn; the pilot corrects it"}
                proposal = profile_proposal(args, self.turn.situation, self.loop.clock.utc_now())
                pid = await g.propose_search_profile(mid, proposal)
                self.profile_proposed = True
                return {
                    "proposal_id": pid,
                    "status": "proposed: the pilot corrects and approves it on the profile card",
                }
        except (GroundError, AdaptError, ValueError, KeyError) as exc:  # ProfileArgsError too
            return {"error": truncate(str(exc), TOOL_RESULT_MAX_CHARS)}
        return {"error": f"unknown tool {name}"}


def _logged_errors(outcome: CharterProposalOutcome | None) -> list[CharterValidationError]:
    if outcome is None:
        return []
    return [
        CharterValidationError(
            code=e.code, detail_sv=truncate(e.detail_sv, 500), field=e.field, limit=e.limit
        )
        for e in outcome.errors[:32]
    ]


def _logged_draft(draft: Any) -> dict[str, Any] | None:
    """The turn's charter draft (a ``CharterDraft``, or a refused draft's raw arguments)
    for the decision log, when it fits."""
    if draft is None:
        return None
    body = draft.model_dump(mode="json") if isinstance(draft, CharterDraft) else draft
    if not isinstance(body, dict):
        return None
    raw = json.dumps(body, ensure_ascii=False, default=str)
    return json.loads(raw) if len(raw.encode()) <= PARAMS_MAX_BYTES else None


def _deadline(why: str) -> OutMessage:
    """The answer when the model did not answer in time (``refs.turn_failed``: no card
    or order follows from this turn)."""
    return OutMessage(text=DEADLINE_TEXT, refs={"turn_failed": why})


def _wire_backend(name: str) -> Literal["llm", "grammar", "scripted"]:
    if name == "grammar" or name == "scripted":
        return name
    return "llm"


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):g}".replace(".", ",")
    except (TypeError, ValueError):
        return "?"


def _what(action: AiAction, req: SkillRequest) -> str:
    names = {
        "takeoff": "start",
        "goto": "flyg till punkten",
        "survey_sector": f"sök sektor {req.params.get('sector_id', '')}".strip(),
        "hold": "håll position",
        "resume": "fortsätt",
        "return_home": "flyg hem",
        "report": "rapport",
        "mark_location": "markering",
        "follow_path": f"flyg längs vägen {req.params.get('label') or ''}".strip(),
    }
    text = names.get(action.skill, action.skill)
    if action.skill == "takeoff" and req.params.get("alt_m") is not None:
        text += f" till {_fmt(req.params['alt_m'])} m"
    if action.skill == "goto" and action.params.get("pin"):
        text = f"flyg till {action.params['pin']}"
    return text
