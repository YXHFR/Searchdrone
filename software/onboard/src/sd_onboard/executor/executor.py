"""The executor: takes ``SkillCommand`` from the ground, checks it again onboard and flies
it through ``sd_mav``. It owns the executor state and lets go at once when a person, the
ground's emergency path or the flight controller takes over.

Everything time-based reads the injected clock. ``tick()`` runs at 10 Hz: the flight
controller's mode and failsafes, the limit monitors, the running skill and the
transitions. Inbound messages from SD-link come in through the ``on_*`` methods; what goes
back (Acks, events, charter answers) leaves through the ``Uplink``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol
from uuid import UUID

from pydantic import ValidationError

from sd_mav.mission import (
    FENCE,
    MISSION,
    MissionError,
    MissionItem,
    download,
    fence_hash,
    mission_hash,
    upload,
)
from sd_mav.modes import HOME_MODES, CopterMode
from sd_mav.vehicle import (
    ControlNotAllowedError,
    ForbiddenMavlinkError,
    MavResult,
    SendOrigin,
    Vehicle,
)
from sd_obs.clock import Clock
from sd_onboard.executor.authority import Authority
from sd_onboard.executor.guards import GuardInput, Refusal, is_airborne, run_guard
from sd_onboard.executor.limits import (
    INTERNAL_ALT_GOTO,
    LimitAction,
    LimitInput,
    LimitMonitor,
    PilotTracker,
    inside_charter,
)
from sd_onboard.executor.override import OverrideState
from sd_onboard.executor.state import (
    AIRBORNE_CONTROLLED,
    MOVING,
    ON_GROUND,
    TRANSITIONS,
    Trigger,
    TriggerKey,
    skill_key,
)
from sd_onboard.io.persistence import StateStore
from sd_onboard.skills import HANDLERS, extra_transitions
from sd_onboard.skills.base import (
    PlanContext,
    PlanError,
    SkillFailedError,
    SkillHandler,
    SkillPlan,
    TakeoffTimeoutError,
    TickContext,
)
from sd_onboard.skills.goto import GotoHandler, goto_plan
from sd_onboard.skills.return_home import ReturnHomeHandler
from sd_safety import evaluate
from sd_safety.constants import (
    CLEARANCE_VALID_S,
    EXECUTOR_TICK_HZ,
    FC_HEARTBEAT_TIMEOUT_S,
    FENCE_ALT_MARGIN_M,
    FENCE_RADIUS_MARGIN_M,
    GCS_SEEN_FOR_RELEASE_S,
    HOME_MATCH_M,
    LINK_LOSS_DETECT_S,
    MODE_CONFIRM_TIMEOUT_S,
    TAKEOFF_BATTERY_MARGIN_PCT,
)
from sd_safety.limits import (
    ABSOLUTE_MAX_ALT_AGL_M,
    ABSOLUTE_MAX_RANGE_FROM_PILOT_M,
    ABSOLUTE_MAX_SPEED_MS,
    DEFAULT_CAPS,
    DEFAULT_LIMITS,
    area_reach_from_home_m,
    charter_violations,
    fc_area_max_from_home_m,
)
from sd_safety.policy import ALWAYS_REQUIRE_APPROVAL, MODE_A_EXEMPT
from sd_safety.targets import Plane, distance_m, polygon_from_geojson
from sd_schemas.approval import command_hash
from sd_schemas.base import LatLon, LatLonAlt, new_id
from sd_schemas.charter import AutonomyMode, MissionCharter
from sd_schemas.executor import ExecutorState, ResumePoint
from sd_schemas.gate import ClearanceState, GateContext, GateDecision, GateRequest
from sd_schemas.link import (
    SCHEMA_MAJOR,
    AckStage,
    CharterAck,
    CharterInstall,
    ClearanceConfirmation,
    CommanderStatus,
    Hello,
    HelloAck,
    ManualTakeover,
    OperatorOverrideNotice,
    OperatorRelease,
    PilotPosition,
    SkillCommand,
    StateSnapshot,
)
from sd_schemas.safety import SafetyLimits
from sd_schemas.skills import (
    SKILL_REGISTRY,
    ReturnHomeParams,
    SkillParams,
    SkillRegistry,
    SkillSpec,
)
from sd_schemas.vehicle import VehicleState

__all__ = ["Executor", "ExecutorOptions", "Uplink"]

log = logging.getLogger("sd_onboard")
S = ExecutorState
Severity = Literal["debug", "info", "warning", "critical"]

TICK_S: Final = 1.0 / EXECUTOR_TICK_HZ
_LANDED_WAIT_MS: Final = 5000
_SAVE_EVERY_MS: Final = 1000
_PREFLIGHT_EVERY_MS: Final = 1000
_ALT_GOTO_SPEED_MS: Final = 3.0
_ALT_GOTO_BELOW_M: Final = 5.0
_RTL_RETRY_MS: Final = 3000
_LOOP_ALIVE_MS: Final = 1500  # our HEARTBEAT stops when the tick has not run for this long
_MIN_GPS_SATS: Final = 8
_CRITICAL: Final = frozenset(
    {
        "fc_silent",
        "fc_failsafe",
        "limit_battery_rtl",
        "limit_battery_unknown_rtl",
        "limit_range_rtl",
        "limit_area",
        "link_loss_rtl",
        "forbidden_mavlink_blocked",
        "gate_disagreement",
        "return_home_failed",
        "return_home_retry",
    }
)
_WARNING: Final = frozenset(
    {
        "control_taken",
        "limit_battery_unknown",
        "limit_altitude",
        "limit_range",
        "limit_speed",
        "limit_nav",
        "link_lost",
        "ai_silent",
        "pilot_position_stale",
        "inbound_command_ignored",
        "onboard_restart",
    }
)
_STOPPING_MODES: Final = frozenset({CopterMode.BRAKE.name, CopterMode.LOITER.name})
HOME_PARAMS: Final = ReturnHomeParams(reason="limit")
_ISSUER_KIND: Final = {
    "operator": "operator",
    "brain": "brain",
    "system": "system",
    "rule": "system",
}


class Uplink(Protocol):
    """What the executor sends to the ground (SD-link queues it while the link is down)."""

    def ack(
        self,
        ref_id: UUID,
        stage: AckStage,
        *,
        reason_codes: tuple[str, ...] = (),
        detail_sv: str | None = None,
        gate: GateDecision | None = None,
    ) -> None: ...

    def event(self, type: str, severity: Severity, data: dict[str, Any]) -> None: ...

    def charter_ack(self, ack: CharterAck) -> None: ...


@dataclass(frozen=True)
class ExecutorOptions:
    vehicle_id: str
    allow_companion_arm: bool = True
    takeoff_delay_after_arm_s: float = 0.0  # test hook (sim and SD_TEST=1 only)
    expected_params: Mapping[str, float] = field(default_factory=dict[str, float])  # PF6
    registry: SkillRegistry = SKILL_REGISTRY


@dataclass
class _Active:
    command_id: UUID
    skill: str  # the registry name, or INTERNAL_ALT_GOTO
    handler: SkillHandler
    plan: SkillPlan
    origin: SendOrigin
    issuer: str
    task: asyncio.Task[None] | None = None
    ack_done: bool = True  # Ack(done)/(failed) for this order (False for internal orders)


_PARAM_TOLERANCE: Final = 1e-3
# The fence and LOITER speed may be raised with the mission limits (by the ground, by
# the pilot's click); PF6 then checks them against the hard floor instead of the file.
_RAISABLE: Final = ("FENCE_ALT_MAX", "FENCE_RADIUS", "LOIT_SPEED_MS")


class _IO:
    """``ExecutorIO`` for one order. Every write passes the executor's send gate."""

    def __init__(self, ex: Executor, origin: SendOrigin) -> None:
        self._ex = ex
        self._origin = origin
        self.clock = ex.clock

    @property
    def vehicle(self) -> VehicleState:
        return self._ex.vehicle_state()

    async def set_mode(self, mode: CopterMode) -> bool:
        return await self._ex.set_mode(mode, self._origin)

    async def arm(self) -> MavResult:
        return await self._ex.vehicle.arm(origin=self._origin)

    async def takeoff(self, alt_m: float) -> MavResult:
        return await self._ex.vehicle.takeoff(alt_m, origin=self._origin)

    async def set_position_target(self, lat: float, lon: float, alt_m: float) -> None:
        await self._ex.vehicle.goto(lat, lon, alt_m, origin=self._origin)

    async def change_speed(self, speed_ms: float) -> MavResult:
        return await self._ex.vehicle.change_speed(speed_ms, origin=self._origin)

    async def upload_mission(self, items: list[MissionItem]) -> str:
        return await self._ex.upload_mission(items, self._origin)

    async def set_mission_current(self, seq: int) -> MavResult:
        return await self._ex.vehicle.set_mission_current(seq, origin=self._origin)

    async def wait_message(self, predicate: Callable[[Any], bool], within_s: float) -> Any:
        return await self._ex.vehicle.wait_message(predicate, within_s)

    async def sleep(self, seconds: float) -> None:
        await self.clock.sleep(seconds)

    def last_statustext(self) -> str:
        return self._ex.last_statustext

    def emit_event(self, type: str, data: dict[str, Any]) -> None:
        self._ex.emit(type, data)


class Executor:
    def __init__(
        self,
        vehicle: Vehicle,
        clock: Clock,
        uplink: Uplink,
        options: ExecutorOptions,
        store: StateStore | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.clock = clock
        self.uplink = uplink
        self.options = options
        self.registry = options.registry
        self.store = store
        self.transitions = {**TRANSITIONS}
        for key, row in extra_transitions().items():
            if key in self.transitions:
                raise ValueError(f"a handler redefines the transition {key}")
            self.transitions[key] = row
        self.state = S.IDLE
        self.hold_reason: str | None = None
        self.held_by: str | None = None
        self.control_suspended = False
        self.auth = Authority()
        self.over = OverrideState()
        self.limits = LimitMonitor()
        self.pilot = PilotTracker()
        self.charter: MissionCharter | None = None
        self.hard_limits: SafetyLimits = DEFAULT_LIMITS
        self.clearance: ClearanceState | None = None
        self.resume_point: ResumePoint | None = None
        self.active: _Active | None = None
        self.boot_id = 1
        self.state_seq = 0
        self.last_transition_state_seq = 0
        self.preflight: dict[str, bool] = {}
        self.params_ok: bool | None = None  # PF6
        self.fence_ok: bool | None = None  # PF7
        self.last_statustext = ""
        self.link_up = False
        self.link_last_inbound_ms: int | None = None
        self.commander_connected: bool | None = None
        self.commander_status_ms: int | None = None
        self.ai_silent_after_step = False
        self.flown = False  # airborne seen since the takeoff began
        self._landed_ms: int | None = None
        self._rtl_retry_ms: int | None = None
        self._last_tick_ms: int | None = None
        self._last_save_ms = -_SAVE_EVERY_MS
        self._last_preflight_ms = -_PREFLIGHT_EVERY_MS
        self._preflight_task: asyncio.Task[None] | None = None
        self._side_tasks: set[asyncio.Task[Any]] = set()
        self._tick_ms: list[float] = []
        self._started = False

    # --- the picture --------------------------------------------------------------------

    def now_ms(self) -> int:
        return self.clock.monotonic_ms()

    def vehicle_state(self) -> VehicleState:
        now = self.now_ms()
        pilot, source = self._pilot(now)
        return self.vehicle.telemetry.to_vehicle_state(self.options.vehicle_id, now, pilot, source)

    def _pilot(self, now: int) -> tuple[LatLon, str]:
        if self.charter is None:
            home = self.vehicle.telemetry.home
            fallback = LatLon(lat=home.lat, lon=home.lon) if home else LatLon(lat=0, lon=0)
            return fallback, "charter"
        position, became_stale = self.pilot.effective(self.charter, now)
        if became_stale:
            self.emit("pilot_position_stale", {})
        fresh = position != self.charter.pilot_position
        return position, "phone" if fresh else "charter"

    def plane(self) -> Plane | None:
        if self.charter is None:
            return None
        return Plane(self.charter.home.lat, self.charter.home.lon)

    def loop_alive(self) -> bool:
        """Our HEARTBEAT goes out only while the tick runs, so a hung loop falls silent."""
        return (
            self._last_tick_ms is not None and self.now_ms() - self._last_tick_ms <= _LOOP_ALIVE_MS
        )

    # --- events and acks -----------------------------------------------------------------

    def emit(self, type: str, data: dict[str, Any] | None = None) -> None:
        severity: Severity = (
            "critical" if type in _CRITICAL else "warning" if type in _WARNING else "info"
        )
        payload = {"state": str(self.state), **(data or {})}
        log.info("executor.%s %s", type, payload)
        self.uplink.event(f"executor.{type}", severity, payload)

    def on_vehicle_event(self, kind: str, details: dict[str, Any]) -> None:
        """Events from ``sd_mav.Vehicle`` (``executor.*`` pass through to the ground)."""
        if kind == "mav.statustext":
            self.last_statustext = str(details.get("text", ""))[:200]
        if kind.startswith("executor."):
            self.emit(kind.removeprefix("executor."), details)

    def _ack(self, ref: UUID, stage: AckStage, codes: tuple[str, ...] = (), **kw: Any) -> None:
        self.uplink.ack(ref, stage, reason_codes=codes, **kw)

    # --- the send gate (compare-and-set) -------------------------------------------------

    def send_gate(self, origin: SendOrigin) -> str | None:
        """Why a message from ``origin`` may not go out now, or None. Asked inside
        ``Vehicle.send`` before every single write."""
        if origin not in (SendOrigin.SKILL, SendOrigin.LIMIT):
            return None  # startup traffic; the fallback path has its own rules
        now = self.now_ms()
        if self.state in (S.HELD, S.FAILSAFE) or self.control_suspended:
            return "control_held"
        if self.auth.notice_recent(now):
            return "override_notice"
        return self.auth.mode_problem(self.vehicle.telemetry, now)

    async def set_mode(self, mode: CopterMode, origin: SendOrigin) -> bool:
        """Change mode and wait for the HEARTBEAT to show it; False when refused or late."""
        if mode in (CopterMode.BRAKE, CopterMode.LAND):
            raise ValueError("BRAKE and LAND are only for the emergency fallback")
        deadline = self.clock.monotonic() + MODE_CONFIRM_TIMEOUT_S
        while self.auth.pending_mode is not None and self.auth.pending_mode != mode:
            if self.clock.monotonic() > deadline:
                return False
            await self.clock.sleep(0.05)
        self.auth.pending_mode = mode
        try:
            result = await self.vehicle.set_mode(mode, origin=origin)
            if result is not MavResult.ACCEPTED:
                return False
            deadline = self.clock.monotonic() + MODE_CONFIRM_TIMEOUT_S
            while self.clock.monotonic() <= deadline:
                if self.vehicle.telemetry.fc_custom_mode == int(mode):
                    self.auth.confirmed(mode)
                    return True
                await self.clock.sleep(0.05)
            return False
        finally:
            if self.auth.pending_mode == mode:
                self.auth.pending_mode = None

    async def upload_mission(self, items: list[MissionItem], origin: SendOrigin) -> str:
        """Upload an AUTO mission and read it back: never while AUTO flies."""
        if self.vehicle.telemetry.fc_custom_mode == int(
            CopterMode.AUTO
        ) and not await self.set_mode(CopterMode.LOITER, origin):
            raise SkillFailedError(
                "ONB_MODE_CHANGE_TIMEOUT", "Flygkontrollern bytte inte till LOITER."
            )
        try:
            await upload(self.vehicle, items, origin=origin)
            back = await download(self.vehicle, MISSION, origin=origin)
        except MissionError as exc:
            raise SkillFailedError("ONB_MISSION_UPLOAD_FAILED", str(exc)) from exc
        sent, stored = mission_hash(items), mission_hash(back)
        if sent != stored:
            raise SkillFailedError("ONB_MISSION_READBACK_MISMATCH", "Sökrutten stämmer inte.")
        return stored

    # --- transitions ---------------------------------------------------------------------

    def _guard_input(self, key: TriggerKey, **kw: Any) -> GuardInput | None:
        row = self.transitions.get((self.state, key))
        if row is None:
            return None
        now = self.now_ms()
        return GuardInput(
            state=self.state,
            targets=row.targets,
            vehicle=self.vehicle_state() if self.vehicle.telemetry.fc_connected else None,
            resume_point=self.resume_point,
            held_by=self.held_by,
            in_classification_window=self.auth.in_window(now),
            battery_rtl_active=self.over.battery_rtl_active,
            failsafe_cause_gone=self._failsafe_cause_gone(),
            **kw,
        )

    def target_of(self, key: TriggerKey, **kw: Any) -> ExecutorState | Refusal:
        """Where ``key`` would take the executor now (nothing changes)."""
        g = self._guard_input(key, **kw)
        if g is None:
            return Refusal("ONB_INVALID_TRANSITION")
        row = self.transitions[(self.state, key)]
        return run_guard(row.guard, g)

    def _enter(
        self,
        target: ExecutorState,
        *,
        hold_reason: str | None = None,
        held_by: str | None = None,
        cause: str = "",
    ) -> None:
        old = self.state
        self.state = target
        self.hold_reason = hold_reason if target == S.HOLD else None
        self.held_by = held_by if target == S.HELD else None
        if target not in (S.HELD, S.FAILSAFE):
            self.control_suspended = False
        if target != old:
            # The next snapshot shows the new state; an order based on an older one is stale.
            self.last_transition_state_seq = self.state_seq + 1
            self.emit(
                "state_changed",
                {
                    "from": str(old),
                    "to": str(target),
                    "cause": cause,
                    "hold_reason": self.hold_reason,
                    "held_by": self.held_by,
                },
            )
        self._save()

    # --- the active order ---------------------------------------------------------------

    def _io(self, origin: SendOrigin) -> _IO:
        return _IO(self, origin)

    def _start_active(self, active: _Active) -> None:
        self.active = active

        async def body() -> None:
            await active.handler.start(active.plan, self._io(active.origin))

        active.task = asyncio.ensure_future(body())

    def _drop_active(
        self, *, stage: AckStage | None, codes: tuple[str, ...] = (), save_resume: bool = True
    ) -> None:
        """Stop the running order without sending anything."""
        active = self.active
        if active is None:
            return
        if save_resume:
            rp = active.handler.resume_point(self.vehicle_state())
            if rp is not None:
                self.resume_point = rp
        self.active = None
        if active.task is not None and not active.task.done():
            active.task.cancel()
        if stage is not None and active.ack_done:
            self._ack(active.command_id, stage, codes)

    def _side(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        self._side_tasks.add(task)
        task.add_done_callback(self._side_done)

    def _side_done(self, task: asyncio.Task[Any]) -> None:
        self._side_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            exc = task.exception()
            if not isinstance(exc, ControlNotAllowedError | ForbiddenMavlinkError):
                log.error("executor side task failed: %r", exc)

    async def _loiter(self, origin: SendOrigin) -> None:
        with contextlib.suppress(ControlNotAllowedError, ForbiddenMavlinkError):
            await self.set_mode(CopterMode.LOITER, origin)

    def _internal(
        self, skill: str, handler: SkillHandler, plan: SkillPlan, origin: SendOrigin
    ) -> None:
        self._start_active(
            _Active(new_id(), skill, handler, plan, origin, "rule:onboard", ack_done=False)
        )

    def _finish_active(self) -> None:
        """Turn a finished skill task into SKILL_DONE or SKILL_FAILED."""
        active = self.active
        if active is None or active.task is None or not active.task.done():
            return
        task = active.task
        if task.cancelled():
            self.active = None
            return
        exc = task.exception()
        if isinstance(exc, ControlNotAllowedError):
            self.active = None  # someone else steers; the takeover already answered the order
            return
        if exc is None:
            self._skill_done(active)
        elif isinstance(exc, SkillFailedError):
            self._skill_failed(active, exc)
        else:
            self._skill_failed(active, SkillFailedError("ONB_SKILL_TIMEOUT", repr(exc)[:200]))

    def _skill_done(self, active: _Active) -> None:
        self.active = None
        if self.state == S.RTL:
            return  # return home is done at LANDED
        target = self.target_of(Trigger.SKILL_DONE)
        if isinstance(target, Refusal):
            return
        reason = "limit" if active.skill == INTERNAL_ALT_GOTO else "skill_done"
        if self.ai_silent_after_step:
            reason = "ai_silent"
        if active.skill in ("goto", "survey_sector", "resume"):
            self.resume_point = None  # finished: nothing left to resume
        if active.ack_done:
            self._ack(active.command_id, "done")
        if active.skill == "takeoff":
            self.flown = True
        self._enter(target, hold_reason=reason, cause=f"{active.skill} done")

    def _skill_failed(self, active: _Active, exc: SkillFailedError) -> None:
        self.active = None
        rp = active.handler.resume_point(self.vehicle_state())
        if rp is not None:
            self.resume_point = rp
        key = (
            Trigger.TAKEOFF_TIMEOUT
            if isinstance(exc, TakeoffTimeoutError)
            else Trigger.SKILL_FAILED
        )
        if active.ack_done:
            self._ack(active.command_id, "failed", (exc.code,), detail_sv=exc.detail_sv or None)
        if self.state == S.RTL:
            # No way out of RTL: the tick asks for SmartRTL/RTL again until it flies home.
            self.emit(
                "return_home_failed", {"code": exc.code, "fc_mode": self.vehicle.telemetry.fc_mode}
            )
            return
        target = self.target_of(key)
        if isinstance(target, Refusal):
            return
        self._enter(
            target,
            hold_reason="skill_done" if target == S.HOLD else None,
            cause=f"{active.skill} failed: {exc.code}",
        )
        if target == S.HOLD and is_airborne(self.vehicle_state()):
            self._side(self._loiter(active.origin))

    # --- inbound: orders -------------------------------------------------------------------

    async def on_skill_command(self, cmd: SkillCommand, *, deadline_ms: int | None) -> None:
        """Check one order and run it, or answer why not (Ack rejected)."""
        if deadline_ms is not None and self.now_ms() >= deadline_ms:
            self._ack(cmd.command_id, "rejected", ("ONB_ORDER_EXPIRED",))
            return
        spec = self.registry.get(cmd.skill) if cmd.skill in self.registry else None
        if spec is None or spec.status == "stub":
            code = "SKILL_NOT_IMPLEMENTED" if spec is not None else "SKILL_UNKNOWN"
            self._ack(cmd.command_id, "rejected", (code,))
            return
        if spec.runs_on != "onboard" or spec.name not in HANDLERS:
            self._ack(cmd.command_id, "rejected", ("ONB_WRONG_EXECUTOR",))
            return
        try:
            params = spec.params_model.model_validate(cmd.params)
        except ValidationError:
            self._ack(cmd.command_id, "rejected", ("SCHEMA_INVALID",))
            return
        code = self._onboard_checks(cmd, spec, params)
        if code is not None:
            self._ack(cmd.command_id, "rejected", (code,))
            return
        target = self.target_of(skill_key(spec.name))
        if isinstance(target, Refusal):
            self._ack(cmd.command_id, "rejected", (target.code or "ONB_INVALID_TRANSITION",))
            return
        if self.charter is None:
            raise RuntimeError("no charter installed")
        handler: SkillHandler = (
            HANDLERS[spec.name](self.options.takeoff_delay_after_arm_s)
            if spec.name == "takeoff"
            else HANDLERS[spec.name]()
        )
        ctx = self._plan_context(cmd.command_id)
        try:
            plan = handler.plan(params, ctx)
        except PlanError as exc:
            self._ack(cmd.command_id, "rejected", (exc.code,), detail_sv=exc.detail or None)
            return
        decision = evaluate(self._gate_request(cmd, plan), self._gate_context(ctx.vehicle))
        if decision.decision != "ALLOW":
            self.emit(
                "gate_disagreement",
                {
                    "command_id": str(cmd.command_id),
                    "skill": spec.name,
                    "codes": decision.reason_codes,
                },
            )
            self._ack(
                cmd.command_id,
                "rejected",
                tuple(decision.reason_codes),
                detail_sv=decision.message_sv,
                gate=decision,
            )
            return
        home_running = self.active is None or self.active.skill == "return_home"
        if self.state == S.RTL and self._flying_home() and home_running:
            self._ack(cmd.command_id, "accepted")  # T11: already flying home
            self._ack(cmd.command_id, "done")
            return
        self._accept(cmd, spec, handler, plan, target)

    def _accept(
        self,
        cmd: SkillCommand,
        spec: SkillSpec,
        handler: SkillHandler,
        plan: SkillPlan,
        target: ExecutorState,
    ) -> None:
        self._ack(cmd.command_id, "accepted")
        moves_on = spec.name in ("goto", "survey_sector")
        self._drop_active(stage="superseded", save_resume=not moves_on)
        if moves_on:
            self.resume_point = None  # a new move replaces whatever was interrupted
        if spec.name == "takeoff":
            if self.clearance is not None:
                self.clearance = self.clearance.model_copy(update={"consumed": True})
            self.flown = False
        origin = SendOrigin.LIMIT if cmd.issuer == "rule:onboard" else SendOrigin.SKILL
        reason = "ai_hold" if spec.name == "hold" else None
        self._enter(target, hold_reason=reason, cause=f"skill:{spec.name}")
        self._start_active(_Active(cmd.command_id, spec.name, handler, plan, origin, cmd.issuer))
        self._ack(cmd.command_id, "running")

    def _onboard_checks(
        self, cmd: SkillCommand, spec: SkillSpec, params: SkillParams
    ) -> str | None:
        """The executor's own checks (beyond the shared gate), in order."""
        brain = cmd.issuer.startswith("brain:")
        if self.state in (S.HELD, S.FAILSAFE) or self.control_suspended or self.over.manual:
            return "ONB_CONTROL_HELD"
        if self.state == S.RTL and spec.name != "return_home":
            return "ONB_RTL_ACTIVE"
        if self.state not in spec.allowed_from:
            return "ONB_INVALID_TRANSITION"
        charter = self.charter
        if charter is None or cmd.charter_version != charter.version:
            return "CHARTER_VERSION_MISMATCH"
        if not brain and (cmd.based_on is None or cmd.based_on.state_epoch != self.boot_id):
            return "ONB_STALE_ORDER"
        approval_needed = brain and (
            spec.requires_approval_in_b
            or spec.name in ALWAYS_REQUIRE_APPROVAL
            or (
                charter.autonomy_mode == AutonomyMode.APPROVE_EACH
                and spec.name not in MODE_A_EXEMPT
            )
        )
        if approval_needed:
            if cmd.approval is None:
                return "ONB_APPROVAL_MISSING"
            if cmd.approval.command_hash != command_hash(spec.name, params, charter.version):
                return "ONB_APPROVAL_MISMATCH"
        if brain and self.limits.battery_rtl_active:
            return "ONB_BATTERY_RTL_ACTIVE"
        if self.over.stop_latch and brain and spec.moves_vehicle:
            return "ONB_STOP_LATCHED"
        if spec.name == "resume" and self.over.stop_latch:
            return "ONB_STOP_LATCHED"
        if brain and spec.moves_vehicle and spec.name != "return_home":
            now = self.now_ms()
            if self.limits.nav_degraded(now):
                return "ONB_NAV_DEGRADED"
            if self.limits.range_active and not self._closer_to_pilot(spec.name, params):
                return "ONB_RANGE_MUST_DECREASE"
        if spec.name == "takeoff":
            return self._takeoff_problem()
        return None

    def _closer_to_pilot(self, skill: str, params: SkillParams) -> bool:
        if self.charter is None:
            return False
        v = self.vehicle_state()
        pilot = v.pilot_position
        now_d = distance_m(pilot.lat, pilot.lon, v.lat, v.lon)
        end = getattr(params, "lat", None), getattr(params, "lon", None)
        if skill == "goto" and end[0] is not None and end[1] is not None:
            return distance_m(pilot.lat, pilot.lon, end[0], end[1]) < now_d
        return False

    def _takeoff_problem(self) -> str | None:
        charter = self.charter
        if self.state != S.PREFLIGHT or not self.preflight_ok():
            return "ONB_PREFLIGHT_NOT_OK"
        if charter is None or charter.approved_at is None or charter.approved_by is None:
            return "CHARTER_VERSION_MISMATCH"
        c = self.clearance
        if c is None or c.consumed:
            return "ONB_CLEARANCE_MISSING"
        if c.origin == "remote":
            return "ONB_CLEARANCE_REMOTE"
        if (
            c.charter_id != charter.charter_id
            or c.charter_version != charter.version
            or c.mission_id != charter.mission_id
            or c.vehicle_id != self.options.vehicle_id
        ):
            return "ONB_CLEARANCE_MISMATCH"
        age = self.now_ms() - c.received_mono_ms
        if age < 0 or age > min(c.valid_for_ms, CLEARANCE_VALID_S * 1000):
            return "ONB_CLEARANCE_STALE"
        v = self.vehicle_state()
        if v.armed or v.landed_state != "on_ground":
            return "ONB_NOT_ON_GROUND"
        if not self.options.allow_companion_arm:
            return "ONB_ARM_NOT_ALLOWED_IN_PROFILE"
        return None

    def _plan_context(self, command_id: UUID) -> PlanContext:
        if self.charter is None:
            raise RuntimeError("no charter installed")
        plane = self.plane()
        if plane is None:
            raise RuntimeError("planned before running")
        return PlanContext(
            command_id=command_id,
            vehicle=self.vehicle_state(),
            charter=self.charter,
            plane=plane,
            resume_point=self.resume_point,
            clock=self.clock,
        )

    def _gate_request(self, cmd: SkillCommand, plan: SkillPlan) -> GateRequest:
        if self.charter is None:
            raise RuntimeError("no charter installed")
        kind = _ISSUER_KIND[cmd.issuer.split(":", 1)[0]]
        return GateRequest(
            command_id=cmd.command_id,
            mission_id=self.charter.mission_id,
            vehicle_id=self.options.vehicle_id,
            skill=cmd.skill,
            params=cmd.params,
            issuer_kind=kind,  # type: ignore[arg-type]
            issuer_ref=cmd.issuer,
            charter_version=cmd.charter_version,
            based_on=cmd.based_on,
            phase="onboard",
            planned_path=plan.path or None,
        )

    def _gate_context(self, vehicle: VehicleState) -> GateContext:
        now = self.now_ms()
        t = self.vehicle.telemetry
        age = None if t.position_mono_ms is None else now - t.position_mono_ms
        clearance = self.clearance
        return GateContext(
            side="onboard",
            now_mono_ms=now,
            registry=self.registry.as_mapping(),
            limits=self.hard_limits,
            charter=self.charter,
            charter_installed=self.charter is not None,
            fence_verified=self.fence_ok is True,
            autonomy_mode=self.charter.autonomy_mode if self.charter else None,
            mission_state=None,
            paused_by=None,
            returning_cause=None,
            ai_enabled=True,
            vehicle=vehicle,
            vehicle_age_ms=age,
            snapshot_age_ms=0,
            executor_state=self.state,
            state_epoch=self.boot_id,
            last_transition_state_seq=self.last_transition_state_seq,
            preflight_ok=self.preflight_ok(),
            sdlink_up=True,
            stop_latch=self.over.stop_latch,
            latch=None,
            manual_control=self.over.manual,
            clearance=clearance,
            recent_brain_actions_mono_ms=(),
            recent_brain_reports_mono_ms=(),
            brain_pending_approvals=0,
            operator_present=True,
        )

    def on_cancel(self, command_id: UUID) -> None:
        active = self.active
        if active is None or active.command_id != command_id:
            return
        target = self.target_of(Trigger.ORDER_CANCEL)
        if isinstance(target, Refusal):
            return
        handler, origin = active.handler, active.origin
        self._drop_active(stage="superseded")
        self._enter(target, hold_reason="skill_done" if target == S.HOLD else None, cause="cancel")
        if target == S.HOLD:
            self._side(self._cancel_into_loiter(handler, origin))

    async def _cancel_into_loiter(self, handler: SkillHandler, origin: SendOrigin) -> None:
        with contextlib.suppress(ControlNotAllowedError, ForbiddenMavlinkError):
            await handler.cancel(self._io(origin))

    # --- inbound: the people in charge -----------------------------------------------------

    def on_override(self, notice: OperatorOverrideNotice) -> None:
        """STOP, FLY HOME, the battery guard or a watchdog: let go at once (T20)."""
        rule = self.over.apply_notice(notice)
        if rule is None:
            return
        now = self.now_ms()
        self.auth.notice(now)
        if rule.sets_stop_latch and self.state in ON_GROUND:
            self.clearance = None  # STOP on the ground also withdraws "clear around the drone"
        target = self.target_of(Trigger.OPERATOR_OVERRIDE)
        if isinstance(target, Refusal):
            self._save()
            return
        self._drop_active(stage="superseded", codes=("ONB_CONTROL_HELD",))
        self._enter(target, held_by=rule.held_by, cause=f"override:{notice.kind}")
        self.emit("control_taken", {"kind": notice.kind, "action_seq": notice.action_seq})

    def on_takeover(self, takeover: ManualTakeover, ref_id: UUID) -> None:
        """The pilot takes over by hand or hands back.

        The ground station puts the vehicle in LOITER and sends the sticks itself (as
        250/190, the way STOP goes). Onboard only the AI's orders pause: the running
        order stops without sending anything and the executor goes to HELD
        (``operator_manual``), following the flight controller through its telemetry.
        """
        was_manual = self.over.manual
        if not self.over.apply_takeover(takeover):
            self._ack(ref_id, "rejected", ("ONB_STALE_ORDER",))
            return
        if takeover.active:
            self.auth.notice(self.now_ms())
            target = self.target_of(Trigger.OPERATOR_OVERRIDE)
            if not isinstance(target, Refusal):
                self._drop_active(stage="superseded", codes=("ONB_CONTROL_HELD",))
                if self.state != S.FAILSAFE:
                    self._enter(target, held_by="operator_manual", cause="takeover")
            self._save()
            self._ack(ref_id, "done")
            return
        if was_manual and self.state == S.HELD and self.held_by == "operator_manual":
            self._release(ref_id, keep_latch=True)
            return
        self._ack(ref_id, "done")

    def on_release(self, release: OperatorRelease, ref_id: UUID) -> None:
        """The pilot's Resume click: the only way out of HELD (T21) and FAILSAFE (T22)."""
        if not self.over.accept_release_seq(release.action_seq):
            self._ack(ref_id, "rejected", ("ONB_STALE_ORDER",))
            return
        self._release(ref_id, keep_latch=False)

    def _release(self, ref_id: UUID, *, keep_latch: bool) -> None:
        if self.state not in (S.HELD, S.FAILSAFE):
            if not self.over.battery_rtl_active and not keep_latch:
                self.over.released()
            self._ack(ref_id, "done")
            self._save()
            return
        target = self.target_of(Trigger.OPERATOR_RELEASE)
        if isinstance(target, Refusal):
            self._ack(ref_id, "rejected", (target.code or "ONB_RELEASE_MODE_UNSUPPORTED",))
            return
        fc_mode = self.vehicle.telemetry.fc_custom_mode
        if not keep_latch:
            self.over.released()
        self.over.manual = False
        self.auth.change = None
        if target in (S.HOLD, S.RTL) and fc_mode is not None:
            self.auth.confirmed(CopterMode(fc_mode))
        self._enter(target, hold_reason="operator_released", cause="release")
        self._ack(ref_id, "done")
        if target == S.HOLD and fc_mode == int(CopterMode.BRAKE):
            self._side(self._loiter(SendOrigin.SKILL))  # out of BRAKE, compare-and-set on BRAKE

    def _failsafe_cause_gone(self) -> bool:
        charter = self.charter
        t = self.vehicle.telemetry
        if charter is None or not t.fc_connected:
            return False
        now = self.now_ms()
        gcs = t.gcs_mavlink_seen_mono_ms
        return (
            t.battery_pct is not None
            and t.battery_pct > charter.battery_rtl_pct
            and t.gps_fix_type >= 3
            and t.ekf_ok
            and inside_charter(t.lat, t.lon, charter)
            and gcs is not None
            and now - gcs <= GCS_SEEN_FOR_RELEASE_S * 1000
        )

    # --- inbound: charter, clearance and the picture of the world ----------------------------

    async def on_charter(self, install: CharterInstall) -> None:
        charter = install.charter
        if self.state in ON_GROUND:
            # On the ground the flight controller's fence is read first: it decides the
            # hard limits (a pilot may have raised them, and the fence with them).
            await self._read_fence_limits()
        codes = self._charter_problems(charter)
        if not codes and self.state not in ON_GROUND and self.charter is None:
            # After a restart in the air: the first charter is taken only when it names
            # the fence that is in the flight controller.
            if not await self._fence_matches(charter):
                codes = ["ONB_CHARTER_IN_FLIGHT"]
        elif not codes and self.state not in ON_GROUND:
            outcome = self.target_of(
                Trigger.CHARTER_INSTALL, old_charter=self.charter, new_charter=charter
            )
            if isinstance(outcome, Refusal):
                codes = [outcome.code or "ONB_CHARTER_IN_FLIGHT"]
        accepted = not codes
        self.uplink.charter_ack(
            CharterAck(
                charter_id=charter.charter_id,
                charter_version=charter.version,
                accepted=accepted,
                reason_codes=codes,
            )
        )
        if not accepted:
            return
        self.charter = charter
        self.clearance = None  # a new charter version always needs a new clearance
        if self.state in ON_GROUND:
            self._enter(S.PREFLIGHT, cause="charter")
            self._start_preflight_reads()
        else:
            self._save()

    def _charter_problems(self, charter: MissionCharter) -> list[str]:
        if charter.vehicle_id != self.options.vehicle_id:
            return ["MISSION_STATE_INVALID"]
        if charter_violations(charter, self.hard_limits):
            return ["ONB_CHARTER_EXCEEDS_HARD_LIMITS"]
        return []

    def on_clearance(self, c: ClearanceConfirmation, ref_id: UUID) -> None:
        self.clearance = ClearanceState(
            clearance_id=c.clearance_id,
            origin=c.origin,
            mission_id=c.mission_id,
            vehicle_id=c.vehicle_id,
            charter_id=c.charter_id,
            charter_version=c.charter_version,
            received_mono_ms=self.now_ms(),
            valid_for_ms=c.valid_for_ms,
            consumed=False,
        )
        self._ack(ref_id, "done")

    def on_pilot_position(self, p: PilotPosition) -> None:
        self.pilot.update(p.lat, p.lon, self.now_ms())

    def on_commander_status(self, s: CommanderStatus) -> None:
        self.commander_connected = s.connected
        self.commander_status_ms = self.now_ms()
        if s.connected and not self.limits.ai_silent:
            self.ai_silent_after_step = False  # the AI is back

    def on_link(self, up: bool, last_inbound_ms: int | None) -> None:
        if up and not self.link_up and self.link_last_inbound_ms is not None:
            self.emit("link_restored", {})
        self.link_up = up
        if last_inbound_ms is not None:
            self.link_last_inbound_ms = last_inbound_ms

    def on_connected(self, ack: HelloAck) -> None:
        """A new connection: forget the clearance; adopt a newer latch from the ground."""
        self.clearance = None
        self.commander_status_ms = None
        self.commander_connected = None
        ref = ack.operator_latch
        if ref is not None and self.over.is_new(ref.action_seq):
            self.on_override(
                OperatorOverrideNotice(
                    override_id=new_id(),
                    kind=ref.action,
                    action_seq=ref.action_seq,
                    issuer_kind="system",
                    operator_id="system:ground",
                )
            )

    def hello(self) -> Hello:
        c = self.charter
        return Hello(
            schema_major=SCHEMA_MAJOR,
            boot_id=self.boot_id,
            executor_state=self.state,
            charter_id=c.charter_id if c else None,
            charter_version=c.version if c else None,
            fc_firmware=self.vehicle.telemetry.fc_firmware,
            operator_latch=self.over.latch_ref(),
            last_action_seq=max(0, self.over.last_action_seq),
        )

    # --- preflight ---------------------------------------------------------------------

    def preflight_ok(self) -> bool:
        return bool(self.preflight) and all(self.preflight.values())

    def _start_preflight_reads(self) -> None:
        if self._preflight_task is not None and not self._preflight_task.done():
            self._preflight_task.cancel()
        self.params_ok = None
        self.fence_ok = None
        self._preflight_task = asyncio.ensure_future(self._preflight_reads())

    async def _preflight_reads(self) -> None:
        """PF6 (parameters) and PF7 (the fence), once per charter, never in flight."""
        try:
            self.params_ok = await self._check_params()
        except (ForbiddenMavlinkError, ControlNotAllowedError):
            self.params_ok = False
        try:
            items = await download(self.vehicle, FENCE, origin=SendOrigin.STARTUP)
            self.fence_ok = (
                self.charter is not None and fence_hash(items) == self.charter.fence_hash
            )
        except (MissionError, ForbiddenMavlinkError, ControlNotAllowedError):
            self.fence_ok = False

    async def _fence_matches(self, charter: MissionCharter) -> bool:
        try:
            items = await download(self.vehicle, FENCE, origin=SendOrigin.STARTUP)
        except (MissionError, ForbiddenMavlinkError, ControlNotAllowedError):
            return False
        return charter.fence_hash is not None and fence_hash(items) == charter.fence_hash

    async def _read_param(self, name: str) -> float | None:
        return await self.vehicle.read_param(name, origin=SendOrigin.STARTUP)

    async def _check_params(self) -> bool:
        ok = True
        for name, want in self.options.expected_params.items():
            got = await self._read_param(name)
            if name in _RAISABLE:
                continue  # checked against the hard floor below
            if got is None or abs(got - want) > _PARAM_TOLERANCE:
                log.warning("PF6 %s = %s, expected %s", name, got, want)
                ok = False
        floors = await self._read_fence_limits()
        return ok and floors

    async def _read_fence_limits(self) -> bool:
        """The flight controller's fence decides the hard limits onboard (raised limits
        move the fence too). PF6 needs it at least at the standard limits + margins."""
        alt = await self._read_param("FENCE_ALT_MAX")
        radius = await self._read_param("FENCE_RADIUS")
        speed = await self._read_param("LOIT_SPEED_MS")
        if alt is None or radius is None or speed is None:
            self.hard_limits = DEFAULT_LIMITS
            return False
        floor_ok = (
            alt >= ABSOLUTE_MAX_ALT_AGL_M + FENCE_ALT_MARGIN_M
            and radius >= ABSOLUTE_MAX_RANGE_FROM_PILOT_M + FENCE_RADIUS_MARGIN_M
            and speed >= ABSOLUTE_MAX_SPEED_MS
        )
        if not floor_ok:
            self.hard_limits = DEFAULT_LIMITS
            return False
        self.hard_limits = SafetyLimits(
            max_alt_agl_m=min(alt - FENCE_ALT_MARGIN_M, DEFAULT_CAPS.max_alt_agl_m),
            max_speed_ms=min(speed, DEFAULT_CAPS.max_speed_ms),
            max_range_from_pilot_m=min(
                radius - FENCE_RADIUS_MARGIN_M, DEFAULT_CAPS.max_range_from_pilot_m
            ),
        )
        return True

    def _run_preflight(self) -> None:
        charter = self.charter
        t = self.vehicle.telemetry
        now = self.now_ms()
        if charter is None:
            self.preflight = {}
            return
        home = t.home
        v = self.vehicle_state()
        plane = Plane(charter.home.lat, charter.home.lon)
        area = polygon_from_geojson(charter.area, plane)
        checks = {
            "PF1": t.fc_connected
            and t.fc_heartbeat_mono_ms is not None
            and now - t.fc_heartbeat_mono_ms <= FC_HEARTBEAT_TIMEOUT_S * 1000,
            "PF2": t.gps_fix_type >= 3 and t.gps_satellites >= _MIN_GPS_SATS,
            "PF3": t.ekf_ok,
            "PF4": home is not None
            and distance_m(home.lat, home.lon, charter.home.lat, charter.home.lon) <= HOME_MATCH_M,
            "PF5": t.battery_pct is not None
            and t.battery_pct >= charter.battery_rtl_pct + TAKEOFF_BATTERY_MARGIN_PCT,
            "PF6": self.params_ok is True,
            "PF7": self.fence_ok is True,
            "PF8": self.link_up
            and self.link_last_inbound_ms is not None
            and now - self.link_last_inbound_ms <= LINK_LOSS_DETECT_S * 1000,
            "PF9": distance_m(
                v.pilot_position.lat, v.pilot_position.lon, charter.home.lat, charter.home.lon
            )
            <= charter.max_range_from_pilot_m,
            "PF10": area.contains(plane.xy(charter.home.lat, charter.home.lon)),
            "PF11": area_reach_from_home_m(charter) <= fc_area_max_from_home_m(self.hard_limits),
        }
        if checks != self.preflight:
            changed = {k: v for k, v in checks.items() if self.preflight.get(k) != v}
            self.preflight = checks
            self.emit("preflight_changed", {"changed": changed, "ok": self.preflight_ok()})

    # --- the tick -------------------------------------------------------------------------

    async def start(self) -> None:
        """Restore what the last run saved and decide the first state: a vehicle that is
        armed or in the air is HELD, whatever the file says."""
        saved = self.store.load() if self.store is not None else None
        if saved is not None:
            self.boot_id = saved.boot_id + 1
            self.resume_point = saved.resume()
        await self.vehicle.wait_connected(3600.0)
        v = self.vehicle_state()
        if v.armed or is_airborne(v):
            self.state = S.HELD
            self.held_by = "onboard_restart"
            self.emit("onboard_restart", {"armed": v.armed, "fc_mode": v.fc_mode})
        else:
            self.state = S.IDLE
            self.resume_point = None
        self._started = True
        self._save()

    async def run(self, stop: asyncio.Event) -> None:
        if not self._started:
            await self.start()
        while not stop.is_set():
            await self.tick()
            await self.clock.sleep(TICK_S)

    async def tick(self) -> None:
        started = self.clock.monotonic()
        now = self.now_ms()
        self._last_tick_ms = now
        t = self.vehicle.telemetry
        self._finish_active()
        if self.state in AIRBORNE_CONTROLLED and not t.fc_connected:
            self._take(Trigger.FC_SILENT, held_by="fc_silent", event="fc_silent")
        if t.fc_connected:
            self._watch_landing()
            self._classify(now)
            self._keep_flying_home(now)
        if self.state in MOVING and self.charter is not None and t.fc_connected:
            if self.state == S.TAKEOFF and is_airborne(self.vehicle_state()):
                self.flown = True
            self._run_limits(now)
        if (
            self.state == S.LANDED
            and self._landed_ms is not None
            and now - self._landed_ms >= _LANDED_WAIT_MS
        ):
            target = self.target_of(Trigger.LANDED_TIMEOUT_5S)
            if not isinstance(target, Refusal) and self.charter is not None:
                self._enter(target, cause="landed")
                self._start_preflight_reads()
        if self.state == S.PREFLIGHT and now - self._last_preflight_ms >= _PREFLIGHT_EVERY_MS:
            self._last_preflight_ms = now
            self._run_preflight()
        if now - self._last_save_ms >= _SAVE_EVERY_MS:
            self._save()
        self._tick_ms.append((self.clock.monotonic() - started) * 1000)
        del self._tick_ms[:-100]

    def _flying_home(self) -> bool:
        t = self.vehicle.telemetry
        return t.fc_custom_mode is not None and CopterMode(t.fc_custom_mode) in HOME_MODES

    def _keep_flying_home(self, now: int) -> None:
        """RTL means the flight controller flies home. If it does not (SmartRTL and RTL
        were refused or never confirmed), ask again every few seconds."""
        if self.state != S.RTL or self.active is not None or self._flying_home():
            return
        if not is_airborne(self.vehicle_state()):
            return
        if self._rtl_retry_ms is not None and now - self._rtl_retry_ms < _RTL_RETRY_MS:
            return
        self._rtl_retry_ms = now
        self.emit("return_home_retry", {"fc_mode": self.vehicle.telemetry.fc_mode})
        handler = ReturnHomeHandler()
        plan = handler.plan(HOME_PARAMS, self._plan_context(new_id()))
        self._internal("return_home", handler, plan, SendOrigin.LIMIT)

    def _take(self, key: Trigger, *, held_by: str, event: str) -> None:
        target = self.target_of(key)
        if isinstance(target, Refusal):
            return
        self._drop_active(stage="superseded", codes=("ONB_CONTROL_HELD",))
        self._enter(target, held_by=held_by, cause=str(key))
        self.emit(event, {})

    def _watch_landing(self) -> None:
        v = self.vehicle_state()
        on_ground = not v.armed and v.landed_state == "on_ground"
        if not on_ground:
            return
        if self.state == S.TAKEOFF and not self.flown:
            return  # not off the ground yet
        if self.state not in AIRBORNE_CONTROLLED | {S.HELD, S.FAILSAFE}:
            return
        target = self.target_of(Trigger.LANDED_DISARMED)
        if isinstance(target, Refusal):
            return
        active = self.active
        if active is not None and active.skill == "return_home" and active.ack_done:
            self._ack(active.command_id, "done")
            active.ack_done = False
        self._drop_active(stage=None, save_resume=False)
        self.resume_point = None
        self.auth.reset()
        self.over.landed()
        self.limits.reset()
        self.clearance = None
        self.ai_silent_after_step = False
        self.flown = False
        self.control_suspended = False
        self._landed_ms = self.now_ms()
        self._enter(target, cause="landed")
        self.emit("disarmed", {})

    def _classify(self, now: int) -> None:
        """A mode change nobody here asked for: HELD at once, FAILSAFE if a failsafe
        signal comes with it (T17, T19)."""
        t = self.vehicle.telemetry
        if self.state in AIRBORNE_CONTROLLED:
            change = self.auth.unexpected_mode(t, now)
            if change is not None:
                self.auth.start_classification(change)
                held_by = (
                    self.over.held_by_for_mode_change()
                    if self.auth.notice_near_change()
                    else "external"
                )
                if change.mode in _STOPPING_MODES and held_by == "external":
                    self.over.stop_latch = True  # an unrequested BRAKE/LOITER means stop
                target = self.target_of(Trigger.UNEXPECTED_MODE)
                if not isinstance(target, Refusal):
                    self.control_suspended = True  # first, so nothing more goes out
                    self._drop_active(stage="superseded", codes=("ONB_CONTROL_HELD",))
                    self._enter(target, held_by=held_by, cause=f"mode {change.mode}")
                    self.emit("control_taken", {"fc_mode": change.mode, "held_by": held_by})
        if self.state == S.HELD and self.auth.change is not None:
            if self.auth.in_window(now):
                if self.held_by == "external" and self.auth.failsafe_near_change(t):
                    target = self.target_of(Trigger.FAILSAFE_DETECTED)
                    if not isinstance(target, Refusal):
                        self._enter(target, cause="failsafe")
                        self.control_suspended = True
                        sig = t.last_failsafe_signal
                        self.emit("fc_failsafe", {"signal": sig.name if sig else None})
                elif self.over.last_kind is not None and self.auth.notice_near_change():
                    self.held_by = self.over.held_by_for_mode_change()
            else:
                self.auth.change = None
                self.control_suspended = False
        elif (
            self.state == S.FAILSAFE
            and self.auth.change is not None
            and not self.auth.in_window(now)
        ):
            self.auth.change = None

    def _run_limits(self, now: int) -> None:
        if self.charter is None:
            raise RuntimeError("no charter installed")
        v = self.vehicle_state()
        inbound = (
            self.active is not None and self.active.skill == "goto" and self.limits.range_active
        )
        actions = self.limits.check(
            LimitInput(
                now_ms=now,
                state=self.state,
                vehicle=v,
                charter=self.charter,
                pilot=v.pilot_position,
                active_skill=self.active.skill if self.active else None,
                inbound_order_active=inbound,
                link_last_inbound_ms=self.link_last_inbound_ms,
                link_up=self.link_up,
                commander_connected=self.commander_connected,
                commander_status_ms=self.commander_status_ms,
            )
        )
        for action in actions:
            self.emit(action.event, dict(action.data))
            self._apply_limit(action)

    def _apply_limit(self, action: LimitAction) -> None:
        match action.kind:
            case "after_step":
                self.ai_silent_after_step = True
                if self.state == S.HOLD and self.active is None:
                    self.hold_reason = "ai_silent"
                    self._save()
            case "hold":
                target = self.target_of(Trigger.LIMIT_HOLD)
                if isinstance(target, Refusal):
                    return
                was = self.state
                if was == S.HOLD and self.active is None:
                    self._enter(target, hold_reason=action.hold_reason, cause=action.limit)
                    return
                self._drop_active(stage="superseded", codes=(action.limit,))
                self._enter(target, hold_reason=action.hold_reason, cause=action.limit)
                if was == S.TAKEOFF and not is_airborne(self.vehicle_state()):
                    return
                self._side(self._loiter(SendOrigin.LIMIT))
            case "rtl":
                target = self.target_of(Trigger.LIMIT_RTL)
                if isinstance(target, Refusal):
                    return
                self._drop_active(stage="superseded", codes=(action.limit,))
                self._enter(target, cause=action.limit)
                handler = ReturnHomeHandler()
                plan = handler.plan(HOME_PARAMS, self._plan_context(new_id()))
                self._internal("return_home", handler, plan, SendOrigin.LIMIT)
            case "alt":
                target = self.target_of(Trigger.LIMIT_ALT)
                if isinstance(target, Refusal) or self.charter is None:
                    return
                self._drop_active(stage="superseded", codes=(action.limit,))
                v = self.vehicle_state()
                alt = self.charter.alt_max_agl_m - _ALT_GOTO_BELOW_M
                goto = GotoHandler()
                ctx = self._plan_context(new_id())
                goto.prepare(ctx, v.lat, v.lon, alt, _ALT_GOTO_SPEED_MS)
                plan = goto_plan(ctx, v.lat, v.lon, alt, _ALT_GOTO_SPEED_MS)
                self._enter(target, cause=action.limit)
                self._internal(INTERNAL_ALT_GOTO, goto, plan, SendOrigin.LIMIT)

    # --- snapshot and persistence --------------------------------------------------------

    def snapshot(self) -> StateSnapshot:
        """The next StateSnapshot (2 Hz); ``state_seq`` grows by one per snapshot."""
        self.state_seq += 1
        v = self.vehicle_state()
        t = self.vehicle.telemetry
        charter = self.charter
        progress = None
        if self.active is not None:
            progress = self.active.handler.on_tick(
                TickContext(vehicle=v, clock=self.clock, now_mono_ms=self.now_ms())
            ).progress_pct
        ticks = sorted(self._tick_ms)
        p95 = ticks[int(0.95 * (len(ticks) - 1))] if ticks else 0.0
        return StateSnapshot(
            state_seq=self.state_seq,
            state_epoch=self.boot_id,
            last_transition_state_seq=min(self.last_transition_state_seq, self.state_seq),
            executor_state=self.state,
            hold_reason=self.hold_reason,  # type: ignore[arg-type]
            held_by=self.held_by,  # type: ignore[arg-type]
            control_suspended=self.control_suspended,
            expected_mode=self.auth.expected_mode.name if self.auth.expected_mode else None,
            flight_mode=t.fc_mode,
            armed=t.armed,
            position=LatLonAlt(
                lat=v.lat, lon=v.lon, alt_amsl_m=max(-500.0, min(9000.0, v.alt_amsl_m))
            ),
            alt_rel_home_m=v.alt_rel_home_m,
            groundspeed_ms=v.groundspeed_ms,
            heading_deg=v.heading_deg,
            battery_pct=v.battery_pct,
            active_command_id=self.active.command_id
            if self.active and self.active.ack_done
            else None,
            active_skill=self.active.skill if self.active else None,
            progress_pct=progress,
            resume_available=self.resume_point is not None,
            charter_id=charter.charter_id if charter else None,
            charter_version=charter.version if charter else None,
            clearance_available=self._clearance_valid(),
            preflight_ok=self.preflight_ok(),
            preflight=dict(self.preflight),
            limits_active=self.limits.active(),
            stop_latch=self.over.stop_latch,
            battery_rtl_active=self.over.battery_rtl_active or self.limits.battery_rtl_active,
            manual_control=self.over.manual,
            tick_ms_p95=round(p95, 2),
            sector_progress={},
        )

    def _clearance_valid(self) -> bool:
        c = self.clearance
        if c is None or c.consumed or c.origin == "remote":
            return False
        age = self.now_ms() - c.received_mono_ms
        return 0 <= age <= min(c.valid_for_ms, CLEARANCE_VALID_S * 1000)

    def _save(self) -> None:
        self._last_save_ms = self.now_ms()
        if self.store is None:
            return
        rp = self.resume_point
        c = self.charter
        try:
            self.store.save(
                {
                    "vehicle_id": self.options.vehicle_id,
                    "mission_id": str(c.mission_id) if c else None,
                    "boot_id": self.boot_id,
                    "state": str(self.state),
                    "hold_reason": self.hold_reason,
                    "held_by": self.held_by,
                    "expected_mode": self.auth.expected_mode.name
                    if self.auth.expected_mode
                    else None,
                    "charter_id": str(c.charter_id) if c else None,
                    "charter_version": c.version if c else None,
                    "active_command_id": str(self.active.command_id) if self.active else None,
                    "resume_point": rp.model_dump(mode="json") if rp is not None else None,
                }
            )
        except OSError as exc:
            log.error("could not save the executor state: %s", exc)

    async def close(self) -> None:
        self._drop_active(stage=None)
        tasks = [*self._side_tasks, *([self._preflight_task] if self._preflight_task else [])]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
