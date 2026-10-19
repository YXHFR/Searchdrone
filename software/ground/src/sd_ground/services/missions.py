"""The mission flow: one place that ties charter, approvals, fence, gate, AI and the
onboard computer together into a mission with a state.

``MissionFlow`` lives in the main loop. The routes (panel and commander) call it; the
SD-link session hands it what the onboard computer says (``on_inbound``); its ``run``
task follows the latches, the flight controller and the snapshots and moves the mission
through ``mission_state.TRANSITIONS``. It never sits on the emergency path: STOP and FLY
HOME reach the flight controller without it, and it only follows what they did.

What each part does is in its own module: ``charters`` and ``fence`` (pure checks),
``approvals`` (cards), ``commands`` (orders and the gate), ``clearance``, ``chat``,
``annotations``, ``coverage`` and ``files``. This module holds the shared state, the
state machine's side and the mission's life from opening to closing.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

from sd_ground.api.deps import ApiStatusError
from sd_ground.db.flow import FlowRepo, MissionExistsError
from sd_ground.logging import get_logger
from sd_ground.mav.thread import MavSnapshot
from sd_ground.services import fence as fence_rules
from sd_ground.services import mission_state
from sd_ground.services.clearance import Clearances
from sd_ground.services.coverage import Coverage
from sd_ground.services.gate_context import BrainRate, build_gate_context
from sd_ground.services.latches import ADOPTED_BY
from sd_ground.services.limits import check_change
from sd_mav.modes import CopterMode
from sd_mav.vehicle import Vehicle
from sd_safety.constants import OPERATOR_ABSENT_S
from sd_schemas.api import Charter, Mission, MissionDetail
from sd_schemas.base import new_id
from sd_schemas.charter import AutonomyMode, MissionCharter
from sd_schemas.executor import ExecutorState
from sd_schemas.gate import GateContext
from sd_schemas.link import Hello, StateSnapshot
from sd_schemas.mission import (
    MissionState,
    MissionStatusFull,
    OperatorLatch,
    TransitionCtx,
    Trigger,
)
from sd_schemas.safety import LimitCaps, SafetyLimits
from sd_schemas.skills import SKILL_REGISTRY, SkillSpec

if TYPE_CHECKING:
    from sd_ground.context import Ground
    from sd_ground.link import GroundLink
    from sd_ground.services.vehicle_actions import VehicleActions
    from sd_link.session import Expired, Inbound

__all__ = [
    "CLOSE_AFTER_DISARM_S",
    "TICK_S",
    "FlowError",
    "MavCaller",
    "MissionFlow",
]

TLOG_NAME = "flight.tlog"  # the ground station's own MAVLink channel, per mission

CLOSE_AFTER_DISARM_S: Final = 10.0
TICK_S: Final = 0.2
_HOME_MODES: Final = frozenset({CopterMode.RTL, CopterMode.SMART_RTL})
_LATCH_TRIGGER: Final = {
    "stop": Trigger.OPERATOR_STOP,
    "return_home": Trigger.OPERATOR_RTL,
    "battery_rtl": Trigger.BATTERY_RTL,
    "watchdog_stop": Trigger.WATCHDOG_STOP,
    "watchdog_rtl": Trigger.WATCHDOG_RTL,
}
# A latch the ground took over from the onboard computer's Hello (``GroundLink``).
_ADOPTED_TRIGGER: Final = {
    "stop": Trigger.LATCH_ADOPTED_STOP,
    "watchdog_stop": Trigger.LATCH_ADOPTED_STOP,
    "return_home": Trigger.LATCH_ADOPTED_RTL,
    "battery_rtl": Trigger.LATCH_ADOPTED_RTL,
    "watchdog_rtl": Trigger.LATCH_ADOPTED_RTL,
}
_LATCH_REASON: Final = {
    "stop": "OPERATOR_STOP",
    "return_home": "OPERATOR_RTL",
    "battery_rtl": "BATTERY_RTL",
    "watchdog_stop": "WATCHDOG_STOP",
    "watchdog_rtl": "WATCHDOG_RTL",
}
# Onboard events the panel shows as they arrive (most only go to the event log).
_PANEL_EVENTS: Final = frozenset({"executor.find_hold_ended"})
log = get_logger("sd_ground.missions")


class FlowError(ApiStatusError):
    """A refusal the routes return as is (status, code, Swedish text, optional items)."""

    def __init__(
        self, status: int, code: str, detail_sv: str, errors: list[Any] | None = None
    ) -> None:
        items = [e.model_dump(mode="json") if hasattr(e, "model_dump") else e for e in errors or []]
        super().__init__(status, code, detail_sv, items)


class MavCaller(Protocol):
    """The MAVLink thread for the fence and ``flight.tlog`` (``mav.thread.ThreadSafeMav``)."""

    def snapshot(self) -> MavSnapshot | None: ...

    def start_tlog(self, path: Path) -> bool: ...

    def stop_tlog(self) -> None: ...

    async def call[T](
        self, work: Callable[[Vehicle], Awaitable[T]], timeout_s: float = 30.0
    ) -> T: ...


@dataclass
class _VehicleWatch:
    """What the run task last saw for a vehicle."""

    latch_seq: int = 0
    latch: OperatorLatch | None = None
    resume_pending: bool = False
    executor: ExecutorState | None = None
    hold_reason: str | None = None
    disarmed_mono: float | None = None
    closing: bool = False
    armed: bool = False
    flown: bool = False  # in the air since it last stood disarmed outside a flight state
    battery_rtl: bool = False  # the onboard battery rule is flying home


@dataclass
class MissionFlow:
    g: Ground
    repo: FlowRepo
    caps: LimitCaps = field(default_factory=LimitCaps)
    registry: Mapping[str, SkillSpec] = field(default_factory=SKILL_REGISTRY.as_mapping)
    link: GroundLink | None = None
    mav: MavCaller | None = None
    actions: VehicleActions | None = None
    coverage: Coverage = field(default_factory=Coverage)
    clearances: Clearances = field(default_factory=Clearances)
    rates: dict[str, BrainRate] = field(default_factory=dict[str, BrainRate])
    finds: dict[str, Any] = field(default_factory=dict[str, Any])  # mission id -> Finds
    reports: dict[str, Any] = field(default_factory=dict[str, Any])  # vehicle id -> ReportQueue
    link_watch: dict[str, Any] = field(default_factory=dict[str, Any])  # vehicle id -> LinkWatch
    _watch: dict[str, _VehicleWatch] = field(default_factory=dict[str, _VehicleWatch])
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _tasks: set[asyncio.Task[Any]] = field(default_factory=set[asyncio.Task[Any]])
    _status: dict[str, str] = field(default_factory=dict[str, str])
    _tlog_mission: str | None = None  # whose flight.tlog the MAVLink channel records
    # Missions not to start recording for again: their files are closing, or opening the
    # file failed once (logged).
    _tlog_done: set[str] = field(default_factory=set[str])

    # --- small helpers -----------------------------------------------------------------

    @property
    def vehicle_ids(self) -> list[str]:
        return [self.g.config.default_vehicle_id]

    def now(self) -> dt.datetime:
        return self.g.clock.utc_now()

    def now_ms(self) -> int:
        return self.g.clock.monotonic_ms()

    def data_root(self) -> Path:
        return self.g.loaded.data_dir

    def mission_dir(self, mission_id: str) -> Path:
        return self.data_root() / "missions" / mission_id

    def publish(
        self,
        topic: str,
        type_: str,
        data: dict[str, Any],
        *,
        vehicle_id: str | None,
        mission_id: str | None,
    ) -> None:
        self.g.bus.publish(
            topic, type_, data, ts=self.now(), vehicle_id=vehicle_id, mission_id=mission_id
        )

    def alert(
        self,
        code: str,
        severity: str,
        message_sv: str,
        *,
        vehicle_id: str | None,
        mission_id: str | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        payload = {"code": code, "severity": severity, "message_sv": message_sv, **(details or {})}
        self.g.add_event(
            f"alarm.{code}",
            severity="critical" if severity == "critical" else "warning",
            mission_id=mission_id,
            vehicle_id=vehicle_id,
            payload=payload,
        )
        self.publish("alerts", "alert", payload, vehicle_id=vehicle_id, mission_id=mission_id)

    def spawn(self, work: Awaitable[Any], name: str) -> None:
        task = asyncio.ensure_future(work)
        task.set_name(name)
        self._tasks.add(task)
        task.add_done_callback(self._done)

    def _done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            exc = task.exception()
            log.error(
                "flow.task_failed",
                task=task.get_name(),
                error=type(exc).__name__,
                detail=str(exc)[:300],
            )

    def mav_snapshot(self, vehicle_id: str) -> MavSnapshot | None:
        snap = None if self.mav is None else self.mav.snapshot()
        return snap if snap is not None and snap.vehicle_id == vehicle_id else None

    def seen_flying(self, vehicle_id: str) -> bool:
        """Whether the run task has seen the vehicle in the air since it last stood
        disarmed outside a flight state (what makes a later disarm a landing)."""
        watch = self._watch.get(vehicle_id)
        return watch is not None and watch.flown

    def in_flight(self, vehicle_id: str) -> bool:
        snap = self.mav_snapshot(vehicle_id)
        return snap is not None and snap.connected and snap.in_flight

    def on_ground_disarmed(self, vehicle_id: str) -> bool:
        snap = self.mav_snapshot(vehicle_id)
        return (
            snap is not None and snap.connected and not snap.in_flight and not snap.telemetry.armed
        )

    def may_be_airborne(self, mission: Mission) -> bool:
        """Fail closed: true unless the flight controller reports the vehicle disarmed on
        the ground and the mission is not in a flight state (no telemetry = maybe flying).
        For the checks that only the ground may pass (limits, larger areas, clearance)."""
        status = MissionStatusFull(mission.status, mission.paused_by, mission.returning_cause)
        return status in mission_state.AIR or not self.on_ground_disarmed(mission.vehicle_id)

    def latch(self, vehicle_id: str) -> OperatorLatch | None:
        return None if self.actions is None else self.actions.latches.get(vehicle_id)

    def snapshot(self, vehicle_id: str) -> tuple[StateSnapshot | None, int | None]:
        if self.link is None:
            return None, None
        seen = self.link.snapshot_seen(vehicle_id)
        return (None, None) if seen is None else (seen.snapshot, seen.received_mono_ms)

    def sdlink_up(self, vehicle_id: str) -> bool:
        return self.link is not None and self.link.up(vehicle_id)

    def operator_present(self) -> bool:
        seen = self.g.operator_seen_mono
        return seen is not None and self.g.clock.monotonic() - seen < OPERATOR_ABSENT_S

    # --- missions ----------------------------------------------------------------------

    async def mission(self, mission_id: str) -> Mission:
        found = await self.repo.mission(mission_id)
        if found is None:
            raise FlowError(404, "MISSION_NOT_FOUND", "Uppdraget finns inte.")
        return found

    async def open_mission_of(self, mission_id: str) -> Mission:
        """The mission, which must still be open (409 ``MISSION_CLOSED`` otherwise)."""
        mission = await self.mission(mission_id)
        if mission.status is MissionState.CLOSED:
            raise FlowError(409, "MISSION_CLOSED", "Uppdraget är stängt.")
        return mission

    async def open_mission(self, vehicle_id: str, by: str) -> Mission:
        """The vehicle's open mission, or a new one in DRAFT (titled with UTC time)."""
        if vehicle_id not in self.vehicle_ids:
            raise FlowError(404, "VEHICLE_NOT_FOUND", "Drönaren finns inte.")
        found = await self.repo.open_mission(vehicle_id)
        if found is not None:
            return found
        now = self.now()
        mission_id = str(new_id())
        try:
            await self.repo.create_mission(
                mission_id=mission_id,
                vehicle_id=vehicle_id,
                title=f"Uppdrag {now:%Y-%m-%d %H:%M}",
                created_by=by,
                limits=SafetyLimits(),
                now=now,
            )
        except MissionExistsError:
            found = await self.repo.open_mission(vehicle_id)
            if found is None:
                raise
            return found
        self.mission_dir(mission_id).mkdir(parents=True, exist_ok=True)
        mission = await self.mission(mission_id)
        self.publish(
            "mission",
            "mission.state_changed",
            {"from": None, "to": "DRAFT", "created": True, "state_seq": mission.state_seq},
            vehicle_id=vehicle_id,
            mission_id=mission_id,
        )
        return mission

    def transition_ctx(self, vehicle_id: str) -> TransitionCtx:
        snap = self.mav_snapshot(vehicle_id)
        flying = snap is not None and snap.connected and snap.in_flight
        home = snap is not None and snap.fc_mode in _HOME_MODES
        return TransitionCtx(in_flight=flying, fc_home_mode=home)

    async def transition(
        self, mission_id: str, trigger: Trigger, *, closed_reason: str | None = None
    ) -> Mission | None:
        """Apply ``trigger``; the mission after a change, ``None`` when nothing changed."""
        for _ in range(3):
            mission = await self.repo.mission(mission_id)
            if mission is None:
                return None
            current = MissionStatusFull(mission.status, mission.paused_by, mission.returning_cause)
            nxt = mission_state.next_state(
                current, trigger, self.transition_ctx(mission.vehicle_id)
            )
            if nxt is None:
                return None
            seq = await self.repo.set_state(
                mission_id,
                expect_seq=mission.state_seq,
                status=str(nxt.state),
                paused_by=nxt.paused_by,
                returning_cause=nxt.returning_cause,
                trigger=str(trigger),
                previous=str(mission.status),
                now=self.now(),
                closed_reason=closed_reason,
            )
            if seq is None:
                continue  # moved meanwhile: read again
            self.publish(
                "mission",
                "mission.state_changed",
                {
                    "from": str(mission.status),
                    "to": str(nxt.state),
                    "paused_by": nxt.paused_by,
                    "returning_cause": nxt.returning_cause,
                    "trigger": str(trigger),
                    "state_seq": seq,
                },
                vehicle_id=mission.vehicle_id,
                mission_id=mission_id,
            )
            after = await self.repo.mission(mission_id)
            if after is not None:
                await self._after_transition(after, nxt.state)
            return after
        return None

    async def _after_transition(self, mission: Mission, state: MissionState) -> None:
        vid = mission.vehicle_id
        if state is MissionState.LANDED:
            await self.end_clearance(vid, "invalidated", "LANDED", str(mission.id))
            watch = self._watch.setdefault(vid, _VehicleWatch())
            watch.disarmed_mono = self.g.clock.monotonic()

    async def detail(self, mission_id: str) -> MissionDetail:
        from sd_ground.services import approvals as approvals_service
        from sd_ground.services import commands as commands_service

        mission = await self.mission(mission_id)
        charter = await self.active_charter_row(mission)
        clearance = self.clearances.get(mission.vehicle_id)
        stored = None if clearance is None else await self.repo.clearance(clearance.clearance_id)
        return MissionDetail(
            mission=mission,
            charter=charter,
            sectors=await self.sectors_geojson(mission_id),
            pending_approvals=await approvals_service.pending(self, mission_id),
            active_command=await commands_service.active_command(self, mission),
            latch=self.latch(mission.vehicle_id),
            clearance=stored,
        )

    async def sectors_geojson(self, mission_id: str) -> dict[str, Any]:
        rows = await self.repo.sectors(mission_id)
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": r["sector_id"],
                    "geometry": r["area"],
                    "properties": {
                        "sector_id": r["sector_id"],
                        "name": r["name"],
                        "status": r["status"],
                        "progress": float(r["progress"]),
                        "area_m2": r["area_m2"],
                        "covered": r["covered"],
                    },
                }
                for r in rows
            ],
        }

    async def end(self, mission_id: str, reason: str, by: str) -> Mission:
        """``POST /missions/{id}/end``: close a mission that is not in the air."""
        mission = await self.open_mission_of(mission_id)
        if self.in_flight(mission.vehicle_id):
            raise FlowError(
                409, "MISSION_IN_FLIGHT", "Uppdraget kan inte avslutas medan drönaren flyger."
            )
        self.g.add_event(
            "mission.end_requested",
            mission_id=mission_id,
            vehicle_id=mission.vehicle_id,
            payload={"by": by, "reason": reason},
        )
        return await self.close(mission, Trigger.CLOSE_MANUAL, "manual")

    async def close(self, mission: Mission, trigger: Trigger, reason: str) -> Mission:
        from sd_ground.services import files as files_service

        await self.end_clearance(
            mission.vehicle_id, "invalidated", "MISSION_CLOSED", str(mission.id)
        )
        await files_service.close_files(self, mission)
        after = await self.transition(str(mission.id), trigger, closed_reason=reason)
        self.coverage.forget(str(mission.id))
        return after or await self.mission(str(mission.id))

    # --- limits and autonomy -----------------------------------------------------------

    async def put_limits(self, mission_id: str, limits: SafetyLimits, by: str) -> Mission:
        mission = await self.open_mission_of(mission_id)
        charter = await self.active_charter(mission)
        refusal = check_change(
            limits,
            old=mission.limits,
            caps=self.caps,
            active=charter,
            in_flight=self.may_be_airborne(mission),
        )
        if refusal is not None:
            raise FlowError(refusal.status, refusal.code, refusal.detail_sv, refusal.errors)
        await self.repo.set_limits(mission_id, limits)
        self.g.add_event(
            "mission.limits_changed",
            mission_id=mission_id,
            vehicle_id=mission.vehicle_id,
            payload={"by": by, "from": mission.limits.model_dump(), "to": limits.model_dump()},
        )
        updated = await self.mission(mission_id)
        self.publish(
            "mission",
            "mission.updated",
            updated.model_dump(mode="json"),
            vehicle_id=updated.vehicle_id,
            mission_id=mission_id,
        )
        if charter is not None and fence_rules.fence_param_targets(
            mission.limits
        ) != fence_rules.fence_param_targets(limits):
            # The verified fence belongs to the old limits: check (and move) it again.
            row = await self.active_charter_row(updated)
            if row is not None:
                await self.repo.set_fence(str(row.id), row.fence_hash, None)
                self.spawn(self.install_charter(updated, str(row.id)), "install")
        return updated

    async def put_autonomy(self, mission_id: str, mode: AutonomyMode, by: str) -> Mission:
        """A new charter version with the same content and the new mode, approved by the
        pilot in this call. The clearance becomes invalid."""
        from sd_ground.services import approvals as approvals_service

        mission = await self.open_mission_of(mission_id)
        row = await self.active_charter_row(mission)
        if row is None:
            raise FlowError(409, "CHARTER_NOT_APPROVED", "Det finns ingen godkänd uppdragsram.")
        await approvals_service.new_version_with_mode(self, mission, row, mode, by)
        return await self.mission(mission_id)

    # --- charters ----------------------------------------------------------------------

    async def active_charter_row(self, mission: Mission) -> Charter | None:
        if mission.active_charter_id is None:
            return None
        return await self.repo.charter(str(mission.active_charter_id))

    async def active_charter(self, mission: Mission) -> MissionCharter | None:
        row = await self.active_charter_row(mission)
        return None if row is None else row.body

    async def install_charter(self, mission: Mission, charter_id: str) -> None:
        """After approval: the fence (unless carried over), then ``CharterInstall``.
        ``READY`` follows once both are confirmed."""
        from sd_ground.services import approvals as approvals_service

        await approvals_service.install(self, mission, charter_id)

    # --- the gate's picture ------------------------------------------------------------

    async def gate_context(self, mission: Mission) -> GateContext:
        vid = mission.vehicle_id
        row = await self.active_charter_row(mission)
        snapshot, received = self.snapshot(vid)
        active = self.clearances.get(vid)
        clearance = None
        if active is not None and active.state.mission_id == mission.id:
            clearance = active.state
        return build_gate_context(
            now_ms=self.now_ms(),
            registry=self.registry,
            mission=mission,
            charter=None if row is None else row.body,
            charter_installed=row is not None and row.install_status == "installed",
            fence_verified=row is not None and row.fence_verified_at is not None,
            mav=self.mav_snapshot(vid),
            snapshot=snapshot,
            snapshot_received_ms=received,
            sdlink_up=self.sdlink_up(vid),
            latch=self.latch(vid),
            clearance=clearance,
            rate=self.rates.setdefault(vid, BrainRate()),
            brain_pending_approvals=await self.repo.brain_pending_approvals(vid),
            operator_present=self.operator_present(),
        )

    # --- clearance ---------------------------------------------------------------------

    async def end_clearance(
        self, vehicle_id: str, status: str, reason: str, mission_id: str | None
    ) -> None:
        dropped = self.clearances.drop(vehicle_id)
        count = await self.repo.end_clearances(vehicle_id, status, reason, self.now())
        if dropped is not None or count:
            self.publish(
                "mission",
                "clearance.updated",
                {"status": status, "confirmed_by": None, "valid_until": None, "reason": reason},
                vehicle_id=vehicle_id,
                mission_id=mission_id,
            )

    # --- SD-link: what the onboard computer says ----------------------------------------

    async def on_connected(self, vehicle_id: str, hello: Hello) -> None:
        from sd_ground.services import link_loss as link_loss_service

        await link_loss_service.on_connected(self, vehicle_id)
        mission = await self.repo.open_mission(vehicle_id)
        if mission is None:
            return
        row = await self.active_charter_row(mission)
        if row is not None and row.fence_hash is not None:
            from sd_ground.services import approvals as approvals_service

            await approvals_service.send_install(self, mission, row)

    async def on_disconnected(self, vehicle_id: str) -> None:
        return None

    async def on_expired(self, vehicle_id: str, expired: Expired) -> None:
        from sd_ground.services import commands as commands_service

        await commands_service.on_expired(self, vehicle_id, expired)

    async def on_inbound(self, vehicle_id: str, inbound: Inbound) -> None:
        from sd_ground.services import approvals as approvals_service
        from sd_ground.services import commands as commands_service
        from sd_ground.services import perception as perception_service
        from sd_schemas.link import Ack, CharterAck, CropData, OnboardEvent, PerceptionReport

        body = inbound.body
        if isinstance(body, Ack):
            await commands_service.on_ack(self, vehicle_id, body)
        elif isinstance(body, CharterAck):
            await approvals_service.on_charter_ack(self, vehicle_id, body)
        elif isinstance(body, OnboardEvent):
            mission = await self.repo.open_mission(vehicle_id)
            self.g.add_event(
                body.type,
                severity=body.severity,
                mission_id=None if mission is None else str(mission.id),
                vehicle_id=vehicle_id,
                payload={"source": "onboard", **body.data},
            )
            if body.type in _PANEL_EVENTS:  # the pilot's log line, after the link is back
                self.publish(
                    "vehicle",
                    body.type,
                    dict(body.data),
                    vehicle_id=vehicle_id,
                    mission_id=None if mission is None else str(mission.id),
                )
        elif isinstance(body, StateSnapshot):
            await self._on_snapshot(vehicle_id, body)
        elif isinstance(body, PerceptionReport):
            perception_service.enqueue(self, vehicle_id, body)  # never waits for storage
        elif isinstance(body, CropData):
            perception_service.add_crop(self, vehicle_id, body)  # kept until its report is stored

    async def _on_snapshot(self, vehicle_id: str, snap: StateSnapshot) -> None:
        from sd_ground.services import commands as commands_service

        watch = self._watch.setdefault(vehicle_id, _VehicleWatch())
        mission = await self.repo.open_mission(vehicle_id)
        if mission is None:
            watch.executor = snap.executor_state
            return
        mid = str(mission.id)
        before, before_reason = watch.executor, watch.hold_reason
        watch.executor, watch.hold_reason = snap.executor_state, snap.hold_reason
        if watch.resume_pending and not snap.stop_latch:
            watch.resume_pending = False
            await self.transition(mid, Trigger.OPERATOR_RESUME_CONFIRMED)
        state = snap.executor_state
        if (
            state is ExecutorState.HOLD
            and snap.hold_reason == "ai_hold"
            and (before is not ExecutorState.HOLD or before_reason != "ai_hold")
        ):
            await self.transition(mid, Trigger.BRAIN_HOLD)
        elif before is ExecutorState.HOLD and state in (
            ExecutorState.TRANSIT,
            ExecutorState.SURVEY,
        ):
            await self.transition(mid, Trigger.BRAIN_HOLD_END)
        taken = state is ExecutorState.HELD and snap.held_by in ("external", "onboard_restart")
        if taken and before is not ExecutorState.HELD:
            await self.transition(mid, Trigger.CONTROL_TAKEN)
        if state is ExecutorState.FAILSAFE and before is not ExecutorState.FAILSAFE:
            await self.transition(mid, Trigger.FC_FAILSAFE)
        if snap.battery_rtl_active and not watch.battery_rtl:
            # The onboard battery rule flies home by itself: the mission returns, the
            # pilot is told and the AI's orders meet the battery failsafe.
            changed = await self.transition(mid, Trigger.BATTERY_RTL)
            if changed is not None:
                self.alert(
                    "BATTERY_RTL",
                    "critical",
                    "Batteriet är nere på reserven. Drönaren flyger hem.",
                    vehicle_id=vehicle_id,
                    mission_id=mid,
                    details={"battery_pct": snap.battery_pct, "source": "onboard"},
                )
                await commands_service.cancel_open(self, vehicle_id, "BATTERY_RTL")
        watch.battery_rtl = snap.battery_rtl_active
        await commands_service.on_snapshot(self, mission, snap)

    # --- the run task ------------------------------------------------------------------

    async def start(self) -> None:
        """At start: waiting order cards and clearances from before are void."""
        now = self.now()
        await self._at_start("expire_cards", self.repo.expire_all_pending(now))
        await self._at_start("invalidate_clearances", self.repo.invalidate_all_clearances(now))
        for vid in self.vehicle_ids:
            await self._at_start("ensure_vehicle", self.repo.ensure_vehicle(vid, "sim", 1))
            if self.actions is not None:
                seq = await self._at_start("latch_seq", self.repo.latch_seq(vid))
                if isinstance(seq, int):
                    self.actions.latches.raise_seq(vid, seq)
            latch = self.latch(vid)
            self._watch[vid] = _VehicleWatch(
                latch_seq=0 if latch is None else latch.action_seq, latch=latch
            )

    async def _at_start(self, step: str, work: Awaitable[Any]) -> Any:
        """One start step; a failure is logged and the next step still runs (the
        database may be down at start: the tick catches up once it is back)."""
        try:
            return await work
        except Exception as exc:
            log.error("flow.start_step_failed", step=step, error=type(exc).__name__)
            return None

    async def run(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception as exc:  # the flow must keep following the vehicle
                log.error("flow.tick_failed", error=type(exc).__name__, detail=str(exc)[:300])
            await self.g.clock.sleep(TICK_S)

    async def tick(self) -> None:
        from sd_ground.services import approvals as approvals_service

        if not self.g.db_ok:
            return  # the mission lives in the database; STOP / FLY HOME do not need it
        from sd_ground.services import coverage as coverage_service
        from sd_ground.services import link_loss as link_loss_service

        await approvals_service.expire(self)
        await self._expire_clearances()
        for vid in self.vehicle_ids:
            mission = await self.repo.open_mission(vid)
            await self._follow_latch(vid, mission)
            await self._follow_arming(vid, mission)
            if mission is None:
                continue
            await self._follow_landing(vid, mission)
            await coverage_service.tick(self, mission)
            await link_loss_service.tick(self, mission)
            self._follow_tlog(mission)

    def _follow_tlog(self, mission: Mission) -> None:
        """``flight.tlog`` in the mission's folder from its first activation until its
        files close (also after a restart of this process: the file is appended to)."""
        mid = str(mission.id)
        if self.mav is None or mid == self._tlog_mission or mid in self._tlog_done:
            return
        if mission.activated_at is None or mission.status in (
            MissionState.DRAFT,
            MissionState.CLOSED,
        ):
            return
        # A log must never stop the flow from following the vehicle: one try per mission.
        self._tlog_done.add(mid)
        path = self.mission_dir(mid) / TLOG_NAME
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            started = self.mav.start_tlog(path)
        except OSError as exc:
            log.error("files.tlog_failed", mission_id=mid, error=type(exc).__name__)
            return
        if started:
            self._tlog_mission = mid
            self._tlog_done.discard(mid)

    def stop_tlog(self, mission_id: str) -> bool:
        """Stop recording the mission's ``flight.tlog``; ``True`` when this process wrote it."""
        self._tlog_done.add(mission_id)  # its files are closing: never start it again
        if self.mav is None or self._tlog_mission != mission_id:
            return False
        self.mav.stop_tlog()
        self._tlog_mission = None
        return True

    def mission_active(self, vehicle_id: str) -> bool:
        """For the operator-absence guard: the mission is ACTIVE or PAUSED by the AI."""
        return self._status.get(vehicle_id) in ("ACTIVE", "PAUSED:brain")

    async def _expire_clearances(self) -> None:
        for vid, active in self.clearances.expired(self.now_ms()):
            await self.repo.end_clearances(vid, "expired", "EXPIRED", self.now())
            self.publish(
                "mission",
                "clearance.updated",
                {"status": "expired", "confirmed_by": None, "valid_until": None},
                vehicle_id=vid,
                mission_id=str(active.state.mission_id),
            )

    async def _follow_latch(self, vehicle_id: str, mission: Mission | None) -> None:
        from sd_ground.services import commands as commands_service

        watch = self._watch.setdefault(vehicle_id, _VehicleWatch())
        if mission is not None:
            key = str(mission.status) + (f":{mission.paused_by}" if mission.paused_by else "")
            self._status[vehicle_id] = key
        else:
            self._status.pop(vehicle_id, None)
        latch = self.latch(vehicle_id)
        seq = 0 if self.actions is None else self.actions.latches.override_seq(vehicle_id)
        if seq == watch.latch_seq and latch == watch.latch:
            return
        old = watch.latch
        watch.latch_seq, watch.latch = seq, latch
        mid = None if mission is None else str(mission.id)
        with contextlib.suppress(Exception):
            await self.repo.mirror_latch(vehicle_id, latch, seq, mid)
        via = "button"
        if latch is not None and self.actions is not None:
            via = self.actions.latest_via(vehicle_id) or "button"
        self.publish(
            "vehicle",
            "latch.updated",
            {
                "active": latch is not None,
                "action": None if latch is None else latch.action,
                "set_by": None if latch is None else latch.set_by,
                "action_seq": seq,
                "via": via,
            },
            vehicle_id=vehicle_id,
            mission_id=mid,
        )
        if mission is None:
            return
        if latch is not None and (old is None or old.action_seq != latch.action_seq):
            adopted = latch.set_by == ADOPTED_BY
            trigger = (_ADOPTED_TRIGGER if adopted else _LATCH_TRIGGER)[latch.action]
            await self.transition(mid or "", trigger)
            await commands_service.cancel_open(self, vehicle_id, _LATCH_REASON[latch.action])
        elif latch is None and old is not None:
            watch.resume_pending = True

    async def _follow_arming(self, vehicle_id: str, mission: Mission | None) -> None:
        """Disarming ends a "clear around the drone" confirmation that is still waiting."""
        watch = self._watch.setdefault(vehicle_id, _VehicleWatch())
        snap = self.mav_snapshot(vehicle_id)
        armed = snap is not None and snap.connected and snap.telemetry.armed
        if watch.armed and not armed and self.clearances.get(vehicle_id) is not None:
            mid = None if mission is None else str(mission.id)
            await self.end_clearance(vehicle_id, "invalidated", "DISARMED", mid)
        watch.armed = armed

    async def _follow_landing(self, vehicle_id: str, mission: Mission) -> None:
        """Landed = it has been in the air during this mission and now stands disarmed
        on the ground. (Right after the take-off is accepted the motors are not running
        yet: that is not a landing.)"""
        watch = self._watch.setdefault(vehicle_id, _VehicleWatch())
        status = MissionStatusFull(mission.status, mission.paused_by, mission.returning_cause)
        air = status in mission_state.AIR
        if self.in_flight(vehicle_id):
            watch.flown = True  # also when seen before the state turned ACTIVE
        elif not air and self.on_ground_disarmed(vehicle_id):
            watch.flown = False
        if air and watch.flown and self.on_ground_disarmed(vehicle_id):
            watch.flown = False
            await self.transition(str(mission.id), Trigger.LANDED_DISARMED)
            return
        if mission.status is MissionState.LANDED and not watch.closing:
            since = watch.disarmed_mono
            if since is None:
                watch.disarmed_mono = self.g.clock.monotonic()
                return
            if self.g.clock.monotonic() - since >= CLOSE_AFTER_DISARM_S:
                watch.closing = True
                try:
                    await self.close(mission, Trigger.CLOSE_AUTO, "auto")
                finally:
                    watch.closing = False
                    watch.disarmed_mono = None

    async def takeoff_failed(self, mission: Mission) -> None:
        """The take-off was accepted but never got the vehicle up: back on the ground."""
        watch = self._watch.setdefault(mission.vehicle_id, _VehicleWatch())
        if not watch.flown and self.on_ground_disarmed(mission.vehicle_id):
            await self.transition(str(mission.id), Trigger.LANDED_DISARMED)

    def note_resume(self, vehicle_id: str) -> None:
        """Resume was pressed: the next snapshot without the onboard latch confirms it."""
        self._watch.setdefault(vehicle_id, _VehicleWatch()).resume_pending = True

    async def record_resume(self, vehicle_id: str, reply: Any, by: str) -> None:
        """Resume is kept for the record as an approved card (``kind = resume``)."""
        self.note_resume(vehicle_id)
        mission = await self.repo.open_mission(vehicle_id)
        if mission is None:
            return
        now = self.now()
        with contextlib.suppress(Exception):
            await self.repo.add_approval(
                {
                    "id": str(new_id()),
                    "mission_id": str(mission.id),
                    "vehicle_id": vehicle_id,
                    "kind": "resume",
                    "subject_id": str(reply.action_id),
                    "subject_version": 1,
                    "status": "approved",
                    "summary_sv": "Fortsätt efter stopp",
                    "requested_by": by,
                    "requested_at": now,
                    "decided_by": by,
                    "decided_at": now,
                }
            )

    # --- the AI's picture --------------------------------------------------------------

    async def situation(self, mission_id: str) -> Any:
        """``GET /commander/v1/missions/{id}/situation``."""
        from sd_ground.services import approvals as approvals_service
        from sd_ground.services import commands as commands_service
        from sd_ground.services.situation import build_situation

        mission = await self.mission(mission_id)
        row = await self.active_charter_row(mission)
        proposals = [c for c in await self.repo.charters(mission_id) if c.status == "proposed"]
        annotations = [] if self.g.repo is None else await self.g.repo.annotations(mission_id)
        events = (
            [] if self.g.repo is None else await self.g.repo.events(mission_id, None, None, 1000)
        )
        snapshot, _ = self.snapshot(mission.vehicle_id)
        clearance = self.clearances.get(mission.vehicle_id)
        return build_situation(
            mission=mission,
            charter=row,
            pending_proposal=proposals[-1] if proposals else None,
            pending_approvals=await approvals_service.pending(self, mission_id),
            clearance=clearance,
            now=self.now(),
            now_ms=self.now_ms(),
            latch=self.latch(mission.vehicle_id),
            mav=self.mav_snapshot(mission.vehicle_id),
            snapshot=snapshot,
            sdlink_up=self.sdlink_up(mission.vehicle_id),
            active_command=await commands_service.active_command(self, mission),
            sectors=await self.repo.sectors(mission_id),
            annotations=annotations,
            recent_events=events[-20:],
        )
