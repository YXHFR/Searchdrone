"""STOP, FLY HOME, Resume, the system's own actions and the pilot's takeover.

``VehicleActions`` is the ground station's emergency path (``sources.EmergencyPath``).
The buttons, the CLI socket and the chat's stop path all call ``dispatch``; it runs in
whatever thread calls it (the ingress thread for the buttons) and touches only memory
that is safe across threads: the latches, the MAVLink thread's queue, the outcome cache
and the queue for the main loop. It never waits for the database, the AI or the main
loop, and nothing here goes through the gate.

1. ``dispatch``: a repeated ``client_action_id`` gets the same answer. Otherwise: take
   the next ``action_seq``, set the latch, hand the steps to the MAVLink thread and
   queue the follow-up work. 202 at once; 503 ``NO_PATH`` only when the MAVLink thread
   has seen no flight-controller heartbeat for 3 s and SD-link is not up either.
2. The MAVLink thread sends the steps (``sd_ground.mav.gcs``) and reports back.
3. ``pump`` (main loop): sends the ``OperatorOverrideNotice`` over SD-link, writes the
   events, and publishes ``vehicle.action`` on the WebSocket when the result is known.

Who may press what (F24): every logged-in user may press STOP; FLY HOME, Resume and
steering by hand are for the pilot only (checked by ``sd_ground.emergency`` and the
routes). The battery guard, the companion watchdog and the operator-absence guard use
the same path with their own latch kinds (``system``). Only Resume lifts a latch, and
no chat text ever creates a Resume.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

from sd_ground.auth import Principal
from sd_ground.bus import EventBus
from sd_ground.logging import get_logger
from sd_ground.mav.thread import ActionReport, MavSnapshot, ModeAction
from sd_ground.services.latches import LatchChange, Latches, LatchRefusedError
from sd_ground.sources import EmergencyAction, EmergencyUnavailableError
from sd_obs.clock import Clock
from sd_schemas.api import ActionDispatched, ActionOutcome, ManualControlInput
from sd_schemas.base import SdModel, new_id
from sd_schemas.link import ManualTakeover, OperatorOverrideNotice, OperatorRelease, StickAxes
from sd_schemas.mission import LatchAction
from sd_schemas.sv_commands import OverrideIntent, classify_override

__all__ = [
    "OUTCOMES_MAX",
    "PUMP_S",
    "ActionRefusedError",
    "GroundView",
    "MavBackend",
    "NoNoticeSink",
    "NoticeSink",
    "OperatorPresence",
    "StopCheck",
    "VehicleActions",
]

OUTCOMES_MAX: Final = 512
PUMP_S: Final = 0.05
SYSTEM: Final = "system:ground"
log = get_logger("sd_ground.actions")

Via = Literal["button", "chat", "cli", "system"]
OutcomeAction = Literal[
    "stop",
    "return_home",
    "resume",
    "battery_rtl",
    "watchdog_stop",
    "watchdog_rtl",
    "takeover",
    "handback",
]


class ActionRefusedError(Exception):
    """Resume or a takeover that is not allowed now (409)."""

    def __init__(self, status: int, code: str, detail_sv: str) -> None:
        super().__init__(code)
        self.status, self.code, self.detail_sv = status, code, detail_sv


class MavBackend(Protocol):
    """The MAVLink thread as seen from other threads (``mav.thread.ThreadSafeMav``)."""

    def submit(self, action: ModeAction) -> bool: ...

    def takeover(self, takeover_id: str) -> bool: ...

    def handback(self) -> bool: ...

    def sticks(self, axes: StickAxes) -> bool: ...

    def snapshot(self) -> MavSnapshot | None: ...


class NoticeSink(Protocol):
    """SD-link towards the onboard computer (reliable messages)."""

    def up(self, vehicle_id: str) -> bool: ...

    async def send(self, vehicle_id: str, message: SdModel) -> bool: ...


class NoNoticeSink:
    """Until the ground's SD-link session is wired in: nothing reaches the onboard side."""

    def up(self, vehicle_id: str) -> bool:
        return False

    async def send(self, vehicle_id: str, message: SdModel) -> bool:
        return False


class GroundView(Protocol):
    """What the main loop's part needs from ``sd_ground.context.Ground`` (kept apart so
    the emergency path never imports the shared context)."""

    @property
    def clock(self) -> Clock: ...

    @property
    def bus(self) -> EventBus: ...

    @property
    def operator_seen_mono(self) -> float | None: ...

    def add_event(
        self,
        type_: str,
        *,
        severity: str = "info",
        mission_id: str | None = None,
        vehicle_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None: ...


class StopCheck(Protocol):
    """The quick stop check of the decision interface: "stop", "return_home" or ``None``."""

    def __call__(self, text: str) -> Awaitable[OverrideIntent | None]: ...


@dataclass
class _Post:
    """Work for the main loop after an action (never awaited by the emergency path)."""

    outcome: ActionOutcome
    notice: SdModel | None


@dataclass
class _Report:
    report: ActionReport


@dataclass
class _Event:
    type: str
    severity: str
    vehicle_id: str | None
    payload: dict[str, Any] = field(default_factory=dict[str, Any])


class VehicleActions:
    def __init__(
        self,
        clock: Clock,
        latches: Latches,
        *,
        vehicle_ids: list[str],
        notices: NoticeSink | None = None,
    ) -> None:
        self.clock = clock
        self.latches = latches
        self.vehicle_ids = list(vehicle_ids)
        self.notices: NoticeSink = notices or NoNoticeSink()
        self.mav: MavBackend | None = None  # set once the MAVLink thread runs
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._outcomes: OrderedDict[str, ActionOutcome] = OrderedDict()
        self._by_client: OrderedDict[tuple[str, str, str], ActionDispatched] = OrderedDict()
        self.post: queue.SimpleQueue[_Post | _Report | _Event] = queue.SimpleQueue()

    # --- the emergency path (sources.EmergencyPath) -----------------------------------

    def vehicles(self) -> list[str]:
        """Vehicles with a live MAVLink or SD-link connection (for stop_all)."""
        live = []
        for vid in self.vehicle_ids:
            snap = self._snapshot(vid)
            if (snap is not None and snap.connected) or self.notices.up(vid):
                live.append(vid)
        return live

    def dispatch(
        self,
        action: EmergencyAction,
        vehicle_id: str,
        *,
        client_action_id: str,
        via: Literal["button", "chat", "cli"],
        by: Principal,
    ) -> ActionDispatched:
        """STOP or FLY HOME now. Raises ``EmergencyUnavailableError``."""
        return self._start(action, vehicle_id, client_action_id, via, by.issuer_ref, "operator")

    def system(self, vehicle_id: str, kind: LatchAction) -> ActionDispatched | None:
        """The battery guard, the watchdog or the absence guard (any thread)."""
        try:
            return self._start(kind, vehicle_id, str(new_id()), "system", SYSTEM, "system")
        except EmergencyUnavailableError:
            log.error("actions.system_no_path", kind=kind, vehicle_id=vehicle_id)
            return None

    def _snapshot(self, vehicle_id: str) -> MavSnapshot | None:
        mav = self.mav
        snap = None if mav is None else mav.snapshot()
        return snap if snap is not None and snap.vehicle_id == vehicle_id else None

    def _known(self, vehicle_id: str) -> None:
        if vehicle_id not in self.vehicle_ids:
            raise EmergencyUnavailableError(404, "VEHICLE_NOT_FOUND", "Drönaren finns inte.")

    def _start(
        self,
        kind: LatchAction,
        vehicle_id: str,
        client_action_id: str,
        via: Via,
        by: str,
        issuer_kind: Literal["operator", "system"],
    ) -> ActionDispatched:
        with self._start_lock:  # one action at a time: dedupe, number, latch, hand-off
            return self._start_locked(kind, vehicle_id, client_action_id, via, by, issuer_kind)

    def _start_locked(
        self,
        kind: LatchAction,
        vehicle_id: str,
        client_action_id: str,
        via: Via,
        by: str,
        issuer_kind: Literal["operator", "system"],
    ) -> ActionDispatched:
        key = (vehicle_id, kind, client_action_id)
        with self._lock:
            seen = self._by_client.get(key)
        if seen is not None:
            return seen
        self._known(vehicle_id)
        snap = self._snapshot(vehicle_id)
        mav_ok = snap is not None and snap.connected
        sd_ok = self.notices.up(vehicle_id)
        if not mav_ok and not sd_ok:
            raise EmergencyUnavailableError(
                503,
                "NO_PATH",
                "Ingen väg till drönaren just nu. Använd RC-sändaren; flygkontrollens egen "
                "failsafe tar hem den om markstationen är borta.",
            )
        seq = self.latches.start(vehicle_id, kind, by).action_seq
        action_id = str(new_id())
        submitted = self.mav is not None and self.mav.submit(
            ModeAction(action_id, vehicle_id, kind, seq)
        )
        mav_ok = mav_ok and submitted
        paths: dict[str, Literal["queued", "unavailable"]] = {
            "mavlink": "queued" if mav_ok else "unavailable",
            "sdlink": "queued" if sd_ok else "unavailable",
        }
        reply = ActionDispatched.model_validate(
            {"action_id": action_id, "action_seq": seq, "paths": paths}
        )
        outcome = self._outcome(action_id, vehicle_id, kind, seq, via, by, dict(paths))
        notice = OperatorOverrideNotice(
            override_id=new_id(),
            kind=kind,
            action_seq=seq,
            issuer_kind=issuer_kind,
            operator_id=by,
        )
        self._remember(key, reply)
        self.post.put(_Post(outcome, notice))
        return reply

    def _remember(self, key: tuple[str, str, str], reply: ActionDispatched) -> None:
        with self._lock:
            self._by_client[key] = reply
            while len(self._by_client) > OUTCOMES_MAX:
                self._by_client.popitem(last=False)

    def _outcome(
        self,
        action_id: str,
        vehicle_id: str,
        action: OutcomeAction,
        seq: int,
        via: Via,
        by: str,
        paths: dict[str, Any],
        *,
        result: str = "dispatched",
        detail_sv: str | None = None,
    ) -> ActionOutcome:
        outcome = ActionOutcome.model_validate(
            {
                "action_id": action_id,
                "vehicle_id": vehicle_id,
                "action": action,
                "action_seq": seq,
                "via": via,
                "by": by,
                "result": result,
                "mode_requested": None,
                "mode_confirmed": None,
                "paths": paths,
                "detail_sv": detail_sv,
                "t": self.clock.utc_now(),
            }
        )
        self._store(outcome)
        return outcome

    def _store(self, outcome: ActionOutcome) -> None:
        with self._lock:
            self._outcomes[str(outcome.action_id)] = outcome
            self._outcomes.move_to_end(str(outcome.action_id))
            while len(self._outcomes) > OUTCOMES_MAX:
                self._outcomes.popitem(last=False)

    def outcome(self, vehicle_id: str, action_id: str) -> ActionOutcome | None:
        with self._lock:
            found = self._outcomes.get(action_id)
        return found if found is not None and found.vehicle_id == vehicle_id else None

    # --- called from the MAVLink thread -----------------------------------------------

    def on_report(self, report: ActionReport) -> None:
        self.post.put(_Report(report))

    def on_event(self, type_: str, severity: str, payload: dict[str, Any]) -> None:
        """An event from the MAVLink thread; ``payload["vehicle_id"]`` names the vehicle."""
        vid = payload.get("vehicle_id")
        self.post.put(_Event(type_, severity, vid if isinstance(vid, str) else None, payload))

    # --- Resume and steering by hand (pilot only; the routes check the role) -----------

    def resume(self, vehicle_id: str, *, client_action_id: str, by: Principal) -> ActionDispatched:
        """Lift the latch and send ``OperatorRelease``. Never through MAVLink."""
        return self._pilot_action("resume", vehicle_id, client_action_id, by)

    def takeover(
        self, vehicle_id: str, *, client_action_id: str, by: Principal
    ) -> ActionDispatched:
        """Ta över: LOITER now, the AI's orders pause onboard, the sticks come from here."""
        return self._pilot_action("takeover", vehicle_id, client_action_id, by)

    def handback(
        self, vehicle_id: str, *, client_action_id: str, by: Principal
    ) -> ActionDispatched:
        """Lämna tillbaka: no more sticks; the AI may lead again."""
        return self._pilot_action("handback", vehicle_id, client_action_id, by)

    def _pilot_action(
        self,
        action: Literal["resume", "takeover", "handback"],
        vehicle_id: str,
        client_action_id: str,
        by: Principal,
    ) -> ActionDispatched:
        if by.role != "pilot":
            raise ActionRefusedError(403, "FORBIDDEN_ROLE", "Bara piloten får göra det.")
        with self._start_lock:
            return self._pilot_locked(action, vehicle_id, client_action_id, by)

    def _pilot_locked(
        self,
        action: Literal["resume", "takeover", "handback"],
        vehicle_id: str,
        client_action_id: str,
        by: Principal,
    ) -> ActionDispatched:
        key = (vehicle_id, action, client_action_id)
        with self._lock:
            seen = self._by_client.get(key)
        if seen is not None:
            return seen
        try:
            self._known(vehicle_id)
        except EmergencyUnavailableError as exc:
            raise ActionRefusedError(exc.status, exc.code, exc.detail_sv) from None
        action_id = str(new_id())
        notice: SdModel
        try:
            if action == "resume":
                seq, _lifted = self.latches.release(vehicle_id)  # number and lift in one step
            elif action == "takeover":
                self.latches.check_release(vehicle_id)  # not while flying home
                seq = self.latches.next_seq(vehicle_id)
            else:  # hand-back is always allowed; it stops no running chain
                seq = self.latches.next_seq(vehicle_id)
        except LatchRefusedError as exc:
            raise ActionRefusedError(409, exc.code, exc.detail_sv) from None
        if action == "resume":
            notice = OperatorRelease(action_seq=seq, operator_id=by.issuer_ref)
        else:
            notice = ManualTakeover.model_validate(
                {
                    "takeover_id": action_id,
                    "active": action == "takeover",
                    "action_seq": seq,
                    "operator_id": by.issuer_ref,
                }
            )
            if self.mav is not None:
                if action == "takeover":
                    self.mav.takeover(action_id)
                else:
                    self.mav.handback()
        sd_ok = self.notices.up(vehicle_id)
        paths: dict[str, Literal["queued", "unavailable"]] = {
            "sdlink": "queued" if sd_ok else "unavailable"
        }
        if action == "takeover":
            snap = self._snapshot(vehicle_id)
            paths["mavlink"] = "queued" if snap is not None and snap.connected else "unavailable"
        reply = ActionDispatched.model_validate(
            {"action_id": action_id, "action_seq": seq, "paths": paths}
        )
        outcome = self._outcome(
            action_id, vehicle_id, action, seq, "button", by.issuer_ref, dict(paths)
        )
        self._remember(key, reply)
        self.post.put(_Post(outcome, notice))
        return reply

    def manual_input(self, vehicle_id: str, value: ManualControlInput, by: Principal) -> bool:
        """One stick value from the panel (10-20 Hz). Only the pilot steers."""
        if by.role != "pilot" or vehicle_id not in self.vehicle_ids or self.mav is None:
            return False
        axes = StickAxes.model_validate(value.model_dump(exclude={"source"}))
        return self.mav.sticks(axes)

    # --- the chat's stop path ---------------------------------------------------------

    async def chat_override(
        self,
        text: str,
        *,
        vehicle_id: str,
        message_id: str,
        by: Principal,
        stop_check: StopCheck | None = None,
    ) -> ActionDispatched | None:
        """A pilot's chat line: STOP or FLY HOME when the stop check says so.

        The decision interface's quick stop check decides (no fixed word list); without
        an answer nothing happens and the line goes to the AI as usual. Never Resume.
        """
        intent = classify_override(text)
        if intent is None and stop_check is not None:
            try:
                intent = await stop_check(text)
            except Exception as exc:  # a broken check must not stop the chat
                log.warning("actions.stop_check_failed", error=type(exc).__name__)
                intent = None
        if intent not in ("stop", "return_home"):
            return None
        return self.chat_intent(intent, vehicle_id=vehicle_id, message_id=message_id, by=by)

    def chat_intent(
        self, intent: str, *, vehicle_id: str, message_id: str, by: Principal
    ) -> ActionDispatched:
        """The stop check said yes for the pilot's line ``message_id``: same path as the
        buttons, with ``via = "chat"``. The same line never acts twice."""
        if intent not in ("stop", "return_home"):
            raise ValueError("the chat can only stop or fly home")
        if by.role != "pilot":
            raise EmergencyUnavailableError(403, "FORBIDDEN_ROLE", "Bara pilotens rader räknas.")
        action: EmergencyAction = "stop" if intent == "stop" else "return_home"
        return self.dispatch(
            action, vehicle_id, client_action_id=f"chat:{message_id}", via="chat", by=by
        )

    # --- the main loop's part ---------------------------------------------------------

    async def pump(self, g: GroundView) -> None:
        """Runs in the main loop: notices, events and WebSocket updates."""
        while True:
            try:
                await self.drain(g)
            except Exception as exc:  # keep pumping
                log.error("actions.pump_failed", error=type(exc).__name__)
            await self.clock.sleep(PUMP_S)

    async def drain(self, g: GroundView) -> None:
        while True:
            try:
                change = self.latches.changes.get_nowait()
            except queue.Empty:
                break
            _latch_event(g, change)
        while True:
            try:
                item = self.post.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, _Post):
                await self._handle_post(g, item)
            elif isinstance(item, _Report):
                self._handle_report(g, item.report)
            else:
                g.add_event(
                    item.type,
                    severity=item.severity,
                    vehicle_id=item.vehicle_id,
                    payload=item.payload,
                )
                if item.type.startswith(("alarm.", "vehicle.")):
                    _publish(g, "vehicle", item.type, item.payload, item.vehicle_id)

    async def _handle_post(self, g: GroundView, item: _Post) -> None:
        o = item.outcome
        vid = str(o.vehicle_id)
        if item.notice is not None and self.notices.up(vid):
            sent = False
            try:
                sent = await asyncio.wait_for(self.notices.send(vid, item.notice), 1.0)
            except Exception as exc:
                log.warning("actions.notice_failed", error=type(exc).__name__)
            o = o.model_copy(update={"paths": {**o.paths, "sdlink": "sent" if sent else "failed"}})
            self._store(o)
        g.add_event(
            f"vehicle.{o.action}",
            severity="warning" if o.via == "system" else "info",
            vehicle_id=vid,
            payload={
                "action_id": str(o.action_id),
                "action_seq": o.action_seq,
                "via": o.via,
                "by": o.by,
            },
        )
        _publish(g, "vehicle", "vehicle.action", o.model_dump(mode="json"), vid)

    def _handle_report(self, g: GroundView, r: ActionReport) -> None:
        with self._lock:
            o = self._outcomes.get(r.action_id)
        if o is None:
            return
        result = {"acked": "acked", "already_active": "already_active", "failed": "failed"}.get(
            r.result, o.result
        )
        detail = None
        if r.result == "failed":
            detail = "Flygkontrollen bytte inte läge. Använd RC-sändaren."
        elif r.result == "aborted":
            detail = "Ersatt av en nyare åtgärd."
        elif r.result == "already_active":
            detail = "Drönaren står redan still på marken."
        o = o.model_copy(
            update={
                "result": result,
                "mode_requested": r.mode_requested,
                "mode_confirmed": r.mode_confirmed,
                "paths": {**o.paths, "mavlink": "failed" if r.result == "failed" else "sent"},
                "detail_sv": detail,
                "t": self.clock.utc_now(),
            }
        )
        self._store(o)
        g.add_event(
            "vehicle.action_result",
            severity="critical" if r.alarm else "info",
            vehicle_id=r.vehicle_id,
            payload={
                "action_id": r.action_id,
                "kind": r.kind,
                "result": r.result,
                "mode": r.mode_confirmed,
                "alarm": r.alarm,
            },
        )
        _publish(g, "vehicle", "vehicle.action", o.model_dump(mode="json"), r.vehicle_id)


def _publish(
    g: GroundView, topic: str, type_: str, data: dict[str, Any], vehicle_id: str | None
) -> None:
    g.bus.publish(topic, type_, data, ts=g.clock.utc_now(), vehicle_id=vehicle_id)


def _latch_event(g: GroundView, change: LatchChange) -> None:
    latch = change.latch
    payload: dict[str, Any] = {"action_seq": change.action_seq}
    if latch is not None:
        payload |= {"action": latch.action, "underlying": latch.underlying, "set_by": latch.set_by}
    kind = "latch.set" if change.kind == "set" else "latch.released"
    if change.kind == "landed":
        payload["cause"] = "landed"
    g.add_event(kind, vehicle_id=change.vehicle_id, payload=payload)
    _publish(g, "vehicle", kind, payload, change.vehicle_id)


class OperatorPresence:
    """No pilot WebSocket for ``operator_absent_s`` during an active mission: the
    configured action (``hold`` = STOP's steps with ``watchdog_stop``, ``return_home``
    = FLY HOME's with ``watchdog_rtl``, ``off`` = only the alarm). Once per absence."""

    def __init__(
        self,
        actions: VehicleActions,
        *,
        absent_s: float,
        action: Literal["hold", "return_home", "off"],
        mission_active: Callable[[str], bool],
    ) -> None:
        self.actions = actions
        self.absent_s = absent_s
        self.action = action
        self.mission_active = mission_active
        self._fired: set[str] = set()

    def tick(self, g: GroundView) -> None:
        now = g.clock.monotonic()
        seen = g.operator_seen_mono
        for vid in self.actions.vehicle_ids:
            active = self.mission_active(vid)
            absent = active and (seen is None or now - seen >= self.absent_s)
            if not absent:
                self._fired.discard(vid)
                continue
            if vid in self._fired:
                continue
            self._fired.add(vid)
            g.add_event("vehicle.operator_absent", severity="critical", vehicle_id=vid)
            _publish(g, "vehicle", "alarm.OPERATOR_ABSENT", {}, vid)
            if self.action == "hold":
                self.actions.system(vid, "watchdog_stop")
            elif self.action == "return_home":
                self.actions.system(vid, "watchdog_rtl")

    async def run(self, g: GroundView, every_s: float = 0.1) -> None:
        while True:
            try:
                self.tick(g)
            except Exception as exc:
                log.error("actions.presence_failed", error=type(exc).__name__)
            await g.clock.sleep(every_s)
