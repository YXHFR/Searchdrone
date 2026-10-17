"""SD-link onboard: dials the ground station, feeds the executor and sends back what it
says. Acks and events wait in a bounded queue while the link is down (events: the newest
500, oldest sent first); the state snapshot goes out at 2 Hz while connected.

Stick values (``ManualControl``) are never forwarded from here: the ground station sends
the pilot's sticks to the flight controller itself (as 250/190, the way STOP goes), so
they are only counted onboard.

Inbound order: charters, clearances, orders, cancels, releases and hand-backs are
handled one at a time in the order they arrived (a charter install reads the fence
first, which takes a while). STOP/FLY HOME notices, "take over", the pilot's position
and the AI's status do not wait behind them: what overtakes a queued order can only
make that order be refused, never let one through.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any, Final
from uuid import UUID

from sd_link.client import LinkClient
from sd_link.session import (
    Closed,
    Expired,
    Inbound,
    LinkBusyError,
    LinkClosedError,
    LinkSession,
    LinkState,
    StateChanged,
)
from sd_obs.clock import Clock
from sd_onboard.executor.executor import Executor, Severity
from sd_safety.constants import SNAPSHOT_HZ
from sd_schemas.gate import GateDecision
from sd_schemas.link import (
    AckStage,
    CharterAck,
    CharterInstall,
    ClearanceConfirmation,
    CommandCancel,
    CommanderStatus,
    HelloAck,
    ManualControl,
    ManualTakeover,
    OnboardEvent,
    OperatorOverrideNotice,
    OperatorRelease,
    PilotPosition,
    SkillCommand,
)

__all__ = ["OnboardLink"]

log = logging.getLogger("sd_onboard.link")
MAX_EVENTS: Final = 500
MAX_ACKS: Final = 200
_WATCH_S: Final = 0.1


class OnboardLink:
    """The executor's ``Uplink`` and the inbound side of SD-link."""

    def __init__(self, *, url: str, token: str, vehicle_id: str, clock: Clock) -> None:
        self.clock = clock
        self.executor: Executor | None = None
        self._client = LinkClient(
            url=url,
            token=token,
            vehicle_id=vehicle_id,
            make_hello=self._hello,
            on_session=self._on_session,
            clock=clock,
        )
        self._session: LinkSession | None = None
        self._events: deque[OnboardEvent] = deque(maxlen=MAX_EVENTS)
        self._acks: deque[Callable[[LinkSession], Awaitable[object]]] = deque(maxlen=MAX_ACKS)
        self._wake = asyncio.Event()
        self.manual_values_ignored = 0
        self.dropped_events = 0
        self.dropped_acks = 0
        self._ordered: asyncio.Queue[Inbound] = asyncio.Queue()

    def __repr__(self) -> str:  # never show the token
        return f"OnboardLink({self._client!r})"

    # --- the executor's uplink --------------------------------------------------------

    def ack(
        self,
        ref_id: UUID,
        stage: AckStage,
        *,
        reason_codes: tuple[str, ...] = (),
        detail_sv: str | None = None,
        gate: GateDecision | None = None,
    ) -> None:
        codes = list(reason_codes)
        self._count_ack_drop()
        self._acks.append(
            lambda s: s.ack(ref_id, stage, reason_codes=codes, detail_sv=detail_sv, gate=gate)
        )
        self._wake.set()

    def event(self, type: str, severity: Severity, data: dict[str, Any]) -> None:
        if len(self._events) == MAX_EVENTS:
            self.dropped_events += 1
            log.warning("link down too long: oldest event dropped (%d so far)", self.dropped_events)
        clean = {k: (str(v) if isinstance(v, UUID) else v) for k, v in data.items()}
        self._events.append(OnboardEvent(type=type, severity=severity, data=clean))
        self._wake.set()

    def charter_ack(self, ack: CharterAck) -> None:
        self._count_ack_drop()
        self._acks.append(lambda s: s.send(ack))
        self._wake.set()

    def _count_ack_drop(self) -> None:
        if len(self._acks) == MAX_ACKS:
            self.dropped_acks += 1
            log.warning("link down too long: oldest ack dropped (%d so far)", self.dropped_acks)

    # --- the connection ---------------------------------------------------------------

    def _ex(self) -> Executor:
        if self.executor is None:
            raise RuntimeError("no executor attached")
        return self.executor

    def _hello(self) -> Any:
        return self._ex().hello()

    async def run(self, stop: asyncio.Event) -> None:
        await self._client.run_forever(stop)

    async def _on_session(self, session: LinkSession, ack: HelloAck) -> None:
        ex = self._ex()
        self._session = session
        ex.on_connected(ack)
        ex.on_link(True, self.clock.monotonic_ms())
        self._ordered = asyncio.Queue()  # nothing from an earlier session is handled late
        tasks = [
            asyncio.ensure_future(self._handle_in_order()),
            asyncio.ensure_future(self._snapshots(session)),
            asyncio.ensure_future(self._flush(session)),
            asyncio.ensure_future(self._watch(session)),
        ]
        self._wake.set()
        try:
            while True:
                event = await session.next_event()
                if isinstance(event, Closed):
                    break
                if isinstance(event, StateChanged):
                    ex.on_link(event.state is LinkState.UP, round(session.last_inbound * 1000))
                elif isinstance(event, Inbound):
                    await self._dispatch(event)
                elif isinstance(event, Expired):
                    log.warning("sdlink message expired: %s", event.envelope.kind)
        finally:
            self._session = None
            ex.on_link(False, round(session.last_inbound * 1000))
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _watch(self, session: LinkSession) -> None:
        ex = self._ex()
        while True:
            ex.on_link(session.state is LinkState.UP, round(session.last_inbound * 1000))
            await self.clock.sleep(_WATCH_S)

    async def _dispatch(self, inbound: Inbound) -> None:
        ex = self._ex()
        body, env = inbound.body, inbound.envelope
        match body:
            case (
                SkillCommand()
                | CommandCancel()
                | OperatorRelease()
                | ClearanceConfirmation()
                | CharterInstall()
            ):
                self._ordered.put_nowait(inbound)
            case OperatorOverrideNotice():
                ex.on_override(body)
            case PilotPosition():
                ex.on_pilot_position(body)
            case CommanderStatus():
                ex.on_commander_status(body)
            case ManualTakeover() if body.active:
                ex.on_takeover(body, env.msg_id)
            case ManualTakeover():
                self._ordered.put_nowait(inbound)  # a hand-back lets go of HELD: in order
            case ManualControl():
                if self.manual_values_ignored == 0:
                    log.warning("stick values over SD-link are ignored onboard")
                self.manual_values_ignored += 1
            case _:
                pass

    async def _handle_in_order(self) -> None:
        while True:
            inbound = await self._ordered.get()
            try:
                await self._handle(inbound)
            except Exception:
                # One bad message must not stop the ones behind it; the ground sees no
                # answer for it and the event says why.
                log.exception("sdlink %s could not be handled", inbound.envelope.kind)
                self.event(
                    "executor.inbound_failed",
                    "critical",
                    {"kind": inbound.envelope.kind, "msg_id": inbound.envelope.msg_id},
                )

    async def _handle(self, inbound: Inbound) -> None:
        ex = self._ex()
        body, env = inbound.body, inbound.envelope
        match body:
            case SkillCommand():
                deadline = None if inbound.deadline is None else round(inbound.deadline * 1000)
                await ex.on_skill_command(body, deadline_ms=deadline)
            case CommandCancel():
                ex.on_cancel(body.command_id)
            case OperatorRelease():
                ex.on_release(body, env.msg_id)
            case ClearanceConfirmation():
                ex.on_clearance(body, env.msg_id)
            case CharterInstall():
                await ex.on_charter(body)
            case ManualTakeover():
                ex.on_takeover(body, env.msg_id)
            case _:
                pass

    async def _snapshots(self, session: LinkSession) -> None:
        ex = self._ex()
        while True:
            with contextlib.suppress(LinkBusyError, ValueError):
                await session.send(ex.snapshot())
            await self.clock.sleep(1.0 / SNAPSHOT_HZ)

    async def _flush(self, session: LinkSession) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            try:
                while self._acks:
                    await self._acks[0](session)
                    self._acks.popleft()
                while self._events:
                    await session.send(self._events[0])
                    self._events.popleft()
            except LinkBusyError:
                await self.clock.sleep(0.2)
                self._wake.set()
            except LinkClosedError:
                return
