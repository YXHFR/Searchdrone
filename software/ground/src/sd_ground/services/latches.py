"""The override latches per vehicle: STOP, FLY HOME, the battery guard and the watchdog.

Kept in memory behind a ``threading.Lock``: the ingress thread, the main loop and the
MAVLink thread all read and write them, and the emergency path never waits for the
database. Every change is also put on a queue (``changes``) that the main loop mirrors
into ``operator_latches`` and ``events`` when it can.

Rules:

* A newer latch replaces an older one (STOP during FLY HOME gives ``stop``). The battery
  latch stays underneath: STOP during a battery flight home gives ``stop`` with
  ``underlying = "battery_rtl"``, so after Resume the battery guard starts the flight
  home again if the battery is still low.
* ``action_seq`` never goes back. It starts at ``max(saved, now in UTC milliseconds)``
  and every new action gets the next number, so even a restart without the database
  hands out larger numbers than before.
* Only Resume lifts a latch (``release``), and never during FLY HOME, the watchdog's
  flight home or the battery's: those answer ``RETURN_HOME_ACTIVE`` or
  ``BATTERY_RTL_ACTIVE``. Every latch goes at ``LANDED``.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

from sd_obs.clock import Clock
from sd_schemas.mission import LatchAction, OperatorLatch

__all__ = ["ADOPTED_BY", "HOME_LATCHES", "LatchChange", "LatchRefusedError", "Latches"]

HOME_LATCHES: Final = frozenset({"return_home", "watchdog_rtl"})
# set_by of a latch taken over from the onboard Hello (the ground's own system latches,
# battery and watchdog, use "system:ground").
ADOPTED_BY: Final = "system:onboard"


class LatchRefusedError(Exception):
    def __init__(self, code: str, detail_sv: str) -> None:
        super().__init__(code)
        self.code, self.detail_sv = code, detail_sv


@dataclass(frozen=True, slots=True)
class LatchChange:
    kind: Literal["set", "released", "landed"]
    vehicle_id: str
    latch: OperatorLatch | None  # the new latch for "set", the lifted one otherwise
    action_seq: int


class Latches:
    def __init__(self, clock: Clock, epoch_ms: Callable[[], int] | None = None) -> None:
        self._clock = clock
        self._epoch_ms = epoch_ms or (lambda: int(clock.utc_now().timestamp() * 1000))
        self._lock = threading.Lock()
        self._latch: dict[str, OperatorLatch] = {}
        self._seq: dict[str, int] = {}
        self._override: dict[str, int] = {}
        self.changes: queue.SimpleQueue[LatchChange] = queue.SimpleQueue()

    # --- action numbers ---------------------------------------------------------------

    def _take(self, vehicle_id: str) -> int:
        seq = max(self._seq.get(vehicle_id, 0) + 1, self._epoch_ms())
        self._seq[vehicle_id] = seq
        return seq

    def next_seq(self, vehicle_id: str) -> int:
        """A number for an action that sets or lifts no latch (takeover, hand-back)."""
        with self._lock:
            return self._take(vehicle_id)

    def raise_seq(self, vehicle_id: str, seen: int) -> None:
        """After a restart or an onboard ``Hello``: the next number is above ``seen``."""
        with self._lock:
            self._seq[vehicle_id] = max(self._seq.get(vehicle_id, 0), seen)

    def latest_seq(self, vehicle_id: str) -> int:
        with self._lock:
            return self._seq.get(vehicle_id, 0)

    def override_seq(self, vehicle_id: str) -> int:
        """The number of the newest STOP, FLY HOME, system action or Resume. A mode chain
        stops once this is not its own number; a takeover or hand-back never stops one."""
        with self._lock:
            return self._override.get(vehicle_id, 0)

    # --- the latch --------------------------------------------------------------------

    def get(self, vehicle_id: str) -> OperatorLatch | None:
        with self._lock:
            return self._latch.get(vehicle_id)

    def start(self, vehicle_id: str, action: LatchAction, by: str) -> OperatorLatch:
        """Take the next number and set the latch, in one step (any thread)."""
        with self._lock:
            seq = self._take(vehicle_id)
            old = self._latch.get(vehicle_id)
            underlying: Literal["battery_rtl"] | None = None
            under_battery = old is not None and "battery_rtl" in (old.action, old.underlying)
            if action != "battery_rtl" and under_battery:
                underlying = "battery_rtl"
            latch = OperatorLatch(
                vehicle_id=vehicle_id,
                action=action,
                underlying=underlying,
                action_seq=seq,
                set_by=by,
                set_at=self._clock.utc_now(),
                set_mono_ms=self._clock.monotonic_ms(),
            )
            self._latch[vehicle_id] = latch
            self._override[vehicle_id] = seq
            # Queued under the lock, so the main loop sees the changes in number order.
            self.changes.put(LatchChange("set", vehicle_id, latch, seq))
        return latch

    def check_release(self, vehicle_id: str) -> OperatorLatch | None:
        """The latch Resume would lift; ``LatchRefusedError`` while flying home."""
        latch = self.get(vehicle_id)
        _refuse_while_home(latch)
        return latch

    def release(self, vehicle_id: str) -> tuple[int, OperatorLatch | None]:
        """Resume: take the next number and lift the latch, in one step. Raises
        ``LatchRefusedError`` while flying home."""
        with self._lock:
            latch = self._latch.get(vehicle_id)
            _refuse_while_home(latch)
            seq = self._take(vehicle_id)
            self._latch.pop(vehicle_id, None)
            self._override[vehicle_id] = seq
            self.changes.put(LatchChange("released", vehicle_id, latch, seq))
        return seq, latch

    def landed(self, vehicle_id: str) -> OperatorLatch | None:
        with self._lock:
            latch = self._latch.pop(vehicle_id, None)
        if latch is not None:
            self.changes.put(LatchChange("landed", vehicle_id, latch, latch.action_seq))
        return latch

    def adopt(self, latch: OperatorLatch) -> None:
        """A latch read back after a restart (database or the onboard ``Hello``)."""
        with self._lock:
            self._latch[latch.vehicle_id] = latch
            self._seq[latch.vehicle_id] = max(self._seq.get(latch.vehicle_id, 0), latch.action_seq)


def _refuse_while_home(latch: OperatorLatch | None) -> None:
    if latch is None:
        return
    if latch.action in HOME_LATCHES:
        raise LatchRefusedError(
            "RETURN_HOME_ACTIVE",
            "Drönaren flyger hem. Tryck STOPP först om du vill avbryta hemflygningen.",
        )
    if latch.action == "battery_rtl":
        raise LatchRefusedError(
            "BATTERY_RTL_ACTIVE",
            "Batteriet är lågt och drönaren flyger hem. Tryck STOPP om vägen är farlig.",
        )
