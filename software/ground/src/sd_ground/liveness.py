"""Signs of life from the event loops, for the MAVLink thread.

Each loop beats regularly. When a loop has not beaten for ``LOOP_STALL_S`` the MAVLink
thread stops its GCS heartbeat, so the flight controller's own GCS failsafe takes over
instead of a hung ground station pretending to be alive.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

__all__ = ["LOOP_STALL_S", "LoopLiveness"]

# Same value as sd_safety.constants.LOOP_STALL_S.
LOOP_STALL_S = 8.0


class LoopLiveness:
    def __init__(self, now: Callable[[], float] | None = None) -> None:
        self._now = now or time.monotonic
        self._lock = threading.Lock()
        self._beats: dict[str, float] = {}

    def beat(self, loop: str) -> None:
        with self._lock:
            self._beats[loop] = self._now()

    def stalled(self, stall_s: float = LOOP_STALL_S) -> list[str]:
        now = self._now()
        with self._lock:
            return [name for name, t in self._beats.items() if now - t > stall_s]

    def alive(self, stall_s: float = LOOP_STALL_S) -> bool:
        return not self.stalled(stall_s)
