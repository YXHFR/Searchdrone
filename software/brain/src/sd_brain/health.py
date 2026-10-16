"""``HealthState``: when the language model counts as down and the command mode takes over."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from sd_brain.constants import DEGRADED_MIN_S, LLM_FAILS_TO_DEGRADE, PROBE_FAILS_TO_DEGRADE

__all__ = ["HealthState"]

Cause = Literal["probe", "turn"]


class HealthState:
    """Two counters: failed probes of the model server, and failed turns.

    ``degraded`` after 3 failed probes or 3 failed turns; then the command mode answers.
    Back to ``ready``: after a probe cause, two good probes in a row; after a turn cause,
    at least 60 s, two good probes in a row and a good test call (``recover_ok``). A
    backend without probes (the script) only needs the 60 s.
    """

    def __init__(self, monotonic: Callable[[], float], *, has_probe: bool) -> None:
        self._mono = monotonic
        self.has_probe = has_probe
        self.probe_fails = 0
        self.turn_fails = 0
        self.good_probes = 0
        self.degraded_at: float | None = None
        self.cause: Cause | None = None

    @property
    def degraded(self) -> bool:
        return self.degraded_at is not None

    def _degrade(self, cause: Cause) -> None:
        if self.degraded_at is None:
            self.degraded_at = self._mono()
            self.cause = cause
            self.good_probes = 0

    def probe(self, ok: bool) -> None:
        if ok:
            self.probe_fails = 0
            self.good_probes += 1
        else:
            self.probe_fails += 1
            self.good_probes = 0
            if self.probe_fails >= PROBE_FAILS_TO_DEGRADE:
                self._degrade("probe")

    def turn(self, ok: bool) -> None:
        if ok:
            self.turn_fails = 0
            return
        self.turn_fails += 1
        if self.turn_fails >= LLM_FAILS_TO_DEGRADE:
            self._degrade("turn")

    def force_degrade(self) -> None:
        """E.g. three rate-limit rejections in a row."""
        self.turn_fails = max(self.turn_fails, LLM_FAILS_TO_DEGRADE)
        self._degrade("turn")

    def may_recover(self) -> bool:
        """Whether recovery may be tried now (the caller makes the test call if needed)."""
        if self.degraded_at is None:
            return False
        if not self.has_probe:
            return self._mono() - self.degraded_at >= DEGRADED_MIN_S
        if self.cause == "probe":
            return self.good_probes >= 2
        return self._mono() - self.degraded_at >= DEGRADED_MIN_S and self.good_probes >= 2

    def needs_test_call(self) -> bool:
        return self.has_probe and self.cause == "turn"

    def recovered(self) -> None:
        self.degraded_at = None
        self.cause = None
        self.probe_fails = 0
        self.turn_fails = 0
        self.good_probes = 0
