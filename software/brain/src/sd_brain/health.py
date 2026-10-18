"""``HealthState``: when the commander's backend counts as down and the command mode takes over."""

from __future__ import annotations

from collections.abc import Callable

from sd_brain.constants import DEGRADED_MIN_S, LLM_FAILS_TO_DEGRADE

__all__ = ["HealthState"]


class HealthState:
    """Failed turns in a row.

    ``degraded`` after 3 failed turns (or a forced degrade); then the command mode answers.
    Back to ``ready`` after at least ``DEGRADED_MIN_S``. The cloud runtime keeps its own
    pause after a failure, so the next turn finds out whether the cloud answers again.
    """

    def __init__(self, monotonic: Callable[[], float]) -> None:
        self._mono = monotonic
        self.turn_fails = 0
        self.degraded_at: float | None = None

    @property
    def degraded(self) -> bool:
        return self.degraded_at is not None

    def _degrade(self) -> None:
        if self.degraded_at is None:
            self.degraded_at = self._mono()

    def turn(self, ok: bool) -> None:
        if ok:
            self.turn_fails = 0
            return
        self.turn_fails += 1
        if self.turn_fails >= LLM_FAILS_TO_DEGRADE:
            self._degrade()

    def force_degrade(self) -> None:
        """E.g. three rate-limit rejections in a row."""
        self.turn_fails = max(self.turn_fails, LLM_FAILS_TO_DEGRADE)
        self._degrade()

    def may_recover(self) -> bool:
        return self.degraded_at is not None and self._mono() - self.degraded_at >= DEGRADED_MIN_S

    def recovered(self) -> None:
        self.degraded_at = None
        self.turn_fails = 0
