"""``BestAvailableRuntime``: the cloud model decides when it can, the local model otherwise.

Per call: when the cloud runtime is configured and has not failed recently, it is asked
first. If it fails or does not answer in time, the local model answers in the same call
(the same turn). After a cloud failure, calls go straight to the local model for
``retry_after_s`` so later turns are not slowed down.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Literal

from sd_brain.runtime.base import ModelError, ModelReply, ModelRuntime, Sampling
from sd_brain.runtime.cloud import CloudModelRuntime

__all__ = ["BestAvailableRuntime"]

log = logging.getLogger("sd_brain.runtime")

# Leave the local model at least this long after a slow cloud call.
LOCAL_RESERVE_S = 5.0


class BestAvailableRuntime:
    name = "best"
    supports_images = False

    def __init__(
        self,
        cloud: CloudModelRuntime | None,
        local: ModelRuntime,
        *,
        retry_after_s: float = 60.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cloud = cloud
        self.local = local
        self.retry_after_s = retry_after_s
        self._mono = monotonic
        self._cloud_failed_at: float | None = None
        self.last_engine: Literal["cloud", "local"] | None = None

    @property
    def model_id(self) -> str:
        return self.cloud.model_id if self._cloud_ready() and self.cloud else self.local.model_id

    def _cloud_ready(self) -> bool:
        if self.cloud is None or not self.cloud.configured:
            return False
        failed = self._cloud_failed_at
        return failed is None or self._mono() - failed >= self.retry_after_s

    def engine(self) -> Literal["cloud", "local"]:
        """Which engine would answer now (for the health line)."""
        return "cloud" if self._cloud_ready() else "local"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        sampling: Sampling,
    ) -> ModelReply:
        start = self._mono()
        if self._cloud_ready() and self.cloud is not None:
            budget = (
                max(0.1, timeout_s - LOCAL_RESERVE_S)
                if timeout_s > LOCAL_RESERVE_S * 2
                else timeout_s / 2
            )
            try:
                reply = await self.cloud.chat(messages, tools, budget, sampling)
                self._cloud_failed_at = None
                self.last_engine = "cloud"
                return reply
            except ModelError as exc:
                self._cloud_failed_at = self._mono()
                log.warning("molnmodellen svarade inte (%s); lokal modell tar över", exc.kind)
        remaining = max(0.1, timeout_s - (self._mono() - start))
        reply = await self.local.chat(messages, tools, remaining, sampling)
        self.last_engine = "local"
        return reply
