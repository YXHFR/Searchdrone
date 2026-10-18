"""``BestAvailableRuntime``: the cloud model decides when it can, the command mode otherwise.

There is no local language model. When the cloud runtime is not configured, or failed
within the last ``retry_after_s``, :meth:`engine` says ``"command"`` and :meth:`chat`
raises :class:`CloudUnavailableError` at once, so the turn goes to the command mode
(``GrammarBackend``) without waiting. A failed call starts that pause, so later turns are
not slowed down by a cloud that does not answer.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Literal

from sd_brain.runtime.base import ModelError, ModelReply, Sampling
from sd_brain.runtime.cloud import CloudModelRuntime

__all__ = ["BestAvailableRuntime", "CloudUnavailableError", "EngineName"]

log = logging.getLogger("sd_brain.runtime")

# Which engine answers the chat: the cloud model, or the command mode (fixed commands in
# code, no AI). Shown in the panel's health strip.
EngineName = Literal["cloud", "command"]


class CloudUnavailableError(ModelError):
    """No cloud model now; the command mode answers."""


class BestAvailableRuntime:
    name = "best"
    supports_images = False

    def __init__(
        self,
        cloud: CloudModelRuntime | None,
        *,
        retry_after_s: float = 60.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cloud = cloud
        self.retry_after_s = retry_after_s
        self._mono = monotonic
        self._cloud_failed_at: float | None = None
        self.last_engine: EngineName | None = None

    @property
    def model_id(self) -> str:
        return self.cloud.model_id if self.cloud is not None and self.cloud.configured else ""

    def _cloud_ready(self) -> bool:
        if self.cloud is None or not self.cloud.configured:
            return False
        failed = self._cloud_failed_at
        return failed is None or self._mono() - failed >= self.retry_after_s

    def engine(self) -> EngineName:
        """Which engine would answer now (for the health strip)."""
        return "cloud" if self._cloud_ready() else "command"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        sampling: Sampling,
    ) -> ModelReply:
        if not self._cloud_ready() or self.cloud is None:
            self.last_engine = "command"
            raise CloudUnavailableError("ingen molnmodell nu; kommandoläget svarar")
        try:
            reply = await self.cloud.chat(messages, tools, timeout_s, sampling)
        except ModelError as exc:
            self._cloud_failed_at = self._mono()
            self.last_engine = "command"
            log.warning("molnmodellen svarade inte (%s); kommandoläget tar över", exc.kind)
            raise
        self._cloud_failed_at = None
        self.last_engine = "cloud"
        return reply
