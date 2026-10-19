"""``CloudModelRuntime``: the cloud model over an OpenAI-compatible API.

Default provider OpenRouter; the model is a setting (``[cloud] model``) and the key is
``SD_CLOUD_API_KEY`` in ``.env``. Every order still goes through the safety gate. Each
request carries the reasoning effort and the provider routing from ``[cloud]``. When the
main model fails (HTTP error, bad answer), ``fallback_model`` is asked with the time that
is left; a timeout leaves no time, so the caller goes to the command mode. Without a key
or model it is not configured.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import httpx

from sd_brain.config import CloudConfig
from sd_brain.runtime.base import (
    ModelError,
    ModelReply,
    ModelTimeoutError,
    NotConfiguredError,
    Sampling,
    openai_chat,
)

__all__ = ["CloudModelRuntime"]

log = logging.getLogger("sd_brain.runtime")

_OPENROUTER_HEADERS = {"X-Title": "Searchdrone"}
# A fallback call is not started with less time than this.
MIN_FALLBACK_S = 1.0


class CloudModelRuntime:
    name = "cloud"
    supports_images = True

    def __init__(
        self,
        cfg: CloudConfig,
        api_key: str | None,
        client: httpx.AsyncClient | None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.model_id = cfg.model
        self._key = api_key
        self._client = client
        self._mono = monotonic

    @property
    def configured(self) -> bool:
        return bool(self._key and self.cfg.model and self._client is not None)

    async def _call(
        self,
        key: str,
        client: httpx.AsyncClient,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        sampling: Sampling,
    ) -> ModelReply:
        openrouter = urlsplit(self.cfg.base_url).hostname == "openrouter.ai"
        headers = _OPENROUTER_HEADERS if openrouter else None
        body = self.cfg.extra_body()
        if openrouter:
            body["usage"] = {"include": True}  # the real price of the call, for the cost cap
        # The evaluation seed is not sent: cloud providers handle it differently.
        reply = await openai_chat(
            client,
            self.cfg.base_url + "/chat/completions",
            api_key=key,
            model=model,
            messages=messages,
            tools=tools,
            timeout_s=timeout_s,
            sampling=sampling.model_copy(update={"seed": None}),
            extra_headers=headers,
            extra_body=body,
        )
        reply.engine = "cloud"
        reply.model_id = reply.model_id or model
        return reply

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        sampling: Sampling,
    ) -> ModelReply:
        key, client = self._key, self._client
        if not self.configured or key is None or client is None:
            raise NotConfiguredError("molnmodellen är inte konfigurerad (nyckel eller modell)")
        budget = min(timeout_s, self.cfg.call_timeout_s)
        start = self._mono()
        try:
            return await self._call(key, client, self.cfg.model, messages, tools, budget, sampling)
        except ModelTimeoutError:
            raise
        except ModelError as exc:
            left = budget - (self._mono() - start)
            fallback = self.cfg.fallback_model
            if not fallback or fallback == self.cfg.model or left < MIN_FALLBACK_S:
                raise
            log.warning("huvudmodellen svarade fel (%s); reservmodellen tar över", exc.kind)
        return await self._call(key, client, fallback, messages, tools, left, sampling)
