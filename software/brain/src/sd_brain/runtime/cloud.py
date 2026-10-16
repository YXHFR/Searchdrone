"""``CloudModelRuntime``: the strongest cloud model over an OpenAI-compatible API.

Default provider OpenRouter; the model is a setting (``[cloud] model``) and the key is
``SD_CLOUD_API_KEY`` in ``.env``. It gets the same tools as the local model, and every
order still goes through the safety gate. Without a key or model it is not configured.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import httpx

from sd_brain.config import CloudConfig
from sd_brain.runtime.base import ModelReply, NotConfiguredError, Sampling, openai_chat

__all__ = ["CloudModelRuntime"]

_OPENROUTER_HEADERS = {"X-Title": "Searchdrone"}


class CloudModelRuntime:
    name = "cloud"
    supports_images = True

    def __init__(
        self, cfg: CloudConfig, api_key: str | None, client: httpx.AsyncClient | None
    ) -> None:
        self.cfg = cfg
        self.model_id = cfg.model
        self._key = api_key
        self._client = client

    @property
    def configured(self) -> bool:
        return bool(self._key and self.cfg.model and self._client is not None)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        sampling: Sampling,
    ) -> ModelReply:
        if not self.configured or self._key is None or self._client is None:
            raise NotConfiguredError("molnmodellen är inte konfigurerad (nyckel eller modell)")
        headers = (
            _OPENROUTER_HEADERS if urlsplit(self.cfg.base_url).hostname == "openrouter.ai" else None
        )
        # The local model's seed is not sent: cloud providers handle it differently.
        cloud_sampling = sampling.model_copy(update={"seed": None})
        reply = await openai_chat(
            self._client,
            self.cfg.base_url + "/chat/completions",
            api_key=self._key,
            model=self.cfg.model,
            messages=messages,
            tools=tools,
            timeout_s=min(timeout_s, self.cfg.call_timeout_s),
            sampling=cloud_sampling,
            extra_headers=headers,
        )
        reply.engine = "cloud"
        reply.model_id = reply.model_id or self.model_id
        return reply
