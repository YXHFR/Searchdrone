"""``LocalModelRuntime``: the local model (Qwen3-VL-4B in llama-server) on loopback."""

from __future__ import annotations

from typing import Any

import httpx

from sd_brain.config import LlmConfig
from sd_brain.runtime.base import ModelError, ModelReply, Sampling, openai_chat

__all__ = ["LocalModelRuntime"]


class LocalModelRuntime:
    name = "local"
    supports_images = False  # the image part (mmproj) is loaded in stage 2

    def __init__(self, cfg: LlmConfig, api_key: str, client: httpx.AsyncClient) -> None:
        self.cfg = cfg
        self.model_id = cfg.model
        self._key = api_key
        self._client = client

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        sampling: Sampling,
    ) -> ModelReply:
        reply = await openai_chat(
            self._client,
            self.cfg.base_url + "/chat/completions",
            api_key=self._key,
            model=self.cfg.model,
            messages=messages,
            tools=tools,
            timeout_s=min(timeout_s, self.cfg.call_timeout_s),
            sampling=sampling,
        )
        reply.engine = "local"
        reply.model_id = reply.model_id or self.model_id
        return reply

    async def health(self) -> bool:
        """``GET /health`` (open without a key in llama-server)."""
        try:
            resp = await self._client.get(
                self.cfg.root_url + "/health", timeout=self.cfg.health_timeout_s
            )
        except httpx.HTTPError:
            return False
        return resp.status_code == 200

    async def check_key(self) -> None:
        """A one-token call; raise ``ModelError`` with a Swedish text on 401."""
        try:
            resp = await self._client.post(
                self.cfg.base_url + "/chat/completions",
                json={
                    "model": self.cfg.model,
                    "messages": [{"role": "user", "content": "ok"}],
                    "max_tokens": 1,
                },
                headers={"Authorization": f"Bearer {self._key}"},
                timeout=self.cfg.call_timeout_s,
            )
        except httpx.HTTPError as exc:
            raise ModelError(f"llama-server svarar inte ({type(exc).__name__})") from None
        if resp.status_code == 401:
            raise ModelError("fel nyckel till llama-server (SD_LLM_API_KEY)")
        if resp.status_code != 200:
            raise ModelError(f"llama-server svarade HTTP {resp.status_code}")

    async def count_tokens(self, text: str) -> int | None:
        """``POST /tokenize`` (with the key); ``None`` when it does not answer."""
        try:
            resp = await self._client.post(
                self.cfg.root_url + "/tokenize",
                json={"content": text},
                headers={"Authorization": f"Bearer {self._key}"},
                timeout=self.cfg.health_timeout_s,
            )
            if resp.status_code == 200:
                return len(resp.json().get("tokens", []))
        except (httpx.HTTPError, ValueError):
            return None
        return None
