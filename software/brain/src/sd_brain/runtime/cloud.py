"""``CloudModelRuntime``: the cloud model over an OpenAI-compatible API.

Default provider OpenRouter; the model is a setting (``[cloud] model``) and the key is
``SD_CLOUD_API_KEY`` in ``.env``. Every order still goes through the safety gate. Each
request carries the reasoning effort and the provider routing from ``[cloud]``. When the
main model fails (HTTP error, bad answer), ``fallback_model`` is asked with the time that
is left; a timeout leaves no time, so the caller goes to the command mode. Without a key
or model it is not configured.

Every request first goes through the cloud image policy (only small, clean crops; see
:mod:`sd_brain.runtime.images`). With a ``budget`` hook, each call (the fallback too)
first asks the ground station for a reservation; refused, nothing is sent and
:class:`~sd_brain.runtime.budget.BudgetDeniedError` is raised. The real use is committed
after the call, or the reservation released when the request never left. Only the
offline tools (evaluation, latency measurement) run without the hook, under their own
caps.
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
from sd_brain.runtime.budget import (
    BudgetClient,
    BudgetContext,
    BudgetDeniedError,
    estimate_prompt_tokens,
)
from sd_brain.runtime.images import check_messages
from sd_schemas.cloud_budget import BudgetCommit, BudgetReserve

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
        budget: BudgetClient | None = None,
        budget_context: Callable[[], BudgetContext] = lambda: BudgetContext("chat"),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.model_id = cfg.model
        self._key = api_key
        self._client = client
        self._mono = monotonic
        self.budget = budget
        self._budget_context = budget_context

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
        ctx: BudgetContext,
        response_format: dict[str, Any] | None,
    ) -> ModelReply:
        openrouter = urlsplit(self.cfg.base_url).hostname == "openrouter.ai"
        headers = _OPENROUTER_HEADERS if openrouter else None
        body = self.cfg.extra_body()
        if openrouter:
            body["usage"] = {"include": True}  # the real price of the call, for the cost cap
        if response_format is not None:
            body["response_format"] = response_format
        reservation = await self._reserve(model, messages, sampling, ctx)
        # The evaluation seed is not sent: cloud providers handle it differently.
        try:
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
        except ModelError as exc:
            await self._settle_failed(reservation, exc)
            raise
        except BaseException:
            await self._settle_failed(reservation, None)
            raise
        reply.engine = "cloud"
        reply.model_id = reply.model_id or model
        if reservation is not None and self.budget is not None:
            booked = await self.budget.commit(
                BudgetCommit.model_validate(
                    {
                        "reservation_id": reservation[0],
                        "prompt_tokens": reply.prompt_tokens,
                        "completion_tokens": reply.completion_tokens,
                        "cost_usd": reply.cost_usd,
                    }
                )
            )
            reply.cost_sek = booked.actual_sek if booked is not None else None
        return reply

    async def _reserve(
        self,
        model: str,
        messages: list[dict[str, Any]],
        sampling: Sampling,
        ctx: BudgetContext,
    ) -> tuple[str, int] | None:
        """(reservation id, estimated prompt tokens), or None without a budget hook."""
        if self.budget is None:
            return None
        prompt_tokens, images = estimate_prompt_tokens(messages)
        grant = await self.budget.reserve(
            BudgetReserve.model_validate(
                {
                    "purpose": ctx.purpose,
                    "mission_id": ctx.mission_id,
                    "model_id": model,
                    "est_prompt_tokens": prompt_tokens,
                    "est_completion_tokens": sampling.max_tokens,
                    "image_count": images,
                }
            )
        )
        if not grant.granted or grant.reservation_id is None:
            raise BudgetDeniedError(grant.reason or "nekad")
        return str(grant.reservation_id), prompt_tokens

    async def _settle_failed(
        self, reservation: tuple[str, int] | None, exc: ModelError | None
    ) -> None:
        """A failed call: released when the provider surely got nothing to bill (it was
        unreachable or answered with an error), otherwise (a timeout, a broken answer,
        a cancel) committed at the estimate, since it may have been billed."""
        if reservation is None or self.budget is None:
            return
        rid, prompt_tokens = reservation
        never_billed = exc is not None and exc.kind == "http_error"
        if never_billed:
            await self.budget.release(rid)
        else:
            await self.budget.commit(
                BudgetCommit.model_validate(
                    {"reservation_id": rid, "prompt_tokens": prompt_tokens, "completion_tokens": 0}
                )
            )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        sampling: Sampling,
        *,
        response_format: dict[str, Any] | None = None,
        budget_context: BudgetContext | None = None,
    ) -> ModelReply:
        """One answer; ``response_format`` asks for a JSON schema, ``budget_context`` says
        what the call is for (the runtime's own context otherwise)."""
        key, client = self._key, self._client
        if not self.configured or key is None or client is None:
            raise NotConfiguredError("molnmodellen är inte konfigurerad (nyckel eller modell)")
        check_messages(messages)  # before any reservation or network use
        ctx = budget_context or self._budget_context()
        limit = min(timeout_s, self.cfg.call_timeout_s)
        start = self._mono()
        args = (messages, tools)
        try:
            return await self._call(
                key, client, self.cfg.model, *args, limit, sampling, ctx, response_format
            )
        except (ModelTimeoutError, BudgetDeniedError):
            raise
        except ModelError as exc:
            left = limit - (self._mono() - start)
            fallback = self.cfg.fallback_model
            if not fallback or fallback == self.cfg.model or left < MIN_FALLBACK_S:
                raise
            log.warning("huvudmodellen svarade fel (%s); reservmodellen tar över", exc.kind)
        return await self._call(key, client, fallback, *args, left, sampling, ctx, response_format)
