"""The budget hook of ``CloudModelRuntime``: no cloud call without a granted reservation.

The ground station keeps the books (``/internal/v1/cloud-budget``) for the commander and
the search service together. Before each call the runtime reserves the estimated price;
a refusal, or a ground station that does not answer, raises :class:`BudgetDeniedError`
and nothing is sent. After the call it commits what the provider says was used, or
releases the reservation when the request never reached the provider.

The client posts to the loopback listener with the process's own token; the ground takes
the process (commander or search) from the token.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import httpx

from sd_brain.runtime.base import ModelError
from sd_schemas.cloud_budget import (
    BudgetCommit,
    BudgetCommitted,
    BudgetGrant,
    BudgetRelease,
    BudgetReserve,
)

__all__ = [
    "PREFIX",
    "BudgetClient",
    "BudgetContext",
    "BudgetDeniedError",
    "HttpBudgetClient",
    "estimate_prompt_tokens",
]

log = logging.getLogger("sd_brain.runtime.budget")
PREFIX = "/internal/v1/cloud-budget"
# Characters per token for the estimate: low on purpose, so the estimate is high.
CHARS_PER_TOKEN = 3.0


class BudgetDeniedError(ModelError):
    """No reservation: the cap is reached, the price unknown or the books unreachable."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"molnbudgeten nekade anropet ({reason})")
        self.reason = reason


@dataclass(frozen=True)
class BudgetContext:
    purpose: Literal["chat", "match"]
    mission_id: str | None = None


class BudgetClient(Protocol):
    async def reserve(self, req: BudgetReserve) -> BudgetGrant: ...
    async def commit(self, done: BudgetCommit) -> BudgetCommitted | None: ...
    async def release(self, reservation_id: str) -> None: ...


def estimate_prompt_tokens(messages: Sequence[dict[str, Any]]) -> tuple[int, int]:
    """(text tokens, images) of a request: the text counted high, images counted apart
    (the ground prices each image by the model's image tokens)."""
    chars = 0
    images = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            for raw in cast("list[Any]", content):
                part = cast("dict[str, Any]", raw) if isinstance(raw, dict) else {}
                if part.get("type") == "image_url":
                    images += 1
                else:
                    chars += len(str(part.get("text", "")))
        for call in msg.get("tool_calls") or []:
            chars += len(json.dumps(call, ensure_ascii=False))
    return int(chars / CHARS_PER_TOKEN) + 16 * len(messages), images


class HttpBudgetClient:
    """The ground station's budget routes on the loopback listener."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_s: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_s,
            trust_env=False,  # loopback only: never through a proxy
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        try:
            return await self._http.post(PREFIX + path, json=body)
        except httpx.HTTPError as exc:
            raise BudgetDeniedError(f"markstationen svarar inte: {type(exc).__name__}") from None

    async def reserve(self, req: BudgetReserve) -> BudgetGrant:
        res = await self._post("/reserve", req.model_dump(mode="json"))
        if res.status_code != 200:
            raise BudgetDeniedError(f"HTTP {res.status_code}")
        try:
            return BudgetGrant.model_validate(res.json())
        except ValueError:
            raise BudgetDeniedError("oväntat svar") from None

    async def commit(self, done: BudgetCommit) -> BudgetCommitted | None:
        """The booked price, or None when the ground did not take it (the reservation
        then expires and is booked at its estimate)."""
        try:
            res = await self._post("/commit", done.model_dump(mode="json"))
            if res.status_code == 200:
                return BudgetCommitted.model_validate(res.json())
        except (BudgetDeniedError, ValueError):
            pass
        log.warning("molnkostnaden kunde inte bokföras; reservationen bokförs vid utgång")
        return None

    async def release(self, reservation_id: str) -> None:
        try:
            await self._post(
                "/release",
                BudgetRelease.model_validate({"reservation_id": reservation_id}).model_dump(
                    mode="json"
                ),
            )
        except BudgetDeniedError:
            log.warning("reservationen kunde inte släppas; den bokförs vid utgång")
