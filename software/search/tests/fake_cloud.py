"""A fake OpenAI-compatible cloud server (``httpx.MockTransport``) and a fake budget.

Tests never reach a real provider. ``FakeCloud`` answers ``/chat/completions`` with a
fixed assessment (or a broken one, a server error or a timeout) and keeps every request
body, so a test can check what would have left the ground station. ``FakeBudget`` grants
reservations until its cap and records each booking.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from sd_brain.config import CloudConfig, ProviderConfig
from sd_brain.runtime.budget import BudgetClient
from sd_brain.runtime.cloud import CloudModelRuntime
from sd_schemas.base import new_id
from sd_schemas.cloud_budget import (
    BudgetCommit,
    BudgetCommitted,
    BudgetGrant,
    BudgetReserve,
)

BASE_URL = "https://cloud.test/api/v1"
MODEL = "anthropic/claude-haiku-5.5"


def answer(**kw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "person": "yes",
        "person_p": 0.9,
        "fields": {},
        "overall": 0.8,
        "explanation_sv": "En person i röd jacka står på gräset.",
    }
    return out | kw


@dataclass
class FakeCloud:
    reply: dict[str, Any] | str | Callable[[dict[str, Any]], dict[str, Any] | str] = field(
        default_factory=answer
    )
    fail: str | None = None  # "503", "timeout" or None
    cost_usd: float | None = 0.0004
    requests: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions"), request.url
        body = json.loads(request.content)
        self.requests.append(body)
        if self.fail == "timeout":
            raise httpx.ReadTimeout("fake timeout", request=request)
        if self.fail == "503":
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        reply = self.reply(body) if callable(self.reply) else self.reply
        content = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        usage: dict[str, Any] = {"prompt_tokens": 900, "completion_tokens": 60}
        if self.cost_usd is not None:
            usage["cost"] = self.cost_usd
        return httpx.Response(
            200,
            json={
                "id": "fake",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": usage,
            },
        )

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


@dataclass
class FakeBudget:
    cap_sek: float = 100.0
    per_call_sek: float = 0.01
    spent_sek: float = 0.0
    reserved: list[BudgetReserve] = field(default_factory=list[BudgetReserve])
    committed: list[BudgetCommit] = field(default_factory=list[BudgetCommit])
    released: list[str] = field(default_factory=list[str])

    async def reserve(self, req: BudgetReserve) -> BudgetGrant:
        self.reserved.append(req)
        if self.spent_sek + self.per_call_sek > self.cap_sek:
            return BudgetGrant(granted=False, reason="MISSION_CAP", remaining_sek=0.0)
        self.spent_sek += self.per_call_sek
        return BudgetGrant.model_validate(
            {
                "granted": True,
                "reservation_id": str(new_id()),
                "est_sek": self.per_call_sek,
                "remaining_sek": self.cap_sek - self.spent_sek,
            }
        )

    async def commit(self, done: BudgetCommit) -> BudgetCommitted | None:
        self.committed.append(done)
        return BudgetCommitted(actual_sek=self.per_call_sek)

    async def release(self, reservation_id: str) -> None:
        self.released.append(reservation_id)


def runtime(
    cloud: FakeCloud, budget: BudgetClient | None, *, timeout_s: float = 8.0
) -> CloudModelRuntime:
    """The fast tier's runtime as ``sd-search`` builds it, against the fake server."""
    cfg = CloudConfig(
        base_url=BASE_URL,
        model=MODEL,
        fallback_model="",
        reasoning_effort="",
        provider=ProviderConfig(data_collection="deny", zdr=False),
        call_timeout_s=timeout_s,
    )
    return CloudModelRuntime(cfg, "fake-key", cloud.client(), budget=budget)
