"""E6: the cloud runtime makes no call without a granted reservation from the ground."""

from __future__ import annotations

import json

import httpx
import pytest
from brain_helpers import chat_reply, mock_client

from sd_brain.config import CloudConfig
from sd_brain.runtime.base import ModelError, ModelTimeoutError, Sampling
from sd_brain.runtime.budget import (
    PREFIX,
    BudgetContext,
    BudgetDeniedError,
    HttpBudgetClient,
    estimate_prompt_tokens,
)
from sd_brain.runtime.cloud import CloudModelRuntime
from sd_brain.runtime.fallback import BestAvailableRuntime, CloudUnavailableError
from sd_schemas.base import new_id
from sd_schemas.cloud_budget import (
    BudgetCommit,
    BudgetCommitted,
    BudgetGrant,
    BudgetReserve,
)

KEY = "k" * 40
MSG = [{"role": "user", "content": "hej"}]


class Books:
    """The ground's budget in memory: grants while ``left`` holds, records everything."""

    def __init__(self, *, grant: bool = True, reason: str | None = None) -> None:
        self.grant = grant
        self.reason = reason
        self.reserved: list[BudgetReserve] = []
        self.committed: list[BudgetCommit] = []
        self.released: list[str] = []

    async def reserve(self, req: BudgetReserve) -> BudgetGrant:
        self.reserved.append(req)
        if not self.grant:
            return BudgetGrant(granted=False, reason=self.reason or "DAY_CAP", remaining_sek=0)  # type: ignore[arg-type]
        return BudgetGrant(granted=True, reservation_id=new_id(), est_sek=0.01, remaining_sek=1)

    async def commit(self, done: BudgetCommit) -> BudgetCommitted | None:
        self.committed.append(done)
        return BudgetCommitted(actual_sek=0.02)

    async def release(self, reservation_id: str) -> None:
        self.released.append(reservation_id)


def _rt(handler, books: Books, **cfg) -> CloudModelRuntime:  # type: ignore[no-untyped-def]
    return CloudModelRuntime(
        CloudConfig(**cfg),
        KEY,
        mock_client(handler),
        budget=books,
        budget_context=lambda: BudgetContext("chat", "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a21"),
    )


async def test_denied_reservation_sends_nothing() -> None:
    calls: list[httpx.Request] = []
    books = Books(grant=False, reason="MISSION_CAP")
    rt = _rt(lambda r: calls.append(r) or httpx.Response(200, json=chat_reply("x")), books)
    with pytest.raises(BudgetDeniedError) as exc:
        await rt.chat(MSG, [], 5, Sampling())
    assert exc.value.reason == "MISSION_CAP" and calls == []
    assert books.committed == [] and books.released == []


async def test_granted_call_is_committed_with_the_real_use() -> None:
    books = Books()

    def ok(_r: httpx.Request) -> httpx.Response:
        body = chat_reply("svar")
        body["usage"] = {"prompt_tokens": 120, "completion_tokens": 30, "cost": 0.0004}
        return httpx.Response(200, json=body)

    reply = await _rt(ok, books, model="vendor/m").chat(MSG, [], 5, Sampling(max_tokens=300))
    (req,) = books.reserved
    assert req.purpose == "chat" and req.model_id == "vendor/m"
    assert req.est_completion_tokens == 300 and req.image_count == 0
    assert str(req.mission_id) == "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a21"
    (done,) = books.committed
    assert (done.prompt_tokens, done.completion_tokens, done.cost_usd) == (120, 30, 0.0004)
    assert reply.cost_sek == 0.02


async def test_an_error_answer_releases_and_the_fallback_reserves_again() -> None:
    books = Books()
    models: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        model = json.loads(req.content)["model"]
        models.append(model)
        if model == "vendor/main":
            return httpx.Response(503)
        return httpx.Response(200, json=chat_reply("spare answer"))

    rt = _rt(handler, books, model="vendor/main", fallback_model="vendor/spare")
    reply = await rt.chat(MSG, [], 10, Sampling())
    assert reply.content == "spare answer" and models == ["vendor/main", "vendor/spare"]
    assert [r.model_id for r in books.reserved] == ["vendor/main", "vendor/spare"]
    assert len(books.released) == 1 and len(books.committed) == 1


async def test_a_timeout_is_committed_at_the_estimate() -> None:
    books = Books()

    def slow(_r: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ModelTimeoutError):
        await _rt(slow, books).chat(MSG, [], 5, Sampling())
    (done,) = books.committed
    assert done.completion_tokens == 0 and done.prompt_tokens == books.reserved[0].est_prompt_tokens
    assert books.released == []


async def test_a_denied_main_model_never_tries_the_fallback() -> None:
    calls: list[httpx.Request] = []
    books = Books(grant=False)
    rt = _rt(lambda r: calls.append(r) or httpx.Response(200), books, fallback_model="v/b")
    with pytest.raises(BudgetDeniedError):
        await rt.chat(MSG, [], 10, Sampling())
    assert len(books.reserved) == 1 and calls == []


async def test_best_available_pauses_on_a_denial() -> None:
    now = [0.0]
    books = Books(grant=False)
    rt = _rt(lambda _r: httpx.Response(200, json=chat_reply("x")), books)
    best = BestAvailableRuntime(rt, retry_after_s=60, monotonic=lambda: now[0])
    with pytest.raises(BudgetDeniedError):
        await best.chat(MSG, [], 5, Sampling())
    assert best.engine() == "command"
    with pytest.raises(CloudUnavailableError):
        await best.chat(MSG, [], 5, Sampling())
    assert len(books.reserved) == 1  # the pause asks nobody
    now[0] = 61.0
    books.grant = True
    assert (await best.chat(MSG, [], 5, Sampling())).content == "x"


def test_estimate_counts_text_high_and_images_apart() -> None:
    msgs = [
        {"role": "system", "content": "a" * 300},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "b" * 300},
                {"type": "image_url", "image_url": {"url": "data:"}},
            ],
        },
    ]
    tokens, images = estimate_prompt_tokens(msgs)
    assert tokens >= 200 and images == 1


async def test_http_client_routes_and_failures() -> None:
    seen: list[str] = []
    rid = str(new_id())

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        assert req.headers["authorization"] == "Bearer tok"
        if req.url.path.endswith("/reserve"):
            return httpx.Response(
                200, json={"granted": True, "reservation_id": rid, "remaining_sek": 3.0}
            )
        if req.url.path.endswith("/commit"):
            return httpx.Response(200, json={"actual_sek": 0.1})
        return httpx.Response(204)

    client = HttpBudgetClient(
        "http://127.0.0.1:7700", "tok", transport=httpx.MockTransport(handler)
    )
    req = BudgetReserve(purpose="match", model_id="m", est_prompt_tokens=1, est_completion_tokens=1)
    grant = await client.reserve(req)
    assert grant.granted and str(grant.reservation_id) == rid
    done = await client.commit(
        BudgetCommit.model_validate(
            {"reservation_id": rid, "prompt_tokens": 1, "completion_tokens": 1}
        )
    )
    assert done is not None and done.actual_sek == 0.1
    await client.release(rid)
    assert seen == [PREFIX + "/reserve", PREFIX + "/commit", PREFIX + "/release"]
    await client.aclose()

    def down(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    dead = HttpBudgetClient("http://127.0.0.1:7700", "tok", transport=httpx.MockTransport(down))
    with pytest.raises(BudgetDeniedError):
        await dead.reserve(req)
    assert (
        await dead.commit(
            BudgetCommit.model_validate(
                {"reservation_id": rid, "prompt_tokens": 1, "completion_tokens": 1}
            )
        )
        is None
    )
    await dead.release(rid)  # logged, never raised

    bad = HttpBudgetClient(
        "http://127.0.0.1:7700",
        "tok",
        transport=httpx.MockTransport(lambda _r: httpx.Response(403)),
    )
    with pytest.raises(BudgetDeniedError, match="403"):
        await bad.reserve(req)
    junk = HttpBudgetClient(
        "http://127.0.0.1:7700",
        "tok",
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, text="x")),
    )
    with pytest.raises(BudgetDeniedError):
        await junk.reserve(req)
    assert isinstance(BudgetDeniedError("x"), ModelError)
