"""B2-06: the cloud budget guard (E6). Caps per mission and per day, the outer cap in code,
reservations that expire, unknown prices and which token may book what."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sd_ground.app import commander_asgi, create_commander_app
from sd_ground.auth import ServiceToken
from sd_ground.config import ConfigError, check, cloud_budget_errors, load
from sd_ground.context import Ground
from sd_ground.db.cloud_usage import UsageRow
from sd_ground.services import health
from sd_ground.services.cloud_budget import (
    EXHAUSTED_TEXT,
    CloudBudget,
    ModelPrice,
    Prices,
    load_prices,
)
from sd_schemas.cloud_budget import (
    CLOUD_BUDGET_CEILING_SEK_PER_DAY,
    BudgetCommit,
    BudgetReserve,
)
from sd_testing import FakeClock

REPO = Path(__file__).resolve().parents[4]
MODEL = "test/model"
MISSION = "0192f49e-0b3a-7c55-8e2f-1a2b3c4d5e6f"
OTHER_MISSION = "0192f49e-0b3a-7c55-8e2f-1a2b3c4d5e70"
SEARCH = "search-token-for-tests-0123456789abcdef"
BRAIN = "brain-token-for-tests-0123456789abcdef"  # same as conftest.BRAIN_TOKEN
# 1 kr per 100 000 prompt tokens: easy sums.
PRICES = Prices(
    usd_to_sek=10.0,
    models={MODEL: ModelPrice(in_usd_per_mtok=1.0, out_usd_per_mtok=1.0, image_tokens_per_crop=0)},
)


class FakeStore:
    def __init__(self) -> None:
        self.rows: list[UsageRow] = []
        self.caps: dict[str, float] = {}
        self.fail = False

    async def add(self, row: UsageRow) -> None:
        if self.fail:
            raise OSError("db down")
        self.rows.append(row)

    async def since(self, t: dt.datetime) -> Sequence[UsageRow]:
        return list(self.rows)

    async def mission_caps(self) -> dict[str, float]:
        return dict(self.caps)

    def statuses(self) -> list[str]:
        return [r.status for r in self.rows]


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


def _budget(g: Ground, store: FakeStore) -> CloudBudget:
    g.db_ok = True
    b = CloudBudget(g, PRICES, store)  # pyright: ignore[reportArgumentType]
    g.cloud_budget = b
    return b


@pytest.fixture
def budget(make_ground: Any, store: FakeStore) -> CloudBudget:
    return _budget(make_ground(cloud_budget={"mission_cap_sek": 1.0, "day_cap_sek": 3.0}), store)


def req(tokens: int = 10_000, mission: str | None = MISSION, model: str = MODEL) -> BudgetReserve:
    return BudgetReserve.model_validate(
        {
            "purpose": "match",
            "mission_id": mission,
            "model_id": model,
            "est_prompt_tokens": tokens,
            "est_completion_tokens": 0,
        }
    )


def done(rid: Any, tokens: int, cost_usd: float | None = None) -> BudgetCommit:
    return BudgetCommit.model_validate(
        {
            "reservation_id": rid,
            "prompt_tokens": tokens,
            "completion_tokens": 0,
            "cost_usd": cost_usd,
        }
    )


async def test_budget_cap(budget: CloudBudget, store: FakeStore) -> None:
    """Mission cap 1 kr: 0.1 kr calls fit until the sum would go over; the refusal says
    which cap, the state is exhausted and the panel and chat are told."""
    granted = 0
    for _ in range(12):
        grant = await budget.reserve("search", req())
        if not grant.granted:
            assert grant.reason == "MISSION_CAP"
            break
        granted += 1
        assert grant.est_sek == pytest.approx(0.1)
        await budget.commit("search", done(grant.reservation_id, 10_000))
    assert granted == 10
    status = budget.status(MISSION)
    assert status.state == "exhausted"
    assert status.mission_sek == pytest.approx(1.0)
    assert health.system_status(budget.g).cloud_budget == "exhausted"
    types = [e["type"] for e in budget.g.queued_events()]
    assert "cloud_budget.warning" in types
    assert "cloud_budget.exhausted" in types
    assert store.statuses().count("committed") == 10
    # Another mission still has its own cap; the day cap (3 kr) counts both.
    assert (await budget.reserve("commander", req(mission=OTHER_MISSION))).granted


async def test_exhausted_refuses_smaller_calls(budget: CloudBudget) -> None:
    """A large call that does not fit makes the mission exhausted, and then a smaller one
    that would fit is refused too: the panel's "no cloud calls" holds."""
    g = await budget.reserve("search", req(90_000))
    await budget.commit("search", done(g.reservation_id, 90_000))
    big = await budget.reserve("commander", req(20_000))
    assert not big.granted and big.reason == "MISSION_CAP"
    assert budget.status(MISSION).state == "exhausted"
    small = await budget.reserve("search", req(1_000))
    assert not small.granted and small.reason == "MISSION_CAP"
    await budget.set_mission_cap(MISSION, 2.0, "operator:pilot")
    assert (await budget.reserve("search", req(1_000))).granted


async def test_day_cap_counts_both_processes(budget: CloudBudget) -> None:
    for mission in (MISSION, OTHER_MISSION, None):
        for process in ("search", "commander"):
            grant = await budget.reserve(process, req(50_000, mission=mission))
            assert grant.granted, (mission, process)
    grant = await budget.reserve("commander", req(1_000, mission=None))
    assert not grant.granted
    assert grant.reason == "DAY_CAP"
    assert budget.status(None).state == "exhausted"


async def test_open_reservations_count(budget: CloudBudget) -> None:
    """Two processes cannot both spend the last krona: reserved counts until booked."""
    first = await budget.reserve("search", req(60_000))
    second = await budget.reserve("commander", req(60_000))
    assert first.granted
    assert not second.granted
    await budget.release("search", str(first.reservation_id))
    assert (await budget.reserve("commander", req(60_000))).granted


async def test_unknown_price_is_refused(budget: CloudBudget, store: FakeStore) -> None:
    grant = await budget.reserve("search", req(model="someone/unpriced"))
    assert not grant.granted
    assert grant.reason == "UNKNOWN_PRICE"
    assert store.rows == []


async def test_no_ledger_refuses(make_ground: Any, store: FakeStore) -> None:
    b = _budget(make_ground(), store)
    b.g.db_ok = False
    assert (await b.reserve("search", req())).reason == "NO_LEDGER"
    b.g.db_ok = True
    store.fail = True
    assert (await b.reserve("search", req())).reason == "NO_LEDGER"
    assert b.status(MISSION).mission_sek == 0


async def test_reservation_expires_at_its_estimate(
    budget: CloudBudget, store: FakeStore, clock: FakeClock
) -> None:
    grant = await budget.reserve("search", req(30_000))
    clock.advance(121)
    await budget.tick()
    assert store.statuses() == ["reserved", "expired"]
    assert store.rows[-1].actual_sek == pytest.approx(0.3)
    assert budget.status(MISSION).mission_sek == pytest.approx(0.3)
    # A late commit is refused; the estimate stays booked.
    assert await budget.commit("search", done(grant.reservation_id, 1)) == "closed"


async def test_provider_cost_wins(budget: CloudBudget) -> None:
    grant = await budget.reserve("search", req(10_000))
    booked = await budget.commit("search", done(grant.reservation_id, 10_000, cost_usd=0.002))
    assert not isinstance(booked, str)
    assert booked.actual_sek == pytest.approx(0.02)


async def test_only_the_owner_books(budget: CloudBudget) -> None:
    grant = await budget.reserve("search", req())
    rid = str(grant.reservation_id)
    assert await budget.commit("commander", done(rid, 1)) == "unknown"
    assert await budget.release("commander", rid) == "unknown"
    assert await budget.release("search", rid) == "ok"
    assert await budget.release("search", rid) == "closed"


async def test_rebuilt_from_the_books(make_ground: Any, store: FakeStore) -> None:
    """After a restart the sums come back; an open reservation is booked as expired."""
    first = _budget(make_ground(cloud_budget={"mission_cap_sek": 1.0}), store)
    a = await first.reserve("search", req(40_000))
    await first.commit("search", done(a.reservation_id, 40_000))
    await first.reserve("commander", req(30_000))
    again = _budget(make_ground(cloud_budget={"mission_cap_sek": 1.0}), store)
    grant = await again.reserve("search", req(40_000))
    assert not grant.granted  # 0.4 + 0.3 (expired) + 0.4 > 1.0
    assert store.statuses().count("expired") == 1


async def test_refresh_shows_the_books_before_any_call(make_ground: Any, store: FakeStore) -> None:
    """The panel sees what was spent before a restart without waiting for a call, and a
    cap the pilot sets in this run wins over an older stored one."""
    first = _budget(make_ground(cloud_budget={"mission_cap_sek": 1.0}), store)
    a = await first.reserve("search", req(40_000))
    await first.commit("search", done(a.reservation_id, 40_000))
    store.caps[MISSION] = 0.5
    again = _budget(make_ground(cloud_budget={"mission_cap_sek": 1.0}), store)
    await again.set_mission_cap(MISSION, 2.0, "operator:pilot")
    await again.refresh()
    status = again.status(MISSION)
    assert status.mission_sek == pytest.approx(0.4)
    assert status.mission_cap_sek == 2.0
    store.fail = True  # a failing store is not an error for the background task
    await again.tick()


async def test_pilot_raises_the_mission_cap(budget: CloudBudget) -> None:
    for _ in range(10):
        g = await budget.reserve("search", req())
        await budget.commit("search", done(g.reservation_id, 10_000))
    assert not (await budget.reserve("search", req())).granted
    status = await budget.set_mission_cap(MISSION, 2.0, "operator:pilot")
    assert status.mission_cap_sek == 2.0
    assert status.state == "ok"
    assert (await budget.reserve("search", req())).granted
    logged = [e for e in budget.g.queued_events() if e["type"] == "cloud_budget.mission_cap"]
    assert logged[-1]["payload"]["cap_sek"] == 2.0
    # Never above the outer cap.
    status = await budget.set_mission_cap(MISSION, 5000.0, "operator:pilot")
    assert status.mission_cap_sek == CLOUD_BUDGET_CEILING_SEK_PER_DAY


def test_exhausted_text() -> None:
    assert EXHAUSTED_TEXT == "Molnbudget slut – matchning pausad, ingen AI-chatt"  # noqa: RUF001


# --- configuration ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "toml",
    ["[cloud_budget]\nday_cap_sek = 1001\n", "[cloud_budget]\nmission_cap_sek = 5000\n"],
)
def test_budget_ceiling_config(tmp_path: Path, toml: str) -> None:
    """A cap above the outer cap stops the server with a Swedish line naming the key."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "ground.toml").write_text(toml)
    loaded = load(tmp_path, {})
    errors = cloud_budget_errors(loaded.config)
    assert len(errors) == 1
    assert "yttre taket" in errors[0]
    assert any("yttre taket" in e for e in check(loaded))


def test_example_prices_load() -> None:
    prices = load_prices(REPO)
    assert prices.usd_to_sek == 10.5
    for model in ("anthropic/claude-haiku-5.5", "z-ai/glm-5.3-flash"):
        assert model in prices.models


def test_local_prices_merge_and_bad_file_stops(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "cloud_prices.example.toml").write_text(
        'usd_to_sek = 10.5\n[models."a/b"]\nin_usd_per_mtok = 1\nout_usd_per_mtok = 2\n'
    )
    (tmp_path / "config" / "cloud_prices.toml").write_text(
        '[models."c/d"]\nin_usd_per_mtok = 3\nout_usd_per_mtok = 4\n'
    )
    assert set(load_prices(tmp_path).models) == {"a/b", "c/d"}
    (tmp_path / "config" / "cloud_prices.toml").write_text('[models."c/d"]\nbogus = 1\n')
    with pytest.raises(ConfigError):
        load_prices(tmp_path)


# --- the routes ------------------------------------------------------------------------


@pytest.fixture
def loopback(budget: CloudBudget) -> Iterator[TestClient]:
    g = budget.g
    g.search_token = ServiceToken(SEARCH)
    app = commander_asgi(g, create_commander_app(g))
    with TestClient(app, base_url="http://127.0.0.1:7700") as client:  # pyright: ignore[reportArgumentType]
        yield client


def _body(tokens: int = 10_000) -> dict[str, Any]:
    return req(tokens).model_dump(mode="json")


def test_routes_take_the_process_from_the_token(loopback: TestClient, store: FakeStore) -> None:
    for token, process in ((SEARCH, "search"), (BRAIN, "commander")):
        auth = {"Authorization": f"Bearer {token}"}
        res = loopback.post(  # a process named in the body is refused outright
            "/internal/v1/cloud-budget/reserve",
            json={**_body(), "process": "commander"},
            headers=auth,
        )
        assert res.status_code == 422
        res = loopback.post("/internal/v1/cloud-budget/reserve", json=_body(), headers=auth)
        assert res.status_code == 200, res.text
        assert res.json()["granted"]
        assert store.rows[-1].process == process


def test_routes_need_a_service_token(loopback: TestClient) -> None:
    for headers in ({}, {"Authorization": "Bearer nope"}):
        res = loopback.post("/internal/v1/cloud-budget/reserve", json=_body(), headers=headers)
        assert res.status_code == 401


def test_commit_and_release_routes(loopback: TestClient) -> None:
    auth = {"Authorization": f"Bearer {SEARCH}"}
    rid = loopback.post("/internal/v1/cloud-budget/reserve", json=_body(), headers=auth).json()[
        "reservation_id"
    ]
    other = {"Authorization": f"Bearer {BRAIN}"}
    commit = {"reservation_id": rid, "prompt_tokens": 10_000, "completion_tokens": 0}
    assert (
        loopback.post("/internal/v1/cloud-budget/commit", json=commit, headers=other).status_code
        == 404
    )
    res = loopback.post("/internal/v1/cloud-budget/commit", json=commit, headers=auth)
    assert res.status_code == 200
    assert res.json()["actual_sek"] == pytest.approx(0.1)
    res = loopback.post(
        "/internal/v1/cloud-budget/release", json={"reservation_id": rid}, headers=auth
    )
    assert res.status_code == 409


def test_panel_sees_the_budget(ground: Ground, store: FakeStore, logged_in: TestClient) -> None:
    _budget(ground, store)
    res = logged_in.get("/api/v1/cloud-budget")
    assert res.status_code == 200
    assert res.json()["day_cap_sek"] == 150.0
    assert res.json()["state"] == "ok"


def test_panel_budget_needs_login(ground: Ground, store: FakeStore, panel: TestClient) -> None:
    _budget(ground, store)
    assert panel.get("/api/v1/cloud-budget").status_code == 401
