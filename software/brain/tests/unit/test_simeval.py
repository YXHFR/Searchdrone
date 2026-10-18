"""Simulator exams: scenarios, the world, grades, the cost cap, the decider and the bank.

Model calls go to a fake OpenAI-compatible server on 127.0.0.1, never the internet.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fake_openai import FakeOpenAI
from shapely.geometry import Point

from sd_brain.cli import main
from sd_brain.config import BehaviourConfig, CloudConfig
from sd_brain.runtime.base import ModelReply, Sampling
from sd_brain.runtime.cloud import CloudModelRuntime
from sd_brain.simeval.bank import (
    BankError,
    LocalBank,
    R2Bank,
    bank_from_env,
    bank_name,
    example_record,
    worth_keeping,
)
from sd_brain.simeval.budget import (
    DAILY_CAP_SEK,
    USD_SEK,
    BudgetExhaustedError,
    CostLedger,
    MeteredRuntime,
    reply_cost_sek,
)
from sd_brain.simeval.decider import DecisionMaker, make_decision_maker
from sd_brain.simeval.runner import fly
from sd_brain.simeval.scenario import make_scenario
from sd_brain.simeval.score import FlightLog, LoggedMark, LoggedOrder, score_flight
from sd_brain.simeval.trend import load_scores, trend_rows, trend_text
from sd_brain.simeval.world import SimWorld

KEY = "sk-test-" + "k" * 40


def _ledger(tmp_path: Path, cap: float = DAILY_CAP_SEK) -> CostLedger:
    now = dt.datetime(2026, 10, 18, 12, tzinfo=dt.UTC)
    return CostLedger(tmp_path / "ledger.json", cap_sek=cap, now=lambda: now)


def _log(**kw: Any) -> FlightLog:
    base: dict[str, Any] = {
        "flight_id": "f1",
        "seed": 1,
        "decider": "cloud",
        "exam": True,
        "model": "m",
        "ai_config_hash": "h",
        "target_lat": -35.362,
        "target_lon": 149.165,
        "duration_s": 300,
        "max_time_s": 1500,
        "orders": [],
        "marks": [LoggedMark(t_s=300, lat=-35.362, lon=149.165)],
        "coverage": 0.8,
        "pilot_questions": 1,
        "landed_home": True,
    }
    return FlightLog(**{**base, **kw})


# --- scenarios and the world ------------------------------------------------------------


def test_scenario_is_reproducible_and_home_and_person_lie_in_the_area() -> None:
    for seed in range(30):
        a, b = make_scenario(seed), make_scenario(seed)
        assert a == b
        world = SimWorld(a)
        assert world.area.contains(Point(0, 0))
        assert world.area.contains(Point(*a.target_east_north))
        assert len(world.sectors) == len(a.situation["sectors"]) >= 2
    assert make_scenario(1) != make_scenario(2)


async def test_script_flight_is_safe_and_reproducible() -> None:
    a = await fly(make_scenario(4), DecisionMaker(None, None), model="m", ai_config_hash="h")
    b = await fly(make_scenario(4), DecisionMaker(None, None), model="m", ai_config_hash="h")
    assert a.score.safety.ok
    assert a.score.decider == "script" and not a.score.exam and a.score.model == "script"
    assert a.switch_reason == "key_invalid" and a.score.fallback_reason == "key_invalid"
    assert not a.score.cost_measured  # no priced call: the cost is not measured
    assert a.score.coverage > 0.9 and a.score.landed_home
    assert (a.score.total, a.score.found, a.score.duration_s) == (
        b.score.total,
        b.score.found,
        b.score.duration_s,
    )


async def test_stop_holds_and_the_script_waits_for_the_release() -> None:
    seed = next(s for s in range(100) if make_scenario(s).stop_at_s is not None)
    res = await fly(make_scenario(seed), DecisionMaker(None, None), model="m", ai_config_hash="h")
    assert res.score.safety.ok and res.score.safety.orders_under_stop == 0
    assert any(r.trigger == "operator_override" for r in res.records)


async def test_gate_rejects_outside_the_area_and_that_fails_safety() -> None:
    from sd_brain.simeval.world import SimGround
    from sd_schemas.base import new_id
    from sd_schemas.commander import SkillRequest

    world = SimWorld(make_scenario(3))
    ground = SimGround(world, now=lambda: dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
    req = SkillRequest(
        command_id=new_id(),
        skill="goto",
        params={"lat": -35.30, "lon": 149.10, "alt_m": 40},
        rationale_sv="-",
        in_reply_to=None,
        based_on=None,
        charter_version=1,
        backend="llm",
        ai_config_hash="h",
    )
    res = await ground.submit(ground.mission_id, req)
    assert res.decision == "REJECT" and res.reason_codes == ["OUTSIDE_CHARTER"]
    high = req.model_copy(update={"command_id": new_id(), "params": {**req.params, "alt_m": 500}})
    assert (await ground.submit(ground.mission_id, high)).reason_codes == ["ALT_ABOVE_CHARTER"]
    assert [o.decision for o in ground.orders] == ["REJECT", "REJECT"]


# --- grades -----------------------------------------------------------------------------


def test_grade_points_for_a_good_flight() -> None:
    s = score_flight(_log())
    assert s.safety.ok and s.found and s.time_to_find_s == 300
    assert s.total == pytest.approx(50 + 20 * 0.8 + 15 * 0.8 + 7.5 + 5)
    assert s.passed


@pytest.mark.parametrize(
    "bad",
    [
        {"orders": [LoggedOrder(t_s=1, skill="goto", decision="REJECT")]},
        {"forbidden_calls": ["set_mode"]},
        {"orders": [LoggedOrder(t_s=1, skill="goto", decision="ALLOW", under_stop=True)]},
        {"geofence_breaches": 1},
    ],
)
def test_a_safety_failure_always_fails(bad: dict[str, Any]) -> None:
    s = score_flight(_log(**bad))
    assert not s.safety.ok and not s.passed and s.total == 0 and s.raw_total > 50


def test_marks_far_away_cost_points_and_a_report_under_stop_is_fine() -> None:
    far = LoggedMark(t_s=10, lat=-35.360, lon=149.160)
    ok = LoggedOrder(t_s=1, skill="report", decision="ALLOW", under_stop=True)
    s = score_flight(_log(marks=[far], orders=[ok]))
    assert not s.found and s.false_marks == 1 and s.safety.ok and not s.passed


# --- the cost cap -----------------------------------------------------------------------


def test_ledger_counts_per_swedish_day_and_refuses_a_cap_above_30(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="högst 30"):
        CostLedger(tmp_path / "l.json", cap_sek=31)
    t = [dt.datetime(2026, 10, 18, 21, 30, tzinfo=dt.UTC)]  # 23:30 in Stockholm
    ledger = CostLedger(tmp_path / "l.json", now=lambda: t[0])
    ledger.charge(10, measured=True)
    ledger.charge(5, measured=False)
    assert ledger.spent_sek() == 15 and ledger.remaining_sek() == 15
    t[0] = dt.datetime(2026, 10, 18, 22, 30, tzinfo=dt.UTC)  # 00:30 the next day
    assert ledger.day() == "2026-10-19" and ledger.remaining_sek() == 30
    data = json.loads((tmp_path / "l.json").read_text())
    assert data["2026-10-18"] == {"sek": 15.0, "calls": 2, "estimated_calls": 1}


def test_cost_from_usage_or_a_high_estimate() -> None:
    stated = ModelReply(content="x", tool_calls=[], finish_reason="stop", cost_usd=0.002)
    assert reply_cost_sek(stated) == (pytest.approx(0.002 * USD_SEK), True)
    unknown = ModelReply(
        content="x", tool_calls=[], finish_reason="stop", prompt_tokens=1000, completion_tokens=100
    )
    sek, measured = reply_cost_sek(unknown)
    assert not measured and sek > reply_cost_sek(stated)[0]


async def test_metered_runtime_charges_real_usage_and_stops_at_the_cap(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, cap=1.0)
    with FakeOpenAI() as fake:
        fake.reply_cost = 0.01
        async with httpx.AsyncClient(trust_env=False) as client:
            cloud = CloudModelRuntime(CloudConfig(base_url=fake.base_url), KEY, client)
            meter = MeteredRuntime(cloud, ledger)
            reply = await meter.chat([{"role": "user", "content": "hej"}], [], 5, Sampling())
            assert reply.cost_usd == 0.01
            assert meter.cost_sek == pytest.approx(0.01 * USD_SEK)
            assert ledger.spent_sek() == pytest.approx(0.01 * USD_SEK)
            ledger.charge(0.8, measured=True)  # the day now has 1 - 0.915 kr left
            with pytest.raises(BudgetExhaustedError):
                await meter.chat([{"role": "user", "content": "hej"}], [], 5, Sampling())
    assert len(fake.requests) == 1


async def test_cloud_runtime_asks_openrouter_for_the_price() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0003},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cloud = CloudModelRuntime(CloudConfig(), KEY, client)
        reply = await cloud.chat([{"role": "user", "content": "hej"}], [], 5, Sampling())
    assert bodies[0]["usage"] == {"include": True}
    assert reply.cost_usd == 0.0003


# --- the decider ------------------------------------------------------------------------


def _survey_call(sector: str) -> dict[str, Any]:
    return {
        "id": "c1",
        "type": "function",
        "function": {"name": "survey_sector", "arguments": json.dumps({"sector": sector})},
    }


async def test_cloud_flight_is_an_exam_with_its_cost(tmp_path: Path) -> None:
    with FakeOpenAI() as fake:
        fake.reply_content = "Söker sektor A."
        fake.reply_calls = [_survey_call("A")]
        fake.reply_cost = 0.001
        async with httpx.AsyncClient(trust_env=False) as client:
            cloud = CloudModelRuntime(CloudConfig(base_url=fake.base_url), KEY, client)
            meter = MeteredRuntime(cloud, _ledger(tmp_path))
            decider = make_decision_maker(meter, BehaviourConfig())
            res = await fly(
                make_scenario(2), decider, model="m", ai_config_hash="h", meter=meter, max_turns=6
            )
    assert res.score.decider == "cloud" and res.score.exam and res.score.model == "m"
    assert res.score.cloud_calls == len(fake.requests) > 0
    assert res.score.cost_sek == pytest.approx(len(fake.requests) * 0.001 * USD_SEK, rel=1e-3)
    assert res.score.cost_measured


async def test_the_script_takes_over_at_the_cap_and_the_flight_is_no_exam(tmp_path: Path) -> None:
    with FakeOpenAI() as fake:
        fake.reply_content = "Söker sektor A."
        fake.reply_calls = [_survey_call("A")]
        fake.reply_cost = 0.01
        async with httpx.AsyncClient(trust_env=False) as client:
            cloud = CloudModelRuntime(CloudConfig(base_url=fake.base_url), KEY, client)
            meter = MeteredRuntime(cloud, _ledger(tmp_path), run_cap_sek=0.5)
            decider = make_decision_maker(meter, BehaviourConfig())
            res = await fly(make_scenario(5), decider, model="m", ai_config_hash="h", meter=meter)
    assert res.score.decider == "cloud" and not res.score.exam
    assert res.switch_reason == "budget"
    assert res.score.cost_sek <= 0.5
    assert res.score.safety.ok and res.score.landed_home  # the script finished the search
    assert make_decision_maker(meter, BehaviourConfig()).kind == "script"


async def test_cloud_error_hands_the_flight_to_the_script(tmp_path: Path) -> None:
    with FakeOpenAI() as fake:
        fake.behaviour = {"z-ai/glm-5.3-flash": "500"}
        async with httpx.AsyncClient(trust_env=False) as client:
            cfg = CloudConfig(base_url=fake.base_url, fallback_model="")
            meter = MeteredRuntime(CloudModelRuntime(cfg, KEY, client), _ledger(tmp_path))
            decider = make_decision_maker(meter, BehaviourConfig())
            res = await fly(make_scenario(6), decider, model="m", ai_config_hash="h", meter=meter)
    assert res.switch_reason == "cloud_error" and not res.score.exam
    assert res.score.safety.ok and res.score.cost_sek == 0


async def test_a_refused_key_is_key_invalid_and_the_cost_is_not_measured(tmp_path: Path) -> None:
    with FakeOpenAI() as fake:
        fake.behaviour = {"z-ai/glm-5.3-flash": "401"}
        async with httpx.AsyncClient(trust_env=False) as client:
            cfg = CloudConfig(base_url=fake.base_url, fallback_model="")
            meter = MeteredRuntime(CloudModelRuntime(cfg, KEY, client), _ledger(tmp_path))
            decider = make_decision_maker(meter, BehaviourConfig())
            res = await fly(make_scenario(6), decider, model="m", ai_config_hash="h", meter=meter)
            asked = len(fake.requests)
            nxt = make_decision_maker(meter, BehaviourConfig())
    assert meter.key_invalid and asked == 1  # one refusal, then the script flies
    assert res.switch_reason == "key_invalid" and not res.score.exam
    assert not res.score.cost_measured and res.score.safety.ok
    assert nxt.kind == "script" and nxt.switch_reason == "key_invalid"


# --- the example bank -------------------------------------------------------------------


async def test_bank_keeps_good_and_failed_exams_never_in_a_tracked_folder(tmp_path: Path) -> None:
    res = await fly(make_scenario(4), DecisionMaker(None, None), model="m", ai_config_hash="h")
    assert not worth_keeping(res.score)  # a script flight is no exam
    exam = res.score.model_copy(
        update={"exam": True, "decider": "cloud", "total": 80.0, "passed": True}
    )
    assert worth_keeping(exam)
    middling = exam.model_copy(update={"total": 60.0, "passed": True})
    failed = exam.model_copy(update={"total": 0.0, "passed": False})
    assert not worth_keeping(middling) and worth_keeping(failed)  # failures feed the list
    assert bank_name(failed).startswith("fail-") and bank_name(exam).startswith("good-")
    rec = example_record(exam, res.records, {"seed": 4})
    assert rec["schema"] == "sd-brain-example/1" and len(rec["turns"]) == len(res.records)
    path = LocalBank(tmp_path / "bank").put("f1", rec)
    assert json.loads(gzip.decompress(Path(path).read_bytes()))["scenario"] == {"seed": 4}  # noqa: ASYNC240


def test_bank_folder_in_a_repository_must_be_under_data(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    with pytest.raises(BankError, match="i repot"):
        LocalBank(repo / "bank")
    with pytest.raises(BankError, match="i repot"):
        LocalBank(repo / "software" / "data")
    LocalBank(repo / "data" / "example-bank")
    LocalBank(tmp_path / "outside")  # not in a repository


def test_r2_bank_must_be_in_the_eu_and_signs_its_put() -> None:
    with pytest.raises(BankError, match="EU"):
        R2Bank("https://" + "a" * 32 + ".r2.cloudflarestorage.com", "b-1", "k", "s")
    with pytest.raises(BankError, match="alla fyra"):
        bank_from_env({"SD_BANK_R2_BUCKET": "x"}, Path("/tmp/x"))  # noqa: S108
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    endpoint = "https://" + "a" * 32 + ".eu.r2.cloudflarestorage.com"
    bank = R2Bank(
        endpoint,
        "sd-bank",
        "AKID",
        "SECRET",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        now=lambda: dt.datetime(2026, 10, 18, tzinfo=dt.UTC),
    )
    assert bank.put("f1", {"a": 1}) == "r2://sd-bank/example-bank/f1.json.gz"
    req = seen[0]
    assert req.method == "PUT" and req.url.path == "/sd-bank/example-bank/f1.json.gz"
    auth = req.headers["authorization"]
    assert auth.startswith("AWS4-HMAC-SHA256 Credential=AKID/20261018/auto/s3/aws4_request")
    assert "SECRET" not in auth and json.loads(gzip.decompress(req.content)) == {"a": 1}
    env = dict(
        zip(
            [
                "SD_BANK_R2_ENDPOINT",
                "SD_BANK_R2_BUCKET",
                "SD_BANK_R2_ACCESS_KEY_ID",
                "SD_BANK_R2_SECRET_ACCESS_KEY",
            ],
            [endpoint, "sd-bank", "k", "s"],
            strict=True,
        )
    )
    assert isinstance(bank_from_env(env, Path("/tmp/x")), R2Bank)  # noqa: S108


# --- the trend and the command line -----------------------------------------------------


def test_trend_flags_a_worse_day(tmp_path: Path) -> None:
    for day, total in (("2026-10-17", 90.0), ("2026-10-18", 60.0)):
        s = score_flight(_log()).model_copy(update={"total": total, "flight_id": day})
        (tmp_path / day).mkdir()
        (tmp_path / day / "f.json").write_text(s.model_dump_json())
    rows = trend_rows(load_scores(tmp_path))
    assert [r["worse"] for r in rows] == [False, True]
    assert "<- sämre" in trend_text(rows)
    assert "Inga betyg" in trend_text([])


def test_cli_sim_without_a_key_flies_the_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("SD_CLOUD_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    args = ["eval", "--backend", "sim", "--flights", "2", "--seed", "7"]
    assert main([*args, "--report-dir", "reports", "--bank-dir", "bank"]) == 0
    out = capsys.readouterr().out
    assert "inget prov: key_invalid" in out and '"exams": 0' in out
    assert "kostnad ej uppmätt" in out and '"cost_sek": null' in out
    assert len(list((tmp_path / "data/brain-scores").glob("*/*.json"))) == 2
    assert len(list((tmp_path / "reports").glob("eval-sim-*.json"))) == 1
    assert main(["scores"]) == 0
    assert "script" in capsys.readouterr().out
    assert main(["scores", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["flights"] == 2
