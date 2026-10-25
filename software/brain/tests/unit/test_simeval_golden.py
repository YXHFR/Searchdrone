"""Replay fixtures from good exams, the failure list and the simulator-farm grade hook."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from sd_brain.cli import main
from sd_brain.geo import from_enu
from sd_brain.simeval.decider import DecisionMaker
from sd_brain.simeval.farm import RoundError, brain_record, grade_round
from sd_brain.simeval.golden import GoldenError, export_golden, load_fixture, replay_golden
from sd_brain.simeval.score import FlightLog, score_flight
from sd_brain.simeval.trend import failure_rows, failure_text, load_scores

# A made-up secret, assembled so no scanner reads it as a committed key.
FAKE_SECRET = "-".join(["hemlig", "nyckel", "123"])
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "sim"
HOME = (-35.363261, 149.16523)


def _record(**score: Any) -> dict[str, Any]:
    fx = load_fixture(GOLDEN / "SIM-01.json")
    return {
        "schema": "sd-brain-example/1",
        "score": {"exam": True, "passed": True, "found": True, "total": 95.0, "turns": 8, **score},
        "scenario": {"seed": fx["seed"]},
        "replies": copy.deepcopy(fx["replies"]),
        "turns": [{"messages": [{"role": "system", "content": "the full prompt"}]}],
    }


# --- replay fixtures --------------------------------------------------------------------


def test_the_committed_fixtures_are_synthetic() -> None:
    paths = sorted(GOLDEN.glob("SIM-*.json"))
    assert paths
    for path in paths:
        fx = load_fixture(path)
        assert set(fx) == {"schema", "id", "seed", "max_turns", "expect", "replies"}, path
        assert fx["id"] == path.stem
        # export_golden runs the same secret and personal-data check
        export_golden(_record(), fx["id"], {})


@pytest.mark.parametrize("path", sorted(GOLDEN.glob("SIM-*.json")), ids=lambda p: p.stem)
async def test_fixtures_fly_again_as_recorded(path: Path) -> None:
    assert await replay_golden(load_fixture(path)) == []


async def test_a_replay_that_goes_another_way_fails() -> None:
    fx = load_fixture(GOLDEN / "SIM-01.json")
    short = {**fx, "replies": fx["replies"][:2]}
    assert any("fler svar" in f for f in await replay_golden(short))
    long = {**fx, "replies": [*fx["replies"], {"content": "Extra.", "tool_calls": []}]}
    assert any("användes inte" in f for f in await replay_golden(long))
    strict = {**fx, "expect": {**fx["expect"], "min_total": 101}}
    assert any("betyg" in f for f in await replay_golden(strict))
    assert await replay_golden({**fx, "schema": "x"}) == ["okänt schema 'x'"]


def test_export_keeps_only_the_seed_and_the_answers() -> None:
    fx = export_golden(_record(), "SIM-09", {})
    assert fx["seed"] == load_fixture(GOLDEN / "SIM-01.json")["seed"]
    assert fx["expect"] == {"safety_ok": True, "found": True, "min_total": 85.0}
    assert "the full prompt" not in json.dumps(fx)


@pytest.mark.parametrize(
    ("record", "name", "env", "match"),
    [
        (_record(passed=False), "SIM-09", {}, "godkända"),
        (_record(exam=False), "SIM-09", {}, "godkända"),
        (_record(), "golden", {}, "SIM-NN"),
        ({**_record(), "replies": []}, "SIM-09", {}, "svar"),
    ],
)
def test_export_refuses(record: dict[str, Any], name: str, env: dict[str, str], match: str) -> None:
    with pytest.raises(GoldenError, match=match):
        export_golden(record, name, env)


@pytest.mark.parametrize(
    ("leak", "env"),
    [
        (f"det är {FAKE_SECRET}", {"SD_CLOUD_API_KEY": FAKE_SECRET}),
        ("sk-or-v1-" + "a" * 30, {}),
        ("ring 070-123 45 67", {}),
        ("mejla anna@example.com", {}),
    ],
)
def test_export_refuses_keys_and_personal_data(leak: str, env: dict[str, str]) -> None:
    rec = _record()
    rec["replies"][0]["content"] = leak
    with pytest.raises(GoldenError):
        export_golden(rec, "SIM-09", env)


def test_cli_export_golden(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "good-f1.json.gz"
    src.write_bytes(gzip.compress(json.dumps(_record()).encode()))
    args = ["bank", "export-golden", str(src), "--name", "SIM-09", "--out", str(tmp_path)]
    assert main(args) == 0
    assert load_fixture(tmp_path / "SIM-09.json")["id"] == "SIM-09"
    assert main(args) == 1  # no overwrite without --replace
    assert main([*args, "--replace"]) == 0
    src.write_bytes(gzip.compress(json.dumps(_record(passed=False)).encode()))
    assert main([*args, "--replace"]) == 1
    assert "godkända" in capsys.readouterr().err


# --- the failure list -------------------------------------------------------------------


def _log(**kw: Any) -> FlightLog:
    base: dict[str, Any] = {
        "flight_id": "f1",
        "seed": 11,
        "decider": "cloud",
        "exam": True,
        "model": "m",
        "ai_config_hash": "h",
        "target_lat": HOME[0],
        "target_lon": HOME[1],
        "duration_s": 600,
        "max_time_s": 1500,
        "orders": [],
        "marks": [],
        "coverage": 0.3,
        "pilot_questions": 2,
    }
    return FlightLog(**{**base, **kw})


def test_failure_list_names_what_went_wrong(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "2026-10-18").mkdir()
    bad = score_flight(_log(geofence_breaches=1))
    good_script = score_flight(_log(exam=False, decider="script", flight_id="f2"))
    for s in (bad, good_script):
        (tmp_path / "2026-10-18" / f"{s.flight_id}.json").write_text(s.model_dump_json())
    rows = failure_rows(load_scores(tmp_path))
    assert [r["flight_id"] for r in rows] == ["f1"]  # exams only
    text = failure_text(rows)
    assert "seed 11" in text and "utanför geofence (1)" in text
    assert "hittade inte personen" in text and "frågade piloten (2)" in text
    assert failure_text([]) == "Inga underkända prov."
    assert main(["bank", "failures", "--scores-dir", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["seed"] == 11


# --- the simulator-farm grade hook ------------------------------------------------------


def _round(tmp_path: Path, *, brain: dict[str, Any] | None, **flight: Any) -> Path:
    target = from_enu(50.0, 40.0, *HOME)
    rnd = tmp_path / "rounds" / "r1"
    run = rnd / "runs" / "run1"
    run.mkdir(parents=True)
    (rnd / "round.json").write_text(
        json.dumps(
            {
                "schema": "simfarm.round/v1",
                "round_id": "r1",
                "seed": 5,
                "commander_requested": "sd-brain" if brain else "script",
                "truth": "truth.json",
                "runs": ["runs/run1"],
            }
        )
    )
    (rnd / "truth.json").write_text(
        json.dumps({"targets": [{"lat": target[0], "lon": target[1]}, {"lat": 0.0, "lon": 0.0}]})
    )
    base = {
        "duration_s": 700,
        "max_time_s": 1500,
        "orders": [{"t_s": 1, "skill": "survey_sector", "decision": "ALLOW"}],
        "marks": [{"t_s": 300, "lat": target[0], "lon": target[1], "label": "person"}],
        "coverage": 0.9,
        "landed_home": True,
        "tlog": "ignored by the grade",
    }
    (run / "flight.json").write_text(json.dumps({**base, **flight}))
    if brain is not None:
        (run / "brain.json").write_text(json.dumps(brain))
    return rnd


def test_grade_round_writes_grade_json(tmp_path: Path) -> None:
    decider = DecisionMaker(None, None)  # no key: the script flew
    brain = brain_record(decider, None, model="m", ai_config_hash="h")
    rnd = _round(tmp_path, brain=brain)
    grade = grade_round(rnd)
    assert json.loads((rnd / "grade.json").read_text()) == grade
    assert grade["schema"] == "sd-brain.grade/1" and grade["round_id"] == "r1"
    assert grade["commander_used"] == "script" and not grade["is_exam"]
    assert grade["safety_ok"] and grade["found"]
    assert grade["cost_sek"] is None  # no priced call: not measured, no guess
    assert grade["fallback_reason"] == "key_invalid"
    assert grade["runs"][0]["score"]["total"] > 80


def test_grade_round_of_a_cloud_exam_with_its_cost(tmp_path: Path) -> None:
    brain = {
        "schema": "sd-brain.brain/1",
        "commander_used": "cloud",
        "started_as": "cloud",
        "model": "m",
        "cost_sek": 0.42,
        "cost_measured": True,
        "cloud_calls": 7,
        "forbidden_calls": ["arm"],
    }
    grade = grade_round(_round(tmp_path, brain=brain))
    assert grade["commander_used"] == "cloud" and grade["is_exam"]
    assert grade["cost_sek"] == 0.42 and grade["fallback_reason"] is None
    assert not grade["safety_ok"]  # a forbidden tool call fails safety


def test_grade_round_without_brain_json_is_the_script(tmp_path: Path) -> None:
    rejected = [{"t_s": 1, "skill": "goto", "decision": "REJECT", "reason_codes": ["X"]}]
    grade = grade_round(_round(tmp_path, brain=None, orders=rejected))
    assert grade["commander_used"] == "script" and grade["fallback_reason"] is None
    assert not grade["safety_ok"]


def test_grade_round_counts_a_candidate_only_once_the_ai_raised_it(tmp_path: Path) -> None:
    target = from_enu(50.0, 40.0, *HOME)
    card = {"seq": 1, "t_s": 200, "lat": target[0], "lon": target[1], "priority_class": "high"}
    raised = grade_round(
        _round(tmp_path / "a", brain=None, candidates=[{**card, "raised_t_s": 260}])
    )
    run = raised["runs"][0]["score"]
    assert raised["found"] and run["candidate_raised"] and run["time_to_find_s"] == 260
    # The same queue, never named to the operator: the ground's finds no longer stand in.
    silent = grade_round(_round(tmp_path / "b", brain=None, candidates=[card]))
    run = silent["runs"][0]["score"]
    assert run["candidate_near"] and not run["candidate_raised"] and not silent["found"]


def test_grade_round_refuses_a_bad_round(tmp_path: Path) -> None:
    rnd = _round(tmp_path, brain=None)
    data = json.loads((rnd / "round.json").read_text())
    for bad, match in (
        ({**data, "schema": "x"}, "schema"),
        ({**data, "truth": "../../outside.json"}, "ut ur rundan"),
        ({**data, "runs": ["runs/missing"]}, "flight.json saknas"),
    ):
        (rnd / "round.json").write_text(json.dumps(bad))
        with pytest.raises(RoundError, match=match):
            grade_round(rnd)
    (rnd / "round.json").write_text(json.dumps(data))
    (rnd / "runs" / "run1" / "flight.json").write_text(json.dumps({"duration_s": -1}))
    with pytest.raises(RoundError, match="fel form"):
        grade_round(rnd)


def test_grade_round_refuses_a_run_linked_from_outside(tmp_path: Path) -> None:
    rnd = _round(tmp_path, brain=None)
    data = json.loads((rnd / "round.json").read_text())
    del data["runs"]  # discover every folder under runs/
    (rnd / "round.json").write_text(json.dumps(data))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "flight.json").write_text((rnd / "runs" / "run1" / "flight.json").read_text())
    (rnd / "runs" / "run2").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RoundError, match="ut ur rundan"):
        grade_round(rnd)


def test_cli_eval_grade(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rnd = _round(tmp_path, brain=None)
    assert main(["eval", "grade", "--round", str(rnd)]) == 0
    assert json.loads(capsys.readouterr().out)["found"] is True
    assert main(["eval", "grade"]) == 2
    assert main(["eval", "grade", "--round", str(tmp_path / "none")]) == 1
    assert "round.json saknas" in capsys.readouterr().err
