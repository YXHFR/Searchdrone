"""The search runs' measures, their report and the nightly campaign's requirements."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sd_sim.eval_search import (
    LABEL,
    RunMetrics,
    aggregate,
    main,
    wilson_lower,
    write_run,
)


def run(scenario: str, seed: int, **kw: object) -> RunMetrics:
    m = RunMetrics(scenario=scenario, seed=seed, detector="det_oracle_v0", matcher="vlm_oracle_v0")
    for k, v in kw.items():
        setattr(m, k, v)
    return m.finish()


def test_found_needs_the_target_within_the_limit() -> None:
    assert run("SK01", 1, time_to_confirmed_s=300.0).found
    assert not run("SK01", 1, time_to_confirmed_s=950.0).found
    assert not run("SK03", 1, has_target=False, time_to_confirmed_s=10.0).found
    assert run("SK01", 1, pos_end=0.5, battery_used_pct=25.0).pos_per_batt_pct == 0.02


def test_wilson() -> None:
    assert wilson_lower(50, 50) == pytest.approx(0.9287, abs=1e-3)
    assert wilson_lower(0, 0) == 0.0


def test_the_report_is_swedish_labelled_and_has_no_profile_text(tmp_path: Path) -> None:
    card = {"seq": 1, "state": "confirmed", "priority": 0.8, "priority_class": "high",
            "engine": "oracle", "truth": "target"}  # fmt: skip
    m = run("SK01", 2, time_to_confirmed_s=321.0, queue_rank_target=1,
            engines=["code", "oracle"], queue_end=[card])  # fmt: skip
    path = write_run(tmp_path, m)
    body = path.read_text(encoding="utf-8")
    assert "Sökrapport SK01, frö 2" in body and LABEL in body
    assert "321 s" in body and "code, oracle" in body and "K1" in body
    assert "red" not in body and "röd" not in body
    data = json.loads((tmp_path / "search_metrics.json").read_text(encoding="utf-8"))
    assert data["found"] is True and data["label"] == LABEL


def _by(reqs: list, scenario: str) -> list:
    return [r for r in reqs if r.scenario == scenario]


def test_sk01_requirements() -> None:
    good = [run("SK01", s, time_to_confirmed_s=300.0, geo_error_m=3.0, queue_rank_target=1)
            for s in range(1, 51)]  # fmt: skip
    assert all(r.ok for r in _by(aggregate(r.as_dict() for r in good), "SK01"))
    bad = good[:45] + [run("SK01", s) for s in range(46, 51)]
    assert not _by(aggregate(r.as_dict() for r in bad), "SK01")[0].ok


def test_sk03_fails_on_any_confirmation() -> None:
    runs = [run("SK03", s, has_target=False) for s in range(1, 5)]
    assert all(r.ok for r in _by(aggregate(r.as_dict() for r in runs), "SK03"))
    runs[2].false_confirms = 1
    assert not all(r.ok for r in _by(aggregate(r.as_dict() for r in runs), "SK03"))


def test_sk04_needs_the_hikers_conflict() -> None:
    fields = {"green-jacket": {"garment:0": "conflict"}}
    runs = [run("SK04", s, time_to_confirmed_s=200.0, confirmed_thing="target",
                fields_by_thing=fields) for s in range(1, 11)]  # fmt: skip
    assert all(r.ok for r in _by(aggregate(r.as_dict() for r in runs), "SK04"))
    for r in runs[:3]:
        r.fields_by_thing = {"green-jacket": {"garment:0": "match"}}
    assert not all(r.ok for r in _by(aggregate(r.as_dict() for r in runs), "SK04"))


def test_sk06_pairs_seeds_and_counts_a_miss_as_its_limit() -> None:
    runs = []
    for s in range(1, 11):
        runs.append(run("SK06", s, with_map=True, time_to_confirmed_s=200.0 + s))
        runs.append(run("SK06", s, with_map=False, time_to_confirmed_s=400.0 + 3 * s))
    (req,) = _by(aggregate(r.as_dict() for r in runs), "SK06")
    assert req.ok and req.n == 10
    slow = [run("SK06", s, with_map=m) for s in range(1, 11) for m in (True, False)]
    (req,) = _by(aggregate(r.as_dict() for r in slow), "SK06")
    assert not req.ok  # 900 vs 0.8 x 900


def test_every_run_needs_a_fast_queue_and_no_cylinder_entry() -> None:
    ok = run("SK01", 1, latency_still_to_queue_ms={"p50": 1200, "p95": 2000})
    slow = run("SK01", 2, latency_still_to_queue_ms={"p50": 3500, "p95": 5000})
    reqs = _by(aggregate([ok.as_dict(), slow.as_dict()]), "alla")
    assert not reqs[0].ok
    entered = run("SK01", 3, invariants_broken=["E8: banan gick 3.0 m in"])
    assert not _by(aggregate([entered.as_dict()]), "alla")[1].ok


def test_the_cli_writes_the_requirements_and_fails_on_a_miss(tmp_path: Path) -> None:
    for s in range(1, 4):
        write_run(tmp_path / "runs" / f"r{s}", run("SK03", s, has_target=False))
    assert main(["aggregate", str(tmp_path / "runs"), str(tmp_path / "out")]) == 0
    assert "SK03" in (tmp_path / "out" / "search_requirements.md").read_text(encoding="utf-8")
    write_run(tmp_path / "runs" / "r9", run("SK03", 9, has_target=False, false_confirms=1))
    assert main(["aggregate", str(tmp_path / "runs"), str(tmp_path / "out")]) == 1
    assert main(["aggregate", str(tmp_path / "empty"), str(tmp_path / "out")]) == 1
