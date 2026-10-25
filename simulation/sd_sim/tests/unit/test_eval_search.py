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


def test_sk02_judges_the_share_and_shows_its_wilson_bound() -> None:
    runs = [run("SK02", s, time_to_confirmed_s=300.0, queue_rank_target=1,
                target_ever_low=s <= 2) for s in range(1, 51)]  # fmt: skip
    never_low = _by(aggregate(r.as_dict() for r in runs), "SK02")[2]
    assert never_low.ok  # 48/50 = 0.96, its lower bound below 0.95 does not decide
    assert never_low.value == f"48/50 = 0.96, Wilson-nedre {wilson_lower(48, 50):.2f}"
    runs[2].target_ever_low = True
    assert not _by(aggregate(r.as_dict() for r in runs), "SK02")[2].ok  # 47/50


def _sk04(tops: list[list[str]]) -> list:
    runs = [run("SK04", s, time_to_confirmed_s=200.0, confirmed_thing="target",
                upper_by_thing={"green-jacket": t, "target": ["match"]})
            for s, t in enumerate(tops, 1)]  # fmt: skip
    return _by(aggregate(r.as_dict() for r in runs), "SK04")


def test_sk04_counts_the_hikers_verdicts_that_saw_the_top() -> None:
    # 8 conflict and 2 match among the verdicts that saw the top; not_visible is left out.
    tops = [["conflict", "not_visible"]] * 8 + [["match"], ["match", "not_visible"]]
    first, hiker = _sk04(tops)
    assert first.ok and hiker.ok and not hiker.too_few
    assert hiker.value == "8/10 = 0.80 (19 kort)"
    first, hiker = _sk04([["conflict"]] * 7 + [["match"]] * 3)
    assert not hiker.ok and not hiker.too_few  # 0.70


def test_sk04_with_too_few_sightings_says_so_and_never_passes(tmp_path: Path) -> None:
    _, hiker = _sk04([["conflict"]] * 9 + [["not_visible"]] * 5)
    assert hiker.too_few and not hiker.ok
    assert hiker.value == "9 sedda av 14 kort"
    _, hiker = _sk04([[]] * 3)  # the hiker never seen
    assert hiker.too_few and not hiker.ok
    m = run("SK04", 1, upper_by_thing={"green-jacket": ["conflict"]})
    write_run(tmp_path / "runs" / "r1", m)
    assert main(["aggregate", str(tmp_path / "runs"), str(tmp_path / "out")]) == 1
    md = (tmp_path / "out" / "search_requirements.md").read_text(encoding="utf-8")
    assert "| för få data |" in md


def test_sk06_pairs_seeds_and_counts_a_miss_as_its_limit() -> None:
    runs = []
    for s in range(1, 11):
        runs.append(run("SK06", s, with_map=True, time_to_confirmed_s=200.0 + s))
        runs.append(run("SK06", s, with_map=False, time_to_confirmed_s=400.0 + 3 * s))
    (req,) = _by(aggregate(r.as_dict() for r in runs), "SK06")
    assert req.ok and req.n == 10 and "0,9 x" in req.text
    slow = [run("SK06", s, with_map=m) for s in range(1, 11) for m in (True, False)]
    (req,) = _by(aggregate(r.as_dict() for r in slow), "SK06")
    assert not req.ok  # 900 vs 0.9 x 900


def test_sk06_needs_ten_percent_off() -> None:
    def pairs(ratio: float) -> list[dict]:
        return [r.as_dict() for s in range(1, 41) for r in (
            run("SK06", s, with_map=True, time_to_confirmed_s=ratio * (200.0 + s)),
            run("SK06", s, with_map=False, time_to_confirmed_s=200.0 + s))]  # fmt: skip

    assert _by(aggregate(pairs(0.85)), "SK06")[0].ok
    assert not _by(aggregate(pairs(0.92)), "SK06")[0].ok


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
