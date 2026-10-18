"""sd_sim.report: when a scenario counts as passed, and what result.json holds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sd_sim.report import Report, measure, take_measured


def make(tmp_path: Path) -> Report:
    return Report("SC00", tmp_path, 0, "builtin", "scripted")


def test_passes_with_true_and_not_built(tmp_path: Path) -> None:
    rep = make(tmp_path)
    rep.check("A", "a", True, measured=1, limit=2, unit="s")
    rep.not_built("B", "b", "inte byggt än (X)")
    assert rep.passed
    data = json.loads(rep.write().read_text())
    assert data["schema"] == 1 and data["passed"] is True
    assert data["not_built"] == 1
    crit = {c["id"]: c for c in data["criteria"]}
    assert crit["B"] == {"id": "B", "text": "b", "passed": None, "note": "inte byggt än (X)"}
    assert crit["A"]["unit"] == "s" and "note" not in crit["A"]


def test_false_infra_error_and_empty_fail(tmp_path: Path) -> None:
    rep = make(tmp_path)
    assert not rep.passed  # nothing checked
    assert "inget kriterium kontrollerades" in rep.failures()
    rep.check("A", "a", True)
    rep.check("B", "b", False, measured=3, limit=2)
    assert not rep.passed
    assert rep.failures() == ["B b: 3 (gräns 2)"]
    ok = make(tmp_path)
    ok.check("A", "a", True)
    ok.infra_failure = "port busy"
    assert not ok.passed
    ok.infra_failure = None
    ok.error = "KeyError: x"
    assert not ok.passed


def test_only_not_built_may_be_null(tmp_path: Path) -> None:
    rep = make(tmp_path)
    with pytest.raises(TypeError):
        rep.check("A", "a", None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ange"):
        rep.not_built("B", "b", " ")


def test_artifacts_listed(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "sitl.log").write_text("x")
    rep = make(tmp_path)
    rep.check("A", "a", True)
    data = json.loads(rep.write().read_text())
    assert data["artifacts"] == ["logs/sitl.log"]


def test_measure_records_against_the_limit() -> None:
    take_measured()
    assert measure("BRAKE", 0.43219, 2.0) is True
    assert measure("BRAKE", 2.5, 2.0) is False
    assert measure("coverage", 0.97, 0.95, at_most=False, unit="") is True
    got = take_measured()
    assert [(c.id, c.passed, c.measured, c.bound) for c in got] == [
        ("M1", True, 0.432, "max"),
        ("M2", False, 2.5, "max"),
        ("M3", True, 0.97, "min"),
    ]
    assert take_measured() == []
