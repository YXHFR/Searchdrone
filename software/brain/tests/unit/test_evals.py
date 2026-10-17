"""The golden evaluation: every scenario green with written answers (AC-B03)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from brain_helpers import GOLDEN, SCENARIOS

from sd_brain.evals.runner import (
    ScenarioResult,
    language_ok,
    load_scenario,
    run_scenario,
    summarize,
)
from sd_schemas.reason_codes import CHARTER_VALIDATION_CODES, GATE_REASON_CODES

PATHS = sorted(SCENARIOS.glob("*.yaml"))


def test_there_are_33_scenarios() -> None:
    ids = [load_scenario(p)["id"] for p in PATHS]
    assert ids == [f"EV-{i:02d}" for i in range(1, 34)]


@pytest.mark.parametrize("path", PATHS, ids=[p.stem for p in PATHS])
async def test_scenario_golden(path: Path) -> None:
    result = await run_scenario(path, golden_dir=GOLDEN)
    assert result.ok, result.failures


def test_language_check() -> None:
    for ok in ("Startar till 40 m.", "Stoppat. Drönaren står still.", "Sektor A är sökt."):
        assert language_ok(ok), ok
    assert not language_ok("I will fly to P1 now")


def test_summarize_rules() -> None:
    good = ScenarioResult(
        "EV-01", False, True, latencies_ms=[1000], turns=1, texts=["Klart och klart."]
    )
    bad = ScenarioResult("EV-02", True, False, failures=["x"])
    report = summarize([[good, bad]])
    assert not report["safety_ok"] and not report["passed"] and "EV-02" in report["failures"]
    assert summarize([[good]])["passed"]


def test_reason_codes_known() -> None:
    """Every reason code named in sd_brain or the scenarios exists (AC-B16)."""
    known = (
        set(GATE_REASON_CODES) | set(CHARTER_VALIDATION_CODES) | {"DRAFT_INVALID", "GROUND_ERROR"}
    )
    root = SCENARIOS.parents[1]
    texts = [p.read_text(encoding="utf-8") for p in root.rglob("*.py")]
    texts += [p.read_text(encoding="utf-8") for p in PATHS]
    found = set()
    for t in texts:
        found |= set(
            re.findall(
                r"\b(?:[A-Z]+_){1,4}(?:LIMIT|CHARTER|REQUIRED|ACTIVE|OPERATOR|HELD|CHANGED|LIMITS|INVALID|ERROR)\b",
                t,
            )
        )
    found -= {"ABSOLUTE_LIMIT", "WITHIN_CHARTER"}  # an autonomy mode, not a code
    assert found, "the pattern found nothing"
    assert found <= known, found - known


def test_no_old_names() -> None:
    root = SCENARIOS.parents[1]
    pattern = re.compile(
        r"G11_RATE|GM_CLEARANCE_REQUIRED|GM-CLEAR|GM-FORBIDDEN|SkillSpecWire|/heartbeat"
        r"|/stream|status: sent|config/local/brain|8081"
    )
    for p in [*root.rglob("*.py"), *root.rglob("*.yaml"), *root.rglob("*.md")]:
        assert not pattern.search(p.read_text(encoding="utf-8")), p
