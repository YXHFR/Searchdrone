"""sd_sim.pytest_plugin: shard plan, the CI time budget and the worker instance."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from sd_sim import paths
from sd_sim.pytest_plugin import (
    MARKERS,
    MAX_NODE_S,
    SHARD_LIMIT_S,
    budget_problems,
    plan_shards,
    worker_instance,
)

SITL_YML = paths.REPO / ".github" / "workflows" / "sitl.yml"


def test_lpt_balances_and_is_deterministic() -> None:
    budgets = [("a", 300.0), ("b", 200.0), ("c", 200.0), ("d", 100.0), ("e", 100.0)]
    plan = plan_shards(budgets, 2)
    assert sorted(n for s in plan.shards for n in s) == ["a", "b", "c", "d", "e"]
    assert sorted(plan.sums) == [400.0, 500.0]
    assert plan_shards(list(reversed(budgets)), 2) == plan


def test_one_shard_takes_everything() -> None:
    plan = plan_shards([("x", 1.0), ("y", 2.0)], 1)
    assert plan.shards == [["y", "x"]] and plan.sums == [3.0]


def test_zero_shards_rejected() -> None:
    with pytest.raises(ValueError, match="minst 1"):
        plan_shards([], 0)


def test_budget_problems() -> None:
    assert budget_problems([("a", 120.0)], 1, 4) == []
    assert any("budget_s" in p for p in budget_problems([("a", MAX_NODE_S + 1.0)], 1, 4))
    too_much = [(f"n{i}", 800.0) for i in range(10)]  # 8000 s on one worker
    problems = budget_problems(too_much, 1, 1)
    assert any("minst 6 skärvor" in p for p in problems), problems


@pytest.mark.parametrize(("env", "want"), [({}, 0), ({"PYTEST_XDIST_WORKER": "gw3"}, 3)])
def test_worker_instance(env: dict[str, str], want: int) -> None:
    assert worker_instance(env) == want


def test_worker_instance_limit() -> None:
    with pytest.raises(pytest.UsageError):
        worker_instance({"PYTEST_XDIST_WORKER": "gw8"})


def collected_budgets() -> list[tuple[str, float]]:
    """Collect the CI selection with the plugin and read each test's budget."""
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "sd_sim.pytest_plugin",
            "--collect-only",
            "-q",
            "-m",
            "sitl and not gz and not llm and not nightly",
            "-p",
            "no:cacheprovider",
            "simulation/tests",
            "simulation/sd_sim/tests/sitl",
        ],
        cwd=paths.REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode in (0, 5), out.stdout + out.stderr
    nodes = [line for line in out.stdout.splitlines() if "::" in line]
    budgets = []
    for node in nodes:
        path = paths.REPO / node.split("::")[0]
        text = path.read_text(encoding="utf-8")
        found = re.findall(r"budget_s\((\d+)\)", text)
        assert found, f"{node}: budget_s saknas"
        budgets.append((node, float(found[0])))
    return budgets


def test_sitl_budget() -> None:
    """The blocking SITL suite fits the shards and workers that sitl.yml uses."""
    env = yaml.safe_load(SITL_YML.read_text(encoding="utf-8"))["env"]
    shards, workers = int(env["SHARDS"]), int(env["WORKERS"])
    matrix = yaml.safe_load(SITL_YML.read_text(encoding="utf-8"))["jobs"]["sitl"]["strategy"]
    assert matrix["matrix"]["shard"] == list(range(1, shards + 1))
    budgets = collected_budgets()
    assert budgets, "inga SITL-tester samlades in"
    assert budget_problems(budgets, shards, workers) == [], (SHARD_LIMIT_S, budgets)


def test_missing_budget_is_a_usage_error(tmp_path: Path) -> None:
    test = tmp_path / "test_x.py"
    test.write_text("import pytest\n\n@pytest.mark.sitl\ndef test_x():\n    pass\n")
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "sd_sim.pytest_plugin",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "--rootdir",
            str(tmp_path),
            "-c",
            "/dev/null",
            str(test),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode != 0
    assert "budget_s" in out.stdout + out.stderr


def test_markers_match_pyproject() -> None:
    import tomllib

    data = tomllib.loads((paths.REPO / "pyproject.toml").read_text(encoding="utf-8"))
    names = {m.split(":")[0] for m in data["tool"]["pytest"]["ini_options"]["markers"]}
    assert {m.split(":")[0] for m in MARKERS} <= names


def test_required_sitl_skip_fails(tmp_path: Path) -> None:
    test = tmp_path / "test_x.py"
    test.write_text(
        "import pytest\n\n@pytest.mark.sitl\n@pytest.mark.budget_s(10)\n"
        "@pytest.mark.skipif(True, reason='no sitl')\ndef test_x():\n    pass\n"
    )
    base = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "sd_sim.pytest_plugin",
        "-q",
        "-p",
        "no:cacheprovider",
        "--rootdir",
        str(tmp_path),
        "-c",
        "/dev/null",
        str(test),
    ]
    env = {k: v for k, v in os.environ.items() if k != "SD_SIM__REQUIRE_SITL"}
    assert subprocess.run(base, env=env, capture_output=True, check=False).returncode == 0
    env["SD_SIM__REQUIRE_SITL"] = "1"
    env["SD_ARDUCOPTER_BIN"] = sys.executable  # an existing file, so collection passes
    out = subprocess.run(base, env=env, capture_output=True, text=True, check=False)
    assert out.returncode == 1, out.stdout
    assert "hoppades över" in out.stdout
    env["SD_ARDUCOPTER_BIN"] = str(tmp_path / "missing")
    out = subprocess.run(base, env=env, capture_output=True, text=True, check=False)
    assert out.returncode != 0
    assert "SD_SIM__REQUIRE_SITL=1" in out.stdout + out.stderr
