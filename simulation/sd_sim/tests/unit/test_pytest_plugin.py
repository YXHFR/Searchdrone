"""sd_sim.pytest_plugin: shard plan, the CI time budget and the worker instance."""

from __future__ import annotations

import json
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
    scenario_id,
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


def test_first_instance_moves_the_workers_up() -> None:
    """``just e2e``: the demo keeps instance 0, the rest start at 1."""
    assert worker_instance({"PYTEST_XDIST_WORKER": "gw0", "SD_SIM__FIRST_INSTANCE": "1"}) == 1
    assert worker_instance({"SD_SIM__FIRST_INSTANCE": "1"}) == 1
    with pytest.raises(pytest.UsageError):
        worker_instance({"PYTEST_XDIST_WORKER": "gw7", "SD_SIM__FIRST_INSTANCE": "1"})


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
            "sitl and not gz and not llm and not nightly and not panel",
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
    env["SD_SIM__RUNS_DIR"] = str(tmp_path / "runs")
    assert subprocess.run(base, env=env, capture_output=True, check=False).returncode == 0
    assert not (tmp_path / "runs").exists()  # skipped on purpose: no result
    env["SD_SIM__REQUIRE_SITL"] = "1"
    env["SD_ARDUCOPTER_BIN"] = sys.executable  # an existing file, so collection passes
    out = subprocess.run(base, env=env, capture_output=True, text=True, check=False)
    assert out.returncode == 1, out.stdout
    assert "hoppades över" in out.stdout
    (result,) = (tmp_path / "runs").glob("*-test_x-i*/result.json")
    assert json.loads(result.read_text())["passed"] is False
    env["SD_ARDUCOPTER_BIN"] = str(tmp_path / "missing")
    out = subprocess.run(base, env=env, capture_output=True, text=True, check=False)
    assert out.returncode != 0
    assert "SD_SIM__REQUIRE_SITL=1" in out.stdout + out.stderr


def test_scenario_id() -> None:
    assert scenario_id("test_sc02_stop_and_resume") == "SC02"
    assert scenario_id("test_sc10a_ground_restart_mid_search") == "SC10A"
    assert scenario_id("test_sc04_sc05_refusals_in_the_air") == "SC04"
    assert scenario_id("test_sc09_x[case-a]") == "SC09"
    assert scenario_id("test_demo_flight") == "test_demo_flight"
    assert scenario_id("test_scale") == "test_scale"
    assert scenario_id("test_demo_flight", "test_sc01") == "SC01"
    assert scenario_id("test_sc02_x", "test_sc01") == "SC02"  # the test's own wins
    assert scenario_id("test_gps_lost", "test_sc_failsafe") == "test_gps_lost"


def test_every_simulator_test_writes_its_measurements(tmp_path: Path) -> None:
    test = tmp_path / "test_x.py"
    test.write_text(
        "import pytest\n"
        "from sd_sim.report import measure\n\n"
        "@pytest.mark.sitl\n@pytest.mark.budget_s(10)\n"
        "def test_sc02_ok():\n"
        "    measure('BRAKE', 0.4, 2.0)\n"
        "    measure('holds', 20.1, 20.0, at_most=False)\n\n"
        "@pytest.mark.sitl\n@pytest.mark.budget_s(10)\n"
        "def test_sc03_late():\n"
        "    assert measure('flying home', 2.5, 2.0)\n\n"
        "@pytest.mark.video\n"
        "def test_video_plain():\n"
        "    pass\n\n"
        "def test_unit():\n"
        "    pass\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "SD_SIM__REQUIRE_SITL"}
    env["SD_SIM__RUNS_DIR"] = str(tmp_path / "runs")
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "sd_sim.pytest_plugin", "-q", "-p",
         "no:cacheprovider", "--rootdir", str(tmp_path), "-c", "/dev/null", str(test)],
        env=env, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert out.returncode == 1, out.stdout
    results = {
        r["variant"]: r
        for r in (json.loads(p.read_text()) for p in (tmp_path / "runs").glob("*/result.json"))
    }
    assert set(results) == {"test_sc02_ok", "test_sc03_late", "test_video_plain"}
    ok = results["test_sc02_ok"]
    assert ok["scenario"] == "SC02" and ok["passed"] is True
    assert [(c["text"], c["measured"], c["limit"], c["bound"]) for c in ok["criteria"][:2]] == [
        ("BRAKE", 0.4, 2.0, "max"),
        ("holds", 20.1, 20.0, "min"),
    ]
    assert ok["criteria"][-1]["id"] == "T" and ok["criteria"][-1]["passed"] is True
    late = results["test_sc03_late"]
    assert late["passed"] is False
    assert [c["passed"] for c in late["criteria"]] == [False, False]
    assert results["test_video_plain"]["passed"] is True


def test_a_repeated_attempt_leaves_one_result(tmp_path: Path) -> None:
    test = tmp_path / "test_x.py"
    marker = tmp_path / "first-done"
    test.write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "from sd_sim.report import measure\n\n"
        "class InfraError(Exception):\n    pass\n\n"
        "@pytest.mark.sitl\n@pytest.mark.budget_s(10)\n"
        "def test_sc02_flaky_rig():\n"
        f"    first = not Path({str(marker)!r}).exists()\n"
        f"    Path({str(marker)!r}).touch()\n"
        "    measure('BRAKE', 9.0 if first else 0.5, 2.0)\n"
        "    if first:\n"
        "        raise InfraError('port busy')\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "SD_SIM__REQUIRE_SITL"}
    env["SD_SIM__RUNS_DIR"] = str(tmp_path / "runs")
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "sd_sim.pytest_plugin", "-q", "-p",
         "no:cacheprovider", "--reruns", "1", "--only-rerun", "InfraError",
         "--rootdir", str(tmp_path), "-c", "/dev/null", str(test)],
        env=env, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert out.returncode == 0, out.stdout
    assert "1 rerun" in out.stdout
    results = [json.loads(p.read_text()) for p in (tmp_path / "runs").glob("*/result.json")]
    assert len(results) == 1, results
    assert results[0]["passed"] is True
    assert [c["measured"] for c in results[0]["criteria"]] == [0.5, "passed"]
