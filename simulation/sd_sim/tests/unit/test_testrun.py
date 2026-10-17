"""sd_sim.testrun: the pytest command line, exit codes and the run summary."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from sd_sim import paths, testrun


def test_pr_selection_matches_ci() -> None:
    argv = testrun.pytest_argv(
        {"physics": "builtin", "brain": "scripted", "shard": "2", "shards": "4", "workers": "4"}
    )
    assert argv[1:5] == ["-m", "pytest", "-p", "sd_sim.pytest_plugin"]
    assert argv[argv.index("-m", 2) + 1] == "sitl and not gz and not llm and not nightly"
    assert argv[argv.index("-n") + 1] == "4" and "--dist" in argv
    assert argv[argv.index("--shard-id") + 1] == "2"
    assert argv[argv.index("--num-shards") + 1] == "4"
    assert argv[argv.index("--reruns") : argv.index("--reruns") + 4] == [
        "--reruns",
        "1",
        "--only-rerun",
        "InfraError",
    ]
    assert argv[-2:] == list(testrun.TEST_PATHS)


def test_nightly_and_gz_selection() -> None:
    assert testrun.selection("nightly", "builtin") == "sitl and nightly"
    assert testrun.selection("pr", "gz") == "sitl and not llm and not nightly"


def test_single_worker_without_xdist_and_names() -> None:
    argv = testrun.pytest_argv({"sc": "SC04", "variant": "b"})
    assert "-n" not in argv
    assert argv[argv.index("-k") + 1] == "SC04 and b"


@pytest.mark.parametrize(
    "values",
    [
        {"mode": "weekly"},
        {"physics": "unity"},
        {"shard": "5", "shards": "4"},
        {"workers": "9"},
        {"shards": "x"},
    ],
)
def test_bad_values_exit_2(values: dict[str, str]) -> None:
    with pytest.raises(SystemExit) as exc:
        testrun.pytest_argv(values)
    assert exc.value.code == 2


class FakeRun:
    def __init__(self, code: int) -> None:
        self.code = code
        self.cmd: list[str] = []

    def __call__(self, cmd: list[str], **_kw: Any) -> subprocess.CompletedProcess[str]:
        self.cmd = cmd
        return subprocess.CompletedProcess(cmd, self.code)


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(paths, "runs_dir", lambda repo=paths.REPO: tmp_path / "runs")
    return tmp_path / "runs"


@pytest.mark.parametrize(
    ("mode", "pytest_code", "want"),
    [
        ("pr", 0, 0),
        ("pr", 1, 1),
        ("pr", 5, 1),
        ("nightly", 5, 0),
        ("nightly", 1, 1),
    ],
)
def test_exit_codes(
    mode: str, pytest_code: int, want: int, runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRun(pytest_code)
    monkeypatch.setattr(subprocess, "run", fake)
    assert testrun.main([f"mode={mode}"]) == want
    assert (runs / "summary.md").is_file()


@pytest.mark.parametrize(
    ("brain", "marks"),
    [
        ("scripted", "sitl and not gz and not llm and not nightly"),
        ("grammar", "sitl and not gz and not llm and not nightly"),
        ("llm_local", "sitl and not gz and not nightly"),
        ("none", "sitl and not gz and not llm and not nightly and not brain"),
    ],
)
def test_brain_reaches_pytest(brain: str, marks: str) -> None:
    argv = testrun.pytest_argv({"brain": brain})
    assert argv[argv.index("-m", 2) + 1] == marks
    assert argv[argv.index("--brain") + 1] == brain


def test_brain_is_not_ignored(
    runs: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(subprocess, "run", FakeRun(0))
    testrun.main(["brain=scripted"])
    assert "inte byggt än" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        testrun.pytest_argv({"brain": "cloud"})


def test_summary_lists_new_results(tmp_path: Path) -> None:
    old = tmp_path / "old" / "result.json"
    old.parent.mkdir()
    old.write_text(json.dumps({"scenario": "OLD", "passed": True}))
    start = time.time() + 1
    os.utime(old, (start - 100, start - 100))
    new = tmp_path / "new" / "result.json"
    new.parent.mkdir()
    new.write_text(json.dumps({"scenario": "SC00", "passed": False, "duration_s": 12.5}))
    os.utime(new, (start + 1, start + 1))
    bad = tmp_path / "bad" / "result.json"
    bad.parent.mkdir()
    bad.write_text("{")
    os.utime(bad, (start + 1, start + 1))
    results = testrun.results_since(start, tmp_path)
    text = testrun.summary_markdown(results, 1)
    assert "SC00 | - | nej | - | 12.5 | new" in text
    assert "OLD" not in text
    assert "| ? | - | nej" in text
