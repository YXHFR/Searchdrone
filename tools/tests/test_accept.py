"""``tools/accept/v0_1_0.py`` (``just accept``) with faked ``gh`` answers and commands."""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "accept"))

import v0_1_0 as accept  # pyright: ignore[reportMissingImports]

SHA = "a" * 40
NOW = dt.datetime(2026, 10, 18, 12, 0, tzinfo=dt.UTC)
IDS = """
[meta]
scenarios = ["SC01"]

[[id]]
id = "R-05"
typ = "A"
dod_ref = "3"

[[id]]
id = "F-01"
typ = "A"
dod_ref = "7"

[[id]]
id = "F-02"
typ = "M"
dod_ref = "6"

[[id]]
id = "R-06"
typ = "A + M"
dod_ref = "4"

[[id]]
id = "F-06"
typ = "A"
dod_ref = "6"

[[id]]
id = "V-04"
typ = "A"
dod_ref = "13"
"""
MANUAL_OK = """
[F-02]
pass = true
by = "YXHFR"
date = "2026-10-18"
note = "3 av 3"

[R-06]
pass = true
by = "YXHFR"
date = "2026-10-18"
"""


class FakeGh:
    """The GitHub API as the owner sees it when everything is green."""

    def __init__(self, overrides: dict[str, Any] | None = None) -> None:
        self.calls: list[str] = []
        self.overrides = overrides or {}

    def api(self, path: str) -> Any:
        self.calls.append(path)
        if path in self.overrides:
            return self.overrides[path]
        if path.endswith("/check-runs?per_page=100"):
            names = ("ci-ok", "sitl-ok", "hardware-ok", "e2e")
            return {"check_runs": [{"name": n, "conclusion": "success"} for n in names]}
        if "/actions/workflows/sitl.yml/runs" in path:
            return {"workflow_runs": [{"id": 7, "event": "workflow_dispatch"}]}
        if path.endswith("/actions/runs/7/jobs?per_page=100"):
            return {
                "jobs": [{"name": f"sitl-e2e ({n})", "conclusion": "success"} for n in range(1, 11)]
            }
        if path.endswith("/rulesets"):
            return [{"name": "main-protection"}, {"name": "release-tags"}]
        if path.endswith("/commits/main"):
            return {"sha": SHA}
        raise AssertionError(f"unexpected gh api {path}")


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run(self, argv: Sequence[str]) -> tuple[int, str]:
        self.calls.append(list(argv))
        return 0, ""


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    ids = tmp_path / "accept-ids.toml"
    ids.write_text(IDS, encoding="utf-8")
    monkeypatch.setenv("SD_ACCEPT_IDS", str(ids))
    reports = tmp_path / "data/reports"
    reports.mkdir(parents=True)
    perf = reports / "perf-20261018T100000Z.json"
    perf.write_text(json.dumps({"gpu_name": "NVIDIA GeForce RTX 3070", "git_sha": SHA}), "utf-8")
    return tmp_path


def _report(root: Path, phase: str = "pre") -> list[dict[str, Any]]:
    [path] = sorted((root / "data/reports").glob(f"accept-v0.1.0-{phase}-*.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def _run(root: Path, *argv: str, gh: FakeGh | None = None) -> int:
    return accept.main(list(argv), gh=gh or FakeGh(), runner=FakeRunner(), root=root, now=NOW)


def test_everything_green_exits_0(root: Path) -> None:
    (root / accept.MANUAL_FILE).write_text(MANUAL_OK, encoding="utf-8")
    assert _run(root, "--phase", "pre") == 0
    rows = _report(root)
    assert [r["id"] for r in rows] == ["R-05", "F-01", "F-02", "R-06", "F-06"]  # not V-04
    assert all(r["pass"] is True and r["sha"] == SHA for r in rows), rows
    assert set(rows[0]) == {"id", "typ", "pass", "detalj", "dod_ref", "sha"}


def test_a_missing_manual_row_is_null_and_exits_1(root: Path) -> None:
    manual = tomllib.loads(MANUAL_OK)
    del manual["F-02"]
    (root / accept.MANUAL_FILE).write_text(
        '[R-06]\npass = true\nby = "YXHFR"\ndate = "2026-10-18"\n', encoding="utf-8"
    )
    assert _run(root) == 1
    by_id = {r["id"]: r for r in _report(root)}
    assert by_id["F-02"]["pass"] is None
    assert accept.MANUAL_FILE in by_id["F-02"]["detalj"]
    assert by_id["R-05"]["pass"] is True


def test_without_a_manual_file_every_manual_row_is_null(root: Path) -> None:
    assert _run(root) == 1
    by_id = {r["id"]: r for r in _report(root)}
    assert by_id["F-02"]["pass"] is None and by_id["R-06"]["pass"] is None


def test_pass_true_needs_who_and_when(root: Path) -> None:
    (root / accept.MANUAL_FILE).write_text(
        MANUAL_OK.replace('by = "YXHFR"\ndate = "2026-10-18"\nnote = "3 av 3"', ""),
        encoding="utf-8",
    )
    assert _run(root) == 1
    assert {r["id"]: r for r in _report(root)}["F-02"]["pass"] is False


def test_a_red_check_fails_its_row(root: Path) -> None:
    (root / accept.MANUAL_FILE).write_text(MANUAL_OK, encoding="utf-8")
    gh = FakeGh(
        {
            f"repos/{accept.REPO}/commits/{SHA}/check-runs?per_page=100": {
                "check_runs": [
                    {"name": "ci-ok", "conclusion": "success"},
                    {"name": "sitl-ok", "conclusion": "failure"},
                ]
            }
        }
    )
    assert _run(root, "--sha", SHA, gh=gh) == 1
    by_id = {r["id"]: r for r in _report(root)}
    assert by_id["R-05"]["pass"] is False and "sitl-ok=failure" in by_id["R-05"]["detalj"]
    assert by_id["F-01"]["pass"] is False


def test_an_old_or_foreign_machine_report_is_red(root: Path) -> None:
    (root / accept.MANUAL_FILE).write_text(MANUAL_OK, encoding="utf-8")
    perf = root / "data/reports/perf-20261018T100000Z.json"
    perf.write_text(json.dumps({"gpu_name": "Some GPU", "git_sha": "b" * 40}), "utf-8")
    old = (NOW - dt.timedelta(days=8)).timestamp()
    os.utime(perf, (old, old))
    assert _run(root) == 1
    detail = {r["id"]: r for r in _report(root)}["F-06"]["detalj"]
    assert "7 dagar" in detail and "RTX 3070" in detail and "git_sha" in detail


def test_post_phase_checks_only_the_release_rows(root: Path) -> None:
    gh = FakeGh(
        {
            f"repos/{accept.REPO}/git/ref/tags/v0.1.0": {"object": {"type": "commit", "sha": SHA}},
            f"repos/{accept.REPO}/releases/tags/v0.1.0": {"draft": False},
        }
    )
    assert _run(root, "--phase", "post", gh=gh) == 0
    assert [r["id"] for r in _report(root, "post")] == ["V-04"]


def test_an_automatic_row_without_a_check_is_red(root: Path) -> None:
    ids = Path(os.environ["SD_ACCEPT_IDS"])
    ids.write_text(IDS + '\n[[id]]\nid = "X-99"\ntyp = "A"\ndod_ref = "1"\n', "utf-8")
    (root / accept.MANUAL_FILE).write_text(MANUAL_OK, encoding="utf-8")
    assert _run(root) == 1
    row = {r["id"]: r for r in _report(root)}["X-99"]
    assert row["pass"] is False and "ingen automatisk kontroll" in row["detalj"]


def test_missing_acceptance_list_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SD_ACCEPT_IDS", str(tmp_path / "nope.toml"))
    assert _run(tmp_path) == 2
    assert "SD_ACCEPT_IDS" in capsys.readouterr().err


def test_gh_only_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    class Done:
        returncode = 0
        stdout = "[[1, 2], [3]]"
        stderr = ""

    def fake_run(argv: list[str], **_: Any) -> Done:
        seen.append(argv)
        return Done()

    monkeypatch.setattr(accept.subprocess, "run", fake_run)
    assert accept.GhCli().api("repos/x/y/pulls") == [1, 2, 3]
    assert seen[0][:4] == ["gh", "api", "--method", "GET"]


def test_the_template_never_passes() -> None:
    template = Path(__file__).resolve().parents[1] / "accept/accept-manual.example.toml"
    data = tomllib.loads(template.read_text(encoding="utf-8"))
    assert data and all(v["pass"] is False for v in data.values())
