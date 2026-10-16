"""tools/ci_aggregate.jq: the logic behind ci-ok, sitl-ok, hardware-ok and security-ok."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

JQ_FILE = Path(__file__).resolve().parents[1] / "ci_aggregate.jq"
JQ = shutil.which("jq")
# The required checks depend on this logic: in CI a missing jq is an error, not a skip.
pytestmark = pytest.mark.skipif(
    JQ is None and os.environ.get("CI") != "true", reason="jq is not installed"
)


def ok(needs: dict[str, str], run: str | None = None) -> bool:
    assert JQ is not None, "jq is required"
    doc = {name: {"result": result, "outputs": {}} for name, result in needs.items()}
    if "changes" in doc and run is not None:
        doc["changes"]["outputs"] = {"run": run}
    res = subprocess.run(
        [JQ, "-e", "-f", str(JQ_FILE)],
        input=json.dumps(doc),
        capture_output=True,
        text=True,
        check=False,
    )
    return res.returncode == 0


@pytest.mark.parametrize(
    ("jobs", "ok_job"),
    [(["sitl"], "sitl-ok"), (["hw-files", "hw-calc"], "hardware-ok")],
)
class TestFilteredAggregate:
    def test_changes_failed_is_red(self, jobs: list[str], ok_job: str) -> None:
        assert not ok({"changes": "failure", **dict.fromkeys(jobs, "skipped")}, run="false")

    def test_nothing_to_run_and_skipped_is_green(self, jobs: list[str], ok_job: str) -> None:
        assert ok({"changes": "success", **dict.fromkeys(jobs, "skipped")}, run="false")

    def test_job_failed_is_red(self, jobs: list[str], ok_job: str) -> None:
        needs = {"changes": "success", **dict.fromkeys(jobs, "success")}
        needs[jobs[0]] = "failure"
        assert not ok(needs, run="true")

    def test_skipped_although_filter_matched_is_red(self, jobs: list[str], ok_job: str) -> None:
        assert not ok({"changes": "success", **dict.fromkeys(jobs, "skipped")}, run="true")

    def test_all_success_is_green(self, jobs: list[str], ok_job: str) -> None:
        assert ok({"changes": "success", **dict.fromkeys(jobs, "success")}, run="true")


@pytest.mark.parametrize("bad", ["failure", "cancelled", "skipped"])
def test_ci_ok_requires_every_job_to_succeed(bad: str) -> None:
    jobs = dict.fromkeys(["lint", "typecheck", "unit", "contract", "build"], "success")
    assert ok(jobs)
    jobs["unit"] = bad
    assert not ok(jobs)


def test_missing_filter_output_never_allows_a_skip() -> None:
    assert not ok({"changes": "success", "sitl": "skipped"})


def test_ci_ok_with_no_jobs_is_red() -> None:
    assert not ok({})
