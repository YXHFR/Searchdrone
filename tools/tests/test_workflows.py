"""Static rules for every workflow and composite action in .github/."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
ACTIONS = sorted((ROOT / ".github" / "actions").glob("*/action.yml"))
ALL_FILES = WORKFLOWS + ACTIONS
PINNED_RE = re.compile(r"^[\w.-]+/[\w.-]+(/[\w./-]+)?@[0-9a-f]{40}$")
USES_LINE_RE = re.compile(r"^\s*(-\s+)?uses:\s*(?P<ref>\S+)(?P<rest>.*)$")


def load(path: Path) -> dict[Any, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ids(paths: list[Path]) -> list[str]:
    return [str(p.relative_to(ROOT)) for p in paths]


def steps(doc: dict[Any, Any]) -> list[dict[str, Any]]:
    if "jobs" in doc:
        return [s for job in doc["jobs"].values() for s in job.get("steps", [])]
    return doc["runs"].get("steps", [])


def test_files_exist() -> None:
    names = {p.name for p in WORKFLOWS}
    assert {
        "ci.yml",
        "pr-title.yml",
        "security.yml",
        "review-gate.yml",
        "hardware.yml",
        "owner-gate.yml",
    } <= names
    assert {"sitl.yml", "sitl-image.yml"} <= names
    assert (ROOT / ".github" / "actions" / "setup" / "action.yml").is_file()


@pytest.mark.parametrize("path", ALL_FILES, ids=ids(ALL_FILES))
def test_actions_pinned_to_commit_sha(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        m = USES_LINE_RE.match(line)
        if not m:
            continue
        if m["ref"].startswith("./"):
            assert (ROOT / m["ref"] / "action.yml").is_file(), f"{path.name}: {line}"
            continue
        assert PINNED_RE.match(m["ref"]), f"{path.name}: not pinned to a commit SHA: {line}"
        assert re.search(r"#\s*v\d", m["rest"]), f"{path.name}: missing version comment: {line}"


@pytest.mark.parametrize("path", WORKFLOWS, ids=ids(WORKFLOWS))
def test_top_level_permissions(path: Path) -> None:
    doc = load(path)
    assert "permissions" in doc, f"{path.name}: no top-level permissions"
    assert doc["permissions"] in ({}, {"contents": "read"}), path.name


@pytest.mark.parametrize("path", WORKFLOWS, ids=ids(WORKFLOWS))
def test_jobs_run_on_ubuntu_2404_with_timeout(path: Path) -> None:
    for name, job in load(path)["jobs"].items():
        assert job.get("runs-on") == "ubuntu-24.04", f"{path.name}:{name}"
        assert isinstance(job.get("timeout-minutes"), int), f"{path.name}:{name}"


@pytest.mark.parametrize("path", ALL_FILES, ids=ids(ALL_FILES))
def test_checkout_never_persists_credentials(path: Path) -> None:
    for step in steps(load(path)):
        if str(step.get("uses", "")).startswith("actions/checkout@"):
            assert step.get("with", {}).get("persist-credentials") is False, path.name


@pytest.mark.parametrize("path", ALL_FILES, ids=ids(ALL_FILES))
def test_no_expressions_inside_run(path: Path) -> None:
    # Untrusted input (titles, file names, inputs) reaches scripts only through env:.
    for step in steps(load(path)):
        assert "${{" not in str(step.get("run", "")), f"{path.name}: {step.get('name')}"


@pytest.mark.parametrize("path", ALL_FILES, ids=ids(ALL_FILES))
def test_only_python_312(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for m in re.finditer(r"python-version:\s*\[?([^\]\n]*)", text):
        assert set(re.findall(r"3\.\d+", m[1])) <= {"3.12"}, path.name
    assert "3.13" not in text and "3.11" not in text, path.name


def test_ci_ok_needs_every_ci_job() -> None:
    jobs = load(ROOT / ".github" / "workflows" / "ci.yml")["jobs"]
    expected = {"lint", "typecheck", "unit", "contract", "panel-e2e", "reuse", "secrets", "build"}
    assert set(jobs) == expected | {"ci-ok"}
    assert set(jobs["ci-ok"]["needs"]) == expected
    assert jobs["ci-ok"]["if"] == "always()"


def test_security_ok_needs_every_security_job() -> None:
    jobs = load(ROOT / ".github" / "workflows" / "security.yml")["jobs"]
    assert set(jobs["security-ok"]["needs"]) == set(jobs) - {"security-ok"}
    assert jobs["security-ok"]["if"] == "always()"


@pytest.mark.parametrize("name", ["pr-title.yml", "review-gate.yml", "owner-gate.yml"])
def test_target_workflows_never_check_out_pr_code(name: str) -> None:
    doc = load(ROOT / ".github" / "workflows" / name)
    for step in steps(doc):
        if str(step.get("uses", "")).startswith("actions/checkout@"):
            ref = str(step.get("with", {}).get("ref", ""))
            assert "head" not in ref and "pull_request" not in ref, name
        assert "uv sync" not in str(step.get("run", "")), name
        assert "pnpm install" not in str(step.get("run", "")), name


def triggers(doc: dict[Any, Any]) -> dict[str, Any]:
    # PyYAML reads the bare key `on` as the boolean True.
    on = doc.get("on", doc.get(True))
    assert isinstance(on, dict)
    return on


def test_sitl_runs_on_every_pull_request() -> None:
    doc = load(ROOT / ".github" / "workflows" / "sitl.yml")
    on = triggers(doc)
    # No path filter on the workflow, or the required check would never report.
    assert on["pull_request"] is None
    assert on["push"] == {"branches": ["main"]}
    assert "merge_group" in on
    assert on["schedule"]
    assert set(on["workflow_dispatch"]["inputs"]) == {"mode", "ref"}
    assert on["workflow_dispatch"]["inputs"]["mode"]["options"] == ["pr", "nightly"]
    assert "pull_request_target" not in on


def test_sitl_jobs_and_aggregate() -> None:
    jobs = load(ROOT / ".github" / "workflows" / "sitl.yml")["jobs"]
    assert set(jobs) == {
        "changes",
        "sitl",
        "sitl-ok",
        "sitl-nightly",
        "sitl-nightly-gz",
        "sitl-e2e",
        "video",
    }
    assert jobs["sitl-ok"]["if"] == "always()"
    assert set(jobs["sitl-ok"]["needs"]) == {"changes", "sitl", "video"}
    assert "ci_aggregate.jq" in jobs["sitl-ok"]["steps"][-1]["run"]
    assert jobs["sitl"]["if"] == "needs.changes.outputs.run == 'true'"
    for name in ("sitl-nightly", "sitl-nightly-gz"):
        assert "schedule" in jobs[name]["if"] and "nightly" in jobs[name]["if"], name
    tests = [s for s in jobs["sitl"]["steps"] if "ci-sitl" in str(s.get("run", ""))]
    assert len(tests) == 1
    assert tests[0]["timeout-minutes"] <= 35
    run = tests[0]["run"]
    assert "physics=builtin" in run and "brain=scripted" in run


def test_sitl_change_filter_fails_closed() -> None:
    jobs = load(ROOT / ".github" / "workflows" / "sitl.yml")["jobs"]
    script = next(s["run"] for s in jobs["changes"]["steps"] if s.get("id") == "filter")
    # The diff goes to a file first, so a failing git diff fails the step (bash -e)
    # instead of looking like "nothing changed".
    assert "< <(" not in script
    assert 'git diff --name-only "$base...$head" > "$RUNNER_TEMP/changed.txt"' in script
    assert 'if [ -z "$base" ] || [ -z "$head" ]' in script
    # Deny-list: every file not listed as harmless runs the tests.
    assert "*) echo" in script and "run=true" in script


def test_package_write_only_when_pushing_from_main() -> None:
    for path in WORKFLOWS:
        for name, job in load(path)["jobs"].items():
            perms = job.get("permissions") or {}
            if perms.get("packages") == "write":
                assert (path.name, name) == ("sitl-image.yml", "build-push"), (path.name, name)
    jobs = load(ROOT / ".github" / "workflows" / "sitl-image.yml")["jobs"]
    assert "refs/heads/main" in jobs["build-push"]["if"]
    assert "pull_request" in jobs["build-push"]["if"]
    assert jobs["build"]["permissions"] == {"contents": "read"}


def test_image_scanned_before_push() -> None:
    steps = load(ROOT / ".github" / "workflows" / "sitl-image.yml")["jobs"]["build-push"]["steps"]
    names = [s.get("name", "") for s in steps]
    scan = next(i for i, n in enumerate(names) if "secrets" in n)
    push = names.index("Push")
    assert scan < push
    assert "--exit-code 1" in steps[scan]["run"]
    assert "docker push" not in "".join(str(s.get("run", "")) for s in steps[:push])


def test_sitl_image_tag_from_script() -> None:
    for name in ("sitl.yml", "sitl-image.yml"):
        assert "tools/sitl_image_tag.sh" in (ROOT / ".github" / "workflows" / name).read_text()


def test_video_job_runs_the_video_tests_without_skipping() -> None:
    job = load(ROOT / ".github" / "workflows" / "sitl.yml")["jobs"]["video"]
    assert job["if"] == "needs.changes.outputs.run == 'true'"
    runs = [s for s in job["steps"] if "test-video" in str(s.get("run", ""))]
    assert len(runs) == 1
    assert runs[0]["env"]["SD_SIM__REQUIRE_VIDEO"] == "1"
    assert any("just video-up" in str(s.get("run", "")) for s in job["steps"])
