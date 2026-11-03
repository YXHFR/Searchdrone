"""Static rules for every workflow and composite action in .github/."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import check_owner_gate
import pytest
import yaml
from workflow_rules import GATE_WORKFLOWS, approval_writers, gate_trigger_problems, status_writers

ROOT = Path(__file__).resolve().parents[2]
# GitHub runs both extensions; a rule that only read *.yml could be dodged with *.yaml.
WORKFLOWS = sorted(
    p for ext in ("*.yml", "*.yaml") for p in (ROOT / ".github" / "workflows").glob(ext)
)
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


# Only the three gate workflows may write statuses or check runs or name a job like a
# gate, and they may only run from the base branch; see tools/workflow_rules.py (A18).
STATUS_WRITERS = GATE_WORKFLOWS


def test_workflows_include_yaml_files() -> None:
    assert all(p.suffix in (".yml", ".yaml") for p in WORKFLOWS)
    assert any(p.name == "ci.yml" for p in WORKFLOWS)


@pytest.mark.parametrize("path", WORKFLOWS, ids=ids(WORKFLOWS))
def test_only_the_gate_workflows_write_statuses(path: Path) -> None:
    if path.name in STATUS_WRITERS:
        return
    where = status_writers(load(path))
    assert not where, f"{path.name}: can write a gate ({', '.join(where)}) outside the gates"


@pytest.mark.parametrize("path", WORKFLOWS, ids=ids(WORKFLOWS))
def test_only_the_gate_workflows_can_approve(path: Path) -> None:
    if path.name in STATUS_WRITERS:
        return
    where = approval_writers(load(path))
    assert not where, f"{path.name}: can approve for the owner ({', '.join(where)})"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("permissions:\n  pull-requests: write\njobs: {}\n", ["top"]),
        ("permissions: {}\njobs:\n  a:\n    permissions:\n      issues: write\n", ["job a"]),
        ("permissions:\n  pull-requests: read\n  issues: read\njobs: {}\n", []),
        (
            "jobs:\n  a:\n    steps:\n      - uses: x/y@1\n        with:\n"
            "          github-token: ${{ secrets.OWNER_PAT }}\n",
            ["secret a"],
        ),
        ("env:\n  GH_TOKEN: ${{ secrets.BOT }}\njobs: {}\n", ["secret top"]),
        # Whatever the key, or straight in a run line.
        ("jobs:\n  a:\n    env:\n      GH_PAT: ${{ secrets.OWNER_PAT }}\n", ["secret a"]),
        ("jobs:\n  a:\n    steps:\n      - run: echo ${{ secrets['PAT'] }}\n", ["secret a"]),
        ("jobs:\n  a:\n    steps:\n      - run: echo ${{ toJSON(secrets) }}\n", ["secret a"]),
        (
            "jobs:\n  a:\n    uses: o/r/.github/workflows/x.yml@1\n    secrets: inherit\n",
            ["secret a"],
        ),
        (
            "jobs:\n  a:\n    env:\n      GH_TOKEN: ${{ secrets.GITHUB_TOKEN || secrets.PAT }}\n",
            ["secret a"],
        ),
        # A }} inside a quoted string does not end the expression, nor does an escaped quote.
        (
            "jobs:\n  a:\n    env:\n      T: ${{ format('}}') || secrets.PAT }}\n",
            ["secret a"],
        ),
        (
            "jobs:\n  a:\n    env:\n      T: ${{ format('it''s }}') || secrets.PAT }}\n",
            ["secret a"],
        ),
        ("jobs:\n  a:\n    env:\n      T: ${{ secrets.PAT\n", ["secret a"]),
        # Allowed: the workflow token, the listed API key, and the word outside ${{ }}.
        ("jobs:\n  a:\n    env:\n      GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}\n", []),
        ("jobs:\n  a:\n    env:\n      GH_TOKEN: ${{ github.token }}\n", []),
        ("jobs:\n  a:\n    env:\n      K: ${{ secrets.OPENROUTER_API_KEY != '' }}\n", []),
        ("jobs:\n  secrets:\n    name: No secrets in the image\n", []),
        ("jobs:\n  a:\n    steps:\n      - run: echo ${{ 'x' }} secrets.PAT ${{ 'y' }}\n", []),
    ],
)
def test_approval_writers(text: str, expected: list[str]) -> None:
    assert approval_writers(yaml.safe_load(text)) == expected


def test_the_gate_workflows_exist_and_write_statuses() -> None:
    # A renamed gate workflow must not leave a stale name on the list above.
    for name in sorted(STATUS_WRITERS):
        path = ROOT / ".github" / "workflows" / name
        assert path.is_file(), name
        assert status_writers(load(path)), f"{name} no longer writes statuses"


@pytest.mark.parametrize("name", sorted(GATE_WORKFLOWS))
def test_the_gate_workflows_run_only_from_the_base_branch(name: str) -> None:
    # A pull_request (or push) trigger would run the pull request's own copy of the gate.
    problems = gate_trigger_problems(load(ROOT / ".github" / "workflows" / name))
    assert not problems, f"{name}: {', '.join(problems)}"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("on:\n  pull_request_target:\n  issue_comment:\n", []),
        ("on: [pull_request_target, pull_request]\n", ["trigger pull_request"]),
        ("on: push\n", ["trigger push"]),
        ("jobs: {}\n", ["inga triggers"]),
    ],
)
def test_gate_trigger_problems(text: str, expected: list[str]) -> None:
    assert gate_trigger_problems(yaml.safe_load(text)) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("permissions:\n  statuses: write\njobs: {}\n", ["top"]),
        ("permissions: write-all\njobs: {}\n", ["top"]),
        (
            "permissions: {}\njobs:\n  a:\n    permissions:\n      contents: read\n"
            "  b:\n    permissions:\n      statuses: write\n",
            ["job b"],
        ),
        ("permissions: {}\njobs:\n  a:\n    permissions:\n      checks: write\n", ["job a"]),
        ("permissions: {}\njobs:\n  a:\n    name: gate/review\n", ["name a"]),
        ("permissions: {}\njobs:\n  review/x:\n    runs-on: x\n", ["name review/x"]),
        ("permissions:\n  statuses: read\njobs:\n  a:\n    permissions: read-all\n", []),
        ("permissions: {}\njobs:\n  a:\n    name: lint\n", []),
        ("permissions: {}\njobs:\n  a:\n    name: Gate/Review\n", ["name a"]),
        ("permissions: {}\njobs:\n  a:\n    name: ${{ matrix.n }}\n", ["name a"]),
        ("permissions: {}\njobs:\n  a:\n    name: gat${{ matrix.n }}\n", ["name a"]),
        ("permissions: {}\njobs:\n  a:\n    name: codeql (${{ matrix.l }})\n", []),
        ('{"permissions": {"statuses": "write"}, "jobs": {}}\n', ["top"]),
        ("x: &w write\npermissions:\n  checks: *w\njobs: {}\n", ["top"]),
        ('permissions:\n  "\\x73tatuses": write\njobs: {}\n', ["top"]),
    ],
)
def test_status_writers_finds_every_grant(text: str, expected: list[str]) -> None:
    assert status_writers(yaml.safe_load(text)) == expected


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
    expected = {
        "lint",
        "typecheck",
        "unit",
        "contract",
        "panel-e2e",
        "panel-screens",
        "reuse",
        "secrets",
        "build",
        "rl",
        "train",
    }
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
        "sitl-search",
        "sitl-ok",
        "sitl-nightly",
        "sitl-search-nightly",
        "search-nightly-report",
        "sitl-nightly-gz",
        "sitl-e2e",
        "video",
        "e2e",
        "e2e-offline",
        "e2e-search",
        "e2e-nightly",
    }
    assert jobs["sitl-ok"]["if"] == "always()"
    pr_jobs = {"sitl", "sitl-search", "video", "e2e", "e2e-offline", "e2e-search"}
    assert set(jobs["sitl-ok"]["needs"]) == {"changes"} | pr_jobs
    assert "ci_aggregate.jq" in jobs["sitl-ok"]["steps"][-1]["run"]
    for name in pr_jobs:
        assert jobs[name]["if"] == "needs.changes.outputs.run == 'true'", name
    nightly = ("sitl-nightly", "sitl-search-nightly", "search-nightly-report", "sitl-nightly-gz")
    for name in (*nightly, "e2e-nightly"):
        assert "schedule" in jobs[name]["if"] and "nightly" in jobs[name]["if"], name
    runs = {
        name: " ".join(str(s.get("run", "")) for s in jobs[name]["steps"])
        for name in ("e2e", "e2e-offline", "e2e-nightly")
    }
    assert "just e2e runs=1" in runs["e2e"]
    assert "just e2e offline=1 runs=1" in runs["e2e-offline"]
    assert "just e2e runs=10" in runs["e2e-nightly"]
    tests = [s for s in jobs["sitl"]["steps"] if "ci-sitl" in str(s.get("run", ""))]
    assert len(tests) == 1
    assert tests[0]["timeout-minutes"] <= 35
    run = tests[0]["run"]
    assert "physics=builtin" in run and "brain=scripted" in run


def _run_of(job: dict[str, Any], needle: str) -> dict[str, Any]:
    found = [s for s in job["steps"] if needle in str(s.get("run", ""))]
    assert len(found) == 1, needle
    return found[0]


def test_search_jobs() -> None:
    doc = load(ROOT / ".github" / "workflows" / "sitl.yml")
    env, jobs = doc["env"], doc["jobs"]
    for name, shards, mode in (
        ("sitl-search", "SEARCH_SHARDS", "mode=search "),
        ("sitl-search-nightly", "SEARCH_NIGHTLY_SHARDS", "mode=search-nightly "),
    ):
        job = jobs[name]
        assert job["strategy"]["matrix"]["shard"] == list(range(1, int(env[shards]) + 1)), name
        step = _run_of(job, "ci-sitl")
        assert step["timeout-minutes"] <= 35
        assert mode in step["run"] and f'"shards=${shards}"' in step["run"], name
        assert "physics=builtin" in step["run"] and "brain=scripted" in step["run"], name
    nightly_run = _run_of(jobs["sitl-search-nightly"], "ci-sitl")["run"]
    assert "seeds=1-50 " in nightly_run and "sk06_seeds=1-100" in nightly_run
    e2e = jobs["e2e-search"]
    _run_of(e2e, "just e2e-search runs=1")
    assert "GITHUB_STEP_SUMMARY" in _run_of(e2e, "search_report.md")["run"]
    uploads = [str(s.get("with", {}).get("path", "")) for s in e2e["steps"]]
    assert any("search_report.md" in p and "search_metrics.json" in p for p in uploads)
    report = jobs["search-nightly-report"]
    assert report["needs"] == "sitl-search-nightly"
    assert report["if"].startswith("always() && ")
    assert report["permissions"] == {"contents": "read", "actions": "read"}
    _run_of(report, "sd_sim.eval_search aggregate")
    nightly_uploads = [
        s["with"]
        for name in ("sitl-search-nightly", "search-nightly-report")
        for s in jobs[name]["steps"]
        if str(s.get("uses", "")).startswith("actions/upload-artifact@")
    ]
    assert all(w["retention-days"] == 14 for w in nightly_uploads)


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


def test_brain_nightly_runs_only_from_main_with_the_secret() -> None:
    doc = load(ROOT / ".github" / "workflows" / "brain-nightly.yml")
    assert set(triggers(doc)) == {"schedule", "workflow_dispatch"}  # never pull requests
    job = doc["jobs"]["exam"]
    assert "github.ref == 'refs/heads/main'" in job["if"]
    assert job["permissions"] == {"contents": "read"}
    assert job["env"]["HAS_KEY"] == "${{ secrets.OPENROUTER_API_KEY != '' }}"
    for step in job["steps"][1:]:
        assert "env.HAS_KEY == 'true'" in step["if"]
    upload = next(s for s in job["steps"] if "upload-artifact" in str(s.get("uses")))
    assert all(p.endswith(".json") for p in upload["with"]["path"].split())  # no images
    exam = next(s for s in job["steps"] if s.get("name") == "Short cloud exam")
    assert "--max-cost-sek" in exam["run"]


def test_bench_mini_runs_only_from_main_and_uploads_json() -> None:
    """The one job that fetches real aerial pictures keeps them on the runner."""
    doc = load(ROOT / ".github" / "workflows" / "bench-mini.yml")
    assert set(triggers(doc)) == {"schedule", "workflow_dispatch"}  # never pull requests
    assert doc["permissions"] == {}
    job = doc["jobs"]["bench-mini"]
    assert "github.ref == 'refs/heads/main'" in job["if"]
    assert job["permissions"] == {"contents": "read"}
    steps = job["steps"]
    for step in steps:
        assert "cache" not in str(step.get("uses", ""))
    check = next(i for i, s in enumerate(steps) if "check_no_orthophotos" in str(s.get("run")))
    upload = next(i for i, s in enumerate(steps) if "upload-artifact" in str(s.get("uses")))
    assert check < upload
    assert "steps.numbers.outcome == 'success'" in steps[upload]["if"]
    assert steps[upload]["with"]["path"].strip().endswith("/*.json")
    for step in steps:
        if "bench-fetch" in str(step.get("run")) or "bench-detector" in str(step.get("run")):
            assert "$RUNNER_TEMP" in step["run"]  # pictures and results off the checkout


# Folders a workflow may cache or upload from. Under build/ only the listed subfolders:
# everything else there can hold data that must not leave the runner (the tree list and
# raster crop in build/trees, whose terms forbid redistribution, a site's map data in
# build/sites, the manufacturer models in build/cad/vendor, the generated world in
# build/sim). A cache saved on main can be restored by any pull request's workflow, also
# from a fork, so a cached path counts as published. Any other first folder, a leading
# wildcard or another expression fails the test until it is added here on purpose.
STORE_ROOTS = ("data", "dist", "runs", "software", "${{ runner.temp }}")
SHAREABLE_BUILD_DIRS = (("build", "cad", "fpv7"), ("build", "tree_models"))
STORING_ACTIONS = ("actions/cache", "actions/upload-artifact", "actions/upload-pages-artifact")


def storable(entry: str) -> bool:
    parts = tuple(p for p in entry.split("/") if p)
    if not parts or "*" in parts[0] or any(p in (".", "..") or "${{" in p for p in parts[1:]):
        return False
    if parts[0] == "build":
        return any(parts[: len(d)] == d for d in SHAREABLE_BUILD_DIRS)
    return parts[0] in STORE_ROOTS


def test_storable_paths() -> None:
    assert storable("build/tree_models") and storable("build/cad/fpv7/")
    assert storable("runs/**/result.json") and storable("${{ runner.temp }}/film/")
    for bad in (
        "build/trees",
        "build/tree_models_raw",
        "build/cad/vendor",
        "build/",
        "**/trees",
        "${{ env.DIR }}",
        "trees.csv",
        "runs/../build/trees",
        "build/tree_models/../trees",
        "./build/trees",
    ):
        assert not storable(bad), bad


def stored(step: dict[str, Any]) -> list[str] | None:
    """What a storing step stores (its ``path``, exclusions left out), or None when it is
    not a storing step. A step without a path stores the action's default (all of
    ``_site`` for upload-pages-artifact), which no rule here has seen: it gets ``[]``."""
    if not str(step.get("uses", "")).startswith(STORING_ACTIONS):
        return None
    lines = str((step.get("with") or {}).get("path", "")).splitlines()
    return [e for e in (line.strip() for line in lines) if e and not e.startswith("!")]


@pytest.mark.parametrize("path", ALL_FILES, ids=ids(ALL_FILES))
def test_only_shareable_paths_are_cached_or_uploaded(path: Path) -> None:
    for step in steps(load(path)):
        entries = stored(step)
        if entries is None:
            continue
        action = step["uses"].split("@")[0]
        assert entries, f"{path.name}: {action} without a path"
        for entry in entries:
            assert storable(entry), f"{path.name}: {action} stores {entry!r}"


def test_a_storing_step_without_a_path_is_caught() -> None:
    up = "actions/upload-artifact@0123456789abcdef0123456789abcdef01234567"
    assert stored({"uses": up}) == []
    assert stored({"uses": up, "with": {"name": "x"}}) == []
    assert stored({"uses": up, "with": {"path": "!runs/secrets\n"}}) == []
    assert stored({"uses": "actions/cache@abc", "with": {"path": "runs/a\n!runs/b"}}) == ["runs/a"]
    assert stored({"uses": "actions/checkout@abc"}) is None


UPLOADS = ("actions/upload-artifact@", "actions/upload-pages-artifact@")
CACHES = ("actions/cache@", "actions/cache/save@")
ORTHO_CHECK = "tools/check_no_orthophotos.py"
# Caches that only ever hold downloaded packages, never anything a job made: uv's wheel
# cache (setup-uv's enable-cache). Anything else cached is checked after it is written.
PACKAGE_CACHES = ("astral-sh/setup-uv@",)


def _steps(path: Path) -> list[tuple[str, list[dict[str, Any]]]]:
    """(job, steps) of a workflow, or of a composite action as one job."""
    data = load(path)
    if "runs" in data:
        return [("composite", data["runs"].get("steps", []))]
    return [(n, j.get("steps", [])) for n, j in data.get("jobs", {}).items()]


def _checked(step: dict[str, Any]) -> list[str]:
    """The paths an orthophoto check step checks."""
    out = []
    for line in str(step.get("run", "")).splitlines():
        if ORTHO_CHECK in line:
            args = line.split(ORTHO_CHECK, 1)[1].split()
            out += [a.strip('"').replace("$RUNNER_TEMP", "${{ runner.temp }}") for a in args]
    return [a.rstrip("/") for a in out if not a.startswith("-")]


def _roots(paths: str) -> list[str]:
    """The folders or files an upload or cache ``path`` reaches, before any wildcard."""
    out = []
    for line in paths.splitlines():
        line = line.strip()
        if not line or line.startswith("!"):
            continue
        parts = []
        for part in line.split("/"):
            if any(ch in part for ch in "*?["):
                break
            parts.append(part)
        out.append("/".join(parts).rstrip("/"))
    return out


def _caches(inputs: dict[str, Any]) -> bool:
    """Whether a setup action's inputs turn on its own cache (setup-uv, setup-node,
    setup-python and the like)."""
    on = str(inputs.get("enable-cache", "")).lower() == "true" or bool(inputs.get("cache"))
    pm = inputs.get("package-manager-cache")
    return on or (pm is not None and str(pm).lower() != "false")


def _covered(root: str, checked: list[str]) -> bool:
    return any(c in (".", "") or root == c or root.startswith(c + "/") for c in checked)


@pytest.mark.parametrize("path", ALL_FILES, ids=ids(ALL_FILES))
def test_nothing_is_uploaded_or_cached_without_the_orthophoto_check(path: Path) -> None:
    """Orthophotos stay on the owner's computer: every job checks what it uploads before
    the upload (and an upload that runs after a failure waits for that check to pass),
    and what it caches after the last step that could have written it. The checks cover
    every path the upload or cache reaches."""
    for name, job_steps in _steps(path):
        checks = [i for i, s in enumerate(job_steps) if ORTHO_CHECK in str(s.get("run", ""))]
        for i, step in enumerate(job_steps):
            uses = str(step.get("uses", ""))
            where = f"{path.name} {name}: {step.get('name', uses)}"
            paths = str(step.get("with", {}).get("path", ""))
            if uses.startswith(UPLOADS):
                before = [job_steps[c] for c in checks if c < i]
                assert before, f"{where}: no {ORTHO_CHECK} before the upload"
                cond = str(step.get("if", ""))
                if "always()" in cond or "failure()" in cond:
                    ids_ok = [f"steps.{s['id']}.outcome == 'success'" for s in before if "id" in s]
                    assert any(x in cond for x in ids_ok), f"{where}: uploads after a failed check"
                checked = [p for s in before for p in _checked(s)]
                for root in _roots(paths):
                    assert _covered(root, checked), f"{where}: {root} is uploaded unchecked"
            elif uses.startswith(CACHES):
                after = [job_steps[c] for c in checks if c > i]
                assert after, f"{where}: no {ORTHO_CHECK} after the cache"
                checked = [p for s in after for p in _checked(s)]
                for root in _roots(paths):
                    assert _covered(root, checked), f"{where}: {root} is cached unchecked"
            elif _caches(step.get("with") or {}):
                assert uses.startswith(PACKAGE_CACHES), f"{where}: a cache the test does not know"


def test_the_coverage_rules_see_a_gap() -> None:
    assert _caches({"cache": "pip"}) and _caches({"package-manager-cache": True})
    assert not _caches({"package-manager-cache": False}) and not _caches({})
    assert _roots("runs/**/x.json\n!runs/**/secrets/**\ndist/\n") == ["runs", "dist"]
    assert _covered("build/tree_models", ["build"])
    assert not _covered("build2/x", ["build"])
    assert _checked({"run": 'python3 tools/check_no_orthophotos.py "$RUNNER_TEMP/site" a/'}) == [
        "${{ runner.temp }}/site",
        "a",
    ]


def test_release_workflow_writes_only_from_a_job_that_runs_no_repo_code() -> None:
    doc = load(ROOT / ".github" / "workflows" / "release.yml")
    assert triggers(doc) == {"push": {"tags": ["v[0-9]*"]}}
    assert doc["permissions"] == {"contents": "read"}
    writers = {
        name: job["permissions"]
        for name, job in doc["jobs"].items()
        if "write" in (job.get("permissions") or {}).values()
    }
    assert writers == {"publish": {"contents": "write"}}
    for step in doc["jobs"]["publish"]["steps"]:
        uses = str(step.get("uses", ""))
        assert not uses or uses.startswith("actions/download-artifact@"), uses
        run = str(step.get("run", ""))
        assert not re.search(r"\b(tools/|just|uv|python3?|pnpm|bash \S+\.sh)\b", run), run
    # The gate check runs first, in its own job, and nothing else from the repository
    # runs there: only the protected gate scripts, copied out, with PyYAML pinned by hash.
    assert doc["jobs"]["build"]["needs"] == "gates"
    gates = doc["jobs"]["gates"]["steps"]
    assert [s.get("name") or s.get("uses") for s in gates][1:] == [
        "The tagged commit is on main",
        "Every commit on main came from a pull request with green gates",
    ]
    assert str(gates[0]["uses"]).startswith("actions/checkout@")
    run = str(gates[2]["run"])
    # The anchor comes from GitHub and git, never from repository code: only published
    # release tags on the line count. The anchor's own scripts confirm it (an error stops
    # the step, since it is an assignment), then check everything after it.
    assert "tagged" not in run
    assert 'gh api --paginate "repos/$GITHUB_REPOSITORY/releases?per_page=100"' in run
    assert "--jq '.[] | select(.draft | not) | [.tag_name, .author.login] | @tsv'" in run
    assert 'git rev-parse -q --verify "refs/tags/$name^{commit}"' in run
    # A git error is not "not an ancestor"; releases off the line with none on it, or an
    # anchor not made by this workflow, stop the release before any script runs.
    assert 'git merge-base --is-ancestor "$c" "$GITHUB_SHA" || rc=$?' in run
    assert '[ "$rc" -le 1 ] || {' in run
    assert 'if [ -n "$off$tags" ]; then' in run
    assert "git tag --merged \"$GITHUB_SHA\" --list 'v*'" in run
    # What the loop picks is tested by running it (test_check_merged_pr_gates). Past the
    # API's hourly limit the check waits, so the job has hours, not the default minutes.
    assert doc["jobs"]["gates"]["timeout-minutes"] == 300
    assert '[ "$by" != "github-actions[bot]" ]' in run
    assert 'rules="${since:-$GITHUB_SHA}"' in run
    assert 'git show "$rules:tools/$f" > "$gate/$f"' in run
    assert '--anchor-for-tag "$GITHUB_REF_NAME")' in run
    assert '[ "$anchor" = "$since" ]' in run
    assert '${since:+--since "$since"}' in run
    assert run.index('cd "$gate"') < run.index("anchor=$(")
    loop = re.search(r'scripts="([^"]+)"', run)
    assert loop is not None
    copied = {f"tools/{name}" for name in loop[1].split()}
    # A change to any of them needs owner-ok, and so does a change to this workflow.
    paths = [*copied, ".github/workflows/release.yml"]
    assert check_owner_gate.protected_files([{"filename": p} for p in paths]) == sorted(paths)
    assert "--require-hashes" in run and "--root" in run
    # Python runs only after leaving the clone, and with -m only isolated (-I), so no
    # repository file (venv.py, pip/, yaml.py) can stand in for a module.
    lines = [line.strip() for line in run.splitlines()]
    # (The version check before it runs isolated (-I) with -c: no file is read.)
    first_python = next(
        i for i, line in enumerate(lines) if "python" in line and " -I -c " not in line
    )
    assert 'cd "$gate"' in lines[:first_python]
    assert all(" -I -m " in line for line in lines if " -m " in line)
    assert not re.search(r"\b(just|uv|pnpm)\b|\./", run), run
