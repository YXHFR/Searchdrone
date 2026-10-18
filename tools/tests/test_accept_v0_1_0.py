# pyright: strict
"""tools/accept/v0_1_0.py (B1-16): phases, manual rows, the report and the exit codes.

Nothing here reaches GitHub or runs a real check: the ``gh`` wrapper and the local
command runner are replaced, and the ID list is a small synthetic one in ``tmp_path``.
"""

from __future__ import annotations

import datetime as dt
import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
from accept import v0_1_0 as acc

SHA = "a" * 40
OTHER = "b" * 40
ROOT = Path(__file__).resolve().parents[2]
# The real job names of the workflows (tools/tests/test_workflows.py checks them there).
GREEN_RUNS = [*acc.MAIN_CHECKS, "sitl (1)", "sitl (2)", "panel-e2e", "e2e", acc.OFFLINE_JOB]
EXAMPLE = Path(acc.__file__).with_name("accept-manual.example.toml")

# One row of each kind: automatic, manual, automatic + manual, a CI row with tests,
# and an automatic post-release row.
IDS = """
[[id]]
id = "R-05"
typ = "A"
dod_ref = "3"

[[id]]
id = "R-06"
typ = "A + M"
dod_ref = "4"

[[id]]
id = "F-02"
typ = "M"
dod_ref = "6"

[[id]]
id = "S-01"
typ = "A"
dod_ref = "9 (S3)"
sc = ["SC04"]
tests = ["test_refusal"]

[[id]]
id = "V-04"
typ = "A"
dod_ref = "13"
"""
SYNTHETIC = ("R-05", "R-06", "F-02", "S-01", "V-04")


def manual_row(by: str = "owner", date: str = "2026-10-18", ok: bool = True) -> str:
    return f'pass = {str(ok).lower()}\nby = "{by}"\ndate = "{date}"\nnote = "seen"\n'


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def ruleset_detail(checks: tuple[str, ...] = acc.REQUIRED_CHECKS) -> dict[str, Any]:
    return {
        "enforcement": "active",
        "bypass_actors": [],
        "rules": [
            {"type": "pull_request", "parameters": {"allowed_merge_methods": ["squash"]}},
            {"type": "non_fast_forward"},
            {"type": "deletion"},
            {
                "type": "required_status_checks",
                "parameters": {"required_status_checks": [{"context": c} for c in checks]},
            },
        ],
    }


class FakeGitHub:
    """Answers the read-only ``gh`` calls the synthetic rows make."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.conclusion = "success"
        self.detail = ruleset_detail()
        self.runs = list(GREEN_RUNS)
        self.nightly = dict.fromkeys(acc.NIGHTLY_JOBS, "success")

    def __call__(self, args: list[str]) -> str:
        acc.ensure_read_only(args)  # the real guard still runs on every call
        self.calls.append(args)
        return json.dumps(self.answer(args))

    def answer(self, args: list[str]) -> Any:
        if args[:2] == ["release", "view"]:
            return {"isDraft": False, "body": "What it can do ... run `just demo`."}
        path = args[1] if args[0] == "api" else ""
        if path.endswith("/status?per_page=100"):
            return {"statuses": []}
        if path.endswith("/check-runs?per_page=100"):
            return {
                "check_runs": [
                    {"name": n, "status": "completed", "conclusion": self.conclusion}
                    for n in self.runs
                ]
            }
        if args[:2] == ["run", "list"]:
            return [{"databaseId": 7, "event": "schedule", "headSha": SHA}]
        if args[:2] == ["run", "view"]:
            return {"jobs": [{"name": n, "conclusion": c} for n, c in self.nightly.items()]}
        if path.endswith("/rulesets?includes_parents=false"):
            return [
                {"id": 1, "name": "main-protection"},
                {"id": 2, "name": "release-tags", "target": "tag", "enforcement": "active"},
            ]
        if path.endswith("/rulesets/1"):
            return self.detail
        if path.endswith(f"/git/ref/tags/{acc.TAG}"):
            return {"object": {"type": "commit", "sha": SHA}}
        raise AssertionError(f"unexpected gh call: {args}")


class FakeRunner:
    """Local commands: git answers for ``SHA``; every other command succeeds."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.head = SHA

    def __call__(self, cmd: list[str], cwd: Path) -> acc.Proc:
        self.calls.append(cmd)
        if cmd[:2] == ["git", "rev-parse"]:
            return acc.Proc(0, self.head + "\n", "")
        if cmd[:2] == ["git", "status"]:
            return acc.Proc(0, "", "")
        return acc.Proc(0, "ok\n", "")


@pytest.fixture
def github(monkeypatch: pytest.MonkeyPatch) -> FakeGitHub:
    fake = FakeGitHub()
    monkeypatch.setattr(acc, "gh", fake)
    return fake


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> FakeRunner:
    fake = FakeRunner()
    monkeypatch.setattr(acc, "run", fake)
    return fake


@pytest.fixture
def root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, github: FakeGitHub, runner: FakeRunner
) -> Path:
    """A tiny repository root with one test function and the synthetic ID list."""
    repo = tmp_path / "repo"
    write(repo / "simulation/tests/test_s.py", "def test_refusal():\n    pass\n")
    ids = write(tmp_path / "accept-ids.toml", IDS)
    monkeypatch.setenv("SD_ACCEPT_IDS", str(ids))
    monkeypatch.delenv("SD_ACCEPT_REPO", raising=False)
    monkeypatch.setattr(acc, "KNOWN_IDS", SYNTHETIC)
    return repo


def manual_file(root: Path, rows: dict[str, str]) -> Path:
    text = "".join(f"[{rid}]\n{body}\n" for rid, body in rows.items())
    return write(root / acc.MANUAL_FILE, text)


def report(root: Path, phase: str = "pre") -> dict[str, Any]:
    (path,) = (root / acc.REPORT_DIR).glob(f"accept-v0.1.0-{phase}-*.json")
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_id(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["id"]: r for r in data["rows"]}


# --- exit codes ------------------------------------------------------------------------


def test_all_green_exits_0(root: Path) -> None:
    manual_file(root, {"R-06": manual_row(), "F-02": manual_row()})
    assert acc.main(["phase=pre", f"sha={SHA}"], root=root) == 0
    data = report(root)
    assert data["pass"] is True
    assert {r["id"] for r in data["rows"]} == {"R-05", "R-06", "F-02", "S-01"}
    assert all(r["pass"] is True for r in data["rows"])


def test_missing_manual_row_is_null_and_exits_1(root: Path) -> None:
    manual_file(root, {"R-06": manual_row()})  # F-02 not filled in
    assert acc.main([f"sha={SHA}"], root=root) == 1
    rows = rows_by_id(report(root))
    assert rows["F-02"]["pass"] is None
    assert "saknas" in rows["F-02"]["detalj"]
    assert rows["R-05"]["pass"] is True


def test_no_manual_file_means_every_manual_row_is_null(root: Path) -> None:
    assert acc.main([f"sha={SHA}"], root=root) == 1
    rows = rows_by_id(report(root))
    assert rows["F-02"]["pass"] is None
    assert rows["R-06"]["pass"] is None  # automatic half green, manual half missing
    assert not (root / acc.MANUAL_FILE).exists()


def test_both_halves_must_pass(root: Path, github: FakeGitHub) -> None:
    manual_file(root, {"R-06": manual_row(), "F-02": manual_row()})
    github.detail = ruleset_detail(checks=("ci-ok",))
    assert acc.main([f"sha={SHA}"], root=root) == 1
    r06 = rows_by_id(report(root))["R-06"]
    assert r06["pass"] is False
    assert "gate/review" in r06["detalj"]
    assert r06["detalj"].startswith("A: ")


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (manual_row(ok=False), "inte godkänt"),
        (manual_row(by=""), "by saknas"),
        (manual_row(date="18/10"), "YYYY-MM-DD"),
    ],
)
def test_manual_row_needs_pass_by_and_date(root: Path, body: str, reason: str) -> None:
    manual_file(root, {"R-06": manual_row(), "F-02": body})
    assert acc.main([f"sha={SHA}"], root=root) == 1
    f02 = rows_by_id(report(root))["F-02"]
    assert f02["pass"] is False
    assert reason in f02["detalj"]


def test_red_check_exits_1(root: Path, github: FakeGitHub) -> None:
    manual_file(root, {"R-06": manual_row(), "F-02": manual_row()})
    github.conclusion = "failure"
    assert acc.main([f"sha={SHA}"], root=root) == 1
    rows = rows_by_id(report(root))
    assert rows["R-05"]["pass"] is False
    assert "ci-ok=failure" in rows["R-05"]["detalj"]
    assert rows["S-01"]["pass"] is False


def test_missing_id_list_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SD_ACCEPT_IDS", str(tmp_path / "nope.toml"))
    assert acc.main(["phase=pre"], root=tmp_path) == 2
    assert "SD_ACCEPT_IDS" in capsys.readouterr().err
    assert not (tmp_path / acc.REPORT_DIR).exists()


def test_id_list_path_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("SD_ACCEPT_IDS", raising=False)
    assert acc.main([], root=tmp_path) == 2
    assert "SD_ACCEPT_IDS är inte satt" in capsys.readouterr().err


@pytest.mark.parametrize("arg", ["phase=during", "speed=1", "--phase", "sha"])
def test_bad_arguments_exit_2(root: Path, arg: str) -> None:
    assert acc.main([arg], root=root) == 2


def test_unknown_sha_exits_2(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def unknown(cmd: list[str], cwd: Path) -> acc.Proc:
        return acc.Proc(128, "", "fatal: bad revision")

    monkeypatch.setattr(acc, "run", unknown)
    assert acc.main(["sha=nonexistent"], root=root) == 2


def test_sha_defaults_to_main(root: Path, runner: FakeRunner) -> None:
    acc.main([], root=root)
    assert ["git", "rev-parse", "--verify", "main^{commit}"] in runner.calls
    assert report(root)["sha"] == SHA


# --- phases ------------------------------------------------------------------------------


def test_post_phase_runs_only_the_post_rows(root: Path) -> None:
    assert acc.main(["phase=post", f"sha={SHA}"], root=root) == 0
    data = report(root, "post")
    assert [r["id"] for r in data["rows"]] == ["V-04"]
    assert data["phase"] == "post"


def test_phase_membership_of_the_real_list() -> None:
    every = {i: {"id": i} for i in acc.KNOWN_IDS}
    pre = acc.phase_ids("pre", every)
    post = acc.phase_ids("post", every)
    assert set(post) == {"V-02", "V-04", "R-04b"}
    assert not set(pre) & set(post)
    assert set(pre) | set(post) == set(acc.KNOWN_IDS)
    assert {"R-01", "R-11", "F-01", "F-13", "S-01", "S-25", "V-01", "V-06"} <= set(pre)
    assert len(acc.KNOWN_IDS) == 11 + 1 + 13 + 25 + 6


def test_id_missing_from_the_list_is_red(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acc, "KNOWN_IDS", (*SYNTHETIC, "R-01"))
    manual_file(root, {"R-06": manual_row(), "F-02": manual_row()})
    assert acc.main([f"sha={SHA}"], root=root) == 1
    r01 = rows_by_id(report(root))["R-01"]
    assert r01["pass"] is False
    assert "ID-listan" in r01["detalj"]


# --- the report ---------------------------------------------------------------------------


def test_report_shape(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manual_file(root, {"R-06": manual_row()})
    acc.main([f"sha={SHA}"], root=root)
    data = report(root)
    assert set(data) == {"version", "phase", "sha", "created_at", "pass", "rows"}
    assert data["version"] == "0.1.0"
    assert data["sha"] == SHA
    assert data["pass"] is False
    for row in data["rows"]:
        assert set(row) == {"id", "typ", "pass", "detalj", "dod_ref", "sha"}
        assert row["sha"] == SHA
        assert row["pass"] in (True, False, None)
    rows = rows_by_id(data)
    assert rows["R-06"]["typ"] == "A + M"
    assert rows["S-01"]["dod_ref"] == "9 (S3)"
    out = capsys.readouterr().out
    assert "F-02" in out
    assert "saknas" in out
    assert "Rapport: data/reports/accept-v0.1.0-pre-" in out


def test_manual_file_is_never_written(root: Path) -> None:
    path = manual_file(root, {"R-06": manual_row(), "F-02": manual_row(ok=False)})
    before = path.read_bytes()
    mtime = path.stat().st_mtime_ns
    acc.main([f"sha={SHA}"], root=root)
    acc.main(["phase=post", f"sha={SHA}"], root=root)
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == mtime
    written = {p.name for p in (root / acc.REPORT_DIR).iterdir()}
    assert {n for n in written if not n.startswith("accept-v0.1.0-")} == {path.name}


# --- individual checks --------------------------------------------------------------------


def ctx_for(root: Path, entries: dict[str, dict[str, Any]] | None = None) -> acc.Ctx:
    now = dt.datetime(2026, 10, 18, 12, tzinfo=dt.UTC)
    return acc.Ctx(root=root, sha=SHA, repo="o/r", entries=entries or {}, now=now)


def test_gh_wrapper_refuses_writes() -> None:
    for args in (
        ["api", "-X", "POST", "repos/o/r/rulesets"],
        ["api", "--method=PATCH", "repos/o/r"],
        ["api", "repos/o/r/issues", "-f", "title=x"],
        ["api", "repos/o/r/issues", "-ftitle=x"],
        ["api", "repos/o/r/issues", "--input", "x.json"],
        ["release", "edit", "v0.1.0", "--draft"],
        ["pr", "merge", "1"],
        ["workflow", "run", "sitl.yml"],
        [],
    ):
        with pytest.raises(acc.GhError):
            acc.ensure_read_only(args)
    for args in (
        ["api", "-iXPOST", "repos/o/r/rulesets"],  # a write flag inside a short group
        ["api", "-iX", "POST", "repos/o/r"],
    ):
        with pytest.raises(acc.GhError):
            acc.ensure_read_only(args)
    for args in (
        ["api", "-i", "repos/o/r"],
        ["api", "repos/o/r/rulesets"],
        ["run", "list", "--workflow", "sitl.yml"],
        ["release", "view", "v0.1.0", "--json", "isDraft"],
    ):
        acc.ensure_read_only(args)


def test_gh_failure_turns_into_a_red_row(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(args: list[str]) -> str:
        raise acc.GhError("gh api: kod 1: HTTP 404")

    monkeypatch.setattr(acc, "gh", broken)
    check = acc.automatic(ctx_for(root), "R-05")
    assert check.ok is False
    assert "HTTP 404" in check.detail


def test_local_checks_need_the_release_commit_checked_out(root: Path, runner: FakeRunner) -> None:
    runner.head = OTHER
    check = acc.automatic(ctx_for(root), "R-01")
    assert check.ok is False
    assert "git checkout" in check.detail
    assert not any(c[:2] == ["uv", "run"] for c in runner.calls)


def test_missing_tool_is_red(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_gitleaks(cmd: list[str], cwd: Path) -> acc.Proc:
        if cmd[0] == "gitleaks":
            return acc.Proc(127, "", "gitleaks finns inte")
        return FakeRunner()(cmd, cwd)

    monkeypatch.setattr(acc, "run", no_gitleaks)
    check = acc.automatic(ctx_for(root), "R-03")
    assert check == acc.Check(False, "gitleaks är inte installerat")


def test_private_patterns_file_missing_is_red(root: Path) -> None:
    check = acc.automatic(ctx_for(root), "R-04")
    assert check.ok is False
    assert check.detail.endswith("saknas")


def test_merged_pr_gates_not_built_yet(root: Path) -> None:
    for rid in ("R-08", "R-09"):
        check = acc.automatic(ctx_for(root), rid)
        assert check.ok is False
        assert "inte byggt än (B1-18)" in check.detail


def test_row_without_a_check_is_red(root: Path) -> None:
    check = acc.automatic(ctx_for(root), "X-99")
    assert check.ok is False
    assert "ingen automatisk kontroll" in check.detail


def test_ci_row_needs_its_tests(root: Path) -> None:
    entries = {"S-01": {"id": "S-01", "typ": "A", "tests": ["test_gone"]}}
    check = acc.automatic(ctx_for(root, entries), "S-01")
    assert check.ok is False
    assert "test_gone" in check.detail


def eval_report(root: Path, stamp: str, **fields: Any) -> None:
    data = {"gpu_name": "NVIDIA GeForce RTX 3070", "git_sha": SHA, "passed": True} | fields
    write(root / acc.REPORT_DIR / f"eval-grunden-{stamp}-abcdef12.json", json.dumps(data))


def test_eval_report_must_be_fresh_on_sha_and_on_the_owners_gpu(root: Path) -> None:
    ctx = ctx_for(root)
    assert acc.automatic(ctx, "F-05").ok is False  # no report at all
    eval_report(root, "20261017T120000Z")
    assert acc.automatic(ctx, "F-05").ok is True
    eval_report(root, "20261018T080000Z", gpu_name="Tesla T4")
    assert "gpu_name" in acc.automatic(ctx, "F-05").detail
    eval_report(root, "20261018T090000Z", git_sha=OTHER)
    assert "git_sha" in acc.automatic(ctx, "F-05").detail
    old = ctx_for(root)
    old.now = dt.datetime(2026, 10, 30, tzinfo=dt.UTC)
    eval_report(root, "20261018T100000Z")
    assert "dagar gammal" in acc.automatic(old, "F-05").detail


def test_example_template_lists_every_manual_row() -> None:
    data = tomllib.loads(EXAMPLE.read_text(encoding="utf-8"))
    assert set(data) == set(acc.MANUAL_IDS)
    for rid, row in data.items():
        assert row == {"pass": False, "by": "", "date": "", "note": ""}, rid
        assert acc.manual(row)[0] is False
    assert acc.MANUAL_FILE in EXAMPLE.read_text(encoding="utf-8")


def test_f07_needs_the_offline_end_to_end_job(root: Path, github: FakeGitHub) -> None:
    ctx = ctx_for(root, {"F-07": {"id": "F-07", "tests": []}})
    assert acc.OFFLINE(ctx, "F-07").ok
    github.runs.remove(acc.OFFLINE_JOB)
    ctx = ctx_for(root, {"F-07": {"id": "F-07", "tests": []}})
    got = acc.OFFLINE(ctx, "F-07")
    assert not got.ok and acc.OFFLINE_JOB in got.detail


def test_f01_needs_both_nightly_jobs_on_the_sha(root: Path, github: FakeGitHub) -> None:
    assert acc.f01(ctx_for(root)).ok
    github.nightly["e2e-nightly"] = "failure"  # the ten end-to-end rounds
    got = acc.f01(ctx_for(root))
    assert not got.ok and "e2e-nightly" in got.detail


def test_the_ci_job_names_exist_in_the_workflow() -> None:
    import yaml

    jobs = yaml.safe_load((ROOT / ".github/workflows/sitl.yml").read_text(encoding="utf-8"))["jobs"]
    assert {*acc.NIGHTLY_JOBS, acc.OFFLINE_JOB} <= set(jobs)
