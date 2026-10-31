"""tools/check_merged_pr_gates.py against a fake GitHub: green, red and forged gates."""

from __future__ import annotations

import base64
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import check_merged_pr_gates as cmg
import pytest

REPO = "o/r"
BOT = {"login": "github-actions[bot]"}
TITLE = "feat(tools): a change"
RUNS = {"gate/review": 100, "gate/owner": 200}


def status(context: str, state: str = "success", at: str = "2026-10-01T10:00:00Z", **kw: Any):
    run = RUNS.get(context)
    url = {"target_url": f"https://github.com/{REPO}/actions/runs/{run}"} if run else {}
    return {"context": context, "state": state, "created_at": at, "creator": BOT, **url, **kw}


def green(head: str = "h1") -> list[dict[str, Any]]:
    return [status(g) for g in cmg.GATES]


def gate_run(workflow: str, head: str, number: int, **kw: Any) -> dict[str, Any]:
    """A pull_request_target run of a gate workflow on ``head`` (``number`` is unused: real
    runs carry an empty ``pull_requests`` list)."""
    return {
        "path": f".github/workflows/{workflow}",
        "event": "pull_request_target",
        "head_sha": head,
        "head_branch": "feature",
        "repository": {"full_name": REPO},
        **kw,
    }


class FakeGitHub:
    """Merge commit -> PR, head SHA -> statuses, run id -> run, PR -> changed files."""

    def __init__(self) -> None:
        self.prs: dict[str, dict[str, Any]] = {}
        self.statuses: dict[str, list[dict[str, Any]]] = {}
        self.files: dict[int, list[dict[str, Any]]] = {}
        self.contents: dict[tuple[str, str], str] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.releases: list[dict[str, Any]] = []

    def workflow(self, number: int, head: str, path: str, text: str, status: str = "added"):
        self.files.setdefault(number, []).append({"filename": path, "status": status})
        self.contents[(path, head)] = text

    def merged(self, sha: str, number: int, head: str, statuses: list[dict[str, Any]]) -> None:
        self.prs[sha] = {
            "number": number,
            "title": TITLE,
            "merged_at": "2026-10-01T11:00:00Z",
            "merge_commit_sha": sha,
            "head": {"sha": head},
            "base": {"ref": "main"},
        }
        self.statuses[head] = statuses
        self.runs["100"] = gate_run("review-gate.yml", head, number)
        self.runs["200"] = gate_run("owner-gate.yml", head, number)

    def __call__(self, path: str) -> Any:
        if "/contents/" in path:
            file, ref = path.split("/contents/", 1)[1].split("?ref=")
            raw = self.contents[(file, ref)].encode()
            return [{"content": base64.b64encode(raw).decode()}]
        parts = path.split("?")[0].split("/")
        if parts[3] == "releases":
            return self.releases
        if parts[3] == "actions":
            return [self.runs[parts[5]]]
        if parts[3] == "pulls":
            return self.files.get(int(parts[4]), [])
        sha = parts[4]
        if parts[-1] == "pulls":
            return [
                pr for pr in self.prs.values() if sha in (pr["merge_commit_sha"], pr["head"]["sha"])
            ]
        return self.statuses.get(sha, [])


def test_green_gates_pass() -> None:
    gh = FakeGitHub()
    gh.merged("a" * 40, 1, "h1", green())
    assert cmg.check(gh, REPO, ["a" * 40]) == []


def test_direct_push_fails() -> None:
    (finding,) = cmg.check(FakeGitHub(), REPO, ["b" * 40])
    assert "inte från en pull request sammanslagen till main" in finding.message


def test_a_pr_merged_into_another_branch_does_not_count() -> None:
    gh = FakeGitHub()
    gh.merged("c" * 40, 2, "h2", green())
    gh.prs["c" * 40]["base"] = {"ref": "release-x"}
    assert cmg.check(gh, REPO, ["c" * 40])[0].message.startswith("kom inte från")


def test_open_pr_that_only_contains_the_commit_does_not_count() -> None:
    gh = FakeGitHub()
    gh.merged("c" * 40, 2, "h2", green())
    gh.prs["c" * 40]["merge_commit_sha"] = "d" * 40
    assert cmg.check(gh, REPO, ["c" * 40])[0].message.startswith("kom inte från")


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ([status("gate/review"), status("gate/owner")], "PR #3: gate/pr-title saknas"),
        (
            [status("gate/review", "failure"), status("gate/owner"), status("gate/pr-title")],
            "PR #3: gate/review är failure",
        ),
        (
            [
                status("gate/review"),
                status("gate/owner"),
                status("gate/pr-title", creator={"login": "someone"}),
            ],
            "PR #3: gate/pr-title skrevs av someone, inte github-actions[bot]",
        ),
    ],
)
def test_red_missing_or_forged_gates_fail(statuses: list[dict[str, Any]], expected: str) -> None:
    gh = FakeGitHub()
    gh.merged("e" * 40, 3, "h3", statuses)
    assert [f.message for f in cmg.check(gh, REPO, ["e" * 40])] == [expected]


FORGER = """on: pull_request
permissions: {}
jobs:
  paint:
    permissions:
      statuses: write
    runs-on: ubuntu-24.04
"""
GATE = "on:\n  pull_request_target:\n  issue_comment:\npermissions:\n  statuses: write\n"


@pytest.mark.parametrize(
    ("path", "text", "status", "expected"),
    [
        (".github/workflows/paint.yml", FORGER, "added", "kan skriva en grind (job paint)"),
        (".github/workflows/paint.yaml", "permissions: write-all\n", "modified", "(top)"),
        (".github/workflows/x.yml", "jobs:\n  a:\n    name: gate/review\n", "added", "(name a)"),
        (".github/workflows/x.yml", "jobs:\n  a:\n    name: Gate/Owner\n", "added", "(name a)"),
        (
            ".github/workflows/x.yml",
            "jobs:\n  a:\n    permissions:\n      checks: write\n",
            "added",
            "(job a)",
        ),
        # Spellings a text search misses; YAML reads them all the same.
        (".github/workflows/x.yml", '{"permissions": {"checks": "write"}}\n', "added", "(top)"),
        (".github/workflows/x.yml", "permissions:\n  'statuses': write\n", "added", "(top)"),
        (
            ".github/workflows/x.yml",
            "w: &w write\npermissions:\n  statuses: *w\n",
            "added",
            "(top)",
        ),
        (".github/workflows/x.yml", 'permissions:\n  "\\x73tatuses": write\n', "added", "(top)"),
        (".github/workflows/x.yml", "permissions: [\n", "added", "går inte att läsa"),
        # A gate workflow may change, but must still run only from the base branch.
        (".github/workflows/review-gate.yml", FORGER, "modified", "trigger pull_request"),
        (
            ".github/workflows/owner-gate.yml",
            GATE.replace("issue_comment", "push"),
            "added",
            "trigger push",
        ),
        (".github/workflows/review-gate.yml", GATE, "modified", None),
        (".github/workflows/paint.yml", FORGER, "removed", None),
        (".github/workflows/ci.yml", "permissions:\n  contents: read\n", "modified", None),
        ("docs/statuses.yml", FORGER, "added", None),
    ],
)
def test_a_workflow_from_the_pr_that_can_write_a_gate_fails(
    path: str, text: str, status: str, expected: str | None
) -> None:
    # github-actions[bot] writes for every workflow: green gates from the PR's own
    # workflow look exactly like the real ones, so the workflow itself is checked.
    gh = FakeGitHub()
    gh.merged("9" * 40, 7, "h7", green())
    gh.workflow(7, "h7", path, text, status)
    messages = [f.message for f in cmg.check(gh, REPO, ["9" * 40])]
    if expected is None:
        assert messages == []
    else:
        (message,) = messages
        assert message.startswith(f"PR #7: {path}") and expected in message, message


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (
            {"path": ".github/workflows/paint.yml"},
            "av .github/workflows/paint.yml, inte review-gate.yml",
        ),
        ({"path": ".github/workflows/review-gate.yml@refs/heads/x"}, "inte review-gate.yml"),
        ({"event": "pull_request"}, "startad av pull_request"),
        ({"event": "push"}, "startad av push"),
        ({"repository": {"full_name": "fork/r"}}, "i ett annat repo"),
        ({"head_sha": "other"}, "på en annan commit än PR:ens head"),
        ({"event": "issue_comment", "head_branch": "x", "head_sha": "m"}, "på grenen x, inte main"),
        ({"event": "issue_comment", "head_branch": "main", "head_sha": "m"}, None),
        ({"path": ".github/workflows/review-gate.yml@refs/heads/main"}, None),
    ],
)
def test_a_gate_status_must_come_from_its_own_workflow(
    change: dict[str, Any], expected: str | None
) -> None:
    # Any workflow anywhere in the repo writes as github-actions[bot]; only a run of the
    # gate's own workflow from main counts.
    gh = FakeGitHub()
    gh.merged("5" * 40, 5, "h5", green())
    gh.runs["100"].update(change)
    messages = [f.message for f in cmg.check(gh, REPO, ["5" * 40])]
    if expected is None:
        assert messages == []
    else:
        (message,) = messages
        assert message.startswith("PR #5: gate/review kommer från körning 100")
        assert expected in message, message


def test_a_head_also_proposed_to_another_branch_fails() -> None:
    # pull_request_target runs the base branch's gate workflow; a second pull request with
    # the same head into a branch with an edited gate could have written the status.
    gh = FakeGitHub()
    gh.merged("4" * 40, 4, "h4", green())
    gh.prs["other"] = {
        "number": 9,
        "merge_commit_sha": None,
        "head": {"sha": "h4"},
        "base": {"ref": "evil"},
    }
    messages = [f.message for f in cmg.check(gh, REPO, ["4" * 40])]
    assert messages == [
        f"PR #4: {gate} kommer från körning {run}; head-commiten är också PR #9 mot evil"
        for gate, run in RUNS.items()
    ]


@pytest.mark.parametrize(
    "url",
    [None, "https://example.com/o/r/actions/runs/100", "https://github.com/x/r/actions/runs/100"],
)
def test_a_gate_status_without_a_run_in_the_repo_fails(url: str | None) -> None:
    gh = FakeGitHub()
    owner = status("gate/owner", target_url=url)
    gh.merged("6" * 40, 6, "h6", [status("gate/review"), owner, status("gate/pr-title")])
    assert [f.message for f in cmg.check(gh, REPO, ["6" * 40])] == [
        "PR #6: gate/owner länkar inte till en körning i o/r"
    ]


def test_a_green_pr_title_status_needs_a_title_that_passes() -> None:
    gh = FakeGitHub()
    gh.merged("7" * 40, 8, "h8", green())
    gh.prs["7" * 40]["title"] = "Update stuff"
    assert [f.message for f in cmg.check(gh, REPO, ["7" * 40])] == [
        "PR #8: gate/pr-title är grön men titeln följer inte regeln i pr-title.yml"
    ]


def test_the_title_rule_is_the_one_in_pr_title_yml() -> None:
    text = (cmg.REPO / ".github" / "workflows" / "pr-title.yml").read_text(encoding="utf-8")
    (rule,) = re.findall(r"re='([^']+)'", text)
    assert rule == cmg.PR_TITLE_RE.pattern


def test_the_newest_status_decides() -> None:
    gh = FakeGitHub()
    later_red = [*green(), status("gate/review", "failure", at="2026-10-01T12:00:00Z")]
    gh.merged("f" * 40, 4, "h4", later_red)
    assert [f.message for f in cmg.check(gh, REPO, ["f" * 40])] == ["PR #4: gate/review är failure"]
    later_green = [status("gate/owner", "pending", at="2026-09-30T00:00:00Z"), *green()]
    gh.merged("f" * 40, 4, "h4", later_green)
    assert cmg.check(gh, REPO, ["f" * 40]) == []


def test_exceptions_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "exceptions.txt"
    path.write_text(f"# before the gates\n{'b' * 40}  # bootstrap\n", encoding="utf-8")
    assert cmg.check(FakeGitHub(), REPO, ["b" * 40], cmg.read_exceptions(path)) == []
    path.write_text("abc\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fullständig commit-SHA"):
        cmg.read_exceptions(path)


def test_the_range_starts_at_the_newest_published_release(tmp_path: Path) -> None:
    env = ["-c", "user.name=t", "-c", "user.email=1+t@users.noreply.github.com"]
    env += ["-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"]

    def commit() -> str:
        cmg.git(tmp_path, *env, "commit", "-q", "--allow-empty", "-m", "x")
        return cmg.git(tmp_path, "rev-parse", "HEAD")

    cmg.git(tmp_path, "init", "-q")
    first = commit()
    cmg.git(tmp_path, *env, "tag", "v0.1.0")
    second = commit()
    cmg.git(tmp_path, *env, "tag", "v0.2.0")  # its release failed
    cmg.git(tmp_path, *env, "tag", "v-test")  # not a release tag
    third = commit()
    released = {"v0.1.0"}
    assert cmg.release_base(tmp_path, third, released) == ("v0.1.0", ["v0.2.0"])
    assert cmg.commits(tmp_path, "v0.1.0", third) == [second, third]
    assert cmg.release_base(tmp_path, third, {"v0.1.0", "v0.2.0"}) == ("v0.2.0", [])
    assert cmg.release_base(tmp_path, third, set()) == (None, ["v0.1.0", "v0.2.0"])
    assert cmg.release_base(tmp_path, first, released) == (None, [])
    assert cmg.commits(tmp_path, None, second) == [first, second]


def test_released_tags_ignore_drafts() -> None:
    gh = FakeGitHub()
    gh.releases = [{"tag_name": "v0.1.0", "draft": False}, {"tag_name": "v0.2.0", "draft": True}]
    assert cmg.released_tags(gh, REPO) == {"v0.1.0"}


def test_gh_api_joins_every_page(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_run(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        seen.append(argv)
        out = json.dumps([[{"n": 1}, {"n": 2}], [{"n": 3}], {"id": 9}])
        return subprocess.CompletedProcess(argv, 0, out, "")

    monkeypatch.setattr(cmg.subprocess, "run", fake_run)
    assert cmg.gh_api("repos/o/r/x") == [{"n": 1}, {"n": 2}, {"n": 3}, {"id": 9}]
    assert seen[0][:6] == ["gh", "api", "--method", "GET", "--paginate", "--slurp"]
    monkeypatch.setattr(
        cmg.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 1, "", "boom")
    )
    with pytest.raises(cmg.GitHubError, match="boom"):
        cmg.gh_api("repos/o/r/x")


def test_main_reports_and_fails(capsys: pytest.CaptureFixture[str]) -> None:
    head = cmg.git(cmg.REPO, "rev-parse", "HEAD")
    parent = cmg.git(cmg.REPO, "rev-parse", "HEAD~1")
    gh = FakeGitHub()
    assert cmg.main(["--sha", head, "--since", parent, "--repo", REPO], api=gh) == 1
    assert f"{head[:12]}: kom inte från" in capsys.readouterr().out
    gh.merged(head, 9, "h9", green())
    assert cmg.main(["--sha", head, "--since", parent, "--repo", REPO], api=gh) == 0


def test_github_errors_exit_2() -> None:
    def broken(path: str) -> Any:
        raise cmg.GitHubError("gh api: kod 1")

    head = cmg.git(cmg.REPO, "rev-parse", "HEAD")
    assert cmg.main(["--sha", head, "--since", f"{head}~1"], api=broken) == 2


def test_the_pre_gate_history_is_excepted_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    shas = cmg.read_exceptions(cmg.EXCEPTIONS)
    # The first-parent line up to PR #11, which added owner-gate.yml; nothing later.
    assert len(shas) == 15
    assert "1a64a2804e4ea54d8c910caafbf92ff88ddbab28" in shas
    head = cmg.git(cmg.REPO, "rev-parse", "HEAD")
    assert cmg.main(["--sha", head, "--since", f"{head}~1", "--repo", REPO], api=FakeGitHub()) == 1
    assert "0 commits" not in capsys.readouterr().out
