"""tools/check_merged_pr_gates.py against a fake GitHub: green, red and forged gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import check_merged_pr_gates as cmg
import pytest

REPO = "o/r"
BOT = {"login": "github-actions[bot]"}


def status(context: str, state: str = "success", at: str = "2026-10-01T10:00:00Z", **kw: Any):
    return {"context": context, "state": state, "created_at": at, "creator": BOT, **kw}


def green(head: str = "h1") -> list[dict[str, Any]]:
    return [status(g) for g in cmg.GATES]


class FakeGitHub:
    """Merge commit -> (PR number, head SHA), head SHA -> statuses."""

    def __init__(self) -> None:
        self.prs: dict[str, dict[str, Any]] = {}
        self.statuses: dict[str, list[dict[str, Any]]] = {}

    def merged(self, sha: str, number: int, head: str, statuses: list[dict[str, Any]]) -> None:
        self.prs[sha] = {
            "number": number,
            "merged_at": "2026-10-01T11:00:00Z",
            "merge_commit_sha": sha,
            "head": {"sha": head},
        }
        self.statuses[head] = statuses

    def __call__(self, path: str) -> Any:
        parts = path.split("?")[0].split("/")
        sha = parts[4]
        if parts[-1] == "pulls":
            return [self.prs[sha]] if sha in self.prs else []
        return self.statuses.get(sha, [])


def test_green_gates_pass() -> None:
    gh = FakeGitHub()
    gh.merged("a" * 40, 1, "h1", green())
    assert cmg.check(gh, REPO, ["a" * 40]) == []


def test_direct_push_fails() -> None:
    (finding,) = cmg.check(FakeGitHub(), REPO, ["b" * 40])
    assert "inte från en sammanslagen pull request" in finding.message


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


def test_commits_since_the_previous_tag(tmp_path: Path) -> None:
    env = ["-c", "user.name=t", "-c", "user.email=1+t@users.noreply.github.com"]
    env += ["-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"]

    def commit() -> str:
        cmg.git(tmp_path, *env, "commit", "-q", "--allow-empty", "-m", "x")
        return cmg.git(tmp_path, "rev-parse", "HEAD")

    cmg.git(tmp_path, "init", "-q")
    first = commit()
    cmg.git(tmp_path, *env, "tag", "v0.1.0")
    second = commit()
    third = commit()
    cmg.git(tmp_path, *env, "tag", "v0.2.0")
    assert cmg.previous_tag(tmp_path, third) == "v0.1.0"
    assert cmg.commits(tmp_path, "v0.1.0", third) == [second, third]
    assert cmg.previous_tag(tmp_path, first) is None
    assert cmg.commits(tmp_path, None, second) == [first, second]


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
