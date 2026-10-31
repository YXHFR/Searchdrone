"""tools/check_merged_pr_gates.py against a fake GitHub: gates worked out again from the
pull request's comments, label events, title and the merge commit's files."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import check_merged_pr_gates as cmg
import pytest

REPO = "o/r"
TITLE = "feat(tools): a change"
HEAD_TIME = "2026-10-01T09:00:00Z"
MERGED = "2026-10-01T11:00:00Z"
OWNER = {"login": "o", "type": "User"}
# Head commits: full SHAs, since a verdict must name the whole SHA.
H1, H2, H3, H4, H5, H7, H8, H9 = (str(n) * 40 for n in (1, 2, 3, 4, 5, 7, 8, 9))


def verdict(
    role: str, head: str, word: str = "approve", at: str = "2026-10-01T10:00:00Z", **kw: Any
):
    return {
        "id": hash((role, head, at)) % 10**6,
        "body": f"review/{role}: {word}\n\nHead: {head}\n",
        "author_association": "OWNER",
        "user": {"login": "o", "type": "User"},
        "created_at": at,
        "updated_at": at,
        **kw,
    }


def approvals(head: str) -> list[dict[str, Any]]:
    return [verdict(r, head) for r in ("architecture", "security", "flight-safety")]


def label(event: str = "labeled", at: str = "2026-10-01T10:30:00Z", actor: Any = OWNER, i: int = 1):
    return {
        "id": i,
        "event": event,
        "label": {"name": "owner-ok"},
        "actor": actor,
        "created_at": at,
    }


class FakeGitHub:
    """Merge commit -> PR; PR -> comments and issue events; head -> commit date and check
    suites. Any other request fails the test."""

    ROUTES = re.compile(
        r"repos/o/r/(?:issues/(?P<n>\d+)/(?P<kind>comments|events)"
        r"|commits/(?P<sha>\w+)(?P<rest>/pulls|/check-suites)?)(?:\?per_page=100)?"
    )

    def __init__(self) -> None:
        self.prs: dict[str, dict[str, Any]] = {}
        self.comments: dict[int, list[dict[str, Any]]] = {}
        self.events: dict[int, list[dict[str, Any]]] = {}
        self.suites: dict[str, list[str]] = {}
        self.committed: dict[str, str] = {}

    def merged(
        self, sha: str, number: int, head: str, comments: list[dict[str, Any]] | None = None
    ):
        self.prs[sha] = {
            "number": number,
            "title": TITLE,
            "merged_at": MERGED,
            "merge_commit_sha": sha,
            "head": {"sha": head},
            "base": {"ref": "main"},
        }
        self.comments[number] = approvals(head) if comments is None else comments

    def __call__(self, path: str) -> Any:
        m = self.ROUTES.fullmatch(path)
        if m is None:
            raise AssertionError(f"unexpected request {path}")
        if m["kind"]:
            store = self.comments if m["kind"] == "comments" else self.events
            return store.get(int(m["n"]), [])
        sha = m["sha"]
        if m["rest"] == "/pulls":
            return [pr for pr in self.prs.values() if pr["merge_commit_sha"] == sha]
        if m["rest"] == "/check-suites":
            suites = [{"created_at": at} for at in self.suites.get(sha, [HEAD_TIME])]
            return [{"total_count": len(suites), "check_suites": suites}]
        date = self.committed.get(sha, HEAD_TIME)
        return [{"sha": sha, "commit": {"committer": {"date": date}}}]


class FakeGit:
    """Merge commit -> changed files and their contents (stands in for the local clone)."""

    def __init__(self) -> None:
        self.changed: dict[str, list[dict[str, Any]]] = {}
        self.contents: dict[tuple[str, str], str] = {}

    def add(self, sha: str, path: str, text: str = "", status: str = "added", **kw: Any) -> None:
        self.changed.setdefault(sha, []).append({"filename": path, "status": status, **kw})
        self.contents[(sha, path)] = text

    def files(self, sha: str) -> list[dict[str, Any]]:
        return self.changed.get(sha, [])

    def show(self, sha: str, path: str) -> str:
        return self.contents[(sha, path)]


def run(gh: FakeGitHub, sha: str, git: FakeGit | None = None) -> list[str]:
    return [f.message for f in cmg.check(gh, REPO, [sha], local=git or FakeGit())]


def test_green_gates_pass() -> None:
    gh = FakeGitHub()
    gh.merged("a" * 40, 1, H1)
    assert run(gh, "a" * 40) == []


def test_direct_push_fails() -> None:
    (message,) = run(FakeGitHub(), "b" * 40)
    assert "inte från en pull request sammanslagen till main" in message


def test_a_pr_merged_into_another_branch_does_not_count() -> None:
    gh = FakeGitHub()
    gh.merged("c" * 40, 2, H2)
    gh.prs["c" * 40]["base"] = {"ref": "release-x"}
    assert run(gh, "c" * 40)[0].startswith("kom inte från")


def test_open_pr_that_only_contains_the_commit_does_not_count() -> None:
    gh = FakeGitHub()
    gh.merged("c" * 40, 2, H2)
    gh.prs["c" * 40]["merge_commit_sha"] = "d" * 40
    assert run(gh, "c" * 40)[0].startswith("kom inte från")


@pytest.mark.parametrize(
    ("comments", "expected"),
    [
        ([], "Waiting for review of 3333333"),
        (approvals(H3)[:2], "flight-safety"),
        (approvals("old"), "Waiting for review"),
        (
            [
                *approvals(H3),
                verdict("security", H3, "request_changes", "2026-10-01T10:10:00Z"),
            ],
            "Changes requested by security",
        ),
        ([*approvals(H3)[:2], verdict("flight-safety", H3, updated_at=MERGED)], "Edited"),
        # Written before the head commit, or after the merge: neither counts.
        (
            [*approvals(H3)[:2], verdict("flight-safety", H3, at="2026-10-01T08:00:00Z")],
            "flight",
        ),
        (
            [*approvals(H3)[:2], verdict("flight-safety", H3, at="2026-10-01T12:00:00Z")],
            "flight",
        ),
        # A verdict the owner did not write.
        (
            [*approvals(H3)[:2], verdict("flight-safety", H3, author_association="MEMBER")],
            "flight-safety",
        ),
        (
            [*approvals(H3)[:2], verdict("flight-safety", H3, user={"type": "Bot"})],
            "flight-safety",
        ),
    ],
)
def test_review_is_worked_out_again_from_the_comments(
    comments: list[dict[str, Any]], expected: str
) -> None:
    # A green gate/review status proves nothing (any workflow can write one); the
    # comments before the merge must pass the gate's own rule.
    gh = FakeGitHub()
    gh.merged("e" * 40, 3, H3, comments)
    (message,) = run(gh, "e" * 40)
    assert message.startswith("PR #3: gate/review: ") and expected in message, message


@pytest.mark.parametrize(
    ("events", "ok"),
    [
        ([label()], True),
        ([], False),
        ([label(actor={"login": "mallory", "type": "User"})], False),
        ([label(actor={"login": "o", "type": "Bot"})], False),
        ([label(at="2026-10-01T08:00:00Z")], False),  # before the head commit
        ([label(at="2026-10-01T12:00:00Z")], False),  # after the merge
        (
            [
                label(),
                label("unlabeled", "2026-10-01T10:40:00Z", {"login": "github-actions[bot]"}, 2),
            ],
            False,
        ),
        ([label("unlabeled", "2026-10-01T09:30:00Z", {"login": "x"}, 2), label()], True),
        # Same second: the newer event id decides.
        ([label(i=5), label("unlabeled", actor={"login": "x"}, i=4)], True),
        # A new base, force push or reopening after the label voids it.
        *(
            ([label(), {"id": 9, "event": e, "created_at": "2026-10-01T10:45:00Z"}], False)
            for e in ("base_ref_changed", "head_ref_force_pushed", "reopened")
        ),
        ([{"id": 9, "event": "reopened", "created_at": "2026-10-01T09:10:00Z"}, label()], True),
    ],
)
def test_owner_is_worked_out_again_from_the_label_events(
    events: list[dict[str, Any]], ok: bool
) -> None:
    gh = FakeGitHub()
    gh.merged("1" * 40, 4, H4)
    gh.events[4] = events
    git = FakeGit()
    git.add("1" * 40, "hardware/board.kicad_pcb", status="modified")
    messages = run(gh, "1" * 40, git)
    if ok:
        assert messages == []
    else:
        (message,) = messages
        assert message == (
            "PR #4: gate/owner: ändrar hardware/board.kicad_pcb utan ägarens owner-ok "
            "efter head-commiten"
        )


@pytest.mark.parametrize(
    ("suites", "ok"),
    [
        ([], False),  # no suite: nothing GitHub recorded dates the push
        (["2026-10-01T09:05:00Z", "2026-10-01T10:50:00Z"], True),  # the first suite counts
        # Backdated head: committed 09:00, but it reached GitHub at 10:40, after the label.
        (["2026-10-01T10:40:00Z"], False),
    ],
)
def test_the_owner_label_must_follow_the_push_not_the_committer_date(
    suites: list[str], ok: bool
) -> None:
    gh = FakeGitHub()
    gh.merged("1" * 40, 4, H4)
    gh.events[4] = [label()]
    gh.suites[H4] = suites
    git = FakeGit()
    git.add("1" * 40, "hardware/board.kicad_pcb", status="modified")
    messages = run(gh, "1" * 40, git)
    assert (messages == []) is ok
    if not suites:
        assert messages == [
            "PR #4: gate/owner: ändrar hardware/board.kicad_pcb, men head-commitens push "
            "går inte att datera"
        ]


@pytest.mark.parametrize(
    "change",
    [
        {"filename": ".github/CODEOWNERS", "status": "modified"},
        {"filename": "docs/x.md", "status": "renamed", "previous_filename": "Hardware/x.md"},
        {"filename": "hardware/x", "status": "removed"},
    ],
)
def test_any_protected_path_needs_the_owner(change: dict[str, Any]) -> None:
    gh = FakeGitHub()
    gh.merged("2" * 40, 5, H5)
    git = FakeGit()
    git.changed["2" * 40] = [change]
    (message,) = run(gh, "2" * 40, git)
    assert message.startswith("PR #5: gate/owner: ")


@pytest.mark.parametrize(
    ("parent_rules", "ok"),
    [
        (None, True),  # no list before the merge: only this copy's rules
        ('PROTECTED = (re.compile(r"^docs/keys\\.md$"),)\n', False),
        ('PROTECTED = (re.compile(r"^DOCS/", re.IGNORECASE),)\n', False),
        ('PROTECTED = (re.compile(r"^other/"),)\n', True),
        ('PROTECTED = (re.compile(pattern=r"^docs/", flags=re.I | re.X),)\n', False),
        # A list this check cannot read in full: every changed path needs the owner.
        ("PROTECTED = (re.compile(PATTERN),)\n", False),
        ('PROTECTED = tuple(map(re.compile, ["^a/"]))\n', False),
        ('PROTECTED = (re.compile(r"^a/", 2), *MORE)\n', False),
        ("not python (\n", False),
    ],
)
def test_the_parents_protected_paths_apply(parent_rules: str | None, ok: bool) -> None:
    # A path the merge's parent protects needs owner-ok even if this copy's list
    # (or the list the merge itself leaves behind) no longer names it.
    gh = FakeGitHub()
    gh.merged("3" * 40, 6, H3)
    git = FakeGit()
    git.add("3" * 40, "docs/keys.md", status="modified")
    if parent_rules is not None:
        git.contents[("3" * 40 + "^1", cmg.OWNER_GATE)] = parent_rules
    messages = run(gh, "3" * 40, git)
    assert (messages == []) is ok
    # This copy's own list applies whatever the parent's says.
    git.add("3" * 40, "hardware/board.kicad_pcb", status="modified")
    assert any("hardware/board.kicad_pcb" in m for m in run(gh, "3" * 40, git))
    git.changed["3" * 40].pop()
    if not ok:
        assert messages == [
            "PR #6: gate/owner: ändrar docs/keys.md utan ägarens owner-ok efter head-commiten"
        ]


@pytest.mark.parametrize(
    ("path", "text"),
    [
        # Weakening the gates themselves: each needs owner-ok under the rules before it.
        ("tools/check_owner_gate.py", "PROTECTED = ()\n"),
        ("tools/check_merged_pr_gates.py", "def main(argv=None, api=None):\n    return 0\n"),
        ("tools/accept/pr_gate_exceptions.txt", "a" * 40 + "\n"),
        # release.yml without its gates job.
        (
            ".github/workflows/release.yml",
            "on:\n  push:\n    tags: [v1]\npermissions:\n  contents: read\n"
            "jobs:\n  build:\n    runs-on: x\n",
        ),
    ],
)
def test_weakening_the_release_check_needs_the_owner(path: str, text: str) -> None:
    gh = FakeGitHub()
    gh.merged("4" * 40, 10, H4)
    git = FakeGit()
    git.add("4" * 40, path, text, "modified")
    (message,) = run(gh, "4" * 40, git)
    assert message == f"PR #10: gate/owner: ändrar {path} utan ägarens owner-ok efter head-commiten"
    gh.events[10] = [label()]
    assert run(gh, "4" * 40, git) == []


def test_read_protected_matches_check_owner_gate() -> None:
    text = (cmg.REPO / cmg.OWNER_GATE).read_text(encoding="utf-8")
    import check_owner_gate

    assert cmg.read_protected(text) == list(check_owner_gate.PROTECTED)


def test_local_git_lists_renames_and_deletes(tmp_path: Path) -> None:
    env = ["-c", "user.name=t", "-c", "user.email=1+t@users.noreply.github.com"]
    env += ["-c", "commit.gpgsign=false"]
    cmg.git(tmp_path, "init", "-q")
    (tmp_path / "hardware").mkdir()
    (tmp_path / "hardware" / "a.txt").write_text("same text, long enough to be a rename\n")
    (tmp_path / "gone.txt").write_text("x\n")
    cmg.git(tmp_path, "add", ".")
    cmg.git(tmp_path, *env, "commit", "-q", "-m", "1")
    cmg.git(tmp_path, "mv", "hardware/a.txt", "a b.txt")
    cmg.git(tmp_path, "rm", "-q", "gone.txt")
    (tmp_path / "new.yml").write_text("on: push\n")
    cmg.git(tmp_path, "add", ".")
    cmg.git(tmp_path, *env, "commit", "-q", "-m", "2")
    local = cmg.LocalGit(tmp_path)
    files = sorted(local.files("HEAD"), key=lambda f: f["filename"])
    assert files == [
        {"status": "renamed", "previous_filename": "hardware/a.txt", "filename": "a b.txt"},
        {"status": "removed", "filename": "gone.txt"},
        {"status": "added", "filename": "new.yml"},
    ]
    assert local.show("HEAD", "new.yml") == "on: push\n"
    with pytest.raises(subprocess.CalledProcessError):
        local.files(cmg.git(tmp_path, "rev-list", "--max-parents=0", "HEAD"))
    # A missing file is KeyError (the parent had no list); any other git error is not.
    with pytest.raises(KeyError):
        local.show("HEAD", "gone.txt")
    assert cmg.parent_protected(local, "HEAD") == []
    with pytest.raises(subprocess.CalledProcessError):
        local.show("0" * 40, "new.yml")
    assert cmg.parent_protected(local, "0" * 40) is None


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
        (
            ".github/workflows/x.yml",
            "jobs:\n  a:\n    name: gate/review\n",
            "added",
            "(default a, name a)",
        ),
        (
            ".github/workflows/x.yml",
            "permissions: {}\njobs:\n  a:\n    name: Gate/Owner\n",
            "added",
            "(name a)",
        ),
        # No permissions anywhere: the repository's default token, which may write.
        (
            ".github/workflows/x.yml",
            "on: push\njobs:\n  a:\n    runs-on: x\n",
            "added",
            "(default a)",
        ),
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
        (
            ".github/workflows/x.yml",
            "permissions:\n  pull-requests: write\n",
            "added",
            "kan godkänna i ägarens ställe (top)",
        ),
        (
            ".github/workflows/x.yml",
            "permissions: {}\njobs:\n  a:\n    permissions: {}\n    env:\n"
            "      GH_TOKEN: ${{ secrets.PAT }}\n",
            "modified",
            "kan godkänna i ägarens ställe (secret a)",
        ),
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
def test_a_workflow_from_the_merge_that_can_write_a_gate_fails(
    path: str, text: str, status: str, expected: str | None
) -> None:
    # The gates are worked out again here, but branch protection reads the statuses:
    # no merged workflow may be able to write them.
    gh = FakeGitHub()
    gh.merged("9" * 40, 7, H7)
    git = FakeGit()
    git.add("9" * 40, path, text, status)
    gh.events[7] = [label()]  # the gate workflows are protected paths
    messages = run(gh, "9" * 40, git)
    if expected is None:
        assert messages == []
    else:
        (message,) = messages
        assert message.startswith(f"PR #7: {path}") and expected in message, message


def rename(old: str, new: str, at: str, i: int = 1) -> dict[str, Any]:
    return {"id": i, "event": "renamed", "rename": {"from": old, "to": new}, "created_at": at}


@pytest.mark.parametrize(
    ("title", "events", "ok"),
    [
        ("Update stuff", [], False),
        # Fixed after the merge: the title at the merge decides.
        (TITLE, [rename("Update stuff", TITLE, "2026-10-01T12:00:00Z")], False),
        (
            "Later",
            [
                rename(TITLE, "x", "2026-10-01T12:00:00Z", 1),
                rename("x", "Later", "2026-10-01T13:00:00Z", 2),
            ],
            True,
        ),
        (TITLE, [rename("Update stuff", TITLE, "2026-10-01T10:00:00Z")], True),
        # A rename in the merge's own second counts as after it.
        (TITLE, [rename("Update stuff", TITLE, MERGED)], False),
    ],
)
def test_the_title_at_the_merge_must_pass(
    title: str, events: list[dict[str, Any]], ok: bool
) -> None:
    gh = FakeGitHub()
    gh.merged("7" * 40, 8, H8)
    gh.prs["7" * 40]["title"] = title
    gh.events[8] = events
    expected = (
        []
        if ok
        else ["PR #8: gate/pr-title: titeln vid sammanslagningen följer inte regeln i pr-title.yml"]
    )
    assert run(gh, "7" * 40) == expected


def test_the_title_rule_is_the_one_in_pr_title_yml() -> None:
    text = (cmg.REPO / ".github" / "workflows" / "pr-title.yml").read_text(encoding="utf-8")
    (rule,) = re.findall(r"re='([^']+)'", text)
    assert rule == cmg.PR_TITLE_RE.pattern


def test_exceptions_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "exceptions.txt"
    path.write_text(f"# before the gates\n{'b' * 40}  # bootstrap\n", encoding="utf-8")
    assert cmg.check(FakeGitHub(), REPO, ["b" * 40], cmg.read_exceptions(path)) == []
    path.write_text("abc\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fullständig commit-SHA"):
        cmg.read_exceptions(path)


def test_the_range_is_the_whole_first_parent_line(tmp_path: Path) -> None:
    env = ["-c", "user.name=t", "-c", "user.email=1+t@users.noreply.github.com"]
    env += ["-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"]

    def commit() -> str:
        cmg.git(tmp_path, *env, "commit", "-q", "--allow-empty", "-m", "x")
        return cmg.git(tmp_path, "rev-parse", "HEAD")

    cmg.git(tmp_path, "init", "-q")
    first = commit()
    cmg.git(tmp_path, *env, "tag", "v0.1.0")  # a tag does not shorten the range
    second = commit()
    assert cmg.commits(tmp_path, None, second) == [first, second]
    assert cmg.commits(tmp_path, first, second) == [second]


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
    gh.merged(head, 9, H9)
    gh.events[9] = [label()]  # in case HEAD changes a protected path
    assert cmg.main(["--sha", head, "--since", parent, "--repo", REPO], api=gh) == 0


@pytest.mark.parametrize(
    "since",
    [
        "HEAD",  # nothing to check is not a pass
        "--output=/tmp/x",  # never read as an option
        "no-such-rev",
    ],
)
def test_a_bad_range_exits_2(since: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cmg.main(["--sha", "HEAD", f"--since={since}", "--repo", REPO], api=FakeGitHub()) == 2
    assert "Traceback" not in capsys.readouterr().err


def test_since_must_be_an_ancestor(capsys: pytest.CaptureFixture[str]) -> None:
    root = cmg.git(cmg.REPO, "rev-list", "--max-parents=0", "HEAD").splitlines()[0]
    assert cmg.main(["--sha", root, "--since", "HEAD", "--repo", REPO], api=FakeGitHub()) == 2
    assert "--since HEAD är inte en förfader till --sha" in capsys.readouterr().err


def test_a_copy_outside_the_repo_checks_root(tmp_path: Path) -> None:
    # release.yml runs the protected scripts from a clean directory against --root.
    head = cmg.git(cmg.REPO, "rev-parse", "HEAD")
    gh = FakeGitHub()
    gh.merged(head, 9, H9)
    gh.events[9] = [label()]
    argv = ["--root", str(cmg.REPO), "--sha", head, "--since", f"{head}~1", "--repo", REPO]
    assert cmg.main(argv, api=gh) == 0
    assert cmg.main([*argv[:2], "--root", str(tmp_path)], api=gh) == 2  # not a clone


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


GIT_ENV = ["-c", "user.name=t", "-c", "user.email=1+t@users.noreply.github.com"]
GIT_ENV += ["-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"]
BOT = {"login": "github-actions[bot]"}


def tagged_repo(tmp_path: Path) -> dict[str, str]:
    """main: c1 (v0.1.0) - c2 (v0.2.0) - c3 (released now); side: s1 (v0.9.0), off main."""
    cmg.git(tmp_path, "init", "-q", "-b", "main")
    out: dict[str, str] = {}
    for name in ("c1", "c2", "c3"):
        cmg.git(tmp_path, *GIT_ENV, "commit", "-q", "--allow-empty", "-m", name)
        out[name] = cmg.git(tmp_path, "rev-parse", "HEAD")
    cmg.git(tmp_path, "tag", "v0.1.0", out["c1"])
    cmg.git(tmp_path, "tag", "v0.2.0", out["c2"])
    cmg.git(tmp_path, "tag", "v0.3.0", out["c3"])
    cmg.git(tmp_path, "checkout", "-q", "-b", "side", out["c1"])
    cmg.git(tmp_path, *GIT_ENV, "commit", "-q", "--allow-empty", "-m", "s1")
    out["s1"] = cmg.git(tmp_path, "rev-parse", "HEAD")
    cmg.git(tmp_path, "tag", "v0.9.0", out["s1"])
    return out


class FakeReleases:
    def __init__(self) -> None:
        self.releases: list[dict[str, Any]] = []
        self.runs: dict[str, list[dict[str, Any]]] = {}

    def release(self, tag: str, commit: str, author: Any = BOT, run: bool = True, **kw: Any):
        self.releases.append({"tag_name": tag, "draft": False, "author": author, **kw})
        if run:
            self.runs.setdefault(commit, []).append(
                {
                    "head_branch": tag,
                    "head_sha": commit,
                    "conclusion": "success",
                    "path": ".github/workflows/release.yml",
                }
            )

    def __call__(self, path: str) -> Any:
        if path == "repos/o/r/releases?per_page=100":
            return self.releases
        m = re.fullmatch(
            r"repos/o/r/actions/workflows/release\.yml/runs"
            r"\?event=push&status=success&head_sha=(\w+)&per_page=100",
            path,
        )
        if m is None:
            raise AssertionError(f"unexpected request {path}")
        return [{"workflow_runs": self.runs.get(m[1], [])}]


def anchor(gh: FakeReleases, root: Path, c: dict[str, str]) -> str | None:
    return cmg.release_anchor(gh, REPO, root, c["c3"], "v0.3.0")


def test_the_first_release_has_no_anchor(tmp_path: Path) -> None:
    c = tagged_repo(tmp_path)
    gh = FakeReleases()
    assert anchor(gh, tmp_path, c) is None
    # Not releases on this line: another pattern, a draft, the tag itself, another branch.
    gh.release("x", c["c2"])
    gh.release("v0.2.0", c["c2"], draft=True)
    gh.release("v0.3.0", c["c3"])
    gh.release("v0.9.0", c["s1"])
    assert anchor(gh, tmp_path, c) is None


def test_the_anchor_is_the_newest_release_made_by_release_yml(tmp_path: Path) -> None:
    c = tagged_repo(tmp_path)
    gh = FakeReleases()
    gh.release("v0.2.0", c["c2"])
    gh.release("v0.1.0", c["c1"])
    assert anchor(gh, tmp_path, c) == c["c2"]


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"author": {"login": "mallory"}}, "skapades av mallory"),
        ({"run": False}, "ingen lyckad release.yml-körning"),
        ({"head_branch": "v0.1.0"}, "ingen lyckad release.yml-körning"),
        ({"path": ".github/workflows/other.yml"}, "ingen lyckad release.yml-körning"),
        ({"conclusion": "failure"}, "ingen lyckad release.yml-körning"),
    ],
)
def test_a_release_not_made_by_release_yml_stops_the_release(
    tmp_path: Path, change: dict[str, Any], error: str
) -> None:
    # A hand-made release (or one whose run failed) must not shorten the check or pick
    # its rules, and must not quietly fall back to the tagged commit's own rules either.
    c = tagged_repo(tmp_path)
    gh = FakeReleases()
    gh.release("v0.1.0", c["c1"])
    author = change.pop("author", BOT)
    gh.release("v0.2.0", c["c2"], author=author, run=change.pop("run", True))
    for run in gh.runs.get(c["c2"], []):
        run.update(change)
    with pytest.raises(ValueError, match=error):
        anchor(gh, tmp_path, c)


def test_a_release_without_its_tag_stops_the_release(tmp_path: Path) -> None:
    c = tagged_repo(tmp_path)
    gh = FakeReleases()
    gh.release("v0.4.0", c["c2"])
    with pytest.raises(ValueError, match=r"v0\.4\.0 finns inte i klonen"):
        anchor(gh, tmp_path, c)


def test_main_prints_the_anchor(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    c = tagged_repo(tmp_path)
    gh = FakeReleases()
    argv = ["--root", str(tmp_path), "--sha", c["c3"], "--repo", REPO]
    assert cmg.main([*argv, "--anchor-for-tag", "v0.3.0"], api=gh) == 0
    assert capsys.readouterr().out == "\n"
    gh.release("v0.2.0", c["c2"])
    assert cmg.main([*argv, "--anchor-for-tag", "v0.3.0"], api=gh) == 0
    assert capsys.readouterr().out == c["c2"] + "\n"
    gh.releases[0]["author"] = {"login": "mallory"}
    assert cmg.main([*argv, "--anchor-for-tag", "v0.3.0"], api=gh) == 2
