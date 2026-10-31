"""Every commit on main came from a pull request whose gates really were green.

The stored gate statuses prove nothing afterwards: every workflow in the repository, on
any branch, writes statuses as ``github-actions[bot]`` and may give them any
``target_url``. So for each first-parent commit of ``--sha`` (after ``--since``; by
default the whole history) this check works the gates out again from what GitHub
recorded about the pull request (read-only, ``gh api``) and from the merge commit in the
local clone:

* the commit is the merge commit (squash or merge) of a pull request into ``main``, not
  a direct push;
* ``gate/review``: ``check_review_comments.evaluate`` (the gate's own rule) over the
  comments written before the merge gives ``success``: the owner's unedited
  ``review/<role>: approve`` for all three roles, naming the head commit;
* ``gate/owner``: if the merge changed a path in ``check_owner_gate.PROTECTED``, the
  newest ``owner-ok`` label event before the merge is the owner's account adding it
  after the head commit;
* ``gate/pr-title``: the title follows ``PR_TITLE_RE``, the rule in ``pr-title.yml``
  (kept equal by a test);
* no workflow file the merge adds or changes can write a gate, read as YAML
  (``tools/workflow_rules.py``): outside the three gate workflows no ``statuses: write``,
  ``checks: write``, ``write-all``, missing ``permissions`` or job named like a gate; the
  gate workflows only on ``pull_request_target`` and ``issue_comment``.

Tags and releases do not shorten the default range, since anyone who can push a tag
could then hide older commits. It costs about four API calls per commit.

``--exceptions FILE`` (default ``tools/accept/pr_gate_exceptions.txt``) takes one full
commit SHA per line (``#`` starts a comment); those commits are skipped, for history from
before the gates existed and for commits the owner has accepted. Exit code 1 prints every
finding as ``<sha12>: message``; 2 means GitHub could not be asked.

Used by ``just release-prep``, the release workflow and ``just accept`` (R-08, R-09).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import check_owner_gate
import check_review_comments
import yaml
from workflow_rules import GATE_WORKFLOWS, gate_trigger_problems, status_writers

REPO = Path(__file__).resolve().parent.parent
DEFAULT_REPO_SLUG = "YXHFR/Searchdrone"
EXCEPTIONS = REPO / "tools" / "accept" / "pr_gate_exceptions.txt"
MAIN = "main"
# The same rule as .github/workflows/pr-title.yml (tools/tests/test_workflows.py).
PR_TITLE_RE = re.compile(
    r"^(feat|fix|perf|refactor|test|build|ci|docs|chore|revert|hw)(\([a-z0-9-]+\))?!?: .{3,100}$"
)
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# git diff --name-status letters -> the status names of the GitHub files API.
GIT_STATUS = {"A": "added", "D": "removed", "R": "renamed", "C": "copied"}
WORKFLOW_RE = re.compile(r"^\.github/workflows/[^/]+\.ya?ml$")

# GitHub, read-only: the API path -> parsed JSON. Tests pass a fake.
Api = Callable[[str], Any]


class GitHubError(RuntimeError):
    """``gh`` could not answer."""


@dataclass(frozen=True)
class Finding:
    sha: str
    message: str

    def __str__(self) -> str:
        return f"{self.sha[:12]}: {self.message}"


def gh_api(path: str) -> Any:
    """``gh api --paginate <path>`` (GET only); every page's array joined."""
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["gh", "api", "--method", "GET", "--paginate", "--slurp", path],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        tail = (proc.stderr.strip().splitlines() or [""])[-1]
        raise GitHubError(f"gh api {path}: kod {proc.returncode}: {tail}")
    pages = json.loads(proc.stdout or "[]")
    out: list[Any] = []
    for page in pages:
        out.extend(page if isinstance(page, list) else [page])
    return out


def git(root: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed git arguments, no shell
        ["git", "-C", str(root), *args],  # noqa: S607 - git from PATH
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def commits(root: Path, since: str | None, sha: str) -> list[str]:
    """First-parent commits of ``sha`` after ``since`` (all when ``None``), oldest first."""
    rng = f"{since}..{sha}" if since else sha
    out = git(root, "rev-list", "--first-parent", "--reverse", rng)
    return [line for line in out.splitlines() if line]


def read_exceptions(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    out = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if not SHA_RE.match(line):
            raise ValueError(f"{path}: '{line}' är inte en fullständig commit-SHA")
        out.add(line)
    return out


def merged_pull(api: Api, repo: str, sha: str) -> dict[str, Any] | None:
    """The pull request into ``main`` whose merge commit is ``sha``."""
    for pr in api(f"repos/{repo}/commits/{sha}/pulls"):
        base = str((pr.get("base") or {}).get("ref", ""))
        if pr.get("merged_at") and pr.get("merge_commit_sha") == sha and base == MAIN:
            return pr
    return None


class LocalGit:
    """What a merge commit changed, read from the local clone (no API file-list cap)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def files(self, sha: str) -> list[dict[str, Any]]:
        """``git diff --name-status`` against the first parent, as API-style dicts."""
        out = subprocess.run(  # noqa: S603 - fixed git arguments, no shell
            ["git", "-C", str(self.root), "diff", "--name-status", "-M", "-z", f"{sha}^1", sha],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split("\0")
        files: list[dict[str, Any]] = []
        i = 0
        while i < len(out) and out[i]:
            code = out[i][0]
            if code in "RC":
                files.append(
                    {
                        "status": GIT_STATUS[code],
                        "previous_filename": out[i + 1],
                        "filename": out[i + 2],
                    }
                )
                i += 3
            else:
                files.append({"status": GIT_STATUS.get(code, "modified"), "filename": out[i + 1]})
                i += 2
        return files

    def show(self, sha: str, path: str) -> str:
        return subprocess.run(  # noqa: S603 - fixed git arguments, no shell
            ["git", "-C", str(self.root), "show", f"{sha}:{path}"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout


def review_problems(api: Api, repo: str, pr: dict[str, Any], head_time: str) -> list[str]:
    """gate/review worked out again: the gate's own rule over the comments before the merge."""
    merged = check_review_comments.parse_time(str(pr["merged_at"]))
    comments = [
        c
        for c in api(f"repos/{repo}/issues/{pr['number']}/comments?per_page=100")
        if c.get("created_at") and check_review_comments.parse_time(str(c["created_at"])) <= merged
    ]
    head = str(pr["head"]["sha"])
    result = check_review_comments.evaluate(
        comments, head, check_review_comments.parse_time(head_time)
    )
    return [] if result.state == "success" else [f"gate/review: {result.description}"]


def owner_problems(
    api: Api, repo: str, pr: dict[str, Any], head_time: str, files: list[dict[str, Any]]
) -> list[str]:
    """gate/owner worked out again: protected paths need the owner's newest ``owner-ok``."""
    protected = check_owner_gate.protected_files(files)
    if not protected:
        return []
    owner = repo.split("/", 1)[0]
    parse = check_review_comments.parse_time
    merged = parse(str(pr["merged_at"]))
    events = [
        e
        for e in api(f"repos/{repo}/issues/{pr['number']}/events?per_page=100")
        if e.get("event") in ("labeled", "unlabeled")
        and (e.get("label") or {}).get("name") == check_owner_gate.LABEL
        and e.get("created_at")
        and parse(str(e["created_at"])) <= merged
    ]
    last = max(
        events, key=lambda e: (parse(str(e["created_at"])), int(e.get("id") or 0)), default=None
    )
    actor = (last or {}).get("actor") or {}
    if (
        last is not None
        and last.get("event") == "labeled"
        and actor.get("login") == owner
        and actor.get("type") == "User"
        and parse(str(last["created_at"])) > parse(head_time)
    ):
        return []
    return [
        f"gate/owner: ändrar {', '.join(protected)} utan ägarens "
        f"{check_owner_gate.LABEL} efter head-commiten"
    ]


def workflow_problems(local: LocalGit, sha: str, files: list[dict[str, Any]]) -> list[str]:
    """Workflow files the merge adds or changes that could write a gate status."""
    out = []
    for f in files:
        path = str(f.get("filename", ""))
        if f.get("status") == "removed" or not WORKFLOW_RE.match(path):
            continue
        try:
            doc = yaml.safe_load(local.show(sha, path))
        except yaml.YAMLError:
            doc = None
        if not isinstance(doc, dict):
            out.append(f"{path} går inte att läsa som en workflow")
        elif path.rsplit("/", 1)[-1] in GATE_WORKFLOWS:
            out += [
                f"{path}: {p} (bara pull_request_target/issue_comment)"
                for p in gate_trigger_problems(doc)
            ]
        elif where := status_writers(doc):
            out.append(f"{path} kan skriva en grind ({', '.join(where)})")
    return out


def check_commit(api: Api, repo: str, sha: str, local: LocalGit) -> list[Finding]:
    pr = merged_pull(api, repo, sha)
    if pr is None:
        return [
            Finding(sha, f"kom inte från en pull request sammanslagen till {MAIN} (direkt push?)")
        ]
    head = str(pr["head"]["sha"])
    head_time = str(api(f"repos/{repo}/commits/{head}")[0]["commit"]["committer"]["date"])
    files = local.files(sha)
    problems = review_problems(api, repo, pr, head_time)
    problems += owner_problems(api, repo, pr, head_time, files)
    if not PR_TITLE_RE.match(str(pr.get("title", ""))):
        problems.append("gate/pr-title: titeln följer inte regeln i pr-title.yml")
    problems += workflow_problems(local, sha, files)
    return [Finding(sha, f"PR #{pr.get('number')}: {p}") for p in problems]


def check(
    api: Api,
    repo: str,
    shas: Iterable[str],
    exceptions: set[str] | None = None,
    local: LocalGit | None = None,
) -> list[Finding]:
    local = local or LocalGit(REPO)
    out: list[Finding] = []
    for sha in shas:
        if sha not in (exceptions or set()):
            out += check_commit(api, repo, sha, local)
    return out


def main(argv: list[str] | None = None, api: Api = gh_api) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--sha", default="HEAD", help="the commit to release (default HEAD)")
    parser.add_argument("--since", help="older end, excluded (default: the whole history)")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO_SLUG))
    parser.add_argument("--exceptions", type=Path, default=EXCEPTIONS)
    args = parser.parse_args(argv)
    sha = git(REPO, "rev-parse", f"{args.sha}^{{commit}}")
    try:
        exceptions = read_exceptions(args.exceptions)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    try:
        since = args.since
        shas = commits(REPO, since, sha)
        findings = check(api, args.repo, shas, exceptions)
    except GitHubError as exc:
        print(exc, file=sys.stderr)
        return 2
    for finding in findings:
        print(finding)
    span = f"{since[:12]}..{sha[:12]}" if since else f"hela historiken till {sha[:12]}"
    if findings:
        print(f"{len(findings)} fynd i {len(shas)} commits ({span}).")
        return 1
    print(f"check_merged_pr_gates: {len(shas)} commits ({span}), alla från PR med gröna grindar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
