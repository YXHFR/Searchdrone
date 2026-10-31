"""Every commit on main since the last release came from a pull request with green gates.

For each commit on the first-parent line of ``--sha`` after ``--since`` (by default the
newest ``v*`` tag before ``--sha``; without a tag the whole history), the check asks
GitHub (read-only, through ``gh api``) for the pull request it was merged from and the
gate statuses on that pull request's last head commit:

* the commit must be a pull request's merge commit (squash or merge), not a direct push;
* ``gate/review``, ``gate/owner`` and ``gate/pr-title`` must each be ``success`` in their
  newest status on the head commit;
* each of those statuses must be written by ``github-actions[bot]``, the token of the
  gate workflows. Any other writer (a person's token, another app) could have forged it.

``--exceptions FILE`` takes one full commit SHA per line (``#`` starts a comment); those
commits are skipped, for history from before the gates existed. Exit code 1 prints every
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

REPO = Path(__file__).resolve().parent.parent
DEFAULT_REPO_SLUG = "YXHFR/Searchdrone"
GATES = ("gate/review", "gate/owner", "gate/pr-title")
GATE_WRITER = "github-actions[bot]"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")

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


def previous_tag(root: Path, sha: str) -> str | None:
    """The newest ``v*`` tag reachable from ``sha`` (not ``sha`` itself when it is tagged)."""
    proc = subprocess.run(  # noqa: S603 - fixed git arguments, no shell
        ["git", "-C", str(root), "describe", "--tags", "--abbrev=0", "--match", "v*", f"{sha}^"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    tag = proc.stdout.strip()
    return tag if proc.returncode == 0 and tag else None


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
    """The merged pull request whose merge commit is ``sha``."""
    for pr in api(f"repos/{repo}/commits/{sha}/pulls"):
        if pr.get("merged_at") and pr.get("merge_commit_sha") == sha:
            return pr
    return None


def newest_statuses(statuses: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Context -> its newest status (the API lists the newest first; sorted to be sure)."""
    ordered = sorted(statuses, key=lambda s: (str(s.get("created_at", "")), s.get("id", 0)))
    return {str(s.get("context")): s for s in ordered}


def gate_problems(statuses: Iterable[dict[str, Any]]) -> list[str]:
    newest = newest_statuses(statuses)
    out = []
    for gate in GATES:
        status = newest.get(gate)
        if status is None:
            out.append(f"{gate} saknas")
            continue
        writer = str((status.get("creator") or {}).get("login", "?"))
        if writer != GATE_WRITER:
            out.append(f"{gate} skrevs av {writer}, inte {GATE_WRITER}")
        elif status.get("state") != "success":
            out.append(f"{gate} är {status.get('state')}")
    return out


def check_commit(api: Api, repo: str, sha: str) -> list[Finding]:
    pr = merged_pull(api, repo, sha)
    if pr is None:
        return [Finding(sha, "kom inte från en sammanslagen pull request (direkt push?)")]
    head = str(pr["head"]["sha"])
    number = pr.get("number")
    statuses = api(f"repos/{repo}/commits/{head}/statuses?per_page=100")
    return [Finding(sha, f"PR #{number}: {p}") for p in gate_problems(statuses)]


def check(
    api: Api, repo: str, shas: Iterable[str], exceptions: set[str] | None = None
) -> list[Finding]:
    out: list[Finding] = []
    for sha in shas:
        if sha not in (exceptions or set()):
            out += check_commit(api, repo, sha)
    return out


def main(argv: list[str] | None = None, api: Api = gh_api) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--sha", default="HEAD", help="the commit to release (default HEAD)")
    parser.add_argument("--since", help="older end, excluded (default: the previous v* tag)")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO_SLUG))
    parser.add_argument("--exceptions", type=Path)
    args = parser.parse_args(argv)
    sha = git(REPO, "rev-parse", f"{args.sha}^{{commit}}")
    since = args.since or previous_tag(REPO, sha)
    try:
        exceptions = read_exceptions(args.exceptions)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    shas = commits(REPO, since, sha)
    try:
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
