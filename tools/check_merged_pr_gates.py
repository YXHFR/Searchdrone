"""Every commit on main since the last release came from a pull request with green gates.

For each commit on the first-parent line of ``--sha`` after ``--since`` (by default the
newest ``v*`` tag before ``--sha`` that has a published GitHub release; without one the
whole history), the check asks GitHub (read-only, through ``gh api``) for the pull
request it was merged from and the gate statuses on that pull request's last head commit:

* the commit must be the merge commit (squash or merge) of a pull request into ``main``,
  not a direct push;
* ``gate/review``, ``gate/owner`` and ``gate/pr-title`` must each be ``success`` in their
  newest status on the head commit, written by ``github-actions[bot]``. That is the
  token of every workflow, so the writer alone proves little; each status is also traced:
* ``gate/review`` and ``gate/owner`` must link (``target_url``) to a run of
  ``review-gate.yml`` / ``owner-gate.yml`` in this repository, started by
  ``pull_request_target`` for this pull request into ``main`` on its head commit, or by
  ``issue_comment`` on ``main``. Both events run the workflow file from the base branch,
  never the pull request's copy. A run does not say which pull request it ran for, so no
  other pull request may have the same head commit and a base other than ``main``. (An
  ``issue_comment`` run carries main's commit, not the head, so the head can only be
  matched for ``pull_request_target``.)
* ``gate/pr-title`` has never linked to its run, so the pull request's title is checked
  again with the rule from ``pr-title.yml`` (``PR_TITLE_RE``, kept equal by a test);
* no workflow file the pull request adds or changes may be able to write a gate, read as
  YAML: outside the three gate workflows no ``statuses: write``, ``checks: write``,
  ``write-all`` or job named like ``gate/...`` or ``review/...``; the gate workflows
  themselves only on ``pull_request_target`` and ``issue_comment``
  (``tools/workflow_rules.py``, which ``tools/tests/test_workflows.py`` holds main to).

A ``v*`` tag without a published release (a release that failed) does not end the
range; its commits are checked again and the tag is named in the output.

``--exceptions FILE`` (default ``tools/accept/pr_gate_exceptions.txt``) takes one full
commit SHA per line (``#`` starts a comment); those commits are skipped, for history from
before the gates existed. Exit code 1 prints every
finding as ``<sha12>: message``; 2 means GitHub could not be asked.

Used by ``just release-prep``, the release workflow and ``just accept`` (R-08, R-09).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from workflow_rules import GATE_TRIGGERS, GATE_WORKFLOWS, gate_trigger_problems, status_writers

REPO = Path(__file__).resolve().parent.parent
DEFAULT_REPO_SLUG = "YXHFR/Searchdrone"
EXCEPTIONS = REPO / "tools" / "accept" / "pr_gate_exceptions.txt"
MAIN = "main"
GATES = ("gate/review", "gate/owner", "gate/pr-title")
GATE_WRITER = "github-actions[bot]"
# The gate -> the workflow that alone may write it (gate/pr-title has no target_url).
GATE_SOURCES = {"gate/review": "review-gate.yml", "gate/owner": "owner-gate.yml"}
# The same rule as .github/workflows/pr-title.yml (tools/tests/test_workflows.py).
PR_TITLE_RE = re.compile(
    r"^(feat|fix|perf|refactor|test|build|ci|docs|chore|revert|hw)(\([a-z0-9-]+\))?!?: .{3,100}$"
)
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
WORKFLOW_RE = re.compile(r"^\.github/workflows/[^/]+\.ya?ml$")
RUN_ID_RE = re.compile(r"/actions/runs/(\d+)(?:/|$)")

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


def release_base(root: Path, sha: str, released: set[str]) -> tuple[str | None, list[str]]:
    """The newest ``vN...`` tag before ``sha`` with a published release, and the newer
    tags before ``sha`` that have none (their commits are checked again)."""
    proc = subprocess.run(  # noqa: S603 - fixed git arguments, no shell
        ["git", "-C", str(root), "tag", "--merged", f"{sha}^", "--list", "v[0-9]*"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    tags = proc.stdout.split() if proc.returncode == 0 else []
    depth = {tag: int(git(root, "rev-list", "--count", tag)) for tag in tags}
    done = [tag for tag in tags if tag in released]
    base = max(done, key=lambda tag: depth[tag]) if done else None
    floor = depth[base] if base else -1
    skipped = sorted(
        (t for t in tags if t not in released and depth[t] > floor), key=lambda tag: depth[tag]
    )
    return base, skipped


def released_tags(api: Api, repo: str) -> set[str]:
    """Tags with a published (not draft) GitHub release."""
    return {
        str(r.get("tag_name"))
        for r in api(f"repos/{repo}/releases?per_page=100")
        if not r.get("draft")
    }


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


def newest_statuses(statuses: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Context -> its newest status (the API lists the newest first; sorted to be sure)."""
    ordered = sorted(statuses, key=lambda s: (str(s.get("created_at", "")), s.get("id", 0)))
    return {str(s.get("context")): s for s in ordered}


def run_problems(
    api: Api, repo: str, gate: str, status: dict[str, Any], pr: dict[str, Any]
) -> list[str]:
    """Why the run behind ``status`` (its ``target_url``) is not the gate's own workflow."""
    match = RUN_ID_RE.search(str(status.get("target_url") or ""))
    url_ok = str(status.get("target_url") or "").startswith(f"https://github.com/{repo}/")
    if not match or not url_ok:
        return [f"{gate} länkar inte till en körning i {repo}"]
    (run,) = api(f"repos/{repo}/actions/runs/{match.group(1)}")
    where = f"{gate} kommer från körning {match.group(1)}"
    path, _, ref = str(run.get("path", "")).partition("@")
    out = []
    if path != f".github/workflows/{GATE_SOURCES[gate]}" or ref not in ("", f"refs/heads/{MAIN}"):
        out.append(f"{where} av {run.get('path')}, inte {GATE_SOURCES[gate]}")
    if str((run.get("repository") or {}).get("full_name")) != repo:
        out.append(f"{where} i ett annat repo")
    event = str(run.get("event"))
    if event not in GATE_TRIGGERS:
        out.append(f"{where} startad av {event}")
    elif event == "pull_request_target":
        if run.get("head_sha") != pr["head"]["sha"]:
            out.append(f"{where} på en annan commit än PR:ens head")
        # The run does not say which pull request (or base branch) it ran for, and
        # pull_request_target runs the base branch's copy of the gate. So no other pull
        # request may share the head commit with a base other than main.
        for other in api(f"repos/{repo}/commits/{pr['head']['sha']}/pulls"):
            base = str((other.get("base") or {}).get("ref", ""))
            same_head = (other.get("head") or {}).get("sha") == pr["head"]["sha"]
            if same_head and base != MAIN:
                out.append(f"{where}; head-commiten är också PR #{other.get('number')} mot {base}")
    elif run.get("head_branch") != MAIN:
        out.append(f"{where} på grenen {run.get('head_branch')}, inte {MAIN}")
    return out


def gate_problems(
    api: Api, repo: str, pr: dict[str, Any], statuses: Iterable[dict[str, Any]]
) -> list[str]:
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
        elif gate in GATE_SOURCES:
            out += run_problems(api, repo, gate, status, pr)
        elif not PR_TITLE_RE.match(str(pr.get("title", ""))):
            out.append(f"{gate} är grön men titeln följer inte regeln i pr-title.yml")
    return out


def workflow_problems(api: Api, repo: str, number: object, head: str) -> list[str]:
    """Workflows the pull request adds or changes that could write a gate status."""
    out = []
    for f in api(f"repos/{repo}/pulls/{number}/files?per_page=100"):
        path = str(f.get("filename", ""))
        if f.get("status") == "removed" or not WORKFLOW_RE.match(path):
            continue
        (blob,) = api(f"repos/{repo}/contents/{path}?ref={head}")
        text = base64.b64decode(str(blob.get("content", ""))).decode("utf-8", "replace")
        try:
            doc = yaml.safe_load(text)
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


def check_commit(api: Api, repo: str, sha: str) -> list[Finding]:
    pr = merged_pull(api, repo, sha)
    if pr is None:
        return [
            Finding(sha, f"kom inte från en pull request sammanslagen till {MAIN} (direkt push?)")
        ]
    head = str(pr["head"]["sha"])
    number = pr.get("number")
    statuses = api(f"repos/{repo}/commits/{head}/statuses?per_page=100")
    problems = gate_problems(api, repo, pr, statuses) + workflow_problems(api, repo, number, head)
    return [Finding(sha, f"PR #{number}: {p}") for p in problems]


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
    parser.add_argument(
        "--since", help="older end, excluded (default: the previous v* tag with a release)"
    )
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
        since, skipped = args.since, []
        if since is None:
            since, skipped = release_base(REPO, sha, released_tags(api, args.repo))
        for tag in skipped:
            print(f"{tag} har ingen publicerad release; dess commits granskas igen.")
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
