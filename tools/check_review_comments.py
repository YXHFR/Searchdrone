"""Decide the commit status ``gate/review`` from the reviewers' PR comments.

Three reviewer roles (architecture, security, flight safety) review every pull request
before it is pushed. Each verdict is posted as a PR comment whose first line is exactly

    review/architecture: approve        (or request_changes)
    review/security: approve
    review/flight-safety: approve

A comment counts only if

* it was written by the repository owner (``author_association == "OWNER"``, a ``User``),
* it names the PR's current head commit (the full 40-character SHA) somewhere in the body,
* it was created after the head commit's committer date, and
* it was never edited.

The newest trusted comment per role that names the head commit is that role's verdict,
whatever it says: ``request_changes`` or an edited comment gives ``failure`` (post a new
verdict instead of editing), so an edit can never bring back an older approval. All
three roles with an unedited ``approve`` newer than the head commit give ``success``;
otherwise ``pending``. Trust assumption: only the owner posts verdicts, and a deleted
comment is gone from the API, so deleting a verdict is an owner decision.

Standard library only: the workflow runs it from the base branch with the system Python
and never runs code from the pull request.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROLES = ("architecture", "security", "flight-safety")
FIRST_LINE_RE = re.compile(
    r"^review/(architecture|security|flight-safety): (approve|request_changes)$"
)
SHA_RE = re.compile(r"\b[0-9a-f]{40}\b")
TRUSTED_ASSOCIATION = "OWNER"
MAX_DESCRIPTION = 140


@dataclass(frozen=True)
class Verdict:
    role: str
    verdict: str
    created_at: datetime
    comment_id: int
    edited: bool = False


@dataclass(frozen=True)
class Result:
    state: str  # success | failure | pending
    description: str


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def first_line(body: str) -> str:
    for line in body.splitlines():
        if line.strip():
            return line.strip()
    return ""


def flatten(comments: Any) -> list[dict[str, Any]]:
    """``gh api --paginate --slurp`` gives a list of pages; accept a flat list too."""
    flat: list[dict[str, Any]] = []
    for item in comments:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def candidate(comment: dict[str, Any], head_sha: str) -> Verdict | None:
    """A trusted verdict comment for ``head_sha``, edited or not (``None`` otherwise)."""
    if comment.get("author_association") != TRUSTED_ASSOCIATION:
        return None
    if (comment.get("user") or {}).get("type") != "User":
        return None
    created, updated = comment.get("created_at"), comment.get("updated_at")
    if not created:
        return None
    body = str(comment.get("body") or "")
    if head_sha not in SHA_RE.findall(body):
        return None
    m = FIRST_LINE_RE.match(first_line(body))
    if not m:
        return None
    edited = bool(updated and updated != created)
    return Verdict(m[1], m[2], parse_time(created), int(comment.get("id") or 0), edited)


def evaluate(comments: list[dict[str, Any]], head_sha: str, head_time: datetime) -> Result:
    # The newest trusted verdict per role decides, whatever it says. It counts as an
    # approval only if it is an unedited approve written after the head commit, so an
    # edit can never turn a blocking verdict into an older approval.
    latest: dict[str, Verdict] = {}
    for c in comments:
        v = candidate(c, head_sha)
        if v is None:
            continue
        old = latest.get(v.role)
        if old is None or (v.created_at, v.comment_id) > (old.created_at, old.comment_id):
            latest[v.role] = v
    short = head_sha[:7]
    blocked = [r for r in ROLES if r in latest and latest[r].verdict == "request_changes"]
    edited = [r for r in ROLES if r in latest and r not in blocked and latest[r].edited]
    if blocked or edited:
        parts = []
        if blocked:
            parts.append(f"changes requested by {', '.join(blocked)}")
        if edited:
            parts.append(f"edited verdict from {', '.join(edited)} (post a new one)")
        return Result("failure", f"{'; '.join(parts)} on {short}".capitalize())
    missing = [r for r in ROLES if r not in latest or latest[r].created_at <= head_time]
    if missing:
        return Result("pending", f"Waiting for review of {short}: {', '.join(missing)}")
    return Result("success", f"architecture, security and flight-safety approved {short}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate reviewer comments for gate/review.")
    ap.add_argument("--head-sha", required=True)
    ap.add_argument("--commit", type=Path, required=True, help="GET /commits/<sha> as JSON")
    ap.add_argument("--comments", type=Path, required=True, help="issue comments as JSON")
    ap.add_argument("--github-output", type=Path, help="append state= and description= here")
    args = ap.parse_args(argv)

    head_sha = args.head_sha.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", head_sha):
        print("check_review_comments: --head-sha must be a full commit SHA", file=sys.stderr)
        return 2
    commit = json.loads(args.commit.read_text(encoding="utf-8"))
    if commit.get("sha") != head_sha:
        print("check_review_comments: commit JSON does not match --head-sha", file=sys.stderr)
        return 2
    head_time = parse_time(commit["commit"]["committer"]["date"])
    comments = flatten(json.loads(args.comments.read_text(encoding="utf-8")))

    result = evaluate(comments, head_sha, head_time)
    description = result.description[:MAX_DESCRIPTION]
    print(f"gate/review: {result.state}: {description}")
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as fh:
            fh.write(f"state={result.state}\ndescription={description}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
