"""Decide the commit status ``gate/owner``: the owner's OK for hardware, CODEOWNERS and the
release's gate check.

Pull requests are opened with the owner's own account, so GitHub's Approve button cannot
be used. Instead the owner says OK and the label ``owner-ok`` is set on the pull request
(by the owner's account, a ``User``). The approval is bound to the head commit:

1. No protected path changed (``PROTECTED``) -> ``success``, nothing to approve.
2. New commits (``synchronize``), ``reopened`` or a new base branch (``edited`` with a
   base change, which changes the diff) -> the label is removed -> ``failure``.
3. ``labeled`` with ``owner-ok``: ``success`` only if the owner set it and the PR's head,
   fetched fresh, is still the commit the event names. Otherwise the label is removed and
   the result is ``failure``.
4. Any other event -> ``success`` only if the label is on the PR now **and** the newest
   ``gate/owner`` status that GitHub Actions wrote on the current head is ``success``.

So an approval never carries over to new commits or a new base. Only the owner's account
can set a trusted label; a status can also be written by a workflow added in a pull
request, which the reviewers must catch (the same limit as gate/review). A pull request
whose file list GitHub truncates fails, since its protected paths cannot be verified.
Moving or deleting a file under ``hardware/`` counts as a change.
If a run is skipped in the queue and an old ``owner-ok`` stays on the pull request after
new commits, remove the label and set it again to approve the new head (adding a label
that is already there sends no event). This check says nothing about the hardware
simulations: ``hardware-ok`` must pass too.

Standard library only: owner-gate.yml runs it from the base branch with the system
Python and never runs code from the pull request.
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

LABEL = "owner-ok"
CONTEXT = "gate/owner"
ACTIONS_BOT = "github-actions[bot]"
MAX_FILES = 3000  # GitHub lists at most this many files of a pull request
PROTECTED = (
    re.compile(r"^hardware/", re.IGNORECASE),
    re.compile(r"^\.github/CODEOWNERS$"),
    # The gates and what a release accepts (check_merged_pr_gates.py), so that loosening
    # any of their rules, this list included, needs the owner too.
    re.compile(r"^tools/accept/pr_gate_exceptions\.txt$"),
    re.compile(
        r"^tools/(check_merged_pr_gates|workflow_rules|check_owner_gate|check_review_comments)\.py$"
    ),
    re.compile(r"^\.github/workflows/(review-gate|owner-gate|pr-title|release)\.yml$"),
)
# Issue events after which an earlier owner-ok no longer covers the pull request.
RESETS = ("base_ref_changed", "head_ref_force_pushed", "reopened")


@dataclass(frozen=True)
class Event:
    action: str
    label: str
    sender: str
    sender_type: str
    owner: str
    event_head_sha: str
    base_changed: bool = False


@dataclass(frozen=True)
class Result:
    state: str  # success | failure
    description: str
    remove_label: bool = False


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def flatten(pages: Any) -> list[dict[str, Any]]:
    """``gh api --paginate --slurp`` gives a list of pages; accept a flat list too."""
    flat: list[dict[str, Any]] = []
    for item in pages:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def protected_files(files: list[dict[str, Any]]) -> list[str]:
    hits = []
    for f in files:
        for name in (f.get("filename"), f.get("previous_filename")):
            if isinstance(name, str) and any(p.search(name) for p in PROTECTED):
                hits.append(name)
    return sorted(set(hits))


def approved_by_label_events(
    events: list[dict[str, Any]], owner: str, since: datetime, until: datetime
) -> bool:
    """The rule above, worked out afterwards from a pull request's issue events (used by
    ``check_merged_pr_gates.py``): the newest ``owner-ok`` event up to ``until`` is the
    owner's account adding it after ``since`` (when the head commit reached GitHub) and
    after every base change, force push or reopening. Issue events do not name the head
    commit, so ``since`` stands in for the SHA the live gate compares."""

    def at(e: dict[str, Any]) -> datetime:
        return parse_time(str(e["created_at"]))

    seen = [e for e in events if e.get("created_at") and at(e) <= until]
    ours = [
        e
        for e in seen
        if e.get("event") in ("labeled", "unlabeled")
        and (e.get("label") or {}).get("name") == LABEL
    ]
    if not ours:
        return False
    last = max(ours, key=lambda e: (at(e), int(e.get("id") or 0)))
    actor = last.get("actor") or {}
    resets = [at(e) for e in seen if e.get("event") in RESETS]
    return (
        last.get("event") == "labeled"
        and actor.get("login") == owner
        and actor.get("type") == "User"
        and at(last) > since
        and all(at(last) > r for r in resets)
    )


def approved_status_on(statuses: list[dict[str, Any]]) -> bool:
    """Is the newest gate/owner status from GitHub Actions on this commit a success?"""
    ours = [
        s
        for s in statuses
        if s.get("context") == CONTEXT and (s.get("creator") or {}).get("login") == ACTIONS_BOT
    ]
    if not ours:
        return False
    newest = max(ours, key=lambda s: (str(s.get("created_at", "")), int(s.get("id", 0))))
    return newest.get("state") == "success"


def decide(
    event: Event,
    pr: dict[str, Any],
    files: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
) -> Result:
    changed = pr.get("changed_files")
    if not isinstance(changed, int) or changed != len(files) or len(files) >= MAX_FILES:
        return Result("failure", "Too many files to verify; split the pull request")
    if not protected_files(files):
        return Result("success", "No owner approval needed")

    head = str((pr.get("head") or {}).get("sha", ""))
    has_label = any(lb.get("name") == LABEL for lb in pr.get("labels") or [])

    if event.action in ("synchronize", "reopened"):
        return Result("failure", "Owner approval required (new commits)", remove_label=has_label)
    if event.action == "edited" and event.base_changed:
        return Result("failure", "Owner approval required (new base)", remove_label=has_label)

    if event.action == "labeled" and event.label == LABEL:
        by_owner = event.sender == event.owner and event.sender_type == "User"
        if by_owner and head and head == event.event_head_sha:
            return Result("success", f"Owner approved {head[:7]}")
        why = "owner-ok must be set by the owner" if not by_owner else "head moved; label again"
        return Result("failure", why, remove_label=True)

    if has_label and approved_status_on(statuses):
        return Result("success", f"Owner approved {head[:7]}")
    return Result("failure", f"Owner approval required: label {LABEL}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--action", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--sender", required=True)
    parser.add_argument("--sender-type", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--event-head-sha", required=True)
    parser.add_argument("--base-changed", action="store_true", help="edited: base changed")
    parser.add_argument("--pr", type=Path, required=True, help="GET pulls/<n> as JSON")
    parser.add_argument("--files", type=Path, required=True, help="GET pulls/<n>/files pages")
    parser.add_argument("--statuses", type=Path, required=True, help="GET commits/<sha>/statuses")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args(argv)

    event = Event(
        args.action,
        args.label,
        args.sender,
        args.sender_type,
        args.owner,
        args.event_head_sha,
        args.base_changed,
    )
    pr = json.loads(args.pr.read_text(encoding="utf-8"))
    files = flatten(json.loads(args.files.read_text(encoding="utf-8")))
    statuses = flatten(json.loads(args.statuses.read_text(encoding="utf-8")))
    result = decide(event, pr, files, statuses)

    lines = [
        f"state={result.state}",
        f"description={result.description}",
        f"remove_label={'true' if result.remove_label else 'false'}",
    ]
    print("\n".join(lines))
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
