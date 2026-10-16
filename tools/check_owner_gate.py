"""Decide the commit status ``gate/owner``: the owner's OK for hardware and CODEOWNERS.

Pull requests are opened with the owner's own account, so GitHub's Approve button cannot
be used. Instead the owner says OK and the label ``owner-ok`` is set on the pull request
(by the owner's account, a ``User``). The approval is bound to the head commit:

1. No protected path changed (``PROTECTED``) -> ``success``, nothing to approve.
2. New commits (``synchronize``) or ``reopened`` -> the label is removed -> ``failure``.
3. ``labeled`` with ``owner-ok``: ``success`` only if the owner set it and the PR's head,
   fetched fresh, is still the commit the event names. Otherwise the label is removed and
   the result is ``failure``.
4. Any other event -> ``success`` only if the label is on the PR now **and** the newest
   ``gate/owner`` status that GitHub Actions wrote on the current head is ``success``.

So an approval never carries over to new commits, and nobody but the owner's account can
turn the gate green. Moving or deleting a file under ``hardware/`` counts as a change.
This check says nothing about the hardware simulations: ``hardware-ok`` must pass too.

Standard library only: owner-gate.yml runs it from the base branch with the system
Python and never runs code from the pull request.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LABEL = "owner-ok"
CONTEXT = "gate/owner"
ACTIONS_BOT = "github-actions[bot]"
PROTECTED = (re.compile(r"^hardware/"), re.compile(r"^\.github/CODEOWNERS$"))


@dataclass(frozen=True)
class Event:
    action: str
    label: str
    sender: str
    sender_type: str
    owner: str
    event_head_sha: str


@dataclass(frozen=True)
class Result:
    state: str  # success | failure
    description: str
    remove_label: bool = False


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
    if not protected_files(files):
        return Result("success", "No owner approval needed")

    head = str((pr.get("head") or {}).get("sha", ""))
    has_label = any(lb.get("name") == LABEL for lb in pr.get("labels") or [])

    if event.action in ("synchronize", "reopened"):
        return Result("failure", "Owner approval required (new commits)", remove_label=has_label)

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
    parser.add_argument("--pr", type=Path, required=True, help="GET pulls/<n> as JSON")
    parser.add_argument("--files", type=Path, required=True, help="GET pulls/<n>/files pages")
    parser.add_argument("--statuses", type=Path, required=True, help="GET commits/<sha>/statuses")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args(argv)

    event = Event(
        args.action, args.label, args.sender, args.sender_type, args.owner, args.event_head_sha
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
