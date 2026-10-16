"""tools/check_review_comments.py: the logic behind the commit status gate/review."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import check_review_comments as crc
import pytest

HEAD = "a" * 40
OLD = "b" * 40
HEAD_TIME = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
_next_id = iter(range(1, 10_000))


def comment(
    first_line: str,
    *,
    sha: str = HEAD,
    minute: int = 5,
    association: str = "OWNER",
    user_type: str = "User",
    edited: bool = False,
    extra: str = "",
) -> dict[str, Any]:
    created = f"2026-10-16T12:{minute:02d}:00Z"
    return {
        "id": next(_next_id),
        "author_association": association,
        "user": {"login": "owner", "type": user_type},
        "created_at": created,
        "updated_at": "2026-10-16T13:59:00Z" if edited else created,
        "body": f"{first_line}\n\nReviewed commit: {sha}\n{extra}",
    }


def approvals(**kwargs: Any) -> list[dict[str, Any]]:
    return [comment(f"review/{role}: approve", **kwargs) for role in crc.ROLES]


def state(comments: list[dict[str, Any]]) -> str:
    return crc.evaluate(comments, HEAD, HEAD_TIME).state


def test_three_approvals_pass() -> None:
    result = crc.evaluate(approvals(), HEAD, HEAD_TIME)
    assert result.state == "success"
    assert HEAD[:7] in result.description


def test_missing_role_is_pending() -> None:
    result = crc.evaluate(approvals()[:2], HEAD, HEAD_TIME)
    assert result.state == "pending"
    assert "flight-safety" in result.description


def test_no_comments_is_pending() -> None:
    assert state([]) == "pending"


def test_request_changes_fails() -> None:
    comments = [*approvals()[:2], comment("review/flight-safety: request_changes")]
    result = crc.evaluate(comments, HEAD, HEAD_TIME)
    assert result.state == "failure"
    assert "flight-safety" in result.description


def test_newest_verdict_per_role_wins() -> None:
    comments = [
        *approvals()[:2],
        comment("review/flight-safety: request_changes", minute=5),
        comment("review/flight-safety: approve", minute=9),
    ]
    assert state(comments) == "success"
    comments.append(comment("review/flight-safety: request_changes", minute=30))
    assert state(comments) == "failure"


def test_approval_for_an_older_commit_does_not_count() -> None:
    # A new push needs new reviews.
    assert state(approvals(sha=OLD)) == "pending"


def test_comment_older_than_the_commit_does_not_count() -> None:
    old = [comment(f"review/{r}: approve", minute=0) for r in crc.ROLES]
    for c in old:
        c["created_at"] = c["updated_at"] = "2026-10-16T11:59:00Z"
    assert state(old) == "pending"


@pytest.mark.parametrize("association", ["COLLABORATOR", "MEMBER", "CONTRIBUTOR", "NONE"])
def test_only_the_owner_counts(association: str) -> None:
    assert state(approvals(association=association)) == "pending"


def test_bot_accounts_do_not_count() -> None:
    assert state(approvals(user_type="Bot")) == "pending"


def test_edited_comment_fails() -> None:
    assert state(approvals(edited=True)) == "failure"


def test_edited_request_changes_does_not_bring_back_an_older_approve() -> None:
    comments = [
        *approvals(minute=5),
        comment("review/security: request_changes", minute=10, edited=True),
    ]
    assert state(comments) == "failure"


def test_newer_unedited_approve_replaces_an_edited_verdict() -> None:
    comments = [
        *approvals(minute=5),
        comment("review/security: request_changes", minute=10, edited=True),
        comment("review/security: approve", minute=20),
    ]
    assert state(comments) == "success"


@pytest.mark.parametrize(
    "first_line",
    [
        "Review/architecture: approve",
        "review/architecture: approved",
        "review/architecture: approve please",
        "review/architecture:approve",
        "> review/architecture: approve",
        "review/other: approve",
    ],
)
def test_first_line_must_match_exactly(first_line: str) -> None:
    comments = [comment(first_line), *approvals()[1:]]
    assert state(comments) == "pending"


def test_verdict_must_be_the_first_line() -> None:
    c = comment("Some intro")
    c["body"] = f"Some intro\nreview/architecture: approve\n{HEAD}"
    assert state([c, *approvals()[1:]]) == "pending"


def test_leading_blank_lines_are_ignored() -> None:
    comments = approvals()
    comments[0]["body"] = "\n\n" + comments[0]["body"]
    assert state(comments) == "success"


def test_sha_prefix_is_not_enough() -> None:
    comments = approvals()
    comments[0]["body"] = comments[0]["body"].replace(HEAD, HEAD[:12])
    assert state(comments) == "pending"


def test_flatten_paginated_pages() -> None:
    pages = [approvals()[:2], approvals()[2:]]
    assert state(crc.flatten(pages)) == "success"


def write_inputs(tmp_path: Path, comments: Any, sha: str = HEAD) -> tuple[Path, Path]:
    commit = tmp_path / "commit.json"
    commit.write_text(
        json.dumps({"sha": sha, "commit": {"committer": {"date": "2026-10-16T12:00:00Z"}}}),
        encoding="utf-8",
    )
    comments_path = tmp_path / "comments.json"
    comments_path.write_text(json.dumps(comments), encoding="utf-8")
    return commit, comments_path


def test_main_writes_github_output(tmp_path: Path) -> None:
    commit, comments = write_inputs(tmp_path, [approvals()])
    out = tmp_path / "out"
    rc = crc.main(
        [
            "--head-sha",
            HEAD,
            "--commit",
            str(commit),
            "--comments",
            str(comments),
            "--github-output",
            str(out),
        ]
    )
    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "state=success"
    assert lines[1].startswith("description=")
    assert len(lines[1]) <= len("description=") + crc.MAX_DESCRIPTION


def test_main_rejects_mismatched_commit(tmp_path: Path) -> None:
    commit, comments = write_inputs(tmp_path, [], sha=OLD)
    assert crc.main(["--head-sha", HEAD, "--commit", str(commit), "--comments", str(comments)]) == 2


def test_main_rejects_short_sha(tmp_path: Path) -> None:
    commit, comments = write_inputs(tmp_path, [])
    assert (
        crc.main(["--head-sha", "abc", "--commit", str(commit), "--comments", str(comments)]) == 2
    )
