"""tools/check_owner_gate.py: gate/owner is green only for the owner's OK on this commit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import check_owner_gate as cog
import pytest

HEAD = "a" * 40
OLD = "b" * 40
HW = [{"filename": "hardware/drone.yaml"}]
SW = [{"filename": "software/panel/src/App.tsx"}]


def event(action: str, label: str = "", sender: str = "Owner", sender_type: str = "User",
          head: str = HEAD) -> cog.Event:  # fmt: skip
    return cog.Event(action, label, sender, sender_type, "Owner", head)


def pr(labels: tuple[str, ...] = (), head: str = HEAD) -> dict[str, Any]:
    return {"head": {"sha": head}, "labels": [{"name": n} for n in labels]}


def status(state: str, login: str = "github-actions[bot]", ctx: str = "gate/owner",
           at: str = "2026-10-16T10:00:00Z", sid: int = 1) -> dict[str, Any]:  # fmt: skip
    return {
        "context": ctx,
        "state": state,
        "creator": {"login": login},
        "created_at": at,
        "id": sid,
    }


def test_no_protected_path_is_success() -> None:
    res = cog.decide(event("opened"), pr(), SW, [])
    assert (res.state, res.remove_label) == ("success", False)


@pytest.mark.parametrize(
    "name",
    ["hardware/drone.yaml", "hardware/README.md", "hardware/a/b.step", ".github/CODEOWNERS"],
)
def test_protected_paths_need_approval(name: str) -> None:
    assert cog.decide(event("opened"), pr(), [{"filename": name}], []).state == "failure"


def test_moving_a_file_out_of_hardware_needs_approval() -> None:
    files = [{"filename": "elsewhere/drone.yaml", "previous_filename": "hardware/drone.yaml"}]
    assert cog.decide(event("opened"), pr(), files, []).state == "failure"


def test_similar_names_are_not_protected() -> None:
    files = [{"filename": "hardwarex/a"}, {"filename": "docs/hardware/a"},
             {"filename": ".github/CODEOWNERS.bak"}]  # fmt: skip
    assert cog.decide(event("opened"), pr(), files, []).state == "success"


def test_owner_label_on_current_head_is_success() -> None:
    res = cog.decide(event("labeled", "owner-ok"), pr(("owner-ok",)), HW, [])
    assert res.state == "success"
    assert HEAD[:7] in res.description


@pytest.mark.parametrize(
    ("sender", "sender_type"),
    [("someone", "User"), ("Owner", "Bot"), ("searchdrone-agent[bot]", "Bot")],
)
def test_label_from_anyone_else_is_removed(sender: str, sender_type: str) -> None:
    res = cog.decide(event("labeled", "owner-ok", sender, sender_type), pr(("owner-ok",)), HW, [])
    assert (res.state, res.remove_label) == ("failure", True)


def test_label_for_an_older_head_is_removed() -> None:
    res = cog.decide(event("labeled", "owner-ok", head=OLD), pr(("owner-ok",)), HW, [])
    assert (res.state, res.remove_label) == ("failure", True)


@pytest.mark.parametrize("action", ["synchronize", "reopened"])
def test_new_commits_remove_the_approval(action: str) -> None:
    approved = [status("success")]
    res = cog.decide(event(action), pr(("owner-ok",)), HW, approved)
    assert (res.state, res.remove_label) == ("failure", True)


def test_other_event_keeps_an_approval_on_this_commit() -> None:
    res = cog.decide(event("labeled", "docs"), pr(("owner-ok", "docs")), HW, [status("success")])
    assert res.state == "success"


def test_other_event_without_status_is_failure() -> None:
    # A label alone (e.g. set while the gate was not running) is not enough.
    assert cog.decide(event("ready_for_review"), pr(("owner-ok",)), HW, []).state == "failure"


def test_unlabeled_is_failure() -> None:
    res = cog.decide(event("unlabeled", "owner-ok"), pr(()), HW, [status("success")])
    assert res.state == "failure"


def test_status_from_someone_else_does_not_count() -> None:
    forged = [status("success", login="someone")]
    assert cog.decide(event("opened"), pr(("owner-ok",)), HW, forged).state == "failure"


def test_newest_status_wins() -> None:
    statuses = [
        status("success", at="2026-10-16T10:00:00Z", sid=1),
        status("failure", at="2026-10-16T11:00:00Z", sid=2),
    ]
    assert cog.decide(event("opened"), pr(("owner-ok",)), HW, statuses).state == "failure"
    statuses.append(status("success", at="2026-10-16T12:00:00Z", sid=3))
    assert cog.decide(event("opened"), pr(("owner-ok",)), HW, statuses).state == "success"


def test_other_contexts_are_ignored() -> None:
    other = [status("success", ctx="gate/review")]
    assert cog.decide(event("opened"), pr(("owner-ok",)), HW, other).state == "failure"


def test_cli_reads_paginated_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "pr.json").write_text(json.dumps(pr(("owner-ok",))), "utf-8")
    (tmp_path / "files.json").write_text(json.dumps([SW, HW]), "utf-8")
    (tmp_path / "statuses.json").write_text(json.dumps([[]]), "utf-8")
    out = tmp_path / "out"
    code = cog.main([
        "--action", "labeled", "--label", "owner-ok", "--sender", "Owner",
        "--sender-type", "User", "--owner", "Owner", "--event-head-sha", HEAD,
        "--pr", str(tmp_path / "pr.json"), "--files", str(tmp_path / "files.json"),
        "--statuses", str(tmp_path / "statuses.json"), "--github-output", str(out),
    ])  # fmt: skip
    assert code == 0
    text = out.read_text("utf-8")
    assert "state=success\n" in text
    assert "remove_label=false\n" in text
