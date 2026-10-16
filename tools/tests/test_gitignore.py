"""The .gitignore keeps planning, agent files, secrets and personal config out of git.

Each path is checked with its own `git check-ignore` call: with several paths the
command succeeds as soon as one of them matches.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

IGNORED = [
    "docs/x.md",
    "data/x",
    "runs/x",
    "config/ground.toml",
    "config/sim.toml",
    "config/local/sim.toml",
    "config/local/mediamtx.yml",
    "config/local/private-patterns.txt",
    "software/x/CLAUDE.md",
    "software/x/.claude/settings.json",
    "CLAUDE.md",
    ".claude/settings.json",
    ".mcp.json",
    ".env",
    ".env.local",
    "secrets/x",
    "secrets/brain_token",
    "x.tlog",
    "flight.BIN",
    "video.mp4",
]

NOT_IGNORED = [
    "software/brain/src/sd_brain/planning/x.py",
    "software/panel/src/data/x.ts",
    "software/ground/src/sd_ground/missions/x.py",
    "config/mediamtx.yml",
    "config/ground.example.toml",
    "software/common/schemas/generated/x.json",
    ".env.example",
]


def ignored(path: str) -> bool:
    res = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT, check=False)
    assert res.returncode in (0, 1), res
    return res.returncode == 0


@pytest.mark.parametrize("path", IGNORED)
def test_ignored(path: str) -> None:
    assert ignored(path), path


@pytest.mark.parametrize("path", NOT_IGNORED)
def test_not_ignored(path: str) -> None:
    assert not ignored(path), path


def test_no_internal_files_tracked() -> None:
    files = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    for f in files:
        parts = Path(f).parts
        assert parts[0] != "docs", f
        assert "CLAUDE.md" not in parts, f
        assert ".claude" not in parts, f
