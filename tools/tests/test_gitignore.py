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
    # AI exams: grades, the cost ledger, the example bank, reports and farm rounds.
    "data/brain-scores/2026-10-18/f1.json",
    "data/brain-eval/cost-ledger.json",
    "data/example-bank/good-f1.json.gz",
    "data/example-bank/fail-f1.json.gz",
    "data/reports/eval-sim-20261018T000000Z.json",
    "build/simfarm/rounds/r1/grade.json",
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
    "software/brain/tests/golden/sim/SIM-01.json",
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


def test_ai_exam_outputs_default_to_ignored_folders() -> None:
    """Where ``sd-brain eval --backend sim`` and ``bank failures`` write by default."""
    from sd_brain.cli import parser

    p = parser()
    ev = p.parse_args(["eval", "--backend", "sim"])
    fl = p.parse_args(["bank", "failures"])
    for folder in (ev.scores_dir, ev.bank_dir, ev.report_dir, fl.scores_dir):
        assert ignored(f"{folder}/x.json"), folder
    assert ignored(ev.ledger), ev.ledger
