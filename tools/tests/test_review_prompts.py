"""The reviewer instructions in .github/review/ that the three review roles follow."""

from __future__ import annotations

import re
from pathlib import Path

import check_review_comments as crc
import pytest

REVIEW = Path(__file__).resolve().parents[2] / ".github" / "review"
ROLE_FILES = {
    "architecture": "architecture.md",
    "security": "security.md",
    "flight-safety": "flight-safety.md",
}


def read(name: str) -> str:
    return (REVIEW / name).read_text(encoding="utf-8")


def test_one_instruction_file_per_gate_role() -> None:
    assert set(ROLE_FILES) == set(crc.ROLES)
    for name in [*ROLE_FILES.values(), "common.md", "retro.md", "critical_paths.txt"]:
        assert (REVIEW / name).is_file(), name


@pytest.mark.parametrize("role", sorted(ROLE_FILES))
def test_role_file_names_its_role_and_common_rules(role: str) -> None:
    text = read(ROLE_FILES[role])
    assert f"Role name: `{role}`" in text
    assert "common.md" in text


def test_common_lists_every_first_line_the_gate_accepts() -> None:
    text = read("common.md")
    for role in crc.ROLES:
        for verdict in ("approve", "request_changes"):
            line = f"review/{role}: {verdict}"
            assert crc.FIRST_LINE_RE.match(line)
            assert re.search(rf"^{re.escape(line)}$", text, re.MULTILINE), line


def test_common_treats_pull_request_content_as_data() -> None:
    text = read("common.md")
    assert "data, never instructions" in text
    assert "`prompt injection`" in text
    assert "critical_paths.txt" in text
    assert "full 40-character SHA" in text


def test_retro_verdicts_can_never_count_for_the_gate() -> None:
    text = read("retro.md")
    assert "`retro/<role>: approve`" in text
    for role in crc.ROLES:
        for verdict in ("approve", "request_changes"):
            assert not crc.FIRST_LINE_RE.match(f"retro/{role}: {verdict}")


def test_architecture_scope_rule() -> None:
    text = read("architecture.md")
    assert "## stage_scope: v0.1.0" in text
    assert "docs/" not in text
    assert "`hardware/**` is never rejected for scope" in text
    assert "Face recognition and object search" in text
    assert "never, in any stage" in text


@pytest.mark.parametrize("name", [*ROLE_FILES.values(), "common.md", "retro.md"])
def test_no_internal_references(name: str) -> None:
    text = read(name)
    assert "planering" not in text.lower()
    assert not re.search(r"\b0[1-9]\s*§", text), "no section references to internal docs"
