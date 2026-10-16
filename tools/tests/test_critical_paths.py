""".github/review/critical_paths.txt: every listed path exists, planned ones do not yet."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIST = ROOT / ".github" / "review" / "critical_paths.txt"
PLANNED_MARKER = "# --- planned:"
REQUIRED = (
    "software/common/safety/**",
    "software/common/mav/**",
    "software/common/link/**",
    "software/onboard/**",
    "software/panel/src/safety/**",
    "software/hwcalc/**",
    "simulation/sitl/**",
    ".importlinter",
    "justfile",
    ".github/**",
    "tools/**",
)


def glob_re(pattern: str) -> re.Pattern[str]:
    out = ""
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile(out + r"\Z")


def sections() -> tuple[list[str], list[str]]:
    active: list[str] = []
    planned: list[str] = []
    target = active
    for raw in LIST.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith(PLANNED_MARKER):
            target = planned
            continue
        if line and not line.startswith("#"):
            target.append(line)
    return active, planned


def tracked() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=True
    )
    return out.stdout.splitlines()


def test_glob_semantics() -> None:
    assert glob_re("tools/**").match("tools/a/b.py")
    assert glob_re("config/*.example.toml").match("config/ground.example.toml")
    assert not glob_re("config/*.example.toml").match("config/x/ground.example.toml")
    assert glob_re("**/ruff.toml").match("ruff.toml")
    assert glob_re("**/ruff.toml").match("software/x/ruff.toml")


def test_core_paths_are_listed() -> None:
    active, _ = sections()
    assert set(REQUIRED) <= set(active)


def test_no_duplicates() -> None:
    active, planned = sections()
    both = active + planned
    assert len(both) == len(set(both))


def test_every_active_path_matches_a_tracked_file() -> None:
    files = tracked()
    active, _ = sections()
    dead = [p for p in active if not any(glob_re(p).match(f) for f in files)]
    assert not dead, f"no tracked file matches (moved or renamed?): {dead}"


def test_planned_paths_are_moved_up_when_their_file_lands() -> None:
    files = tracked()
    _, planned = sections()
    landed = [p for p in planned if any(glob_re(p).match(f) for f in files)]
    assert not landed, f"move these above the planned marker: {landed}"
