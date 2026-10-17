"""B1-14: the AGPL detector package Ultralytics is never a dependency of the foundation.

Ultralytics may only ever come as a separate add-on; it must not be in any lock file.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LOCKFILES = [
    ROOT / "uv.lock",
    ROOT / "pnpm-lock.yaml",
    *sorted((ROOT / "simulation").rglob("*.lock")),
    *sorted((ROOT / "software").rglob("*.lock.toml")),
]
BANNED = re.compile(r"ultralytics|yolov8|yolo11", re.IGNORECASE)


def test_lockfiles_exist() -> None:
    assert (ROOT / "uv.lock").is_file()
    assert (ROOT / "pnpm-lock.yaml").is_file()


def test_no_ultralytics_in_lockfiles() -> None:
    hits = [
        f"{p.relative_to(ROOT)}:{n}"
        for p in LOCKFILES
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if BANNED.search(line)
    ]
    assert hits == [], hits
