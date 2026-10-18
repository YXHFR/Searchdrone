"""tools/brain_token.py: the commander token (length, file mode, rotation)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import brain_token


def test_brain_token(tmp_path: Path) -> None:
    path = tmp_path / "secrets" / "brain_token"
    brain_token.write_token(path)
    first = path.read_text().strip()
    assert len(first) >= 43
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    brain_token.main([str(path)])
    assert path.read_text().strip() != first
