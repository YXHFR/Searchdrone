"""tools/search_token.py (B2-01): the search service's token file."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
import search_token


def test_search_token(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "secrets" / "search_token"
    search_token.main([str(path)])
    first = path.read_text().strip()
    assert len(first) >= 43
    assert first not in capsys.readouterr().out  # the value is never printed
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    search_token.main([str(path)])
    assert path.read_text().strip() != first
