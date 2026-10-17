"""The parameter files that preflight check PF6 compares with the flight controller."""

from __future__ import annotations

from pathlib import Path

import pytest

from sd_onboard.io.params import effective_params, parse_parm


def test_parse_names_values_and_comments() -> None:
    text = "# header\nFENCE_ENABLE 1   # on\n\nFS_GCS_TIMEOUT,20\nWP_SPD 5.5\n"
    assert parse_parm(text) == {"FENCE_ENABLE": 1.0, "FS_GCS_TIMEOUT": 20.0, "WP_SPD": 5.5}


@pytest.mark.parametrize("line", ["FENCE_ENABLE", "FENCE_ENABLE one", "A 1 2"])
def test_malformed_line(line: str) -> None:
    with pytest.raises(ValueError):
        parse_parm(line)


def test_later_file_wins(tmp_path: Path) -> None:
    a, b = tmp_path / "a.parm", tmp_path / "b.parm"
    a.write_text("X 1\nY 2\n")
    b.write_text("Y 3\n")
    assert effective_params([a, b]) == {"X": 1.0, "Y": 3.0}
