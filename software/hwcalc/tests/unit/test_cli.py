"""``python -m sd_hwcalc check``: JSON on stdout, Swedish summary on stderr, exit codes."""

import json
from pathlib import Path

import pytest

from sd_hwcalc.__main__ import main

HARDWARE = Path(__file__).resolve().parents[4] / "hardware"


def run(capsys: pytest.CaptureFixture[str], drone: Path, limits: Path) -> tuple[int, dict, str]:
    code = main(["check", str(drone), "--limits", str(limits)])
    out, err = capsys.readouterr()
    return code, json.loads(out), err


def test_committed_files_pass(capsys: pytest.CaptureFixture[str]) -> None:
    code, report, err = run(capsys, HARDWARE / "drone.yaml", HARDWARE / "limits.yaml")
    assert code == 0
    assert report["pass"] is True
    assert report["failures"] == []
    assert report["total_mass_kg"] == pytest.approx(2.3)
    assert report["hover_time_min"] == pytest.approx(36.58, abs=0.05)
    assert set(report) >= {
        "total_mass_kg",
        "hover_power_w",
        "usable_energy_wh",
        "hover_time_min",
        "hover_thrust_fraction",
        "thrust_to_weight",
    }
    for word in ("Total massa", "Svävtid", "Svävgas", "Dragkraft/vikt"):
        assert word in err


def test_broken_limit_exits_1(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    heavy = tmp_path / "drone.yaml"
    text = (HARDWARE / "drone.yaml").read_text(encoding="utf-8")
    heavy.write_text(text.replace("frame_mass_kg: 0.55", "frame_mass_kg: 0.80"), "utf-8")
    code, report, err = run(capsys, heavy, HARDWARE / "limits.yaml")
    assert code == 1
    assert report["pass"] is False
    assert [f["limit"] for f in report["failures"]] == ["max_takeoff_mass_kg"]
    assert "FEL:" in err


def test_invalid_file_exits_1(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    bad = tmp_path / "drone.yaml"
    bad.write_text("schema_version: 1\n", "utf-8")
    code, report, err = run(capsys, bad, HARDWARE / "limits.yaml")
    assert code == 1
    assert report["pass"] is False
    assert "Kan inte läsa" in err


def test_missing_file_exits_1(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code, report, _ = run(capsys, tmp_path / "nope.yaml", HARDWARE / "limits.yaml")
    assert code == 1
    assert report["pass"] is False


def test_non_utf8_file_exits_1(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    bad = tmp_path / "drone.yaml"
    bad.write_bytes(b"name: \xff\xfe\n")
    code, report, _ = run(capsys, bad, HARDWARE / "limits.yaml")
    assert code == 1
    assert report["pass"] is False
