"""tools/check_coverage.py: module mapping, missing files and the exact limit."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import check_coverage as cc
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An importable package ``covpkg`` with ``a.py`` and a sub-package ``sub/b.py``."""
    base = tmp_path / "covpkg"
    (base / "sub").mkdir(parents=True)
    for rel in ("__init__.py", "a.py", "sub/__init__.py", "sub/b.py"):
        (base / rel).write_text("x = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m.startswith("covpkg")]:
        del sys.modules[name]
    return base


def summary(branches: tuple[int, int], lines: tuple[int, int] = (1, 1)) -> dict[str, int]:
    return {
        "covered_branches": branches[0],
        "num_branches": branches[1],
        "covered_lines": lines[0],
        "num_statements": lines[1],
    }


def report(files: dict[Path, dict[str, int]]) -> dict[str, Any]:
    return {"files": {str(p): {"summary": s} for p, s in files.items()}}


def req(module: str, branch: float | None = None, line: float | None = None) -> cc.Requirement:
    return cc.Requirement(module, branch, line)


def test_module_that_is_a_file(pkg: Path) -> None:
    assert cc.module_files("covpkg.a") == [(pkg / "a.py").resolve()]
    rep = report({pkg / "a.py": summary((3, 4))})
    assert cc.problems([req("covpkg.a", branch=75)], rep) == []
    assert cc.problems([req("covpkg.a", branch=100)], rep)


def test_module_that_is_a_package_sums_every_file(pkg: Path) -> None:
    files = cc.module_files("covpkg")
    assert files is not None and len(files) == 4
    rep = report(
        {
            pkg / "__init__.py": summary((0, 0)),
            pkg / "a.py": summary((1, 2)),
            pkg / "sub/__init__.py": summary((0, 0)),
            pkg / "sub/b.py": summary((2, 2)),
        }
    )
    got = cc.measure(files, rep)
    assert (got.covered_branches, got.num_branches) == (3, 4)
    assert cc.problems([req("covpkg", branch=75)], rep) == []


def test_missing_file_counts_as_zero(pkg: Path) -> None:
    rep = report({pkg / "a.py": summary((2, 2))})
    found = cc.problems([req("covpkg.sub", branch=50)], rep)
    assert any("saknas i coverage.json" in f for f in found)
    assert any("grentäckning 0.0 %" in f for f in found)


def test_unknown_module_fails() -> None:
    assert cc.problems([req("no_such_module_xyz", line=1)], report({})) == [
        "no_such_module_xyz: modulen hittades inte (0 %)"
    ]


def test_exactly_on_the_limit_passes_and_just_below_fails(pkg: Path) -> None:
    on = report({pkg / "a.py": summary((1, 1), lines=(850, 1000))})
    assert cc.problems([req("covpkg.a", line=85)], on) == []
    below = report({pkg / "a.py": summary((1, 1), lines=(849, 1000))})  # 84.9 %
    found = cc.problems([req("covpkg.a", line=85)], below)
    assert found == ["covpkg.a: radtäckning 84.9 % < 85 % (849/1000)"]


def test_main_exit_codes(pkg: Path, tmp_path: Path) -> None:
    cov = tmp_path / "coverage.json"
    cov.write_text(json.dumps(report({pkg / "a.py": summary((1, 2))})), encoding="utf-8")
    limits = tmp_path / "thresholds.toml"
    limits.write_text('[[coverage]]\nmodule = "covpkg.a"\nbranch_min = 50\n', encoding="utf-8")
    assert cc.main([str(cov), str(limits)]) == 0
    limits.write_text('[[coverage]]\nmodule = "covpkg.a"\nbranch_min = 51\n', encoding="utf-8")
    assert cc.main([str(cov), str(limits)]) == 1


def test_entry_without_a_limit_is_an_error(tmp_path: Path) -> None:
    bad = tmp_path / "t.toml"
    bad.write_text('[[coverage]]\nmodule = "x"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="branch_min"):
        cc.load_requirements(bad)


def test_repository_thresholds() -> None:
    """The safety gate and the emergency path need every branch; nothing is lower."""
    found = {r.module: r for r in cc.load_requirements(ROOT / "tools/thresholds.toml")}
    for module in (
        "sd_safety",
        "sd_ground.services.commands",
        "sd_ground.services.clearance",
        "sd_ground.services.vehicle_actions",
        "sd_ground.services.latches",
        "sd_ground.mav.gcs",
        "sd_ground.mav.companion_watchdog",
        "sd_onboard.executor.override",
    ):
        assert found[module].branch_min == 100, module
    assert found["sd_ground"].line_min is not None and found["sd_ground"].line_min >= 85
