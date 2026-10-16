"""tools/check_versions.py (ACC-S0-17 and the tool pins in two places)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import check_versions as cv
import pytest

ROOT = Path(__file__).resolve().parents[2]


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_repo(tmp_path: Path, py: str = "0.0.0", npm: str = "0.0.0") -> Path:
    write(tmp_path, "pyproject.toml", f'[project]\nname = "root"\nversion = "{py}"\n')
    write(tmp_path, "software/a/pyproject.toml", f'[project]\nname = "a"\nversion = "{py}"\n')
    write(tmp_path, "package.json", json.dumps({"name": "x", "version": npm}))
    write(tmp_path, "software/p/package.json", json.dumps({"name": "p", "version": npm}))
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    return tmp_path


def test_repository_is_consistent() -> None:
    assert cv.main(["--root", str(ROOT)]) == 0


def package_errors(root: Path, expect: str | None = None) -> list[str]:
    return cv.check_package_versions(cv.package_versions(root), expect)


def test_dev_version_everywhere_passes(tmp_path: Path) -> None:
    assert package_errors(make_repo(tmp_path)) == []


def test_rc_spellings_are_the_same_version(tmp_path: Path) -> None:
    root = make_repo(tmp_path, py="0.0.1rc1", npm="0.0.1-rc.1")
    assert package_errors(root, "0.0.1-rc.1") == []


def test_one_package_differs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = make_repo(tmp_path, py="0.1.0", npm="0.1.0")
    write(root, "software/a/pyproject.toml", '[project]\nname = "a"\nversion = "0.1.1"\n')
    assert cv.main(["--root", str(root)]) == 1
    assert "software/a/pyproject.toml=0.1.1" in capsys.readouterr().out


def test_expect_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cv.main(["--root", str(make_repo(tmp_path)), "--expect", "0.1.0"]) == 1
    assert "väntade 0.1.0" in capsys.readouterr().out


def test_invalid_version(tmp_path: Path) -> None:
    assert cv.main(["--root", str(make_repo(tmp_path, npm="not-a-version"))]) == 1


PRE_COMMIT = """repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: aaaa  # v0.16.9, same version as uv.lock
  - repo: https://github.com/crate-ci/typos
    rev: bbbb  # v1.50.3
  - repo: https://github.com/gitleaks/gitleaks
    rev: cccc  # v8.30.1
"""


def pins_repo(tmp_path: Path, ruff: str = "0.16.9", gitleaks: str = "8.30.1") -> Path:
    write(tmp_path, ".pre-commit-config.yaml", PRE_COMMIT)
    write(tmp_path, "uv.lock", f'[[package]]\nname = "ruff"\nversion = "{ruff}"\n')
    write(
        tmp_path,
        "simulation/versions.lock",
        'typos_version = "1.50.3"\npnpm_version = "10.34.6"\nnode_version = "24.21.0"\n',
    )
    write(
        tmp_path,
        "package.json",
        json.dumps({"packageManager": "pnpm@10.34.6+sha512.abc", "engines": {"node": ">=24 <25"}}),
    )
    write(tmp_path, ".nvmrc", "24\n")
    write(tmp_path, ".github/workflows/ci.yml", f'    env:\n      GITLEAKS_VERSION: "{gitleaks}"\n')
    return tmp_path


def test_tool_pins_agree(tmp_path: Path) -> None:
    assert cv.check_tool_pins(pins_repo(tmp_path)) == []


def test_ruff_mismatch(tmp_path: Path) -> None:
    errors = cv.check_tool_pins(pins_repo(tmp_path, ruff="0.16.10"))
    assert errors == ["ruff: uv.lock har 0.16.10 men .pre-commit-config.yaml har 0.16.9"]


def test_gitleaks_mismatch(tmp_path: Path) -> None:
    errors = cv.check_tool_pins(pins_repo(tmp_path, gitleaks="8.29.0"))
    assert len(errors) == 1 and errors[0].startswith("gitleaks:")


def test_node_major_mismatch(tmp_path: Path) -> None:
    root = pins_repo(tmp_path)
    write(root, ".nvmrc", "22\n")
    assert any(e.startswith("node:") for e in cv.check_tool_pins(root))


def test_pnpm_mismatch(tmp_path: Path) -> None:
    root = pins_repo(tmp_path)
    write(
        root,
        "package.json",
        json.dumps({"packageManager": "pnpm@10.0.0", "engines": {"node": ">=24 <25"}}),
    )
    assert any(e.startswith("pnpm:") for e in cv.check_tool_pins(root))


def test_missing_pin_is_an_error(tmp_path: Path) -> None:
    root = pins_repo(tmp_path)
    write(root, "simulation/versions.lock", 'pnpm_version = "10.34.6"\nnode_version = "24.21.0"\n')
    assert any(e.startswith("typos: hittar ingen version") for e in cv.check_tool_pins(root))
