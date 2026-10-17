"""The import contracts in .importlinter really break when a boundary is crossed.

import-linter splits list options only on newlines, so a comma-separated list would
silently become one module name that never matches.
"""

from __future__ import annotations

import configparser
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LINT_IMPORTS = shutil.which("lint-imports")


def _config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / ".importlinter", encoding="utf-8")
    return cfg


def test_lists_are_one_module_per_line() -> None:
    cfg = _config()
    for section in cfg.sections():
        for key in ("root_packages", "source_modules", "forbidden_modules"):
            if key in cfg[section]:
                assert "," not in cfg[section][key], f"{section}.{key} must be one per line"


def _tree(tmp_path: Path, imports: dict[str, str]) -> Path:
    cfg = _config()
    roots = cfg["importlinter"]["root_packages"].split()
    for pkg in roots:
        d = tmp_path / pkg
        d.mkdir()
        (d / "__init__.py").write_text(imports.get(pkg, ""), encoding="utf-8")
    # Contracts may name submodules (e.g. sd_ground.ingress, sd_ground.services.latches);
    # give each an empty package, so a name and its children can both be listed.
    for section in cfg.sections():
        for key in ("source_modules", "forbidden_modules"):
            for name in cfg[section].get(key, "").split():
                parts = name.split(".")
                if len(parts) > 1 and parts[0] in roots:
                    for depth in range(2, len(parts) + 1):
                        package = tmp_path.joinpath(*parts[:depth])
                        package.mkdir(parents=True, exist_ok=True)
                        (package / "__init__.py").touch()
    shutil.copyfile(ROOT / ".importlinter", tmp_path / ".importlinter")
    return tmp_path


def _lint(root: Path) -> subprocess.CompletedProcess[str]:
    assert LINT_IMPORTS is not None
    env = {**os.environ, "PYTHONPATH": str(root)}
    return subprocess.run(
        [LINT_IMPORTS, "--no-cache"], cwd=root, env=env, capture_output=True, text=True, check=False
    )


pytestmark = pytest.mark.skipif(LINT_IMPORTS is None, reason="import-linter not installed")


def test_clean_tree_is_kept(tmp_path: Path) -> None:
    res = _lint(_tree(tmp_path, {}))
    assert res.returncode == 0, res.stdout + res.stderr


@pytest.mark.parametrize(
    ("pkg", "code", "contract"),
    [
        ("sd_brain", "import sd_mav\n", "the AI cannot talk to the flight controller"),
        ("sd_brain", "import subprocess\n", "the AI has no database, link or subprocess access"),
        ("sd_onboard", "import sd_brain\n", "sd_onboard does not reach ground or AI"),
        ("sd_ground", "import sd_brain\n", "sd_ground does not reach onboard or AI"),
        ("sd_geo", "import sd_mav\n", "sd_geo depends only on sd_schemas"),
        ("sd_hwcalc", "import sd_geo\n", "sd_hwcalc is standalone"),
        (
            "sd_ground",
            "import sd_link_testing\n",
            "test-only packages are never imported by production packages",
        ),
        ("sd_safety", "import sd_obs\n", "sd_safety is pure logic without I/O"),
        (
            "sd_brain",
            "import socket\n",
            "the AI opens no raw sockets",
        ),
    ],
)
def test_crossing_a_boundary_breaks(tmp_path: Path, pkg: str, code: str, contract: str) -> None:
    res = _lint(_tree(tmp_path, {pkg: code}))
    assert res.returncode != 0, res.stdout
    assert f"{contract} BROKEN" in res.stdout
