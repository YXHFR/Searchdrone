"""The justfile: every recipe exists, unbuilt recipes say so, composite recipes skip them."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
JUST = shutil.which("just")
pytestmark = pytest.mark.skipif(JUST is None, reason="just is not installed")

RECIPES = [
    "bootstrap",
    "setup",
    "setup-config",
    "fmt",
    "lint",
    "typecheck",
    "test",
    "test-panel-e2e",
    "db-up",
    "db-down",
    "db-check",
    "migrate",
    "gen",
    "gen-check",
    "test-contract",
    "check",
    "build",
    "ground",
    "dev",
    "sim",
    "stop",
    "sim-doctor",
    "sim-smoke",
    "test-sitl",
    "ci-sitl",
    "perf",
    "demo",
    "sim-headless",
    "e2e",
    "sim-setup",
    "models-fetch",
    "maps-fetch",
    "accept",
    "showcase",
    "release-prep",
    "hw-check",
    "check-traceability",
    "brain-token",
]
ALIASES = ["gen-schemas", "sim-up", "sim-down", "bench"]


def just(*args: str) -> subprocess.CompletedProcess[str]:
    assert JUST is not None
    return subprocess.run([JUST, *args], cwd=ROOT, capture_output=True, text=True, check=False)


def test_all_recipes_listed() -> None:
    summary = just("--summary").stdout.split()
    for name in RECIPES:
        assert name in summary, name
    listing = just("--list").stdout
    for alias in ALIASES:
        assert alias in listing, alias


@pytest.mark.parametrize(
    ("recipe", "ticket"),
    [
        ("db-up", "B0-04"),
        ("gen", "B0-03"),
        ("migrate", "B1-05"),
        ("hw-check", "B0-06"),
        ("models-fetch", "B1-10"),
        ("ci-sitl", "B0-08"),
    ],
)
def test_direct_unbuilt_recipe_exits_2(recipe: str, ticket: str) -> None:
    res = just(recipe)
    assert res.returncode == 2
    assert f"inte byggt än ({ticket})" in res.stdout


def test_composite_skips_unbuilt() -> None:
    res = just("--dry-run", "test-contract")
    # --dry-run prints the script; run the skipping helper for real instead.
    assert res.returncode == 0
    step = just("_step", "db-up")
    assert step.returncode == 0
    assert "hoppar över db-up (inte byggt än)" in step.stdout


def test_test_sitl_passes_key_values_unchanged() -> None:
    res = just("test-sitl", "physics=builtin", "sc=SC00", "dry=1")
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout.strip().splitlines()[-1]) == {
        "args": ["physics=builtin", "sc=SC00", "dry=1"]
    }


def test_dev_dry_run_shows_both_keys() -> None:
    res = just("--dry-run", "dev", "physics=builtin", "gui=false")
    assert res.returncode == 0
    assert "physics=builtin gui=false" in res.stderr


def test_unknown_key_rejected() -> None:
    res = just("e2e", "foo=1")
    assert res.returncode == 2
    assert "okänd nyckel 'foo'" in res.stderr


def test_no_dotenv_export() -> None:
    text = (ROOT / "justfile").read_text(encoding="utf-8")
    assert "dotenv-load" not in text
    assert "source .env" not in text
