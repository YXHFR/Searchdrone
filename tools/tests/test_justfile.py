"""The justfile: every recipe exists, unbuilt recipes say so, composite recipes skip them."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

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
    "maps-fetch",
    "bench-fetch",
    "bench-detector",
    "world-fetch",
    "world-snapshot",
    "sim-farm",
    "sim-farm-status",
    "sim-farm-setup",
    "accept",
    "showcase",
    "release-prep",
    "hw-check",
    "cad",
    "cad-test",
    "cad-detailed",
    "sim-model-fpv7",
    "check-traceability",
    "brain-token",
    "brain-eval",
    "brain-eval-cloud",
    "brain-eval-sim",
    "brain-scores",
    "rl-train",
    "rl-eval",
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
        (("release-prep", "0.1.0"), "B1-18"),
    ],
)
def test_direct_unbuilt_recipe_exits_2(recipe: tuple[str, ...], ticket: str) -> None:
    res = just(*recipe)
    assert res.returncode == 2
    assert f"inte byggt än ({ticket})" in res.stdout


def test_composite_skips_unbuilt() -> None:
    # Every composite step is built now; the skip itself still has to work.
    step = just("--set", "unbuilt", "ci-sitl", "_step", "ci-sitl")
    assert step.returncode == 0
    assert "hoppar över ci-sitl (inte byggt än)" in step.stdout


def _dump() -> dict[str, Any]:
    return json.loads(just("--dump", "--dump-format", "json").stdout)


def test_unbuilt_list_matches_not_built_recipes() -> None:
    dump = _dump()
    not_built = {
        name
        for name, recipe in dump["recipes"].items()
        if any(dep["recipe"] == "_not-built" for dep in recipe["dependencies"])
    }
    unbuilt = set(dump["assignments"]["unbuilt"]["value"].split())
    # Every name the composite recipes skip must really be unbuilt (a built gate must
    # never be skipped silently); the rest are recipes no composite recipe runs.
    assert unbuilt <= not_built, unbuilt - not_built


def test_unbuilt_tools_do_not_exist_yet() -> None:
    names = _dump()["assignments"]["unbuilt_tools"]["value"].split()
    for name in names:
        assert not (ROOT / "tools" / f"{name}.py").exists(), (
            f"tools/{name}.py exists: remove it from unbuilt_tools in the justfile"
        )


def test_test_sitl_passes_key_values_unchanged() -> None:
    res = just("test-sitl", "physics=builtin", "sc=SC00", "dry=1")
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout.strip().splitlines()[-1]) == {
        "args": ["physics=builtin", "sc=SC00", "dry=1"]
    }


def test_dev_passes_both_keys() -> None:
    res = just("dev", "physics=builtin", "gui=false", "dry=1")
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["args"] == ["physics=builtin", "gui=false", "dry=1"]


def test_arguments_are_not_shell_evaluated() -> None:
    res = just("test-sitl", "sc=$(echo hacked)", "dry=1")
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout.strip().splitlines()[-1]) == {
        "args": ["sc=$(echo hacked)", "dry=1"]
    }


def test_unknown_key_rejected() -> None:
    res = just("e2e", "foo=1")
    assert res.returncode == 2
    assert "okänd nyckel 'foo'" in res.stderr


def test_no_dotenv_export() -> None:
    text = (ROOT / "justfile").read_text(encoding="utf-8")
    assert "dotenv-load" not in text
    assert "source .env" not in text
