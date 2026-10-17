"""Traceability from the local acceptance list to the tests, and back.

Usage: ``uv run tools/gen_traceability.py [--accept-ids PATH] [--out PATH]``

The acceptance list is a local TOML file that is never in the repository; its path comes
from ``--accept-ids`` or ``$SD_ACCEPT_IDS``. It has one table per requirement::

    [meta]
    scenarios = ["SC00", "SC01"]           # every one must appear in some entry

    [[id]]
    id = "X-01"
    typ = "A"                               # A = automatic: needs at least one test
    dod_ref = "9"
    sc = ["SC04"]
    tests = ["test_sc04_sc05_refusals_in_the_air"]

The script finds every test function in the repository (``def test_*`` /
``async def test_*`` under ``software/``, ``simulation/`` and ``tools/tests/``, read with
``ast``, nothing is imported or run) and checks that each named test exists, that each
automatic requirement has a test and that every listed scenario is covered. It writes
the map requirement -> ``path::test`` as YAML next to the list (``traceability.yaml``,
local as well). Without the list it says so and exits 0: the list lives only in the
planning folder. Exit code 1 lists every problem.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
TEST_ROOTS = ("software", "simulation", "tools/tests")
SKIP_DIRS = {"node_modules", ".venv", "dist", "build", "__pycache__", "generated"}
MISSING_TEXT = (
    "Spårbarhetsfilen finns bara i planeringsmappen (sätt SD_ACCEPT_IDS); inget att kontrollera."
)


def test_functions(
    root: Path | None = None, roots: Iterable[str] = TEST_ROOTS
) -> dict[str, list[str]]:
    """Test function name -> the files (repository-relative) that define it."""
    root = root or REPO
    found: dict[str, list[str]] = {}
    for top in roots:
        for path in sorted((root / top).rglob("test_*.py")):
            if SKIP_DIRS & set(path.relative_to(root).parts):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and (
                    node.name.startswith("test_")
                ):
                    found.setdefault(node.name, []).append(str(path.relative_to(root)))
    return found


def load(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def problems(data: dict[str, Any], tests: dict[str, list[str]]) -> list[str]:
    out = []
    covered: set[str] = set()
    seen: set[str] = set()
    for entry in data.get("id", []):
        rid = str(entry.get("id", "?"))
        if rid in seen:
            out.append(f"{rid}: finns två gånger")
        seen.add(rid)
        names = [str(t) for t in entry.get("tests", [])]
        covered.update(str(s) for s in entry.get("sc", []))
        if entry.get("typ") == "A" and not names:
            out.append(f"{rid}: automatiskt krav utan test")
        out += [f"{rid}: testet {n} finns inte" for n in names if n not in tests]
    for sc in data.get("meta", {}).get("scenarios", []):
        if sc not in covered:
            out.append(f"{sc}: inget krav nämner scenariot")
    return out


def trace(data: dict[str, Any], tests: dict[str, list[str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for entry in data.get("id", []):
        out[str(entry["id"])] = {
            "typ": entry.get("typ"),
            "sc": list(entry.get("sc", [])),
            "tests": [f"{f}::{n}" for n in entry.get("tests", []) for f in tests.get(n, [])],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--accept-ids", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    given = args.accept_ids or os.environ.get("SD_ACCEPT_IDS")
    path = Path(given) if given else None
    if path is None or not path.is_file():
        print(MISSING_TEXT)
        return 0
    data = load(path)
    tests = test_functions()
    found = problems(data, tests)
    out = args.out or path.with_name("traceability.yaml")
    out.write_text(
        yaml.safe_dump(trace(data, tests), sort_keys=True, allow_unicode=True), encoding="utf-8"
    )
    for line in found:
        print(line)
    if found:
        print(f"Spårbarheten har {len(found)} fel.")
        return 1
    print(f"Spårbarheten stämmer ({len(data.get('id', []))} krav). Skrev {out.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
