"""Check that every test node in a traceability file exists (local only).

    uv run tools/check_traceability.py [FILE]

FILE defaults to ``traceability.yaml`` next to the local id file (see
``gen_traceability.py``). The nodes are compared with ``pytest --collect-only -q`` over
the files the traceability file names. A missing file is not an error: it lives only in
the planning folder, so the check says so and exits 0. Generic: no ids in here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml
from gen_traceability import REPO, ids_path

COLLECT = [
    sys.executable,
    "-m",
    "pytest",
    "-p",
    "sd_sim.pytest_plugin",
    "--collect-only",
    "-q",
]


def nodes(paths: list[str]) -> set[str]:
    res = subprocess.run(  # noqa: S603 - fixed argv
        [*COLLECT, *paths],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return {line.strip() for line in res.stdout.splitlines() if "::" in line}


def missing(listed: list[str], collected: set[str]) -> list[str]:
    """Listed nodes pytest does not know. A node without parameters stands for all of
    its parametrised cases, a file for all of its tests."""
    out = []
    for node in listed:
        if node in collected:
            continue
        if "[" not in node and any(c.startswith(node + "[") for c in collected):
            continue
        if "::" not in node and any(c.startswith(node + "::") for c in collected):
            continue
        out.append(node)
    return out


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else ids_path(None).with_name("traceability.yaml")
    if not path.is_file():
        print("Spårbarhetsfilen finns bara i planeringsmappen; inget att kontrollera.")
        return 0
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    listed = sorted({t for e in data.get("entries", []) for t in e.get("tests", [])})
    files = sorted({t.split("::")[0] for t in listed})
    gone = [f for f in files if not (REPO / f).is_file()]
    found = missing(listed, nodes([f for f in files if f not in gone]))
    for f in gone:
        print(f"filen saknas: {f}")
    for node in found:
        print(f"testet finns inte: {node}")
    print(f"{len(listed)} testnoder kontrollerade, {len(found) + len(gone)} fel.")
    return 1 if found or gone else 0


if __name__ == "__main__":
    sys.exit(main())
