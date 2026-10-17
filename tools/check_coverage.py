"""Check the coverage requirements in ``tools/thresholds.toml`` against ``coverage.json``.

Usage: ``uv run tools/check_coverage.py coverage.json tools/thresholds.toml``

Each ``[[coverage]]`` entry names a module and a ``branch_min`` and/or ``line_min`` in
percent. The module is found with ``importlib.util.find_spec``: a package (its origin is
an ``__init__.py``) covers every ``.py`` file under its directory, a module only its own
file. Covered and total branches (lines) are summed over those files and compared with
the requirement. A file that is not in ``coverage.json``, or a module that cannot be
found, counts as 0 %. Exit code 1 lists every requirement that is not met; the other
packages are only reported by pytest-cov.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Requirement:
    module: str
    branch_min: float | None
    line_min: float | None


@dataclass(frozen=True)
class Measured:
    covered_branches: int = 0
    num_branches: int = 0
    covered_lines: int = 0
    num_statements: int = 0
    missing_files: tuple[str, ...] = ()

    @property
    def branch_pct(self) -> float:
        return _pct(self.covered_branches, self.num_branches)

    @property
    def line_pct(self) -> float:
        return _pct(self.covered_lines, self.num_statements)


def _pct(covered: int, total: int) -> float:
    return 100.0 if total == 0 else 100.0 * covered / total


def load_requirements(path: Path) -> list[Requirement]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    out = []
    for entry in data.get("coverage", []):
        branch, line = entry.get("branch_min"), entry.get("line_min")
        if branch is None and line is None:
            raise ValueError(f"{entry['module']}: branch_min eller line_min saknas")
        out.append(Requirement(str(entry["module"]), branch, line))
    return out


def module_files(module: str) -> list[Path] | None:
    """The source files a requirement covers; None when the module is not found."""
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    origin = Path(spec.origin).resolve()
    if origin.name == "__init__.py":
        return sorted(p.resolve() for p in origin.parent.rglob("*.py"))
    return [origin]


def measure(files: list[Path], report: dict[str, Any]) -> Measured:
    by_path = {Path(name).resolve(): data["summary"] for name, data in report["files"].items()}
    cb = nb = cl = ns = 0
    missing = []
    for path in files:
        summary = by_path.get(path)
        if summary is None:
            missing.append(path.name)
            # A missing file counts as 0 %: at least one uncovered branch and line.
            nb += 1
            ns += 1
            continue
        cb += int(summary.get("covered_branches", 0))
        nb += int(summary.get("num_branches", 0))
        cl += int(summary.get("covered_lines", 0))
        ns += int(summary.get("num_statements", 0))
    return Measured(cb, nb, cl, ns, tuple(missing))


def problems(requirements: list[Requirement], report: dict[str, Any]) -> list[str]:
    out = []
    for req in requirements:
        files = module_files(req.module)
        if not files:
            out.append(f"{req.module}: modulen hittades inte (0 %)")
            continue
        got = measure(files, report)
        if got.missing_files:
            out.append(f"{req.module}: saknas i coverage.json: {', '.join(got.missing_files)}")
        if req.branch_min is not None and got.branch_pct < req.branch_min:
            out.append(
                f"{req.module}: grentäckning {got.branch_pct:.1f} % < {req.branch_min:g} % "
                f"({got.covered_branches}/{got.num_branches})"
            )
        if req.line_min is not None and got.line_pct < req.line_min:
            out.append(
                f"{req.module}: radtäckning {got.line_pct:.1f} % < {req.line_min:g} % "
                f"({got.covered_lines}/{got.num_statements})"
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("coverage_json", type=Path)
    parser.add_argument("thresholds", type=Path)
    args = parser.parse_args(argv)
    report = json.loads(args.coverage_json.read_text(encoding="utf-8"))
    requirements = load_requirements(args.thresholds)
    found = problems(requirements, report)
    for line in found:
        print(line)
    if found:
        print(f"Täckningskraven är inte uppfyllda ({len(found)}).")
        return 1
    print(f"Täckningskraven är uppfyllda ({len(requirements)} moduler).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
