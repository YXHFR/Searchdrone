"""Hardware gate allowlist: every new or changed file under hardware/ needs a built check.

    python3 tools/check_hw_files.py --base <sha> --head <sha>
    python3 tools/check_hw_files.py --changes --base <sha> --head <sha> [--github-output F]

The first form goes through every file under ``hardware/`` (any letter case), and every
hardware design file anywhere (``HARDWARE_SUFFIXES``), that is added, changed or renamed
between the merge base of ``base`` and ``head``. A file passes only if it matches
a row in ``CHECKED``, and every row names the CI job that checks that kind of file. Any
other file fails closed: a KiCad board, a CAD model or an FEA deck cannot be merged until
its automatic check exists. Deleted files pass. Symlinks and submodules never pass.

A row in ``CHECKED`` may only be added in the same pull request that adds its job to
.github/workflows/hardware.yml and to ``hardware-ok``.

The second form is the path filter of the ``changes`` job: it prints ``run=true`` when
the diff touches the hardware gate's own paths (``FILTER``), otherwise ``run=false``.

Standard library only, so CI can run it before any tool is installed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable
from functools import cache
from pathlib import Path

# glob -> CI job in hardware.yml that checks files matching it.
CHECKED: dict[str, str] = {
    "hardware/drone.yaml": "hw-calc",
    "hardware/limits.yaml": "hw-calc",
    # Added in the same PR as their check, e.g.
    # "hardware/mechanical/fea/*.inp": "fea" and "hardware/electronics/**/*.kicad_pro": "kicad"
}

# Paths that make the hardware jobs run (the `changes` job): the hardware itself and
# everything that decides what hw-files, hw-calc and hardware-ok run.
FILTER: tuple[str, ...] = (
    "hardware/**",
    "software/hwcalc/**",
    "tools/check_hw_files.py",
    "tools/ci_aggregate.jq",
    ".github/workflows/hardware.yml",
    ".github/actions/setup/**",
    "justfile",
    "uv.lock",
)

# Hardware design files fail closed anywhere in the repository, not only under hardware/
# (simulation meshes such as .stl or .dae stay allowed for the simulator).
HARDWARE_SUFFIXES = (
    ".kicad_sch", ".kicad_pcb", ".kicad_pro", ".kicad_sym", ".kicad_mod",
    ".step", ".stp", ".fcstd", ".inp", ".frd", ".gbr",
)  # fmt: skip
HARDWARE_DIR_RE = re.compile(r"^hardware/", re.IGNORECASE)

ZERO_SHA_RE = re.compile(r"^0+$")
REGULAR_MODES = {"100644", "100755"}


@cache
def glob_regex(glob: str) -> re.Pattern[str]:
    """``**`` spans directories, ``*`` and ``?`` stay inside one path segment."""
    out: list[str] = []
    i = 0
    while i < len(glob):
        if glob.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif glob.startswith("**", i):
            out.append(".*")
            i += 2
        elif glob[i] == "*":
            out.append("[^/]*")
            i += 1
        elif glob[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(glob[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def matches(path: str, globs: Iterable[str]) -> bool:
    return any(glob_regex(g).match(path) for g in globs)


def annotation(text: str) -> str:
    """Escape a message for a GitHub ``::error::`` line (file names may hold newlines)."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def git(*args: str) -> str:
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def changed_entries(base: str, head: str, pathspec: tuple[str, ...]) -> list[tuple[str, str, str]]:
    """(status, new mode, path) for each changed file. A zero base means 'everything'."""
    if ZERO_SHA_RE.match(base):
        out = git("ls-tree", "-r", "-z", "--full-tree", head, "--", *pathspec)
        entries = []
        for rec in filter(None, out.split("\0")):
            meta, path = rec.split("\t", 1)
            mode = meta.split()[0]
            entries.append(("A", mode, path))
        return entries
    out = git("diff", "--raw", "-z", "--no-renames", f"{base}...{head}", "--", *pathspec)
    fields = out.split("\0")
    entries = []
    i = 0
    while i + 1 < len(fields) and fields[i]:
        meta, path = fields[i], fields[i + 1]
        parts = meta.lstrip(":").split()
        entries.append((parts[4][0], parts[1], path))
        i += 2
    return entries


def in_scope(path: str) -> bool:
    """Under hardware/ (any letter case) or a hardware design file anywhere."""
    return bool(HARDWARE_DIR_RE.match(path)) or path.lower().endswith(HARDWARE_SUFFIXES)


def problems(entries: list[tuple[str, str, str]]) -> list[str]:
    errors = []
    for status, mode, path in entries:
        if status == "D" or not in_scope(path):
            continue
        if mode not in REGULAR_MODES:
            errors.append(f"{path} är inte en vanlig fil (symlänk eller undermodul); godtas inte.")
        elif not matches(path, CHECKED):
            errors.append(
                f"Ingen automatisk kontroll för {path} ännu; bygg den i den här PR:en "
                "(se README, avsnitt Hardware)."
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--changes", action="store_true", help="print run=true|false only")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args(argv)

    if args.changes:
        touched = [p for _, _, p in changed_entries(args.base, args.head, ())]
        hit = any(matches(p, FILTER) or in_scope(p) for p in touched)
        run = "true" if hit else "false"
        line = f"run={run}"
        print(line)
        if args.github_output:
            with args.github_output.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return 0

    entries = changed_entries(args.base, args.head, ())
    errors = problems(entries)
    for err in errors:
        print(f"::error::{annotation(err)}")
    if errors:
        print(f"{len(errors)} fil(er) under hardware/ saknar automatisk kontroll.")
        return 1
    checked = [p for s, _, p in entries if s != "D" and in_scope(p)]
    print(f"Hårdvarufilerna godkända ({len(checked)} ändrade, alla har en kontroll).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
