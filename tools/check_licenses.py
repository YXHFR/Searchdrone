"""License policy for production dependencies (runs in the CI job ``contract``).

Allowed in code we import or link: Apache-2.0, MIT, BSD, ISC, PSF, MPL-2.0 (unmodified)
and LGPL (unmodified, dynamically linked, e.g. pymavlink). GPL and AGPL code is never
imported. Developer tools that are never shipped (reuse, gitleaks, ...) are not checked.

Python: the production dependencies from ``uv.lock`` (``uv export --no-dev``, our own
workspace packages excluded) are installed into a throw-away virtualenv and read with
pip-licenses. npm: ``pnpm licenses list --prod --json``. A package passes when at least
one of its declared licenses matches the allow-list (word match, so ``LGPL-3.0`` passes
and ``GPL-3.0`` or ``AGPL-3.0`` fail; every ``AND`` part must pass, one ``OR``
alternative is enough; long free text counts as unknown). ``UNKNOWN`` fails unless the
package is listed in ``KNOWN_OK`` with the license from its source and the date it was
checked; the security reviewer checks every line there.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PIP_LICENSES = "pip-licenses==5.5.5"
ALLOWED = (
    r"\bMIT\b",
    r"\bBSD\b",
    r"\bApache\b",
    r"\bISC\b",
    r"\bPSF\b",
    r"\bPython Software Foundation\b",
    r"\bMPL\b",
    r"\bMozilla\b",
    r"\bLGPL",
    r"\bLesser General Public\b",
)
ALLOWED_RE = re.compile("|".join(ALLOWED), re.IGNORECASE)
# GPL and AGPL in any spelling, but not LGPL ("Lesser"/"Library" GPL).
DENIED_RE = re.compile(
    r"(?<![L\w])A?GPL|Affero|(?<!Lesser )(?<!Library )(?<!Library or Lesser )"
    r"GNU General Public",
    re.IGNORECASE,
)
MAX_LICENSE_TEXT = 200
# pip-licenses and its own dependencies are tools, not dependencies of our packages.
IGNORED = {"pip-licenses", "prettytable", "wcwidth", "pip", "setuptools"}
# "<package>" or "<package>==<version>": "<license according to its source>, checked
# <YYYY-MM-DD>". With a version the entry covers only that release: a new one is read again.
KNOWN_OK: dict[str, str] = {
    # Pulled in by shapely (sd_geo). The expression is all permissive licenses, but
    # 0BSD, Zlib and CC0-1.0 are not on the word list above.
    "numpy==2.5.3": "BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0"
    " (License-Expression); the wheel bundles OpenBLAS (BSD-3-Clause), LAPACK"
    " (BSD-3-Clause-Open-MPI), libgfortran (GPL-3.0-or-later WITH GCC-exception-3.1)"
    " and libquadmath (LGPL-2.1-or-later), checked 2026-10-17",
}


@dataclass(frozen=True)
class Package:
    name: str
    version: str
    license: str


def term_allowed(term: str) -> bool:
    """One license name: allowed word present and no GPL/AGPL word (LGPL is fine)."""
    return bool(ALLOWED_RE.search(term)) and not DENIED_RE.search(term)


def license_allowed(license_text: str) -> bool:
    """Every ``AND`` part must be allowed; for ``OR`` (or ``;`` / ``/``) one alternative.

    Free text longer than ``MAX_LICENSE_TEXT`` (a pasted license file) is not trusted:
    it counts as unknown and must go through ``KNOWN_OK``.
    """
    text = license_text.strip()
    if not text or len(text) > MAX_LICENSE_TEXT:
        return False
    text = text.replace("(", " ").replace(")", " ")
    # "X WITH <exception>" only adds a permission to X.
    text = re.sub(r"\bWITH\s+\S+", " ", text, flags=re.IGNORECASE)
    alternatives = re.split(r";|/|\bOR\b", text, flags=re.IGNORECASE)
    return any(
        all(term_allowed(t) for t in re.split(r"\bAND\b", alt, flags=re.IGNORECASE))
        for alt in alternatives
        if alt.strip()
    )


def violations(packages: Iterable[Package]) -> list[Package]:
    bad = []
    for pkg in packages:
        if (
            pkg.name.lower() in IGNORED
            or {pkg.name, f"{pkg.name}=={pkg.version}"} & KNOWN_OK.keys()
        ):
            continue
        if pkg.license.strip().upper() in {"", "UNKNOWN"} or not license_allowed(pkg.license):
            bad.append(pkg)
    return bad


def parse_pip_licenses(text: str) -> list[Package]:
    return [
        Package(str(row["Name"]), str(row["Version"]), str(row["License"]))
        for row in json.loads(text)
    ]


def parse_pnpm_licenses(text: str) -> list[Package]:
    text = text.strip()
    if not text.startswith("{"):
        return []  # pnpm prints "No licenses in packages found" when there are none
    packages = []
    for license_name, entries in json.loads(text).items():
        for entry in entries:
            for version in entry.get("versions") or ["?"]:
                packages.append(
                    Package(entry["name"], version, entry.get("license") or license_name)
                )
    return packages


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    return subprocess.run(  # noqa: S603 - fixed tool arguments, no shell
        cmd, capture_output=True, text=True, check=True, cwd=cwd, env=env
    ).stdout


def python_packages(root: Path) -> list[Package]:
    with tempfile.TemporaryDirectory() as tmp:
        req = Path(tmp) / "req.txt"
        req.write_text(
            run(
                [
                    "uv",
                    "export",
                    "--all-packages",
                    "--no-dev",
                    "--no-hashes",
                    "--frozen",
                    "--no-emit-workspace",
                    "--no-header",
                ],
                cwd=root,
            ),
            encoding="utf-8",
        )
        venv = Path(tmp) / "lic"
        run(["uv", "venv", "--quiet", str(venv), "--python", "3.12"])
        py = venv / "bin" / "python"
        run(["uv", "pip", "install", "--quiet", "-p", str(py), "-r", str(req), PIP_LICENSES])
        out = run(
            [
                str(venv / "bin" / "pip-licenses"),
                "--python",
                str(py),
                "--format=json",
                "--from=mixed",
            ]
        )
        return parse_pip_licenses(out)


def npm_packages(root: Path) -> list[Package]:
    env = {**os.environ, "COREPACK_ENABLE_DOWNLOAD_PROMPT": "0"}
    return parse_pnpm_licenses(
        run(["pnpm", "licenses", "list", "--prod", "--json"], cwd=root, env=env)
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="License policy for production dependencies.")
    ap.add_argument("--root", type=Path, default=Path())
    ap.add_argument("--skip-npm", action="store_true", help="only check Python")
    args = ap.parse_args(argv)

    try:
        packages = python_packages(args.root)
        if not args.skip_npm:
            packages += npm_packages(args.root)
    except subprocess.CalledProcessError as exc:
        print(
            f"check_licenses: {' '.join(map(str, exc.cmd))} misslyckades:\n{exc.stderr}",
            file=sys.stderr,
        )
        return 2

    bad = violations(packages)
    for pkg in bad:
        print(
            f"check_licenses: {pkg.name} {pkg.version}: licensen {pkg.license!r} är inte tillåten"
        )
    if bad:
        print("Tillåtet: Apache-2.0, MIT, BSD, ISC, PSF, MPL-2.0, LGPL. Ingen GPL eller AGPL.")
        return 1
    print(f"check_licenses: {len(packages)} produktionsberoenden ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
