"""Check that version numbers that must agree really do.

1. Every ``pyproject.toml`` ``[project] version`` and every ``package.json`` ``version``
   is the same release. Python versions are compared PEP 440-normalised and npm versions
   as SemVer strings mapped to PEP 440, so ``0.0.1rc1`` equals ``0.0.1-rc.1``. ``0.0.0``
   is the development version. ``--expect X`` also requires that every one is ``X``.
2. Tools pinned in two places use the same version: ruff in ``uv.lock`` and in
   ``.pre-commit-config.yaml``; typos and pnpm in ``simulation/versions.lock`` and in
   ``.pre-commit-config.yaml`` / ``package.json``; gitleaks in ``.pre-commit-config.yaml``
   and the CI workflow; the Node major version in ``.nvmrc``, ``package.json`` engines
   and ``simulation/versions.lock``.

Runs in ``just lint`` and in ``just release-prep``. Exit code 1 lists every mismatch.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.version import InvalidVersion, Version

PRE_COMMIT = ".pre-commit-config.yaml"
VERSIONS_LOCK = "simulation/versions.lock"
CI_WORKFLOW = ".github/workflows/ci.yml"
# "repo: <url>" followed by "rev: <sha>  # v1.2.3" in .pre-commit-config.yaml.
PRE_COMMIT_REV_RE = re.compile(
    r"-\s+repo:\s*(?P<repo>\S+)\s*\n\s+rev:\s*\S+\s*#\s*v(?P<version>[0-9][0-9A-Za-z.+-]*)"
)
GITLEAKS_ENV_RE = re.compile(r"^\s*GITLEAKS_VERSION:\s*\"?(?P<version>[0-9][0-9.]*)\"?", re.M)


def git_files(root: Path, *patterns: str) -> list[Path]:
    out = subprocess.run(  # noqa: S603 - fixed git arguments, no shell
        ["git", "-C", str(root), "ls-files", "--", *patterns],  # noqa: S607 - git from PATH
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [root / line for line in out.splitlines() if line]


def normalise(raw: str) -> Version:
    return Version(raw)


def package_versions(root: Path) -> dict[str, str]:
    """Map each manifest (relative path) to its declared version."""
    found: dict[str, str] = {}
    for path in git_files(root, "pyproject.toml", "*/pyproject.toml"):
        project = tomllib.loads(path.read_text(encoding="utf-8")).get("project", {})
        if "version" in project:
            found[str(path.relative_to(root))] = str(project["version"])
    for path in git_files(root, "package.json", "*/package.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "version" in data:
            found[str(path.relative_to(root))] = str(data["version"])
    return found


def check_package_versions(versions: dict[str, str], expect: str | None) -> list[str]:
    errors: list[str] = []
    parsed: dict[str, Version] = {}
    for name, raw in sorted(versions.items()):
        try:
            parsed[name] = normalise(raw)
        except InvalidVersion:
            errors.append(f"{name}: ogiltig version {raw!r}")
    if expect is not None:
        try:
            wanted = normalise(expect)
        except InvalidVersion:
            return [*errors, f"--expect: ogiltig version {expect!r}"]
        errors += [
            f"{name}: version {versions[name]} men väntade {expect}"
            for name, v in sorted(parsed.items())
            if v != wanted
        ]
    elif len(set(parsed.values())) > 1:
        listing = ", ".join(f"{n}={versions[n]}" for n in sorted(parsed))
        errors.append(f"olika versioner: {listing}")
    return errors


def pre_commit_versions(text: str) -> dict[str, str]:
    """Map the last path part of each pre-commit repo URL to its ``# vX.Y.Z`` comment."""
    return {
        m["repo"].rstrip("/").rsplit("/", 1)[-1]: m["version"]
        for m in PRE_COMMIT_REV_RE.finditer(text)
    }


def uv_lock_version(text: str, package: str) -> str | None:
    for pkg in tomllib.loads(text).get("package", []):
        if pkg.get("name") == package:
            return str(pkg.get("version"))
    return None


def check_tool_pins(root: Path) -> list[str]:
    errors: list[str] = []

    def read(rel: str) -> str | None:
        p = root / rel
        return p.read_text(encoding="utf-8") if p.is_file() else None

    hooks = pre_commit_versions(read(PRE_COMMIT) or "")
    lock = tomllib.loads(read(VERSIONS_LOCK) or "")
    uv_lock = read("uv.lock")
    package_json = json.loads(read("package.json") or "{}")

    def same(what: str, a_name: str, a: str | None, b_name: str, b: str | None) -> None:
        if a is None or b is None:
            missing = a_name if a is None else b_name
            errors.append(f"{what}: hittar ingen version i {missing}")
        elif a != b:
            errors.append(f"{what}: {a_name} har {a} men {b_name} har {b}")

    same(
        "ruff",
        "uv.lock",
        uv_lock_version(uv_lock or "", "ruff"),
        PRE_COMMIT,
        hooks.get("ruff-pre-commit"),
    )
    same("typos", VERSIONS_LOCK, lock.get("typos_version"), PRE_COMMIT, hooks.get("typos"))
    manager = str(package_json.get("packageManager", ""))
    pnpm = manager.removeprefix("pnpm@").split("+", 1)[0] if manager else None
    same("pnpm", VERSIONS_LOCK, lock.get("pnpm_version"), "package.json", pnpm)
    workflow = read(CI_WORKFLOW)
    m = GITLEAKS_ENV_RE.search(workflow or "")
    same("gitleaks", PRE_COMMIT, hooks.get("gitleaks"), CI_WORKFLOW, m and m["version"])
    node = lock.get("node_version")
    nvmrc = (read(".nvmrc") or "").strip().removeprefix("v").split(".", 1)[0] or None
    major = str(node).split(".", 1)[0] if node else None
    same("node", VERSIONS_LOCK, major, ".nvmrc", nvmrc)
    if major is not None:
        engines = str(package_json.get("engines", {}).get("node", ""))
        wanted = f">={major} <{int(major) + 1}"
        if engines != wanted:
            errors.append(f"node: package.json engines är {engines!r}, väntade {wanted!r}")
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check that version numbers agree.")
    ap.add_argument("--expect", help="require this release version everywhere")
    ap.add_argument("--root", type=Path, default=Path())
    args = ap.parse_args(argv)

    versions = package_versions(args.root)
    errors = check_package_versions(versions, args.expect) + check_tool_pins(args.root)
    for e in errors:
        print(f"check_versions: {e}")
    if errors:
        return 1
    release = next(iter(versions.values()), "?")
    print(f"check_versions: {len(versions)} manifest har version {release}, verktygen stämmer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
