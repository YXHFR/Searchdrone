"""Prepare a release: one version everywhere and the changelog (``just release-prep X.Y.Z``).

1. Refuses a dirty working tree, a version that is not ``X.Y.Z`` or
   ``X.Y.Z-(alpha|beta|rc).N``, and a version whose tag ``vX.Y.Z`` already exists.
2. Runs ``tools/check_merged_pr_gates.py``: every commit since the previous ``v*`` tag
   with a published release came from a pull request with green gates, each traced to
   its gate workflow (read-only ``gh``).
3. Sets ``[project] version`` in every ``pyproject.toml`` (PEP 440: ``0.1.0rc1``) and
   ``version`` in every ``package.json`` (SemVer: ``0.1.0-rc.1``), refreshes the uv lock
   files, and runs ``tools/check_versions.py --expect``.
4. Turns ``## [Unreleased]`` in ``CHANGELOG.md`` into ``## [X.Y.Z] - <date>`` (the
   hand-written notes stay), adds the commit list from git-cliff (``cliff.toml``) to that
   section and opens a new, empty ``## [Unreleased]``.

``--notes`` prints the version's section of ``CHANGELOG.md`` instead (the release
workflow's release notes) and changes nothing.

It never commits, tags or publishes: the owner reviews the diff, merges it through a
pull request and pushes the tag ``vX.Y.Z``, which starts ``.github/workflows/release.yml``.
``--no-gates`` skips step 2 (no ``gh`` on this computer); the release workflow still
runs it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHANGELOG = "CHANGELOG.md"
UNRELEASED = "## [Unreleased]"
VERSION_RE = re.compile(r"^(?P<base>\d+\.\d+\.\d+)(?:-(?P<pre>alpha|beta|rc)\.(?P<n>\d+))?$")
PEP440_PRE = {"alpha": "a", "beta": "b", "rc": "rc"}
PYPROJECT_VERSION_RE = re.compile(r'^(version\s*=\s*)"[^"]*"', re.M)
PACKAGE_VERSION_RE = re.compile(r'^(\s*"version"\s*:\s*)"[^"]*"', re.M)


class ReleaseError(RuntimeError):
    """A reason not to prepare the release; printed as is."""


def pep440(version: str) -> str:
    """``0.1.0-rc.1`` -> ``0.1.0rc1``; ``0.1.0`` stays."""
    m = VERSION_RE.match(version)
    if m is None:
        raise ReleaseError(
            f"'{version}' är ingen version på formen X.Y.Z eller X.Y.Z-(alpha|beta|rc).N"
        )
    if m["pre"] is None:
        return m["base"]
    return f"{m['base']}{PEP440_PRE[m['pre']]}{m['n']}"


def run(root: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=root, capture_output=True, text=True, check=False)  # noqa: S603


def git_files(root: Path, *patterns: str) -> list[Path]:
    out = run(root, "git", "ls-files", "--", *patterns).stdout
    return [root / line for line in out.splitlines() if line]


def set_pyproject(text: str, version: str) -> str:
    """The ``[project]`` table's ``version``, nothing else."""
    start = text.find("[project]")
    if start < 0:
        return text
    nxt = re.search(r"^\[", text[start + 1 :], re.M)
    end = start + 1 + nxt.start() if nxt else len(text)
    section, n = PYPROJECT_VERSION_RE.subn(rf'\g<1>"{version}"', text[start:end], count=1)
    return text[:start] + section + text[end:] if n else text


def set_package_json(text: str, version: str) -> str:
    """The top-level ``version`` (the first one, before any nested object's)."""
    new = PACKAGE_VERSION_RE.sub(rf'\g<1>"{version}"', text, count=1)
    if json.loads(new).get("version") != version:
        raise ReleaseError("package.json: hittade inte versionen på översta nivån")
    return new


def set_versions(root: Path, version: str) -> list[Path]:
    """Write the version everywhere; returns the files that changed."""
    changed = []
    py = pep440(version)
    for path in git_files(root, "pyproject.toml", "*/pyproject.toml"):
        text = path.read_text(encoding="utf-8")
        new = set_pyproject(text, py)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed.append(path)
    for path in git_files(root, "package.json", "*/package.json"):
        text = path.read_text(encoding="utf-8")
        new = set_package_json(text, version)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed.append(path)
    return changed


def update_changelog(text: str, version: str, date: dt.date, commits: str) -> str:
    """``[Unreleased]`` becomes the release; git-cliff's list ends that section."""
    if f"## [{version}]" in text:
        raise ReleaseError(f"{CHANGELOG} har redan ett avsnitt för {version}")
    start = text.find(UNRELEASED)
    if start < 0:
        raise ReleaseError(f"{CHANGELOG} saknar rubriken '{UNRELEASED}'")
    body_start = start + len(UNRELEASED)
    nxt = re.search(r"^## \[", text[body_start:], re.M)
    end = body_start + nxt.start() if nxt else len(text)
    notes = text[body_start:end].strip("\n")
    parts = [f"{UNRELEASED}\n\n## [{version}] - {date.isoformat()}"]
    if notes:
        parts.append(notes)
    if commits.strip():
        parts.append(commits.strip())
    section = "\n\n".join(parts) + "\n"
    rest = text[end:]
    return text[:start] + section + ("\n" + rest if rest else "")


def notes(text: str, version: str) -> str:
    """The body of ``## [version]`` in the changelog."""
    m = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n", text, re.M)
    if m is None:
        raise ReleaseError(f"{CHANGELOG} har inget avsnitt för {version}; kör just release-prep")
    nxt = re.search(r"^## \[", text[m.end() :], re.M)
    end = m.end() + nxt.start() if nxt else len(text)
    return text[m.end() : end].strip() + "\n"


def cliff_commits(root: Path, version: str) -> str:
    """git-cliff's commit list for the commits since the previous tag."""
    pin = run(root, sys.executable, "tools/versions.py", "git_cliff_version").stdout.strip()
    proc = run(
        root,
        "uvx",
        "--from",
        f"git-cliff=={pin}",
        "git-cliff",
        "--config",
        "cliff.toml",
        "--unreleased",
        "--tag",
        f"v{version}",
        "--strip",
        "all",
    )
    if proc.returncode != 0:
        raise ReleaseError(f"git-cliff misslyckades: {proc.stderr.strip()[-300:]}")
    return proc.stdout


def preflight(root: Path, version: str) -> None:
    pep440(version)
    if run(root, "git", "status", "--porcelain").stdout.strip():
        raise ReleaseError("arbetskopian har ändringar; release-prep kräver ett rent träd")
    if run(root, "git", "rev-parse", "-q", "--verify", f"refs/tags/v{version}").returncode == 0:
        raise ReleaseError(f"taggen v{version} finns redan")


def step(cwd: Path, what: str, *argv: str) -> None:
    print(f"==> {what}", flush=True)
    proc = run(cwd, *argv)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise ReleaseError(f"{what} misslyckades (kod {proc.returncode})")


def prepare(
    root: Path,
    version: str,
    *,
    gates: bool = True,
    today: dt.date | None = None,
    commits: Callable[[Path, str], str] = cliff_commits,
    lock: bool = True,
    verify: bool = True,
) -> list[Path]:
    preflight(root, version)
    if gates:
        step(root, "grindarna på main", sys.executable, "tools/check_merged_pr_gates.py")
    changed = set_versions(root, version)
    if lock:
        # Each uv project has its own lock (the workspace, tools/train, tools/rlmaps).
        for lockfile in git_files(root, "uv.lock", "*/uv.lock"):
            step(lockfile.parent, f"uv lock ({lockfile.parent.relative_to(root)})", "uv", "lock")
    if verify:
        check = REPO / "tools" / "check_versions.py"
        argv = [sys.executable, str(check), "--root", str(root), "--expect", version]
        step(root, "check_versions", *argv)
    changelog = root / CHANGELOG
    text = changelog.read_text(encoding="utf-8")
    date = today or dt.datetime.now(dt.UTC).date()
    new = update_changelog(text, version, date, commits(root, version))
    changelog.write_text(new, encoding="utf-8")
    return [*changed, changelog]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("version", help="X.Y.Z or X.Y.Z-rc.N")
    parser.add_argument("--no-gates", action="store_true", help="skip check_merged_pr_gates")
    parser.add_argument("--notes", action="store_true", help="print the version's notes")
    args = parser.parse_args(argv)
    if args.notes:
        try:
            text = (REPO / CHANGELOG).read_text(encoding="utf-8")
            sys.stdout.write(notes(text, args.version.removeprefix("v")))
        except ReleaseError as exc:
            print(f"release-prep: {exc}", file=sys.stderr)
            return 1
        return 0
    try:
        changed = prepare(REPO, args.version, gates=not args.no_gates)
    except ReleaseError as exc:
        print(f"release-prep: {exc}", file=sys.stderr)
        return 1
    for path in changed:
        print(f"ändrad: {path.relative_to(REPO)}")
    print(
        f"\nKlart. Granska diffen, committa den som 'chore(release): v{args.version}' i en pull "
        f"request och tagga efter merge: git tag v{args.version} <sha> && "
        f"git push origin v{args.version}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
