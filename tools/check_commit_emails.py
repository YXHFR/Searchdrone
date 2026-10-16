"""Fail if a commit has an author or committer e-mail that is not a GitHub noreply address.

Personal data can leak through commit metadata, not only through files. Every commit in
``<since>..<head>`` must have author and committer e-mails that end in
``@users.noreply.github.com`` or are exactly ``noreply@github.com`` (GitHub's web
committer). ``Co-Authored-By`` trailers are not checked.

CI runs it in ``just lint`` with ``--since <PR base>``. Findings name the commit and the
field; the address itself is printed masked (``k***@g***.com``), never in full.

``--exceptions`` takes a file with one full commit SHA per line (``#`` starts a
comment); those commits are skipped. A missing exceptions file means no exceptions.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

NOREPLY_SUFFIX = "@users.noreply.github.com"
WEB_COMMITTER = "noreply@github.com"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# One record per commit: SHA, author e-mail, committer e-mail, separated by NUL.
LOG_FORMAT = "%H%x00%ae%x00%ce"


@dataclass(frozen=True)
class Finding:
    sha: str
    field: str
    masked: str


def is_allowed(email: str) -> bool:
    e = email.strip().lower()
    return e == WEB_COMMITTER or (e.endswith(NOREPLY_SUFFIX) and len(e) > len(NOREPLY_SUFFIX))


def mask(email: str) -> str:
    """``jane@example.com`` -> ``j***@e***.com``. Never reveals more than two letters."""
    local, _, domain = email.strip().partition("@")
    if not domain:
        return (local[:1] or "?") + "***"
    host, dot, tld = domain.rpartition(".")
    if not dot:
        host, tld = domain, ""
    return f"{local[:1] or '?'}***@{host[:1] or '?'}***" + (f".{tld}" if tld else "")


def parse_log(text: str) -> list[tuple[str, str, str]]:
    commits = []
    for line in text.splitlines():
        if not line.strip():
            continue
        sha, author, committer = line.split("\x00")
        commits.append((sha, author, committer))
    return commits


def find_bad(commits: list[tuple[str, str, str]], exceptions: set[str]) -> list[Finding]:
    findings = []
    for sha, author, committer in commits:
        if sha in exceptions:
            continue
        for field, email in (("author", author), ("committer", committer)):
            if not is_allowed(email):
                findings.append(Finding(sha, field, mask(email)))
    return findings


def read_exceptions(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    shas = set()
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if not SHA_RE.match(line):
            raise ValueError(f"{path}:{n}: inte en fullständig commit-SHA")
        shas.add(line)
    return shas


def git_log(root: Path, since: str | None, head: str) -> str:
    rev = f"{since}..{head}" if since else head
    return subprocess.run(  # noqa: S603 - fixed git arguments, no shell
        ["git", "-C", str(root), "log", "--no-mailmap", f"--format={LOG_FORMAT}", rev],  # noqa: S607 - git from PATH
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Require GitHub noreply e-mails on commits.")
    ap.add_argument("--since", help="check commits after this revision (CI: the PR base)")
    ap.add_argument("--head", default="HEAD", help="last revision to check (default HEAD)")
    ap.add_argument("--exceptions", type=Path, help="file with one commit SHA per line")
    ap.add_argument("--root", type=Path, default=Path())
    args = ap.parse_args(argv)

    try:
        exceptions = read_exceptions(args.exceptions)
        commits = parse_log(git_log(args.root, args.since, args.head))
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f"check_commit_emails: {exc}", file=sys.stderr)
        return 2

    findings = find_bad(commits, exceptions)
    for f in findings:
        print(f"{f.sha}: {f.field}: {f.masked} är inte GitHubs noreply-e-post")
    if findings:
        print(
            "Sätt git config user.email till <id>+<användare>@users.noreply.github.com "
            "och skriv om commitarna.",
            file=sys.stderr,
        )
        return 1
    print(f"check_commit_emails: {len(commits)} commits ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
