"""tools/check_commit_emails.py (ACC-S0-24): only GitHub noreply addresses in commits."""

from __future__ import annotations

import subprocess
from pathlib import Path

import check_commit_emails as cce
import pytest

NOREPLY = "12345+someone@users.noreply.github.com"
PRIVATE = "test@example.com"


def git(repo: Path, *args: str, email: str | None = None) -> str:
    env = {"PATH": "/usr/bin:/bin", "HOME": str(repo), "GIT_CONFIG_NOSYSTEM": "1"}
    if email:
        env |= {
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": email,
        }
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=env
    ).stdout.strip()


def commit(repo: Path, email: str) -> str:
    """Commit with ``email`` as both author and committer and return the SHA."""
    (repo / "f.txt").write_text(email + str(len(list(repo.iterdir()))), encoding="utf-8")
    git(repo, "add", "f.txt")
    git(repo, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "x", email=email)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    return tmp_path


@pytest.mark.parametrize(
    ("email", "ok"),
    [
        (NOREPLY, True),
        ("someone@users.noreply.github.com", True),
        ("noreply@github.com", True),
        ("NoReply@GitHub.com", True),
        (PRIVATE, False),
        ("@users.noreply.github.com", False),
        ("x@users.noreply.github.com.evil.io", False),
        ("noreply@github.com.evil.io", False),
    ],
)
def test_is_allowed(email: str, ok: bool) -> None:
    assert cce.is_allowed(email) is ok


@pytest.mark.parametrize(
    ("email", "masked"),
    [
        ("jane@example.com", "j***@e***.com"),
        ("a@b", "a***@b***"),
        ("nobody", "n***"),
        ("", "?***"),
    ],
)
def test_mask(email: str, masked: str) -> None:
    assert cce.mask(email) == masked


def test_noreply_commits_pass(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = commit(repo, PRIVATE)  # before --since: not checked
    commit(repo, NOREPLY)
    commit(repo, "noreply@github.com")
    assert cce.main(["--root", str(repo), "--since", base]) == 0
    assert "2 commits ok" in capsys.readouterr().out


def test_private_author_fails_with_sha_and_field(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = commit(repo, NOREPLY)
    bad = commit(repo, PRIVATE)
    assert cce.main(["--root", str(repo), "--since", base]) == 1
    out = capsys.readouterr()
    assert f"{bad}: author:" in out.out
    assert f"{bad}: committer:" in out.out
    assert PRIVATE not in out.out + out.err
    assert "t***@e***.com" in out.out


def test_head_limits_the_range(repo: Path) -> None:
    base = commit(repo, NOREPLY)
    good = commit(repo, NOREPLY)
    commit(repo, PRIVATE)
    assert cce.main(["--root", str(repo), "--since", base, "--head", good]) == 0


def test_exceptions_file_skips_listed_commits(repo: Path, tmp_path: Path) -> None:
    base = commit(repo, NOREPLY)
    bad = commit(repo, PRIVATE)
    exceptions = tmp_path / "exceptions.txt"
    exceptions.write_text(f"# old commit\n{bad}\n", encoding="utf-8")
    assert cce.main(["--root", str(repo), "--since", base, "--exceptions", str(exceptions)]) == 0


def test_missing_exceptions_file_means_none(repo: Path, tmp_path: Path) -> None:
    base = commit(repo, NOREPLY)
    commit(repo, PRIVATE)
    missing = tmp_path / "nope.txt"
    assert cce.main(["--root", str(repo), "--since", base, "--exceptions", str(missing)]) == 1


def test_bad_exceptions_line_is_an_error(repo: Path, tmp_path: Path) -> None:
    commit(repo, NOREPLY)
    exceptions = tmp_path / "exceptions.txt"
    exceptions.write_text("abc123\n", encoding="utf-8")
    assert cce.main(["--root", str(repo), "--exceptions", str(exceptions)]) == 2


def test_unknown_revision_is_an_error(repo: Path) -> None:
    commit(repo, NOREPLY)
    assert cce.main(["--root", str(repo), "--since", "0" * 40]) == 2
