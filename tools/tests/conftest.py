"""Shared fixtures for the tools/ tests."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


class GitRepo:
    """A throwaway git repository with a hardware/ folder, used as the working directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(path),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "1+t@users.noreply.github.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "1+t@users.noreply.github.com",
        }

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.path), *args],
            capture_output=True,
            text=True,
            check=True,
            env=self.env,
        ).stdout.strip()

    def write(self, rel: str, text: str = "") -> Path:
        path = self.path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def commit(self) -> str:
        self.git("add", "-A")
        self.git("-c", "commit.gpgsign=false", "commit", "-q", "--allow-empty", "-m", "x")
        return self.git("rev-parse", "HEAD")


@pytest.fixture
def hw_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GitRepo:
    repo = GitRepo(tmp_path / "repo")
    repo.path.mkdir()
    repo.git("init", "-q")
    repo.write("hardware/drone.yaml", "a: 1\n")
    repo.write("hardware/limits.yaml", "b: 1\n")
    repo.write("README.md", "x\n")
    repo.commit()
    monkeypatch.chdir(repo.path)
    return repo
