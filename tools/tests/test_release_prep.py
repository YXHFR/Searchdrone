"""tools/release_prep.py: versions, the changelog and the refusals, in a throwaway repo."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import pytest
import release_prep as rp

DAY = dt.date(2026, 11, 2)
CHANGELOG = """# Changelog

Intro.

## [Unreleased]

### Added

- A thing.
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "w"\nversion = "0.0.0"  # dev\n\n[tool.x]\nversion = "9"\n',
        encoding="utf-8",
    )
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "package.json").write_text(
        json.dumps({"name": "p", "version": "0.0.0", "dependencies": {"version": "x"}}, indent=2),
        encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=1+t@users.noreply.github.com", "commit", "-qm", "x")
    return tmp_path


@pytest.mark.parametrize(
    ("semver", "pep"),
    [("0.1.0", "0.1.0"), ("1.2.3-rc.1", "1.2.3rc1"), ("0.2.0-alpha.3", "0.2.0a3")],
)
def test_pep440(semver: str, pep: str) -> None:
    assert rp.pep440(semver) == pep


@pytest.mark.parametrize("bad", ["v0.1.0", "0.1", "0.1.0-dev", "0.1.0rc1", ""])
def test_bad_versions_are_refused(bad: str) -> None:
    with pytest.raises(rp.ReleaseError):
        rp.pep440(bad)


def test_prepare_sets_versions_and_the_changelog(repo: Path) -> None:
    changed = rp.prepare(
        repo, "0.1.0-rc.1", gates=False, lock=False, verify=False, today=DAY, commits=fake_cliff
    )
    assert {p.name for p in changed} == {"pyproject.toml", "package.json", "CHANGELOG.md"}
    py = (repo / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.1.0rc1"  # dev' in py
    assert 'version = "9"' in py  # only the [project] table
    pkg = json.loads((repo / "pkg" / "package.json").read_text(encoding="utf-8"))
    assert pkg["version"] == "0.1.0-rc.1"
    assert pkg["dependencies"] == {"version": "x"}
    log = (repo / "CHANGELOG.md").read_text(encoding="utf-8")
    assert log.endswith(
        "## [Unreleased]\n\n## [0.1.0-rc.1] - 2026-11-02\n\n### Added\n\n- A thing.\n\n"
        "### Commits\n\n- feat: x\n"
    )
    assert rp.notes(log, "0.1.0-rc.1").startswith("### Added\n\n- A thing.")


def fake_cliff(root: Path, version: str) -> str:
    return "### Commits\n\n- feat: x\n"


def test_prepare_in_a_clone_of_this_repo_passes_check_versions(tmp_path: Path) -> None:
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", "--no-hardlinks", str(rp.REPO), str(clone)],
        check=True,
        capture_output=True,
    )
    changed = rp.prepare(clone, "9.8.7", gates=False, lock=False, today=DAY, commits=fake_cliff)
    names = {str(p.relative_to(clone)) for p in changed}
    assert {"pyproject.toml", "CHANGELOG.md"} <= names
    assert "## [9.8.7] - 2026-11-02" in (clone / "CHANGELOG.md").read_text(encoding="utf-8")


def test_a_dirty_tree_is_refused(repo: Path) -> None:
    (repo / "new.txt").write_text("x", encoding="utf-8")
    with pytest.raises(rp.ReleaseError, match="rent träd"):
        rp.prepare(repo, "0.1.0", gates=False, lock=False, commits=fake_cliff)


def test_an_existing_tag_is_refused(repo: Path) -> None:
    subprocess.run(["git", "-C", str(repo), "tag", "v0.1.0"], check=True)
    with pytest.raises(rp.ReleaseError, match="finns redan"):
        rp.prepare(repo, "0.1.0", gates=False, lock=False, commits=fake_cliff)


def test_changelog_keeps_older_releases() -> None:
    text = CHANGELOG + "\n## [0.0.9] - 2026-01-01\n\n- Old.\n"
    new = rp.update_changelog(text, "0.1.0", DAY, "")
    assert "## [0.1.0] - 2026-11-02\n\n### Added\n\n- A thing.\n\n## [0.0.9]" in new
    assert rp.notes(new, "0.0.9") == "- Old.\n"
    with pytest.raises(rp.ReleaseError, match="redan"):
        rp.update_changelog(new, "0.1.0", DAY, "")


def test_notes_for_an_unknown_version() -> None:
    with pytest.raises(rp.ReleaseError, match="inget avsnitt"):
        rp.notes(CHANGELOG, "0.3.0")


def test_cliff_config_is_pinned() -> None:
    lock = (rp.REPO / "simulation" / "versions.lock").read_text(encoding="utf-8")
    assert 'git_cliff_version = "' in lock
    assert (rp.REPO / "cliff.toml").is_file()
