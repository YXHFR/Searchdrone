"""tools/sitl_image_tag.sh: the SITL image tag follows exactly the files that build it."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "sitl_image_tag.sh"
TAG_RE = re.compile(r"^Copter-\d+\.\d+\.\d+-[0-9a-f]{12}$")


def run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(cwd / "tools" / "sitl_image_tag.sh"), *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def copy(tmp_path: Path) -> Path:
    for rel in ("tools/sitl_image_tag.sh", "simulation/versions.lock"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, tmp_path / rel)
    shutil.copytree(
        ROOT / "simulation" / "docker",
        tmp_path / "simulation" / "docker",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return tmp_path


def test_tag_format_and_image() -> None:
    tag = run(ROOT).stdout.strip()
    assert TAG_RE.match(tag), tag
    image = run(ROOT, "--image").stdout.strip()
    assert image == f"ghcr.io/yxhfr/searchdrone-sitl:{tag}"


def test_same_files_same_tag(copy: Path) -> None:
    assert run(copy).stdout == run(ROOT).stdout


def test_dockerfile_change_changes_tag(copy: Path) -> None:
    before = run(copy).stdout
    with (copy / "simulation" / "docker" / "sitl.Dockerfile").open("a") as out:
        out.write("# change\n")
    assert run(copy).stdout != before


def test_new_file_and_lock_change_change_tag(copy: Path) -> None:
    before = run(copy).stdout
    (copy / "simulation" / "docker" / "extra.txt").write_text("x\n")
    after_new = run(copy).stdout
    assert after_new != before
    with (copy / "simulation" / "versions.lock").open("a") as out:
        out.write("# change\n")
    assert run(copy).stdout != after_new


def test_cache_files_ignored(copy: Path) -> None:
    before = run(copy).stdout
    cache = copy / "simulation" / "docker" / "__pycache__"
    cache.mkdir()
    (cache / "x.pyc").write_bytes(b"\0")
    assert run(copy).stdout == before


def test_unknown_argument() -> None:
    res = run(ROOT, "--push")
    assert res.returncode == 2
    assert "okänt argument" in res.stderr


def test_missing_tag_fails(copy: Path) -> None:
    lock = copy / "simulation" / "versions.lock"
    lock.write_text(re.sub(r"(?m)^ardupilot_tag.*$", "", lock.read_text()))
    res = run(copy)
    assert res.returncode == 1
    assert "ardupilot_tag saknas" in res.stderr
