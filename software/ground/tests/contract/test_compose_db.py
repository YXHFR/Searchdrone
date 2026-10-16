"""The running database container: loopback-only port and data kept across restarts."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]

ROOT = Path(__file__).resolve().parents[4]
JUST = shutil.which("just")
DOCKER = shutil.which("docker")
LISTEN = "0A"


def _compose(*args: str) -> subprocess.CompletedProcess[str]:
    assert DOCKER is not None
    return subprocess.run(
        [DOCKER, "compose", "--env-file", ".env", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _listeners(port: int) -> list[str]:
    """Local addresses listening on ``port`` in this network namespace (like `ss -ltn`)."""
    found = []
    for name, width in (("tcp", 8), ("tcp6", 32)):
        table = Path("/proc/net") / name
        if not table.is_file():
            continue
        for line in table.read_text(encoding="ascii").splitlines()[1:]:
            fields = line.split()
            addr, hexport = fields[1].split(":")
            if fields[3] == LISTEN and int(hexport, 16) == port and len(addr) == width:
                found.append(f"{name}:{addr}")
    return found


@pytest.mark.skipif(DOCKER is None, reason="docker is not installed")
def test_port_only_on_loopback() -> None:
    res = _compose("port", "db", "5432")
    assert res.returncode == 0, res.stderr
    assert res.stdout.split() == ["127.0.0.1:5433"]
    # 127.0.0.1 in /proc/net/tcp is little-endian hex 0100007F. No IPv6 or 0.0.0.0 listener.
    for listener in _listeners(5433):
        assert listener == "tcp:0100007F", listener


@pytest.mark.slow
@pytest.mark.skipif(JUST is None, reason="just is not installed")
def test_data_survives_db_down_up(fresh_db: Db) -> None:
    fresh_db.run("CREATE TABLE keep (v text)")
    fresh_db.run("INSERT INTO keep VALUES ('kvar')")
    assert JUST is not None
    for recipe in ("db-down", "db-up"):
        res = subprocess.run([JUST, recipe], cwd=ROOT, capture_output=True, text=True, check=False)
        assert res.returncode == 0, res.stdout + res.stderr
    assert fresh_db.run("SELECT v FROM keep")[0]["v"] == "kvar"
    res = subprocess.run([JUST, "db-check"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr
    assert "POSTGIS=" in res.stdout
