"""The running database container: loopback-only port and data kept across restarts."""

from __future__ import annotations

import os
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
    listeners = _listeners(5433)
    if not listeners:
        pytest.skip("docker publishes the port in another network namespace (like `ss -ltn`)")
    assert listeners == ["tcp:0100007F"], listeners


# Restarting stops the shared local database, so it runs in CI or on request only.
RESTART_OK = os.environ.get("CI") == "true" or os.environ.get("SD_TEST_DB_RESTART") == "1"


@pytest.mark.slow
@pytest.mark.skipif(JUST is None, reason="just is not installed")
@pytest.mark.skipif(not RESTART_OK, reason="set SD_TEST_DB_RESTART=1 (restarts the database)")
def test_data_survives_db_down_up(fresh_db: Db) -> None:
    fresh_db.run("CREATE TABLE keep (v text)")
    fresh_db.run("INSERT INTO keep VALUES ('kvar')")
    assert JUST is not None
    try:
        down = subprocess.run(
            [JUST, "db-down"], cwd=ROOT, capture_output=True, text=True, check=False
        )
        assert down.returncode == 0, down.stdout + down.stderr
    finally:
        # Always bring the database back, so later tests are not left without it.
        up = subprocess.run([JUST, "db-up"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert up.returncode == 0, up.stdout + up.stderr
    assert fresh_db.run("SELECT v FROM keep")[0]["v"] == "kvar"
    res = subprocess.run([JUST, "db-check"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr
    assert "POSTGIS=" in res.stdout
