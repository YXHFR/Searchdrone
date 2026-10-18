"""A test's fault on a service: kill it, then restart it in its old start order."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sd_sim.procs import ProcessGroup, clean_env

SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]


def test_restart_keeps_the_start_order_and_records_the_new_pid(tmp_path: Path) -> None:
    group = ProcessGroup(tmp_path / "pids.json", tmp_path / "logs")
    try:
        for name in ("db", "sd-ground", "sd-onboard"):
            group.start(name, SLEEP, env=clean_env())
        old = group.procs[1].pid
        group.kill("sd-ground")
        assert not group.procs[1].alive()
        new = group.restart("sd-ground", SLEEP, env=clean_env())
        assert [p.name for p in group.procs] == ["db", "sd-ground", "sd-onboard"]
        assert new.pid != old and new.alive() and group.procs[1] is new
        recorded = json.loads((tmp_path / "pids.json").read_text(encoding="utf-8"))
        assert [r["pid"] for r in recorded["procs"] if r["name"] == "sd-ground"] == [new.pid]
    finally:
        group.stop_all()


def test_restart_kills_a_running_one_first(tmp_path: Path) -> None:
    group = ProcessGroup(tmp_path / "pids.json", tmp_path / "logs")
    try:
        first = group.start("sd-ground", SLEEP, env=clean_env())
        second = group.restart("sd-ground", SLEEP, env=clean_env())
        assert not first.alive() and second.alive() and len(group.procs) == 1
    finally:
        group.stop_all()
