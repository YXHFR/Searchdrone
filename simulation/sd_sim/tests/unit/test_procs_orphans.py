"""Stopping a process also stops descendants that moved to their own process group.

``gz sim`` with a window forks a server and a GUI that each call ``setpgid(0, 0)`` and
ignore SIGTERM; stopping only the parent's group used to leave them running.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sd_sim import procs

# The leader forks a child that leaves the group (own session), ignores SIGTERM and
# SIGINT like Gazebo's server, and writes its pid once it has left. argv: ignore_term
# for the leader too ("1"/"0"), pid file.
FORKER = """
import os, signal, sys, time
if sys.argv[1] == "1":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if os.fork() == 0:
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    tmp = sys.argv[2] + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(os.getpid()))
    os.replace(tmp, sys.argv[2])
    time.sleep(60)
    os._exit(0)
time.sleep(60)
"""


@pytest.fixture(autouse=True)
def short_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(procs, "STOP_GRACE_S", 0.5)


def running(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return stat.rsplit(")", 1)[-1].split()[0] != "Z"


def wait_gone(pid: int, limit_s: float = 10.0) -> bool:
    end = time.monotonic() + limit_s
    while running(pid):
        if time.monotonic() > end:
            return False
        time.sleep(0.05)
    return True


def read_pid(pid_file: Path, limit_s: float = 10.0) -> int:
    end = time.monotonic() + limit_s
    while not pid_file.exists():
        assert time.monotonic() < end, "the child did not start"
        time.sleep(0.01)
    return int(pid_file.read_text(encoding="utf-8"))


def kill_quietly(pid: int) -> None:
    if running(pid):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize("leader_ignores_term", [False, True])
def test_stop_all_stops_a_child_in_its_own_group(
    tmp_path: Path, *, leader_ignores_term: bool
) -> None:
    group = procs.ProcessGroup(tmp_path / "pids.json", tmp_path / "logs")
    pid_file = tmp_path / "child.pid"
    flag = "1" if leader_ignores_term else "0"
    group.start("gz", [sys.executable, "-c", FORKER, flag, str(pid_file)], env=procs.clean_env())
    child = read_pid(pid_file)
    try:
        assert running(child)
        group.stop_all()
        assert wait_gone(child), "the child in its own group was left running"
    finally:
        group.stop_all()
        kill_quietly(child)


def test_stop_recorded_without_launcher_stops_the_child(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    leader = subprocess.Popen(
        [sys.executable, "-c", FORKER, "0", str(pid_file)], start_new_session=True
    )
    child = read_pid(pid_file)
    try:
        pids = tmp_path / "pids.json"
        pids.write_text(json.dumps({"procs": [{"name": "gz", **procs.identity(leader.pid)}]}))
        assert procs.stop_recorded(pids, wait_s=0) == ["gz"]
        assert leader.wait(timeout=10) is not None
        assert wait_gone(child)
    finally:
        leader.kill()
        kill_quietly(child)


def test_stop_pid_leaves_unrelated_groups_alone(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    other = subprocess.Popen(
        [sys.executable, "-c", FORKER, "0", str(pid_file)], start_new_session=True
    )
    other_child = read_pid(pid_file)
    target = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        procs.stop_pid(target.pid, target)
        assert target.poll() is not None
        assert other.poll() is None and running(other_child)
    finally:
        target.kill()
        procs.stop_pid(other.pid, other)
        kill_quietly(other_child)


def test_a_reused_pid_is_not_signalled() -> None:
    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        procs._stop_descendants({sleeper.pid: "1"})
        assert sleeper.poll() is None
    finally:
        sleeper.kill()
        sleeper.wait(timeout=10)


def test_descendants_include_children_in_other_groups(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    leader = subprocess.Popen(
        [sys.executable, "-c", FORKER, "0", str(pid_file)], start_new_session=True
    )
    child = read_pid(pid_file)
    try:
        found = procs._descendants(leader.pid)
        assert child in found
        assert found[child] == procs.identity(child)["start"]
    finally:
        procs.stop_pid(leader.pid, leader)
        kill_quietly(child)
