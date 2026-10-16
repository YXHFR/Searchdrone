"""Smoke-test hardening, process-stop safety, Ctrl-C during start-up, linksim edge cases."""

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from sd_sim import config, launch, netcheck, procs, smoke
from sd_sim.errors import StopRequestedError
from sd_sim.linksim import core
from sd_sim.sitl import CMAC_HOME
from sd_sim.udprelay import FROM_VEHICLE, UdpRelay

PROFILES = core.load_profiles()


def test_smoke_harden_ignores_local_qgc_and_home() -> None:
    env = {"SD_SIM__QGC": "true", "SD_SIM__HOME_OVERRIDE": "-35.36,149.16,584,0"}
    cfg = config.load({"physics": "builtin"}, environ=env)
    assert cfg.qgc and cfg.personal_home
    cfg = smoke.harden(cfg)
    assert not cfg.qgc and not cfg.personal_home and cfg.home == CMAC_HOME
    stack = launch.Stack(cfg, Path("/nonexistent"), observer=False)
    assert "--serial5" not in stack.sitl_argv()


def test_owned_pid_listener_is_flagged() -> None:
    odd = netcheck.Listener("udp", "0.0.0.0", 40000, "ruby", 42)  # noqa: S104
    assert netcheck.violations([odd], {}, set(), owned_pids={42}) == [odd]
    assert netcheck.violations([odd], {}, set(), owned_pids={7}) == []
    gz = netcheck.Listener("udp", "0.0.0.0", 10317, "ruby", 42)  # noqa: S104
    allow = netcheck.load_allowlist()
    assert netcheck.violations([gz], {"gz_discovery_msg": 10317}, allow, owned_pids={42}) == []


def test_parse_ss_reads_pid() -> None:
    line = 'udp UNCONN 0 0 0.0.0.0:10317 0.0.0.0:* users:(("ruby",pid=13806,fd=16))'
    assert netcheck.parse_ss(line)[0].pid == 13806


def start_sleeper() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )


def test_stop_recorded_skips_reused_pid(tmp_path: Path) -> None:
    proc = start_sleeper()
    try:
        row = {"name": "x", **procs.identity(proc.pid)}
        assert procs.still_same(row)
        stale = {**row, "start": "1"}  # same pid and command, different process
        assert not procs.still_same(stale)
        assert not procs.still_same({**row, "cmd": row["cmd"] + " --other"})
        pids = tmp_path / "pids.json"
        pids.write_text(json.dumps({"procs": [stale]}))
        procs.stop_recorded(pids, wait_s=0)
        assert proc.poll() is None and not pids.exists()
        pids.write_text(json.dumps({"procs": [row]}))
        assert procs.stop_recorded(pids, wait_s=0) == ["x"]
        assert proc.wait(timeout=10) is not None
    finally:
        proc.kill()


def test_stop_pid_does_not_signal_foreign_group() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert os.getpgid(child.pid) != child.pid  # shares the test's group
        procs.stop_pid(child.pid, child)
        assert child.poll() is not None  # stopped, and the test process is still alive
    finally:
        child.kill()


class NeverHealthy:
    def __init__(self) -> None:
        self.cfg = config.SimConfig(instance=0)
        self.run_dir = Path("/nonexistent")
        self.group = procs.ProcessGroup(Path("/nonexistent/pids.json"), Path("/nonexistent"))
        self.stopped = False

    def up(self) -> None:
        end = time.monotonic() + 30
        while time.monotonic() < end:  # a health wait that never succeeds
            time.sleep(0.05)
        raise AssertionError("not interrupted")

    def down(self) -> None:
        self.stopped = True


def test_ctrl_c_interrupts_start_up() -> None:
    stack = NeverHealthy()
    old = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    timer = threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGINT))
    timer.start()
    try:
        start = time.monotonic()
        assert launch._run_foreground(stack) == 0  # type: ignore[arg-type]
        assert time.monotonic() - start < 5
        assert stack.stopped
    finally:
        signal.signal(signal.SIGINT, old[0])
        signal.signal(signal.SIGTERM, old[1])


def free_port(kind: int = socket.SOCK_STREAM) -> int:
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_tcp_dead_at_start_then_up() -> None:
    accepted = 0

    async def serve(_r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        nonlocal accepted
        accepted += 1
        w.close()

    gport, lport = free_port(), free_port()
    server = await asyncio.start_server(serve, "127.0.0.1", gport)
    flow = core.TcpFlow(lport, gport, core.LinkState(PROFILES["dead"]))
    await flow.start()
    try:
        _r, w = await asyncio.open_connection("127.0.0.1", lport)
        await asyncio.sleep(0.2)
        assert accepted == 0
        flow.state.set_profile(PROFILES["perfect"], 1)
        flow.sync()
        for _ in range(40):
            if accepted:
                break
            await asyncio.sleep(0.05)
        assert accepted == 1
        w.close()
    finally:
        flow.close()
        server.close()


async def test_finished_pairs_are_closed_and_forgotten() -> None:
    async def serve(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        await r.read()
        w.close()

    gport, lport = free_port(), free_port()
    server = await asyncio.start_server(serve, "127.0.0.1", gport)
    flow = core.TcpFlow(lport, gport, core.LinkState(PROFILES["perfect"]))
    await flow.start()
    try:
        r, w = await asyncio.open_connection("127.0.0.1", lport)
        await asyncio.sleep(0.1)
        assert len(flow._pairs) == 1
        w.write_eof()
        assert await asyncio.wait_for(r.read(), 2) == b""
        for _ in range(40):
            if not flow._pairs:
                break
            await asyncio.sleep(0.05)
        assert flow._pairs == []
        w.close()
    finally:
        flow.close()
        server.close()


async def test_udp_in_flight_dropped_after_down() -> None:
    state = core.LinkState(core.Profile("slow", 200, 0, 0))
    target = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target.bind(("127.0.0.1", 0))
    target.setblocking(False)
    relay = UdpRelay(
        "mavlink",
        free_port(socket.SOCK_DGRAM),
        target.getsockname()[1],
        policy=state.udp_policy("mavlink"),
        is_up=state.udp_up("mavlink"),
    )
    await relay.start()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.sendto(b"x", ("127.0.0.1", relay.listen_port))
        await asyncio.sleep(0.05)
        state.up["mavlink"] = False  # radio goes down while the datagram is in flight
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.get_running_loop().sock_recv(target, 10), 0.4)
        assert relay.counters[FROM_VEHICLE].dropped == 1
    finally:
        sender.close()
        target.close()
        relay.close()


def test_started_child_can_be_stopped_with_sigterm(tmp_path: Path) -> None:
    group = procs.ProcessGroup(tmp_path / "pids.json", tmp_path / "logs")
    proc = group.start(
        "sleeper", [sys.executable, "-c", "import time; time.sleep(60)"], env=procs.clean_env()
    )
    try:
        status = Path(f"/proc/{proc.pid}/status").read_text()
        blocked = int(
            next(ln for ln in status.splitlines() if ln.startswith("SigBlk")).split()[1], 16
        )
        assert not blocked & ((1 << (signal.SIGINT - 1)) | (1 << (signal.SIGTERM - 1)))
        start = time.monotonic()
        group.stop_all()
        assert time.monotonic() - start < procs.STOP_GRACE_S
        assert proc.popen.poll() is not None
    finally:
        proc.popen.kill()


def test_stop_during_start_is_deferred_until_recorded(tmp_path: Path) -> None:
    group = procs.ProcessGroup(tmp_path / "pids.json", tmp_path / "logs")
    real_spawn = group._spawn

    def spawn_then_signal(*args, **kwargs):
        proc = real_spawn(*args, **kwargs)
        assert procs.defer_stop()  # a stop arrives while the child is being started
        return proc

    group._spawn = spawn_then_signal  # type: ignore[method-assign]
    with pytest.raises(StopRequestedError):
        group.start("sleeper", [sys.executable, "-c", "import time; time.sleep(60)"], env={})
    assert len(group.procs) == 1  # recorded, so down() will stop it
    group.stop_all()
    assert not procs.defer_stop()
