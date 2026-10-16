"""Start and stop the simulator's processes with a whitelisted environment.

No process gets the whole environment (or ``.env``): each one gets ``PATH``, ``HOME``,
``LANG`` and only the variables it needs. Output goes to ``runs/<run>/logs/<name>.log``.
Started processes are recorded in ``build/sim/i<N>/pids.json`` so ``just stop`` can find
them again.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STOP_GRACE_S = 5.0
BASE_ENV_KEYS = ("PATH", "HOME", "USER", "LOGNAME", "TMPDIR")
# Display and graphics variables a Gazebo window needs under WSLg.
GRAPHICS_ENV_KEYS = (
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "XDG_RUNTIME_DIR",
    "PULSE_SERVER",
    "LIBGL_ALWAYS_SOFTWARE",
    "MESA_D3D12_DEFAULT_ADAPTER_NAME",
    "GALLIUM_DRIVER",
    "LD_LIBRARY_PATH",
)


def clean_env(
    extra: dict[str, str] | None = None, passthrough: tuple[str, ...] = ()
) -> dict[str, str]:
    env = {k: os.environ[k] for k in (*BASE_ENV_KEYS, *passthrough) if k in os.environ}
    env["LANG"] = "C.UTF-8"
    env.update(extra or {})
    return env


@dataclass
class Proc:
    name: str
    popen: subprocess.Popen[bytes]
    log: Path

    @property
    def pid(self) -> int:
        return self.popen.pid

    def alive(self) -> bool:
        return self.popen.poll() is None


@dataclass
class ProcessGroup:
    pids_file: Path
    logs_dir: Path
    procs: list[Proc] = field(default_factory=list)

    def start(
        self,
        name: str,
        argv: list[str],
        *,
        env: dict[str, str],
        cwd: Path | None = None,
    ) -> Proc:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log = self.logs_dir / f"{name}.log"
        with log.open("ab") as out:
            popen = subprocess.Popen(  # noqa: S603 - argv is built by the launcher, never a shell
                argv,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=cwd,
                start_new_session=True,
            )
        proc = Proc(name, popen, log)
        self.procs.append(proc)
        self._save()
        return proc

    def _save(self) -> None:
        self.pids_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "launcher": {"pid": os.getpid(), "cmd": _cmdline(os.getpid())},
            "procs": [
                {"name": p.name, "pid": p.pid, "cmd": _cmdline(p.pid)}
                for p in self.procs
                if p.alive()
            ],
        }
        self.pids_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def stop_all(self) -> None:
        """Stop in reverse start order: SIGTERM, then SIGKILL after the grace time."""
        for proc in reversed(self.procs):
            stop_pid(proc.pid, proc.popen)
        self.procs.clear()
        self.pids_file.unlink(missing_ok=True)


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[0].decode(errors="replace")
    except OSError:
        return ""


def stop_pid(pid: int, popen: subprocess.Popen[bytes] | None = None) -> None:
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGTERM)
    end = time.monotonic() + STOP_GRACE_S
    while time.monotonic() < end:
        if popen is not None:
            if popen.poll() is not None:
                return
        elif not Path(f"/proc/{pid}").exists():
            return
        time.sleep(0.1)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)
    if popen is not None:
        with contextlib.suppress(subprocess.TimeoutExpired):
            popen.wait(timeout=2)


def read_recorded(pids_file: Path) -> dict[str, Any]:
    try:
        return json.loads(pids_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def _still_same(row: dict[str, Any]) -> bool:
    cmd = str(row.get("cmd", ""))
    return bool(cmd) and _cmdline(int(row["pid"])) == cmd


def stop_recorded(pids_file: Path, wait_s: float = 30.0) -> list[str]:
    """Stop what an earlier ``up`` started.

    The running launcher is asked to stop (SIGTERM) so it shuts its processes down in
    the right order. If it is gone, the recorded processes are stopped directly, but
    only those whose command line still matches (a pid can be reused).
    """
    data = read_recorded(pids_file)
    if not data:
        return []
    names = [str(row["name"]) for row in data.get("procs", [])]
    launcher = data.get("launcher")
    if launcher and _still_same(launcher):
        with contextlib.suppress(ProcessLookupError):
            os.kill(int(launcher["pid"]), signal.SIGTERM)
        end = time.monotonic() + wait_s
        while time.monotonic() < end and pids_file.exists():
            time.sleep(0.2)
    for row in reversed(read_recorded(pids_file).get("procs", [])):
        if _still_same(row):
            stop_pid(int(row["pid"]))
    pids_file.unlink(missing_ok=True)
    return names
