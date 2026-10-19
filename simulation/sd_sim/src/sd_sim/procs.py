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

from sd_sim.errors import StopRequestedError

STOP_GRACE_S = 5.0
_starting = 0
_stop_deferred = False


def defer_stop() -> bool:
    """Called by a stop-signal handler: True if a child is being started right now, in
    which case the stop is raised as soon as the child is recorded."""
    global _stop_deferred
    if _starting:
        _stop_deferred = True
        return True
    return False


BASE_ENV_KEYS = ("PATH", "HOME", "USER", "LOGNAME", "TMPDIR")
# Display variables a Gazebo window needs under WSLg.
DISPLAY_ENV_KEYS = ("DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "PULSE_SERVER")
# Mesa's choice of GPU. The headless server renders the camera through Mesa too: without
# these it falls back to llvmpipe on the processor and runs well below real time.
GPU_ENV_KEYS = (
    "LIBGL_ALWAYS_SOFTWARE",
    "MESA_D3D12_DEFAULT_ADAPTER_NAME",
    "GALLIUM_DRIVER",
    "LD_LIBRARY_PATH",
)
GRAPHICS_ENV_KEYS = DISPLAY_ENV_KEYS + GPU_ENV_KEYS


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
        # A stop must not land between starting the child and recording it, or the child
        # would be left running unseen. The stop is deferred (not blocked: a blocked mask
        # would be inherited by the child, which could then not be stopped with SIGTERM).
        global _starting, _stop_deferred
        _starting += 1
        try:
            proc = self._spawn(name, argv, env=env, cwd=cwd, log=log)
        finally:
            _starting -= 1
        if _stop_deferred and not _starting:
            _stop_deferred = False
            raise StopRequestedError
        return proc

    def _spawn(
        self, name: str, argv: list[str], *, env: dict[str, str], cwd: Path | None, log: Path
    ) -> Proc:
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
            "launcher": identity(os.getpid()),
            "procs": [{"name": p.name, **identity(p.pid)} for p in self.procs if p.alive()],
        }
        self.pids_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def owned_pids(self) -> set[int]:
        """Our processes and everything in their process groups (e.g. Gazebo's servers)."""
        leaders = {p.pid for p in self.procs if p.alive()}
        owned = set(leaders)
        for stat in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = (
                    stat.read_text(encoding="utf-8", errors="replace").rsplit(")", 1)[-1].split()
                )
            except OSError:
                continue
            if fields and int(fields[2]) in leaders:  # field 5: process group id
                owned.add(int(stat.parent.name))
        return owned

    def kill(self, name: str) -> int:
        """A fault: SIGKILL ``name`` and its process group at once (no goodbye). Returns
        its place in the start order, so a restart can take the same place."""
        idx = next(i for i, p in enumerate(self.procs) if p.name == name)
        proc = self.procs[idx]
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.popen.wait(timeout=5)
        return idx

    def restart(
        self, name: str, argv: list[str], *, env: dict[str, str], cwd: Path | None = None
    ) -> Proc:
        """Start ``name`` again in its old place in the start (and stop) order."""
        idx = next(i for i, p in enumerate(self.procs) if p.name == name)
        if self.procs[idx].alive():
            self.kill(name)
        proc = self.start(name, argv, env=env, cwd=cwd)
        self.procs.remove(proc)
        self.procs[idx] = proc
        self._save()
        return proc

    def stop_all(self) -> None:
        """Stop in reverse start order: SIGTERM, then SIGKILL after the grace time."""
        for proc in reversed(self.procs):
            stop_pid(proc.pid, proc.popen)
        self.procs.clear()
        self.pids_file.unlink(missing_ok=True)


def _cmdline(pid: int) -> str:
    """The full command line (all arguments), or "" when the process is gone."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return " ".join(part.decode(errors="replace") for part in raw.split(b"\0") if part)


def _start_time(pid: int) -> str:
    """Start time in clock ticks since boot (/proc/<pid>/stat field 22)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    # The command name (field 2) may contain spaces; the fields after it do not.
    return stat.rsplit(")", 1)[-1].split()[19]


def identity(pid: int) -> dict[str, Any]:
    return {"pid": pid, "cmd": _cmdline(pid), "start": _start_time(pid)}


def _proc_table() -> dict[int, tuple[int, int, str, str]]:
    """Every process as pid -> (parent pid, process group, start time, state)."""
    table: dict[int, tuple[int, int, str, str]] = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text(encoding="utf-8", errors="replace").rsplit(")", 1)[-1].split()
            table[int(stat.parent.name)] = (int(fields[1]), int(fields[2]), fields[19], fields[0])
        except (OSError, ValueError, IndexError):
            continue
    return table


def _descendants(pid: int) -> dict[int, str]:
    """Children, grandchildren and so on of ``pid`` as pid -> start time."""
    table = _proc_table()
    children: dict[int, list[int]] = {}
    for child, (ppid, _, _, _) in table.items():
        children.setdefault(ppid, []).append(child)
    found: dict[int, str] = {}
    todo = list(children.get(pid, []))
    while todo:
        child = todo.pop()
        if child not in found:
            found[child] = table[child][2]
            todo.extend(children.get(child, []))
    return found


def _stop_descendants(known: dict[int, str]) -> None:
    """SIGTERM, then SIGKILL, what is left of ``known`` (found before the stop).

    ``gz sim`` with a window forks a server and a GUI that each move to their own process
    group, ignore SIGTERM and outlive their parent; stopping the parent's group alone
    leaves them running. Only processes that are still the same ones (same start time)
    are signalled; a descendant that leads its own group gets the whole group.
    """

    def left() -> dict[int, int]:
        table = _proc_table()
        return {
            pid: row[1]
            for pid, start in known.items()
            if (row := table.get(pid)) and row[2] == start and row[3] != "Z"
        }

    def send(sig: int, alive: dict[int, int]) -> None:
        for pid, pgid in alive.items():
            with contextlib.suppress(ProcessLookupError, PermissionError):
                if pgid in alive and alive[pgid] == pgid:
                    os.killpg(pgid, sig)
                else:
                    os.kill(pid, sig)

    alive = left()
    if not alive:
        return
    send(signal.SIGTERM, alive)
    end = time.monotonic() + STOP_GRACE_S
    while time.monotonic() < end:
        if not (alive := left()):
            return
        time.sleep(0.1)
    send(signal.SIGKILL, left())


def stop_pid(pid: int, popen: subprocess.Popen[bytes] | None = None) -> None:
    """SIGTERM, then SIGKILL. Signals the whole process group only when ``pid`` leads
    its own group (every process we start does); otherwise just that pid. Descendants
    that left the group are stopped afterwards (see ``_stop_descendants``)."""
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    known = _descendants(pid)

    def send(sig: int) -> None:
        with contextlib.suppress(ProcessLookupError):
            if pgid == pid:
                os.killpg(pgid, sig)
            else:
                os.kill(pid, sig)

    def gone() -> bool:
        if popen is not None:
            return popen.poll() is not None
        return not Path(f"/proc/{pid}").exists()

    send(signal.SIGTERM)
    end = time.monotonic() + STOP_GRACE_S
    while time.monotonic() < end and not gone():
        time.sleep(0.1)
    if not gone():
        known.update(_descendants(pid))
        send(signal.SIGKILL)
        if popen is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                popen.wait(timeout=2)
    _stop_descendants(known)


def read_recorded(pids_file: Path) -> dict[str, Any]:
    try:
        return json.loads(pids_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def still_same(row: dict[str, Any]) -> bool:
    """Is the recorded process still the same one (pids are reused after a reboot)?"""
    try:
        pid = int(row["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    cmd, start = str(row.get("cmd", "")), str(row.get("start", ""))
    return bool(cmd and start) and _cmdline(pid) == cmd and _start_time(pid) == start


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
    if launcher and still_same(launcher):
        with contextlib.suppress(ProcessLookupError):
            os.kill(int(launcher["pid"]), signal.SIGTERM)
        end = time.monotonic() + wait_s
        while time.monotonic() < end and pids_file.exists():
            time.sleep(0.2)
    for row in reversed(read_recorded(pids_file).get("procs", [])):
        if still_same(row):
            stop_pid(int(row["pid"]))
    pids_file.unlink(missing_ok=True)
    return names
