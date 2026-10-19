"""Background samplers for the perf run: WSL RAM, VRAM, Gazebo real-time factor and
free Windows memory.

Every sampler has ``start()`` and ``stop()``. ``stop()`` can be called more than once,
never raises and stops the child processes it started. A sampler whose program is
missing (no ``nvidia-smi``, ``gz`` or ``powershell.exe``) has ``available`` False and
just returns no samples, so the report can say what was not measured. The parse
functions are pure and tested on their own.
"""

from __future__ import annotations

import contextlib
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import IO

from sd_sim import procs
from sd_sim.gazebo import gz_env

_RTF = re.compile(r"^\s*real_time_factor:\s*([-+0-9.eE]+)\s*$")
_BOM = "﻿"
# powershell.exe is reached through the Windows mount, which is not always on PATH.
_POWERSHELL_FALLBACK = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
WATCH_MAX_S = 14400
WATCH_EXIT_WAIT_S = 20.0
_QUERY_TIMEOUT_S = 15.0


def _kill(popen: subprocess.Popen[bytes] | None) -> None:
    if popen is None or popen.poll() is not None:
        return
    with contextlib.suppress(Exception):
        procs.stop_pid(popen.pid, popen)


# --- WSL RAM -----------------------------------------------------------------------


def _meminfo_kb(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[key.strip()] = int(parts[0])
    return out


def parse_meminfo(text: str) -> int:
    """Used memory in MB: ``(MemTotal - MemAvailable) // 1024`` from /proc/meminfo."""
    kb = _meminfo_kb(text)
    if "MemTotal" not in kb or "MemAvailable" not in kb:
        raise ValueError("MemTotal eller MemAvailable saknas i meminfo")
    return (kb["MemTotal"] - kb["MemAvailable"]) // 1024


class RamSampler:
    """Reads /proc/meminfo every ``interval_s`` seconds in a thread (first at start)."""

    def __init__(self, interval_s: float = 5.0, meminfo: Path = Path("/proc/meminfo")) -> None:
        self.interval_s = interval_s
        self.meminfo = meminfo
        self._samples: list[tuple[float, int]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return self.meminfo.exists()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="perf-ram", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            with contextlib.suppress(OSError, ValueError):
                used = parse_meminfo(self.meminfo.read_text(encoding="utf-8"))
                with self._lock:
                    self._samples.append((time.monotonic(), used))
            if self._stop.wait(self.interval_s):
                return

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 5)

    @property
    def samples(self) -> list[tuple[float, int]]:
        with self._lock:
            return list(self._samples)


# --- VRAM --------------------------------------------------------------------------


def parse_nvidia_csv(text: str) -> list[int]:
    """``memory.used`` values from ``nvidia-smi --format=csv,noheader,nounits`` output.
    Lines that are not a plain number (warnings, a cut-off last line) are skipped."""
    out: list[int] = []
    for line in text.splitlines():
        value = line.strip().removeprefix(_BOM)
        if value.isdigit():
            out.append(int(value))
    return out


class VramSampler:
    """``nvidia-smi -lms 500`` writing one ``memory.used`` line per sample to ``log``."""

    def __init__(self, log: Path) -> None:
        self.log = log
        self._exe = shutil.which("nvidia-smi")
        self._popen: subprocess.Popen[bytes] | None = None

    @property
    def available(self) -> bool:
        return self._exe is not None

    def start(self) -> None:
        if self._exe is None or self._popen is not None:
            return
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("ab") as out:
            self._popen = subprocess.Popen(  # noqa: S603 - fixed program and arguments
                [
                    self._exe,
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-lms",
                    "500",
                ],
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

    def stop(self) -> None:
        _kill(self._popen)

    @property
    def samples(self) -> list[int]:
        try:
            return parse_nvidia_csv(self.log.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            return []


# --- Gazebo real-time factor -------------------------------------------------------


def parse_rtf_line(line: str) -> float | None:
    """The value of a ``real_time_factor: 0.99`` line of ``gz topic -e`` text output."""
    match = _RTF.match(line)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


class GzStatsSampler:
    """Echoes ``/world/<world>/stats`` and stamps each real-time factor with
    ``time.monotonic()`` when its line arrives. The raw output goes to ``log``."""

    def __init__(self, world: str, instance: int, log: Path) -> None:
        self.world = world
        self.instance = instance
        self.log = log
        self._exe = shutil.which("gz")
        self._popen: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._samples: list[tuple[float, float]] = []
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._exe is not None

    def start(self) -> None:
        if self._exe is None or self._popen is not None:
            return
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self._popen = subprocess.Popen(  # noqa: S603 - fixed program, topic built here
            [self._exe, "topic", "-e", "-t", f"/world/{self.world}/stats"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=gz_env(self.instance, gui=False),
            start_new_session=True,
        )
        stream = self._popen.stdout
        if stream is None:
            return
        self._thread = threading.Thread(
            target=self._read, args=(stream,), name="perf-gz-stats", daemon=True
        )
        self._thread.start()

    def _read(self, stream: IO[bytes]) -> None:
        with contextlib.suppress(OSError, ValueError), self.log.open("ab") as log:
            for raw in iter(stream.readline, b""):
                now = time.monotonic()
                log.write(raw)
                value = parse_rtf_line(raw.decode("utf-8", errors="replace"))
                if value is not None:
                    with self._lock:
                        self._samples.append((now, value))

    def stop(self) -> None:
        _kill(self._popen)
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def samples(self) -> list[tuple[float, float]]:
        with self._lock:
            return list(self._samples)

    def between(self, t_from: float, t_to: float) -> list[tuple[float, float]]:
        return [(t, v) for t, v in self.samples if t_from <= t <= t_to]


# --- Windows memory ----------------------------------------------------------------


def parse_watch_csv(text: str) -> list[tuple[str, int]]:
    """``(utc, available_mb)`` rows of the ``windows-check.ps1 -Watch`` CSV.

    The file is UTF-8 with a byte order mark and CRLF line ends (PowerShell 5). The
    header ``utc,available_mb,gpu_used_mb`` and rows without a number are skipped.
    """
    out: list[tuple[str, int]] = []
    for line in text.replace("\r", "").split("\n"):
        cells = [c.strip() for c in line.replace(_BOM, "").split(",")]
        if len(cells) < 2 or not cells[0] or cells[0] == "utc":
            continue
        try:
            out.append((cells[0], int(float(cells[1]))))
        except ValueError:
            continue
    return out


def _powershell() -> str | None:
    exe = shutil.which("powershell.exe")
    if exe is not None:
        return exe
    return str(_POWERSHELL_FALLBACK) if _POWERSHELL_FALLBACK.exists() else None


def _windows_path(path: Path) -> str | None:
    exe = shutil.which("wslpath")
    if exe is None:
        return None
    try:
        res = subprocess.run(  # noqa: S603 - fixed program
            [exe, "-w", str(path)],
            capture_output=True,
            text=True,
            timeout=_QUERY_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = res.stdout.strip()
    return value if res.returncode == 0 and value else None


class WindowsWatch:
    """Runs ``windows-check.ps1 -Watch`` on the Windows side, which appends free memory
    to ``csv`` every few seconds. ``stop()`` creates ``stop_file`` so the script ends by
    itself, and kills it if it has not ended after 20 seconds."""

    def __init__(self, csv: Path, stop_file: Path, script: Path) -> None:
        self.csv = csv
        self.stop_file = stop_file
        self.script = script
        self._powershell = _powershell()
        self._popen: subprocess.Popen[bytes] | None = None
        self.error = ""

    @property
    def available(self) -> bool:
        return self._powershell is not None and shutil.which("wslpath") is not None

    def start(self) -> None:
        if self._popen is not None:
            return
        if self._powershell is None or not self.available:
            self.error = "powershell.exe eller wslpath saknas"
            return
        self.csv.parent.mkdir(parents=True, exist_ok=True)
        self.stop_file.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            self.stop_file.unlink()
        paths = [_windows_path(p) for p in (self.script, self.csv, self.stop_file)]
        script, csv, stop_file = paths
        if script is None or csv is None or stop_file is None:
            self.error = "kunde inte översätta sökvägarna till Windows-sökvägar"
            return
        argv = [
            self._powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            script,
            "-Watch",
            "-OutFile",
            csv,
            "-StopFile",
            stop_file,
            "-MaxSeconds",
            str(WATCH_MAX_S),
        ]
        try:
            self._popen = subprocess.Popen(  # noqa: S603 - fixed program and script
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            self.error = f"powershell.exe startade inte: {exc.strerror or exc}"

    def stop(self) -> None:
        popen = self._popen
        if popen is None or popen.poll() is not None:
            return
        with contextlib.suppress(OSError):
            self.stop_file.touch()
        with contextlib.suppress(subprocess.TimeoutExpired):
            popen.wait(timeout=WATCH_EXIT_WAIT_S)
        _kill(popen)

    @property
    def available_mb(self) -> list[int]:
        try:
            text = self.csv.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return [mb for _, mb in parse_watch_csv(text)]


# --- Machine facts -----------------------------------------------------------------


def _query(argv: list[str]) -> str:
    try:
        res = subprocess.run(  # noqa: S603 - fixed programs
            argv, capture_output=True, text=True, timeout=_QUERY_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return res.stdout if res.returncode == 0 else ""


def gpu_name() -> str:
    """The first GPU's name from ``nvidia-smi``, "" when it cannot be read."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return ""
    out = _query([exe, "--query-gpu=name", "--format=csv,noheader"])
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    return lines[0] if lines else ""


def ram_total_gb(meminfo: Path = Path("/proc/meminfo")) -> float:
    """WSL's MemTotal in GiB with one decimal (0.0 when it cannot be read)."""
    try:
        kb = _meminfo_kb(meminfo.read_text(encoding="utf-8"))
    except OSError:
        return 0.0
    return round(kb.get("MemTotal", 0) / 1024**2, 1)


def windows_ram_total_gb() -> float | None:
    """The Windows computer's physical memory in GiB with one decimal, None when
    PowerShell is not reachable."""
    exe = _powershell()
    if exe is None:
        return None
    out = _query(
        [
            exe,
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
        ]
    )
    digits = out.strip().removeprefix(_BOM)
    if not digits.isdigit():
        return None
    return round(int(digits) / 1024**3, 1)
