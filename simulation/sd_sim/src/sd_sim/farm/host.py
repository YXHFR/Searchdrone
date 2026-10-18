"""Keeping the farm polite on the owner's computer (WSL2 with an NVIDIA card).

Before every round ``Gate.wait_turn()`` checks, and waits while any of these holds:

* the GPU is busy with something else (``busy_gpu_pct``, measured between rounds, when
  the farm itself renders nothing: a game, a video call) or too hot (``max_gpu_temp_c``);
* the CPU is busy (load per core above ``busy_cpu_load``);
* the farm has used more than its share of the time (``max_duty``: it rests after each
  round so that rounds take at most that share of the wall time);
* ``idle_only``: the computer has not been idle for ``idle_minutes`` (no keyboard or
  mouse in Windows, asked through ``powershell.exe`` from WSL; where that cannot be
  asked, a quiet GPU and CPU for that long count as idle).

Disk: ``disk_ok()`` is false when ``build/simfarm`` uses more than ``max_local_gb`` or
the disk has less than ``min_free_gb`` free; the loop then removes uploaded rounds,
oldest first, and pauses if that is not enough. Rounds also run under ``nice``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

NVIDIA_SMI = ("nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi")
GPU_QUERY = "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total"
IDLE_PS = (
    "Add-Type -TypeDefinition 'using System;using System.Runtime.InteropServices;"
    "public static class SdIdle{[StructLayout(LayoutKind.Sequential)]struct L{public uint s;"
    'public uint t;}[DllImport("user32.dll")]static extern bool GetLastInputInfo(ref L l);'
    "public static uint Ms(){L l=new L();l.s=(uint)Marshal.SizeOf(l);GetLastInputInfo(ref l);"
    "return (uint)Environment.TickCount-l.t;}}';[SdIdle]::Ms()"
)
CHECK_EVERY_S = 60.0


@dataclass(frozen=True)
class Limits:
    busy_gpu_pct: float = 40.0
    max_gpu_temp_c: float = 83.0
    busy_cpu_load: float = 0.8
    max_duty: float = 0.75
    max_local_gb: float = 50.0
    min_free_gb: float = 20.0
    idle_only: bool = False
    idle_minutes: float = 10.0
    nice: int = 10


@dataclass(frozen=True)
class GpuReading:
    util_pct: float
    temp_c: float
    mem_used_mb: float
    mem_total_mb: float


def _run(argv: list[str], timeout: float = 10.0) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv
            argv, capture_output=True, text=True, timeout=timeout, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def gpu(run: Callable[[list[str]], str | None] = _run) -> GpuReading | None:
    """The first GPU's load, or ``None`` without nvidia-smi."""
    for exe in NVIDIA_SMI:
        if shutil.which(exe) is None and not Path(exe).is_file():
            continue
        text = run([exe, GPU_QUERY, "--format=csv,noheader,nounits"])
        if not text:
            continue
        try:
            util, temp, used, total = (float(v) for v in text.splitlines()[0].split(","))
        except (ValueError, IndexError):
            return None
        return GpuReading(util, temp, used, total)
    return None


def cpu_load() -> float:
    """One-minute load average per core."""
    return os.getloadavg()[0] / (os.cpu_count() or 1)


def windows_idle_s(run: Callable[[list[str]], str | None] = _run) -> float | None:
    """Seconds since the last keyboard or mouse input in Windows (WSL), else ``None``."""
    if shutil.which("powershell.exe") is None:
        return None
    text = run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", IDLE_PS])
    try:
        return float((text or "").strip().splitlines()[-1]) / 1000.0
    except (ValueError, IndexError):
        return None


def dir_size_gb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            continue
    return total / 1e9


@dataclass
class Gate:
    limits: Limits
    data_dir: Path
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    read_gpu: Callable[[], GpuReading | None] = gpu
    read_cpu: Callable[[], float] = cpu_load
    read_idle: Callable[[], float | None] = windows_idle_s
    busy_s: float = 0.0  # time spent in rounds
    since: float | None = None  # when the farm started counting
    quiet_since: float | None = None  # when GPU and CPU last turned quiet

    def reason_to_wait(self) -> str | None:
        """Why a round should not start now, or ``None``."""
        now = self.clock()
        if self.since is None:
            self.since = now
        g = self.read_gpu()
        load = self.read_cpu()
        quiet = (g is None or g.util_pct < min(10.0, self.limits.busy_gpu_pct)) and load < min(
            0.3, self.limits.busy_cpu_load
        )
        if not quiet:
            self.quiet_since = None
        elif self.quiet_since is None:
            self.quiet_since = now
        if g is not None and g.util_pct >= self.limits.busy_gpu_pct:
            return f"GPU:n används av annat ({g.util_pct:.0f} %)"
        if g is not None and g.temp_c >= self.limits.max_gpu_temp_c:
            return f"GPU:n är varm ({g.temp_c:.0f} °C)"
        if load >= self.limits.busy_cpu_load:
            return f"processorn är upptagen (last {load:.2f} per kärna)"
        elapsed = now - self.since
        if elapsed > 0 and self.busy_s / elapsed > self.limits.max_duty:
            return "vilar (tak för andel av tiden)"
        if self.limits.idle_only:
            need = self.limits.idle_minutes * 60.0
            idle = self.read_idle()
            if idle is None:
                idle = 0.0 if self.quiet_since is None else now - self.quiet_since
            if idle < need:
                return "väntar på att datorn ska vara ledig"
        return None

    def round_took(self, seconds: float) -> None:
        self.busy_s += max(0.0, seconds)

    def disk_ok(self) -> str | None:
        """Why the disk is full, or ``None``."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(self.data_dir).free / 1e9
        if free_gb < self.limits.min_free_gb:
            return f"för lite ledigt på disken ({free_gb:.1f} GB)"
        used = dir_size_gb(self.data_dir)
        if used > self.limits.max_local_gb:
            return f"build/simfarm använder {used:.1f} GB (tak {self.limits.max_local_gb:g} GB)"
        return None
