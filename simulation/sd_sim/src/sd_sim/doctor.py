"""Check the simulator environment: ``python -m sd_sim.doctor [physics=gz|builtin]``.

Prints one row per check with OK, VARNING or FEL and what to do. Exits 0 when no row
is FEL. With ``physics=builtin`` the Gazebo rows are skipped (CI image, cloud).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sd_sim import _args, config, netcheck, paths, ports
from sd_sim.errors import InfraError
from sd_sim.launch import Stack, new_run_dir, say, tcp_port_taken, udp_port_taken

OK, WARN, FAIL, SKIP = "OK", "VARNING", "FEL", "-"
MIN_FREE_RAM_MB = 6000


@dataclass
class Row:
    name: str
    status: str
    detail: str = ""


def _run(argv: list[str], timeout: float = 20) -> subprocess.CompletedProcess[str] | None:
    exe = shutil.which(argv[0])
    if exe is None:
        return None
    try:
        return subprocess.run(  # noqa: S603 - fixed programs
            [exe, *argv[1:]], capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return None


def check_ubuntu() -> Row:
    text = (
        Path("/etc/os-release").read_text(encoding="utf-8")
        if Path("/etc/os-release").exists()
        else ""
    )
    ok = 'VERSION_ID="24.04"' in text
    return Row("Ubuntu-version", OK if ok else FAIL, "" if ok else "Kräver Ubuntu 24.04.")


def check_repo_location() -> Row:
    repo = str(paths.REPO)
    ok = repo.startswith(str(Path.home())) and not repo.startswith("/mnt/")
    hint = "" if ok else "Klona repot inuti Linux (~/Searchdrone), inte under /mnt/c."
    return Row("Repot ligger under $HOME", OK if ok else FAIL, hint)


def check_glx() -> Row:
    res = _run(["glxinfo", "-B"])
    if res is None:
        return Row("Grafik (glxinfo -B)", WARN, "glxinfo saknas (paketet mesa-utils).")
    renderer = next((ln for ln in res.stdout.splitlines() if "OpenGL renderer" in ln), "")
    if "NVIDIA" in renderer:
        return Row("Grafik (glxinfo -B)", OK, renderer.split(":", 1)[-1].strip())
    return Row(
        "Grafik (glxinfo -B)",
        WARN,
        "Inget NVIDIA-kort syns, Gazebo ritar med processorn. Prova "
        "MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA eller gui=false.",
    )


def check_ardupilot() -> Row:
    tag = str(paths.versions()["ardupilot_tag"])
    marker = paths.ardupilot_home() / ".sd_version"
    found = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
    if found != tag or not paths.arducopter_bin().is_file():
        return Row(
            "ArduPilot SITL",
            FAIL,
            f"Väntade {tag}, hittade {found or 'inget'}. Kör just sim-setup.",
        )
    return Row("ArduPilot SITL", OK, tag)


def check_gz_version() -> Row:
    res = _run(["gz", "sim", "--version"])
    if res is None or not re.search(r"version 8\.", res.stdout + res.stderr):
        return Row("Gazebo Harmonic (gz sim 8)", FAIL, "Kör just sim-setup.")
    return Row("Gazebo Harmonic (gz sim 8)", OK)


def check_gz_plugin() -> Row:
    dirs = [d for d in os.environ.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "").split(":") if d]
    ok = any((Path(d) / "libArduPilotPlugin.so").is_file() for d in dirs)
    hint = "" if ok else "Öppna en ny terminal (läser ~/.searchdrone_env) eller kör just sim-setup."
    return Row("ArduPilots Gazebo-plugin", OK if ok else FAIL, hint)


def check_gz_ip() -> Row:
    ok = os.environ.get("GZ_IP") == "127.0.0.1"
    return Row("GZ_IP = 127.0.0.1", OK if ok else FAIL, "" if ok else "Öppna en ny terminal.")


def check_docker() -> Row:
    res = _run(["docker", "info"])
    ok = res is not None and res.returncode == 0
    hint = "" if ok else "Kör wsl --shutdown i PowerShell och öppna Ubuntu igen (docker-gruppen)."
    return Row("docker utan sudo", OK if ok else FAIL, hint)


def check_ports_free(instance: int = 0) -> Row:
    busy: list[int] = []
    for name, port in ports.for_instance(instance).items():
        if name.startswith(("postgres", "mediamtx", "llm")):
            continue  # shared services may already run
        if udp_port_taken(port) or tcp_port_taken(port):
            busy.append(port)
    if busy:
        return Row("Fria portar (instans 0)", FAIL, f"Upptagna: {busy}. Kör just stop.")
    return Row("Fria portar (instans 0)", OK)


def check_ram() -> Row:
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        avail_kb = int(re.search(r"MemAvailable:\s+(\d+)", meminfo).group(1))  # type: ignore[union-attr]
    except (OSError, AttributeError, ValueError):
        return Row("Ledigt minne", WARN, "Kunde inte läsa /proc/meminfo.")
    mb = avail_kb // 1024
    if mb >= MIN_FREE_RAM_MB:
        return Row("Ledigt minne", OK, f"{mb} MB")
    return Row("Ledigt minne", WARN, f"{mb} MB. Stäng program eller kör med gui=false.")


def check_loopback() -> Row:
    """Start a builtin instance and check every listening simulator port."""
    if shutil.which("ss") is None:
        return Row("Bara loopback lyssnar", FAIL, "ss saknas (installera iproute2).")
    cfg = config.load({"physics": "builtin", "video": "off"})
    cfg.vehicle = "stock"
    stack = Stack(cfg, new_run_dir(0, "doctor"))
    try:
        stack.up()
        bad = netcheck.violations(
            netcheck.parse_ss(netcheck.run_ss()), stack.ports, netcheck.load_allowlist()
        )
    except InfraError as exc:
        return Row("Bara loopback lyssnar", FAIL, str(exc).splitlines()[0])
    finally:
        stack.down()
    if bad:
        return Row("Bara loopback lyssnar", FAIL, "; ".join(str(b) for b in bad))
    return Row("Bara loopback lyssnar", OK)


def run_checks(physics: str) -> list[Row]:
    checks: list[Callable[[], Row]] = [check_ubuntu, check_repo_location, check_ardupilot]
    gz_checks = [check_glx, check_gz_version, check_gz_plugin, check_gz_ip]
    if physics == "gz":
        checks += gz_checks
    checks += [check_docker, check_ports_free, check_ram]
    rows = [check() for check in checks]
    if physics != "gz":
        rows += [
            Row(name, SKIP, "physics=builtin")
            for name in (
                "Grafik (glxinfo -B)",
                "Gazebo Harmonic (gz sim 8)",
                "ArduPilots Gazebo-plugin",
                "GZ_IP = 127.0.0.1",
            )
        ]
    sitl_ok = next(r for r in rows if r.name == "ArduPilot SITL").status == OK
    ports_ok = next(r for r in rows if r.name.startswith("Fria portar")).status == OK
    if sitl_ok and ports_ok:
        rows.append(check_loopback())
    else:
        rows.append(Row("Bara loopback lyssnar", FAIL, "Kräver SITL och fria portar."))
    return rows


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    values = _args.parse(argv, ("physics",), "sd_sim.doctor")
    physics = values.get("physics", "gz")
    if physics not in ("gz", "builtin"):
        _args.fail("sd_sim.doctor: physics= ska vara gz eller builtin")
    rows = run_checks(physics)
    width = max(len(r.name) for r in rows)
    for row in rows:
        say(f"{row.name:<{width}}  {row.status:<8} {row.detail}")
    failed = [r for r in rows if r.status == FAIL]
    say("Allt OK." if not failed else f"{len(failed)} kontroll(er) FEL.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
