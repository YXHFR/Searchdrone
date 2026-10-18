import time
from pathlib import Path

import pytest

from sd_sim.perf import samplers
from sd_sim.perf.samplers import (
    GzStatsSampler,
    RamSampler,
    VramSampler,
    WindowsWatch,
    parse_meminfo,
    parse_nvidia_csv,
    parse_rtf_line,
    parse_watch_csv,
)

MEMINFO = """MemTotal:       16303460 kB
MemFree:         9000000 kB
MemAvailable:   12207364 kB
Buffers:          123456 kB
HugePages_Total:       0
"""


def test_parse_meminfo() -> None:
    assert parse_meminfo(MEMINFO) == (16303460 - 12207364) // 1024
    with pytest.raises(ValueError, match="MemAvailable"):
        parse_meminfo("MemTotal: 100 kB\n")


def test_parse_nvidia_csv_skips_junk() -> None:
    text = "1234\n 2048 \nNVIDIA-SMI has failed\n\n[N/A]\n7\n30"
    assert parse_nvidia_csv(text) == [1234, 2048, 7, 30]
    assert parse_nvidia_csv("") == []


def test_parse_rtf_line() -> None:
    assert parse_rtf_line("real_time_factor: 0.99876") == pytest.approx(0.99876)
    assert parse_rtf_line("  real_time_factor: 1\n") == 1.0
    assert parse_rtf_line("real_time_factor: 9.5e-01") == pytest.approx(0.95)
    assert parse_rtf_line("sim_time {") is None
    assert parse_rtf_line("real_time_factor: abc") is None
    assert parse_rtf_line("iterations: 1234") is None


def test_parse_watch_csv_bom_and_crlf() -> None:
    raw = (
        "﻿utc,available_mb,gpu_used_mb\r\n"
        "2026-10-18T10:00:00.0000000Z,6123,2100\r\n"
        "﻿2026-10-18T10:00:05.0000000Z,5900,\r\n"
        "broken line\r\n"
        "2026-10-18T10:00:10.0000000Z,n/a,2000\r\n"
        "2026-10-18T10:00:15.0000000Z,5800,2000"
    )
    assert parse_watch_csv(raw) == [
        ("2026-10-18T10:00:00.0000000Z", 6123),
        ("2026-10-18T10:00:05.0000000Z", 5900),
        ("2026-10-18T10:00:15.0000000Z", 5800),
    ]
    # Decoded with utf-8 the BOM survives as U+FEFF; decoded with utf-8-sig it does not.
    data = raw.encode("utf-8")
    assert parse_watch_csv(data.decode("utf-8-sig")) == parse_watch_csv(data.decode("utf-8"))
    assert parse_watch_csv("") == []


def test_ram_sampler_reads_file(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(MEMINFO, encoding="utf-8")
    sampler = RamSampler(interval_s=0.05, meminfo=meminfo)
    sampler.start()
    time.sleep(0.2)
    sampler.stop()
    sampler.stop()  # idempotent
    got = sampler.samples
    assert len(got) >= 2
    assert all(mb == (16303460 - 12207364) // 1024 for _, mb in got)
    assert [t for t, _ in got] == sorted(t for t, _ in got)


def test_ram_sampler_missing_file(tmp_path: Path) -> None:
    sampler = RamSampler(interval_s=0.05, meminfo=tmp_path / "nope")
    assert not sampler.available
    sampler.start()
    time.sleep(0.1)
    sampler.stop()
    assert sampler.samples == []


def test_stop_before_start_never_raises(tmp_path: Path) -> None:
    for s in (
        RamSampler(meminfo=tmp_path / "m"),
        VramSampler(tmp_path / "v.log"),
        GzStatsSampler("w", 0, tmp_path / "gz.log"),
        WindowsWatch(tmp_path / "w.csv", tmp_path / "stop", tmp_path / "s.ps1"),
    ):
        s.stop()
        s.stop()


def test_samplers_without_binaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr(samplers, "_POWERSHELL_FALLBACK", tmp_path / "no-powershell.exe")
    vram = VramSampler(tmp_path / "v.log")
    gz = GzStatsSampler("w", 0, tmp_path / "gz.log")
    watch = WindowsWatch(tmp_path / "w.csv", tmp_path / "stop", tmp_path / "s.ps1")
    for s in (vram, gz, watch):
        assert not s.available
        s.start()
        s.stop()
    assert vram.samples == [] and gz.samples == [] and watch.available_mb == []
    assert watch.error
    assert samplers.gpu_name() == ""
    assert samplers.windows_ram_total_gb() is None


def fake_bin(folder: Path, name: str, body: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_vram_sampler_with_fake_nvidia_smi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = tmp_path / "bin"
    fake_bin(bin_dir, "nvidia-smi", "while true; do echo 3100; sleep 0.02; done\n")
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    sampler = VramSampler(tmp_path / "logs" / "vram.log")
    assert sampler.available
    sampler.start()
    time.sleep(0.3)
    sampler.stop()
    sampler.stop()
    got = sampler.samples
    assert got and set(got) == {3100}


def test_gz_stats_sampler_with_fake_gz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = tmp_path / "bin"
    fake_bin(
        bin_dir,
        "gz",
        'echo "$GZ_PARTITION $*" > "$(dirname "$0")/argv"\n'
        "while true; do\n"
        '  echo "sim_time {"; echo "  sec: 1"; echo "}"; echo "real_time_factor: 0.97"\n'
        "  sleep 0.02\n"
        "done\n",
    )
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    sampler = GzStatsSampler("sd_lake_forest", 3, tmp_path / "gz.log")
    assert sampler.available
    t0 = time.monotonic()
    sampler.start()
    time.sleep(0.3)
    sampler.stop()
    t1 = time.monotonic()
    got = sampler.samples
    assert got and all(v == pytest.approx(0.97) for _, v in got)
    assert all(t0 <= t <= t1 for t, _ in got)
    assert sampler.between(t0, t1) == got
    assert sampler.between(t1 + 1, t1 + 2) == []
    assert (bin_dir / "argv").read_text().split() == [
        "sd_i3",
        "topic",
        "-e",
        "-t",
        "/world/sd_lake_forest/stats",
    ]
    assert "real_time_factor" in (tmp_path / "gz.log").read_text()


def test_windows_watch_stop_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fake powershell.exe that writes the CSV until the stop file appears."""
    bin_dir = tmp_path / "bin"
    fake_bin(bin_dir, "wslpath", 'echo "$2"\n')
    fake_bin(
        bin_dir,
        "powershell.exe",
        'echo "$*" > "$(dirname "$0")/argv"\n'
        'out=""; stop=""\n'
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in -OutFile) out="$2";; -StopFile) stop="$2";; esac; shift\n'
        "done\n"
        "printf '\\357\\273\\277utc,available_mb,gpu_used_mb\\r\\n' > \"$out\"\n"
        'while [ ! -e "$stop" ]; do\n'
        "  printf '2026-10-18T10:00:00Z,4321,100\\r\\n' >> \"$out\"; sleep 0.02\n"
        "done\n",
    )
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    watch = WindowsWatch(tmp_path / "w.csv", tmp_path / "stop", tmp_path / "check.ps1")
    assert watch.available
    watch.start()
    deadline = time.monotonic() + 5
    while not watch.available_mb and time.monotonic() < deadline:
        time.sleep(0.05)
    started = time.monotonic()
    watch.stop()
    assert time.monotonic() - started < 5  # ended by the stop file, not the kill
    assert (tmp_path / "stop").exists()
    assert watch.available_mb and set(watch.available_mb) == {4321}
    argv = (bin_dir / "argv").read_text().split()
    assert argv[:5] == [
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(tmp_path / "check.ps1"),
    ]
    assert argv[5:] == [
        "-Watch",
        "-OutFile",
        str(tmp_path / "w.csv"),
        "-StopFile",
        str(tmp_path / "stop"),
        "-MaxSeconds",
        "14400",
    ]
    watch.stop()


def test_windows_watch_kills_stuck_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = tmp_path / "bin"
    fake_bin(bin_dir, "wslpath", 'echo "$2"\n')
    fake_bin(bin_dir, "powershell.exe", "exec sleep 60\n")
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setattr(samplers, "WATCH_EXIT_WAIT_S", 0.2)
    watch = WindowsWatch(tmp_path / "w.csv", tmp_path / "stop", tmp_path / "check.ps1")
    watch.start()
    popen = watch._popen
    assert popen is not None
    watch.stop()
    assert popen.poll() is not None
    assert watch.available_mb == []


def test_machine_facts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(MEMINFO, encoding="utf-8")
    assert samplers.ram_total_gb(meminfo) == round(16303460 / 1024**2, 1)
    assert samplers.ram_total_gb(tmp_path / "missing") == 0.0
    bin_dir = tmp_path / "bin"
    fake_bin(bin_dir, "nvidia-smi", 'echo "NVIDIA GeForce RTX 3070"; echo "Second GPU"\n')
    fake_bin(bin_dir, "powershell.exe", "printf '\\357\\273\\27717179869184\\r\\n'\n")
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    assert samplers.gpu_name() == "NVIDIA GeForce RTX 3070"
    assert samplers.windows_ram_total_gb() == 16.0
