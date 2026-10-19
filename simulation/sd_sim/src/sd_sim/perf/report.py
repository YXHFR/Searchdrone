"""Turn the perf measurements into P0-P8 plus the model check, and write the report.

The limits are the foundation's performance budget. A measurement without data
fails (``passed`` false, ``value`` null and ``detail`` says why), so a broken sampler
can never make the run green. P8 (chat to the first AI reply) is reported without a limit
(``passed`` null). The run passes when P0-P7 and MODEL all passed.
"""

from __future__ import annotations

import json
import statistics
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sd_sim.perf.stats import MAX_DRIFT_MS, TelemetryResult, percentile, window_means

SCHEMA = "searchdrone.perf/1"
COLD_START_MAX_S = 180.0
RTF_MEAN_MIN = 0.90
RTF_WINDOW_MIN = 0.75
RTF_WINDOW_S = 10.0
VIDEO_FPS_MIN = 12.0
WSL_USED_MAX_MB = 9000
WIN_AVAILABLE_MIN_MB = 1500
VRAM_USED_MAX_MB = 7500
STOP_TRIES = 5
STOP_P95_MAX_S = 1.0
TELEMETRY_P95_MAX_MS = 300.0
TELEMETRY_HZ_MIN = 4.0
MODEL_P50_MAX_MS = 5000.0
MODEL_P95_MAX_MS = 12000.0
RAM_TOLERANCE_GB = 1.0
NO_DATA = "inga mätvärden"


@dataclass
class ModelResult:
    p50_ms: float | None
    p95_ms: float | None
    report_file: str
    eval_passed: bool


@dataclass
class Measurements:
    cold_start_s: float | None = None
    rtf: list[tuple[float, float]] = field(default_factory=list)
    video_fps: float | None = None
    video_detail: str = ""
    wsl_used_mb: list[int] = field(default_factory=list)
    win_available_mb: list[int] | None = None  # None = not measured
    vram_used_mb: list[int] = field(default_factory=list)
    stop_latencies_s: list[float] = field(default_factory=list)
    telemetry: TelemetryResult | None = None
    chat_latencies_s: list[float] = field(default_factory=list)
    model: ModelResult | None = None  # None = skipped (eval=0)
    notes: list[str] = field(default_factory=list)


@dataclass
class Metric:
    id: str
    text_sv: str
    value: float | None
    limit_sv: str
    unit: str
    passed: bool | None
    detail: str


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _p0(m: Measurements) -> Metric:
    text, limit = "Kallstart till färdig stack", f"≤ {COLD_START_MAX_S:.0f} s"
    if m.cold_start_s is None:
        return Metric("P0", text, None, limit, "s", False, "stacken blev aldrig klar")
    ok = m.cold_start_s <= COLD_START_MAX_S
    return Metric("P0", text, round(m.cold_start_s, 1), limit, "s", ok, "")


def _p1(m: Measurements) -> Metric:
    text = "Gazebo real-time factor under flygningen"
    limit = f"medel ≥ {RTF_MEAN_MIN:.2f} och sämsta 10-s-fönster ≥ {RTF_WINDOW_MIN:.2f}"
    if not m.rtf:
        return Metric("P1", text, None, limit, "", False, NO_DATA)
    mean = statistics.fmean(v for _, v in m.rtf)
    windows = window_means(m.rtf, RTF_WINDOW_S)
    if not windows:
        detail = f"för kort mätning för ett 10-s-fönster ({len(m.rtf)} värden)"
        return Metric("P1", text, round(mean, 3), limit, "", False, detail)
    worst = min(windows)
    ok = mean >= RTF_MEAN_MIN and worst >= RTF_WINDOW_MIN
    detail = f"sämsta fönster {worst:.3f}, {len(windows)} fönster, {len(m.rtf)} värden"
    return Metric("P1", text, round(mean, 3), limit, "", ok, detail)


def _p2(m: Measurements) -> Metric:
    text, limit = "Video i panelen", f"≥ {VIDEO_FPS_MIN:.0f} fps"
    if m.video_fps is None:
        return Metric("P2", text, None, limit, "fps", False, m.video_detail or NO_DATA)
    ok = m.video_fps >= VIDEO_FPS_MIN
    return Metric("P2", text, round(m.video_fps, 1), limit, "fps", ok, m.video_detail)


def _max_metric(
    mid: str, text: str, values: list[int] | None, limit_mb: int, *, lowest: bool = False
) -> Metric:
    limit = f"{'≥' if lowest else '≤'} {limit_mb} MB"
    if values is None:
        return Metric(mid, text, None, limit, "MB", False, "inte mätt")
    if not values:
        return Metric(mid, text, None, limit, "MB", False, NO_DATA)
    value = min(values) if lowest else max(values)
    ok = value >= limit_mb if lowest else value <= limit_mb
    return Metric(mid, text, float(value), limit, "MB", ok, f"{len(values)} värden")


def _p6(m: Measurements) -> Metric:
    text, limit = "STOPP till kvittens från drönaren (p95)", f"≤ {STOP_P95_MAX_S:.1f} s"
    n = len(m.stop_latencies_s)
    p95 = percentile(m.stop_latencies_s, 0.95)
    if p95 is None:
        return Metric("P6", text, None, limit, "s", False, NO_DATA)
    detail = f"{n} av {STOP_TRIES} försök kvitterade"
    ok = n >= STOP_TRIES and p95 <= STOP_P95_MAX_S
    return Metric("P6", text, round(p95, 3), limit, "s", ok, detail)


def _p7(m: Measurements) -> Metric:
    text = "Telemetri till panelen (p95)"
    limit = (
        f"p95 ≤ {TELEMETRY_P95_MAX_MS:.0f} ms, ≥ {TELEMETRY_HZ_MIN:.0f} Hz, "
        f"klockdrift ≤ {MAX_DRIFT_MS:.0f} ms"
    )
    t = m.telemetry
    if t is None:
        return Metric("P7", text, None, limit, "ms", False, NO_DATA)
    detail = f"{_fmt(t.update_hz, 1)} Hz, klockdrift {_fmt(t.drift_ms, 1)} ms, {t.samples} värden"
    if not t.valid:
        detail = f"ogiltig mätning: {detail}"
    ok = (
        t.valid
        and t.p95_ms is not None
        and t.p95_ms <= TELEMETRY_P95_MAX_MS
        and t.update_hz is not None
        and t.update_hz >= TELEMETRY_HZ_MIN
    )
    value = None if t.p95_ms is None else round(t.p95_ms, 1)
    return Metric("P7", text, value, limit, "ms", ok, detail)


def _p8(m: Measurements) -> Metric:
    text, limit = "Chatt till AI:ns första svar (p50)", "ingen gräns"
    values = m.chat_latencies_s
    p50 = percentile(values, 0.50)
    if p50 is None:
        return Metric("P8", text, None, limit, "s", None, NO_DATA)
    detail = (
        f"p50 {p50:.2f} s, p90 {_fmt(percentile(values, 0.90))} s, "
        f"max {max(values):.2f} s, {len(values)} värden"
    )
    return Metric("P8", text, round(p50, 2), limit, "s", None, detail)


def _model(m: Measurements) -> Metric:
    text = "Modellens svarstid (median)"
    limit = f"median ≤ {MODEL_P50_MAX_MS:.0f} ms och p95 ≤ {MODEL_P95_MAX_MS:.0f} ms"
    r = m.model
    if r is None:
        return Metric("MODEL", text, None, limit, "ms", False, "hoppades över (eval=0)")
    if r.p50_ms is None or r.p95_ms is None:
        return Metric("MODEL", text, r.p50_ms, limit, "ms", False, NO_DATA)
    ok = r.p50_ms <= MODEL_P50_MAX_MS and r.p95_ms <= MODEL_P95_MAX_MS
    detail = (
        f"p95 {r.p95_ms:.0f} ms, eval {'godkänd' if r.eval_passed else 'inte godkänd'}, "
        f"rapport {r.report_file}"
    )
    return Metric("MODEL", text, round(r.p50_ms, 1), limit, "ms", ok, detail)


def evaluate(m: Measurements) -> list[Metric]:
    """P0..P8 and MODEL, in that order."""
    return [
        _p0(m),
        _p1(m),
        _p2(m),
        _max_metric("P3", "Använt minne i WSL (max)", m.wsl_used_mb, WSL_USED_MAX_MB),
        _max_metric(
            "P4",
            "Ledigt minne i Windows (min)",
            m.win_available_mb,
            WIN_AVAILABLE_MIN_MB,
            lowest=True,
        ),
        _max_metric("P5", "Använt grafikminne (max)", m.vram_used_mb, VRAM_USED_MAX_MB),
        _p6(m),
        _p7(m),
        _p8(m),
        _model(m),
    ]


def build_report(
    metrics: list[Metric],
    *,
    git_sha: str,
    dirty: bool,
    gpu_name: str,
    ram_total_gb: float,
    windows_ram_total_gb: float | None,
    settings: dict[str, object],
    warnings: list[str],
    fallback: str | None,
    started_utc: str,
    finished_utc: str,
) -> dict[str, Any]:
    passed = bool(metrics) and all(x.passed is True for x in metrics if x.id != "P8")
    return {
        "schema": SCHEMA,
        "git_sha": git_sha,
        "git_dirty": dirty,
        "gpu_name": gpu_name,
        "ram_total_gb": ram_total_gb,
        "windows_ram_total_gb": windows_ram_total_gb,
        "settings": settings,
        "warnings": warnings,
        "fallback": fallback,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "metrics": [asdict(x) for x in metrics],
        "passed": passed,
    }


def _status(passed: object) -> str:
    if passed is None:
        return "info"
    return "OK" if passed else "FEL"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Prestandatest {report.get('started_utc', '')}",
        "",
        f"Resultat: **{'GODKÄNT' if report.get('passed') else 'INTE GODKÄNT'}**",
        "",
        f"- Grafikkort: {report.get('gpu_name') or 'okänt'}",
        f"- Minne i WSL: {report.get('ram_total_gb')} GB",
        f"- Commit: {report.get('git_sha')}{' (ändrad)' if report.get('git_dirty') else ''}",
    ]
    if report.get("fallback"):
        lines.append(f"- Reservplan: {report['fallback']}")
    lines += ["", "| Mätpunkt | Vad | Värde | Gräns | Resultat |", "|---|---|---|---|---|"]
    for x in report.get("metrics", []):
        value = "-" if x.get("value") is None else f"{x['value']} {x.get('unit', '')}".strip()
        detail = f" ({x['detail']})" if x.get("detail") else ""
        lines.append(
            f"| {x['id']} | {x['text_sv']}{detail} | {value} | {x['limit_sv']} "
            f"| {_status(x.get('passed'))} |"
        )
    warnings = report.get("warnings") or []
    if warnings:
        lines += ["", "## Varningar", "", *(f"- {w}" for w in warnings)]
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out_dir: Path, stamp: str) -> tuple[Path, Path]:
    """Write ``perf-<stamp>.json`` and ``perf-<stamp>.md`` (mode 0644) to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"perf-{stamp}.json"
    md_path = out_dir / f"perf-{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    for path in (json_path, md_path):
        path.chmod(0o644)
    return json_path, md_path


def config_warnings(sim_toml: Path, gpu: str, ram_gb: float) -> list[str]:
    """VARNING lines when ``[perf]`` in sim.toml names another computer than this one.

    ``gpu_name`` is compared with the name nvidia-smi reports (contains, ignoring case)
    and ``ram_gb`` with ``ram_gb`` (pass the Windows total when known, else WSL's). An
    empty name or 0 means "not set" and gives no warning.
    """
    try:
        data = tomllib.loads(sim_toml.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []
    perf = data.get("perf")
    if not isinstance(perf, dict):
        return []
    out: list[str] = []
    want_gpu = str(perf.get("gpu_name") or "").strip()
    if want_gpu and want_gpu.lower() not in gpu.lower():
        found = gpu or "inget grafikkort hittades"
        out.append(f"VARNING: sim.toml [perf] gpu_name är {want_gpu!r} men datorn har {found!r}.")
    want_ram = perf.get("ram_gb") or 0
    if (
        isinstance(want_ram, int | float)
        and not isinstance(want_ram, bool)
        and want_ram > 0
        and abs(want_ram - ram_gb) > RAM_TOLERANCE_GB
    ):
        out.append(f"VARNING: sim.toml [perf] ram_gb är {want_ram} GB men datorn har {ram_gb} GB.")
    return out
