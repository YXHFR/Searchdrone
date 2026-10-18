import json
from pathlib import Path

import pytest

from sd_sim.perf.report import (
    SCHEMA,
    Measurements,
    Metric,
    ModelResult,
    build_report,
    config_warnings,
    evaluate,
    write_report,
)
from sd_sim.perf.stats import TelemetryResult

IDS = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "MODEL"]


def good() -> Measurements:
    return Measurements(
        cold_start_s=120.0,
        rtf=[(float(t), 0.98) for t in range(60)],
        video_fps=14.8,
        video_detail="444 bilder på 30 s",
        wsl_used_mb=[4000, 7000, 6500],
        win_available_mb=[3000, 2500],
        vram_used_mb=[5000, 6100],
        stop_latencies_s=[0.2, 0.3, 0.25, 0.4, 0.35],
        telemetry=TelemetryResult([40.0] * 50, 60.0, 9.8, 1.0, True, 50),
        chat_latencies_s=[2.0, 3.0, 8.0],
        model=ModelResult(3200.0, 9000.0, "data/reports/eval.json", True),
    )


def by_id(metrics: list[Metric]) -> dict[str, Metric]:
    return {m.id: m for m in metrics}


def report(metrics: list[Metric]) -> dict[str, object]:
    return build_report(
        metrics,
        git_sha="0f57504",
        dirty=False,
        gpu_name="NVIDIA GeForce RTX 3070",
        ram_total_gb=15.5,
        windows_ram_total_gb=16.0,
        settings={"physics": "gz", "gui": True},
        warnings=[],
        fallback=None,
        started_utc="2026-10-18T10:00:00Z",
        finished_utc="2026-10-18T10:30:00Z",
    )


def test_all_good_passes() -> None:
    metrics = evaluate(good())
    assert [m.id for m in metrics] == IDS
    m = by_id(metrics)
    assert all(m[i].passed is True for i in IDS if i != "P8")
    assert m["P8"].passed is None
    assert m["P0"].value == 120.0
    assert m["P1"].value == pytest.approx(0.98)
    assert m["P3"].value == 7000 and m["P4"].value == 2500 and m["P5"].value == 6100
    assert m["P6"].value == 0.4
    assert m["P8"].value == 3.0 and "max 8.00" in m["P8"].detail
    assert m["MODEL"].value == 3200.0
    assert report(metrics)["passed"] is True


def test_no_data_fails_every_metric() -> None:
    m = by_id(evaluate(Measurements()))
    for mid in IDS:
        metric = m[mid]
        assert metric.value is None, mid
        assert metric.detail, mid
        assert metric.passed is (None if mid == "P8" else False), mid
    assert m["P4"].detail == "inte mätt"
    assert "eval=0" in m["MODEL"].detail
    assert report(evaluate(Measurements()))["passed"] is False


def test_empty_windows_list_is_no_data() -> None:
    meas = good()
    meas.win_available_mb = []
    assert by_id(evaluate(meas))["P4"].passed is False


@pytest.mark.parametrize(
    ("field", "value", "metric"),
    [
        ("cold_start_s", 180.1, "P0"),
        ("video_fps", 11.9, "P2"),
        ("wsl_used_mb", [9001], "P3"),
        ("win_available_mb", [4000, 1499], "P4"),
        ("vram_used_mb", [7501], "P5"),
        ("stop_latencies_s", [0.2, 0.2, 0.2, 0.2, 1.01], "P6"),
        ("stop_latencies_s", [0.2, 0.2, 0.2, 0.2], "P6"),  # one STOP never acknowledged
    ],
)
def test_limits(field: str, value: object, metric: str) -> None:
    meas = good()
    setattr(meas, field, value)
    m = by_id(evaluate(meas))
    assert m[metric].passed is False
    assert report(list(m.values()))["passed"] is False


def test_limits_inclusive() -> None:
    meas = good()
    meas.cold_start_s = 180.0
    meas.video_fps = 12.0
    meas.wsl_used_mb = [9000]
    meas.win_available_mb = [1500]
    meas.vram_used_mb = [7500]
    meas.stop_latencies_s = [1.0] * 5
    meas.telemetry = TelemetryResult([300.0] * 30, 300.0, 4.0, 20.0, True, 30)
    meas.model = ModelResult(5000.0, 12000.0, "r.json", False)
    assert report(evaluate(meas))["passed"] is True


def test_rtf_mean_and_worst_window() -> None:
    meas = good()
    # mean 0.95 but one 10-s window at 0.70
    meas.rtf = [(float(t), 0.70 if 10 <= t < 20 else 0.995) for t in range(60)]
    m = by_id(evaluate(meas))["P1"]
    assert m.value is not None and m.value >= 0.90
    assert m.passed is False and "0.700" in m.detail
    meas.rtf = [(float(t), 0.85) for t in range(60)]
    assert by_id(evaluate(meas))["P1"].passed is False
    meas.rtf = [(0.0, 1.0), (1.0, 1.0)]  # too short for one window
    m = by_id(evaluate(meas))["P1"]
    assert m.passed is False and "10-s" in m.detail


def test_telemetry_rules() -> None:
    meas = good()
    meas.telemetry = TelemetryResult([50.0] * 30, 50.0, 9.0, 25.0, False, 30)
    m = by_id(evaluate(meas))["P7"]
    assert m.passed is False and m.detail.startswith("ogiltig")
    meas.telemetry = TelemetryResult([50.0] * 30, 50.0, 3.9, 1.0, True, 30)
    assert by_id(evaluate(meas))["P7"].passed is False
    meas.telemetry = TelemetryResult([301.0] * 30, 301.0, 9.0, 1.0, True, 30)
    assert by_id(evaluate(meas))["P7"].passed is False
    meas.telemetry = TelemetryResult([], None, None, None, False, 0)
    m = by_id(evaluate(meas))["P7"]
    assert m.passed is False and m.value is None


def test_model_rules() -> None:
    meas = good()
    meas.model = ModelResult(5001.0, 9000.0, "r.json", True)
    assert by_id(evaluate(meas))["MODEL"].passed is False
    meas.model = ModelResult(3000.0, 12001.0, "r.json", True)
    assert by_id(evaluate(meas))["MODEL"].passed is False
    meas.model = ModelResult(None, None, "r.json", False)
    assert by_id(evaluate(meas))["MODEL"].passed is False


def test_p8_never_blocks() -> None:
    meas = good()
    meas.chat_latencies_s = [600.0]
    metrics = evaluate(meas)
    assert by_id(metrics)["P8"].passed is None
    assert report(metrics)["passed"] is True


def test_build_report_keys() -> None:
    rep = report(evaluate(good()))
    assert rep["schema"] == SCHEMA
    assert set(rep) == {
        "schema",
        "git_sha",
        "git_dirty",
        "gpu_name",
        "ram_total_gb",
        "windows_ram_total_gb",
        "settings",
        "warnings",
        "fallback",
        "started_utc",
        "finished_utc",
        "metrics",
        "passed",
    }
    metrics = rep["metrics"]
    assert isinstance(metrics, list) and len(metrics) == len(IDS)
    assert set(metrics[0]) == {"id", "text_sv", "value", "limit_sv", "unit", "passed", "detail"}
    assert (
        build_report(
            [],
            git_sha="x",
            dirty=True,
            gpu_name="",
            ram_total_gb=0.0,
            windows_ram_total_gb=None,
            settings={},
            warnings=[],
            fallback=None,
            started_utc="",
            finished_utc="",
        )["passed"]
        is False
    )


def test_write_report(tmp_path: Path) -> None:
    rep = report(evaluate(good()))
    rep["warnings"] = ["VARNING: test"]
    rep["fallback"] = "gui=false"
    out = tmp_path / "data" / "reports"
    json_path, md_path = write_report(rep, out, "20261018T100000Z")
    assert json_path == out / "perf-20261018T100000Z.json"
    assert md_path == out / "perf-20261018T100000Z.md"
    assert json.loads(json_path.read_text(encoding="utf-8")) == rep
    assert json_path.stat().st_mode & 0o777 == 0o644
    assert md_path.stat().st_mode & 0o777 == 0o644
    md = md_path.read_text(encoding="utf-8")
    assert "GODKÄNT" in md and "NVIDIA GeForce RTX 3070" in md
    for mid in IDS:
        assert f"| {mid} |" in md
    assert "| OK |" in md and "| info |" in md and "VARNING: test" in md and "gui=false" in md
    rep["passed"] = False
    rep["metrics"][0]["passed"] = False  # type: ignore[index]
    write_report(rep, out, "x")
    md = (out / "perf-x.md").read_text(encoding="utf-8")
    assert "INTE GODKÄNT" in md and "| FEL |" in md


def sim_toml(tmp_path: Path, perf: str) -> Path:
    path = tmp_path / "sim.toml"
    path.write_text(f"[sim]\nphysics = 'gz'\n\n[perf]\n{perf}\n", encoding="utf-8")
    return path


def test_config_warnings(tmp_path: Path) -> None:
    gpu = "NVIDIA GeForce RTX 3070"
    assert config_warnings(sim_toml(tmp_path, 'gpu_name = ""\nram_gb = 0'), gpu, 16.0) == []
    assert (
        config_warnings(sim_toml(tmp_path, 'gpu_name = "rtx 3070"\nram_gb = 16'), gpu, 15.9) == []
    )
    warn = config_warnings(sim_toml(tmp_path, 'gpu_name = "RTX 4090"\nram_gb = 32'), gpu, 16.0)
    assert len(warn) == 2 and all(w.startswith("VARNING") for w in warn)
    assert "RTX 4090" in warn[0] and "32" in warn[1]
    warn = config_warnings(sim_toml(tmp_path, 'gpu_name = "RTX 3070"'), "", 16.0)
    assert len(warn) == 1 and "inget grafikkort" in warn[0]
    assert config_warnings(sim_toml(tmp_path, "ram_gb = 17"), gpu, 16.0) == []  # exactly 1 GB
    assert config_warnings(tmp_path / "missing.toml", gpu, 16.0) == []
    (tmp_path / "bad.toml").write_text("[perf\n", encoding="utf-8")
    assert config_warnings(tmp_path / "bad.toml", gpu, 16.0) == []
    (tmp_path / "noperf.toml").write_text("[sim]\n", encoding="utf-8")
    assert config_warnings(tmp_path / "noperf.toml", gpu, 16.0) == []
