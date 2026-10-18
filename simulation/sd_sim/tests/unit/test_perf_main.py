"""The perf run's own logic (``python -m sd_sim.perf``) with fakes: arguments, preflight,
time marks, the STOP loop, the browser's and the model's results. The live run is not
unit-tested."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from sd_sim import config, procs
from sd_sim.errors import InfraError, StopRequestedError, VehicleError
from sd_sim.perf import __main__ as perf
from sd_sim.perf import report as perf_report
from sd_sim.perf import samplers

# --- arguments -----------------------------------------------------------------------------


def test_defaults() -> None:
    assert perf.parse_args([]) == perf.PerfArgs(runs=1, gui=True, eval=True)


def test_values() -> None:
    assert perf.parse_args(["runs=3", "gui=false", "eval=0"]) == perf.PerfArgs(3, False, False)
    assert perf.parse_args(["gui=1", "eval=true"]) == perf.PerfArgs(1, True, True)


@pytest.mark.parametrize(
    "argv",
    [["runs=0"], ["runs=100"], ["runs=x"], ["gui=maybe"], ["eval=2"], ["foo=1"], ["runs"]],
)
def test_bad_arguments_exit_2(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        perf.parse_args(argv)
    assert exc.value.code == 2
    assert "sd_sim.perf" in capsys.readouterr().err


def test_settings_and_fallback() -> None:
    cfg = config.SimConfig(gui=False)
    args = perf.PerfArgs(runs=2, gui=False, eval=True)
    settings = perf.settings_for(args, cfg, cloud=False, test_rig=True)
    assert settings == {
        "physics": "gz",
        "gui": False,
        "brain": "llm_local",
        "cloud": False,
        "test_rig": True,
        "link_profile": "perfect",
        "eval": True,
        "runs": 2,
    }
    assert perf.fallback_for(args) == "gui=false"
    assert perf.fallback_for(perf.PerfArgs()) is None


def test_perf_config_is_the_demo_with_gazebo_and_the_local_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SD_SIM__HEADLESS", raising=False)
    cfg = perf.perf_config(perf.PerfArgs(gui=False))
    assert (cfg.physics, cfg.gui, cfg.video, cfg.brain) == ("gz", False, "camera", "llm_local")
    assert cfg.instance == 0
    assert cfg.open_browser is False
    assert cfg.personal_home is False


# --- preflight -----------------------------------------------------------------------------


def _facts(tmp_path: Path, **kw: object) -> perf.PreflightFacts:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")
    base: dict[str, object] = {
        "chrome": "/usr/bin/google-chrome",
        "docker": "/usr/bin/docker",
        "docker_runtimes": '{"nvidia":{},"runc":{}}',
        "llm_up": False,
        "model_file": model,
        "nvidia_smi": "/usr/bin/nvidia-smi",
        "busy": [],
    }
    base.update(kw)
    return perf.PreflightFacts(**base)  # type: ignore[arg-type]


def test_preflight_passes(tmp_path: Path) -> None:
    assert perf.preflight_errors(_facts(tmp_path)) == []


@pytest.mark.parametrize(
    ("kw", "needle"),
    [
        ({"chrome": None}, "Google Chrome"),
        ({"docker": None}, "docker saknas"),
        ({"docker_runtimes": '{"runc":{}}'}, "nvidia-runtime"),
        ({"docker_runtimes": ""}, "nvidia-runtime"),
        ({"nvidia_smi": None}, "nvidia-smi"),
        ({"model_file": Path("/nonexistent/m.gguf")}, "just models-fetch"),
        ({"busy": [14550, 8080]}, "Port 14550, 8080"),
    ],
)
def test_preflight_names_each_missing_part(
    tmp_path: Path, kw: dict[str, object], needle: str
) -> None:
    errors = perf.preflight_errors(_facts(tmp_path, **kw))
    assert len(errors) == 1
    assert needle in errors[0]


def test_a_running_model_server_proves_the_gpu_runtime(tmp_path: Path) -> None:
    assert perf.preflight_errors(_facts(tmp_path, docker_runtimes="", llm_up=True)) == []


def test_model_file_comes_from_the_lock(tmp_path: Path) -> None:
    lock = tmp_path / "software/brain/runtime.lock.toml"
    lock.parent.mkdir(parents=True)
    lock.write_text('[model]\nfile = "m.gguf"\n', encoding="utf-8")
    assert perf.model_file(tmp_path) == tmp_path / "models" / "m.gguf"


# --- time marks ------------------------------------------------------------------------------


def _clock(values: list[float]) -> Callable[[], float]:
    it = iter(values)
    return lambda: next(it)


def test_timeline_cold_start_and_flight_window() -> None:
    t = perf.Timeline(clock=_clock([100.0, 160.0, 200.0, 300.0, 500.0]))
    assert t.cold_start_s is None and t.flight_window() is None
    t.mark("start")
    t.mark("ready")
    assert t.cold_start_s == 60.0
    t.mark("airborne")
    assert t.flight_window() == (200.0, 300.0)  # not done yet: up to now
    t.mark("done")
    assert t.flight_window() == (200.0, 500.0)
    assert t.span("ready", "done") == 340.0


# --- P6: the STOP loop -------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.ns = 1_000_000_000

    def now(self) -> int:
        return self.ns

    def sleep(self, seconds: float) -> None:
        self.ns += int(seconds * 1e9)


class FakePilot:
    """BRAKE after a STOP, LOITER after Resume; the ACK appears ``ack_after_s`` later."""

    def __init__(
        self,
        clock: FakeClock,
        acks: list[int],
        *,
        ack_after_s: float | None = 0.4,
        resume_works: bool = True,
        airborne: bool = True,
    ) -> None:
        self.clock = clock
        self.acks = acks
        self.ack_after_s = ack_after_s
        self.resume_works = resume_works
        self._airborne = airborne
        self._mode = "GUIDED"
        self.calls: list[str] = []

    def press_stop(self) -> int:
        self.calls.append("stop")
        self._mode = "BRAKE"
        t0 = self.clock.now()
        if self.ack_after_s is not None:
            self.acks.append(t0 + int(self.ack_after_s * 1e9))
        return t0

    def press_resume(self) -> int:
        self.calls.append("resume")
        if self.resume_works:
            self._mode = "LOITER"
        return self.clock.now()

    def mode(self) -> str | None:
        return self._mode

    def airborne(self) -> bool:
        return self._airborne

    def wait_mode(self, pred: Callable[[str | None], bool], timeout_s: float) -> str | None:
        if not pred(self._mode):
            raise VehicleError(f"flygläget: inte inom {timeout_s:.0f} s")
        return self._mode


def _visible(clock: FakeClock, acks: list[int]) -> Callable[[], list[int]]:
    """radio.tlog as read now: an ACK is there once its time has come."""
    return lambda: [a for a in acks if a <= clock.now()]


def test_five_stops_measure_press_to_ack() -> None:
    clock, acks = FakeClock(), []
    pilot = FakePilot(clock, acks)
    latencies, notes = perf.measure_stops(
        pilot, _visible(clock, acks), clock_ns=clock.now, sleep=clock.sleep
    )
    assert notes == []
    assert pilot.calls == ["stop", "resume"] * 5
    assert len(latencies) == 5
    # The poll runs every 0.25 s, the ACK time itself is what counts.
    assert all(abs(x - 0.4) < 1e-9 for x in latencies)


def test_an_old_ack_does_not_count() -> None:
    clock = FakeClock()
    acks = [clock.now() - 5_000_000]  # before the press
    pilot = FakePilot(clock, acks, ack_after_s=None)
    latencies, notes = perf.measure_stops(
        pilot, _visible(clock, acks), tries=1, clock_ns=clock.now, sleep=clock.sleep
    )
    assert latencies == []
    assert "inget COMMAND_ACK" in notes[0]
    # It waited the 3 s window plus the flush time, then pressed Resume.
    assert pilot.calls == ["stop", "resume"]


def test_a_late_ack_is_missing() -> None:
    clock, acks = FakeClock(), []
    pilot = FakePilot(clock, acks, ack_after_s=3.5)
    latencies, notes = perf.measure_stops(
        pilot, _visible(clock, acks), tries=2, clock_ns=clock.now, sleep=clock.sleep
    )
    assert latencies == []
    assert len(notes) == 2


def test_no_resume_ends_the_loop() -> None:
    clock, acks = FakeClock(), []
    pilot = FakePilot(clock, acks, resume_works=False)
    latencies, notes = perf.measure_stops(
        pilot, _visible(clock, acks), clock_ns=clock.now, sleep=clock.sleep
    )
    assert len(latencies) == 1
    assert pilot.calls == ["stop", "resume"]
    assert "efter Fortsätt" in notes[0]


def test_on_the_ground_nothing_is_pressed() -> None:
    clock, acks = FakeClock(), []
    pilot = FakePilot(clock, acks, airborne=False)
    latencies, notes = perf.measure_stops(
        pilot, _visible(clock, acks), clock_ns=clock.now, sleep=clock.sleep
    )
    assert latencies == [] and pilot.calls == []
    assert "inte i luften" in notes[0]


# --- the browser's result -------------------------------------------------------------------------


def _ping(t: int, offset: int) -> dict[str, int]:
    return {
        "t_send_ns": t,
        "t_recv_ns": t + 2_000_000,
        "server_ns": t,
        "server_mono_ns": t + 1_000_000 + offset,
    }


def test_browser_result_reads_video_and_telemetry() -> None:
    offset = -5_000_000_000  # server monotonic = page clock - 5 s
    base = 10_000_000_000
    start = [_ping(base + i * 10_000_000, offset) for i in range(10)]
    end = [_ping(base + 200_000_000_000 + i * 10_000_000, offset) for i in range(10)]
    arrivals = []
    for i in range(100):
        boot_ms = 5_000 + i * 200  # 5 Hz
        page_ns = boot_ms * 1_000_000 - offset + 50_000_000  # 50 ms on the way
        arrivals.append([boot_ms, page_ns])
    data = {
        "video": {"frames": 450, "seconds": 30, "fps": 15.0, "mode": "webrtc"},
        "telemetry": {"arrivals": arrivals, "pings_start": start, "pings_end": end},
        "errors": ["first", 3],
    }
    res = perf.browser_result(data)
    assert res.video_fps == 15.0
    assert "450 bilder på 30 s, läge webrtc" in res.video_detail
    assert res.notes == ["webbläsaren: first"]
    assert res.telemetry is not None
    assert res.telemetry.samples == 100
    assert res.telemetry.p95_ms is not None and abs(res.telemetry.p95_ms - 50.0) < 1.0


def test_browser_result_without_a_file_or_parts() -> None:
    missing = perf.browser_result(None)
    assert missing.video_fps is None and missing.telemetry is None
    assert "ingen resultatfil" in missing.notes[0]
    broken = perf.browser_result(
        {
            "video": {"frames": 0, "seconds": 0, "fps": None},
            "telemetry": {"arrivals": [["x"], [1]], "pings_start": [{"t_send_ns": 1}, 5]},
        }
    )
    assert broken.video_fps is None
    assert broken.telemetry is not None and broken.telemetry.valid is False


# --- the model's result ---------------------------------------------------------------


def test_eval_report_path_from_its_output(tmp_path: Path) -> None:
    out = "S01: ok\nRapport: data/reports/eval-grunden-x.json\n{}\n"
    assert perf.eval_report_path(out, tmp_path) == tmp_path / "data/reports/eval-grunden-x.json"
    assert perf.eval_report_path("Rapport: /abs/r.json", tmp_path) == Path("/abs/r.json")
    assert perf.eval_report_path("nothing", tmp_path) is None


def test_model_result_reads_the_latencies(tmp_path: Path) -> None:
    path = tmp_path / "data/reports/eval.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"latency_p50_ms": 2100, "latency_p95_ms": 7300.5, "passed": True}),
        encoding="utf-8",
    )
    res = perf.model_result(path, tmp_path)
    assert res == perf_report.ModelResult(2100.0, 7300.5, "data/reports/eval.json", True)


def test_model_result_without_a_report(tmp_path: Path) -> None:
    assert perf.model_result(None, tmp_path) == perf_report.ModelResult(None, None, "", False)
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    res = perf.model_result(bad, tmp_path)
    assert (res.p50_ms, res.p95_ms, res.eval_passed) == (None, None, False)


def test_model_result_with_no_answers_has_no_values(tmp_path: Path) -> None:
    # sd-brain writes 0 ms when no turn got a model answer (llama-server down, all errors).
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps({"latency_p50_ms": 0, "latency_p95_ms": 0, "passed": False}), encoding="utf-8"
    )
    res = perf.model_result(path, tmp_path)
    assert (res.p50_ms, res.p95_ms, res.eval_passed) == (None, None, False)


# --- small parts ----------------------------------------------------------------------


def test_login_code() -> None:
    assert perf.login_code("http://127.0.0.1:8080/#login=abc-123") == "abc-123"
    with pytest.raises(InfraError):
        perf.login_code("http://127.0.0.1:8080/")


def test_summary_lines() -> None:
    metrics = [
        perf_report.Metric("P0", "kallstart", 95.0, "≤ 180 s", "s", True, ""),
        perf_report.Metric("P6", "STOPP", None, "≤ 1.0 s", "s", False, "inga mätvärden"),
        perf_report.Metric("P8", "chatt", 3.2, "-", "s", None, ""),
    ]
    lines = perf.summary_lines(metrics)
    assert "OK" in lines[1] and "95.00 s" in lines[1]
    assert "FEL" in lines[2] and lines[3].strip() == "inga mätvärden"
    assert lines[4].endswith("info")


def test_run_exit_code() -> None:
    assert perf.run_exit_code([0, 0]) == 0
    assert perf.run_exit_code([0, 1]) == 1
    assert perf.run_exit_code([]) == 1


def test_clean_up_runs_newest_first_and_survives_errors() -> None:
    done: list[str] = []

    def broken() -> None:
        done.append("broken")
        raise RuntimeError("boom")

    steps: list[Callable[[], None]] = [lambda: done.append("first")]
    with pytest.raises(ValueError), perf._stopping(steps):
        steps.append(broken)
        steps.append(lambda: done.append("last"))
        raise ValueError
    assert done == ["last", "broken", "first"]


def test_main_runs_each_run_and_stops_on_a_missing_prerequisite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int] = []
    codes = iter([0, 2, 0])

    def fake_run(args: perf.PerfArgs, index: int) -> int:
        seen.append(index)
        return next(codes)

    monkeypatch.setattr(perf, "run_once", fake_run)
    assert perf.main(["runs=3"]) == 2
    assert seen == [1, 2]


def test_main_reports_a_broken_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run(args: perf.PerfArgs, index: int) -> int:
        raise InfraError("MediaMTX startade inte.")

    monkeypatch.setattr(perf, "run_once", fake_run)
    assert perf.main([]) == 1
    assert "MediaMTX startade inte." in capsys.readouterr().err


class _FakeSampler:
    available = False
    available_mb = None

    def __init__(self, *_args: object) -> None:
        self.samples: list[int] = []

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def test_a_stop_request_during_the_flight_ends_the_run_without_a_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    flights: list[int] = []
    reports: list[object] = []

    def fake_fly(*_args: object) -> None:
        flights.append(1)
        raise StopRequestedError

    monkeypatch.setattr(procs, "stop_recorded", lambda _path: [])
    monkeypatch.setattr(perf, "gather_preflight", lambda *_a: None)
    monkeypatch.setattr(perf, "preflight_errors", lambda _p: [])
    monkeypatch.setattr(perf, "new_run_dir", lambda *_a: tmp_path / "run")
    monkeypatch.setattr(perf, "Stack", lambda *_a, **_k: None)
    monkeypatch.setattr(samplers, "gpu_name", lambda: "GPU")
    monkeypatch.setattr(samplers, "ram_total_gb", lambda: 16.0)
    monkeypatch.setattr(samplers, "windows_ram_total_gb", lambda: None)
    for name in ("RamSampler", "VramSampler", "WindowsWatch"):
        monkeypatch.setattr(samplers, name, _FakeSampler)
    monkeypatch.setattr(perf_report, "config_warnings", lambda *_a: [])
    monkeypatch.setattr(perf_report, "write_report", lambda *a: reports.append(a))
    monkeypatch.setattr(perf, "fly_and_measure", fake_fly)

    args = perf.parse_args(["runs=3"])
    with pytest.raises(StopRequestedError):
        perf.run_once(args, 1, tmp_path)
    assert reports == []

    assert perf.main(["runs=3"]) == 1
    assert flights == [1, 1]  # main started no second run
    assert reports == []
