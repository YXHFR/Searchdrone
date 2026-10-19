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
    assert perf.parse_args([]) == perf.PerfArgs(runs=1, gui=False, eval=True)


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
    settings = perf.settings_for(args, cfg, test_rig=True)
    assert settings == {
        "physics": "gz",
        "gui": False,
        "brain": "cloud",
        "test_rig": True,
        "link_profile": "perfect",
        "eval": True,
        "runs": 2,
    }
    assert perf.fallback_for(args) == "gui=false"
    assert perf.fallback_for(perf.PerfArgs(gui=True)) is None


def test_perf_config_is_the_demo_with_gazebo_and_the_cloud_ai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SD_SIM__HEADLESS", raising=False)
    cfg = perf.perf_config(perf.PerfArgs(gui=False))
    assert (cfg.physics, cfg.gui, cfg.video, cfg.brain) == ("gz", False, "camera", "cloud")
    assert cfg.instance == 0
    assert cfg.open_browser is False
    assert cfg.personal_home is False


# --- preflight -----------------------------------------------------------------------------


def _facts(**kw: object) -> perf.PreflightFacts:
    base: dict[str, object] = {
        "chrome": "/usr/bin/google-chrome",
        "docker": "/usr/bin/docker",
        "cloud_key": True,
        "nvidia_smi": "/usr/bin/nvidia-smi",
        "busy": [],
    }
    base.update(kw)
    return perf.PreflightFacts(**base)  # type: ignore[arg-type]


def test_preflight_passes() -> None:
    assert perf.preflight_errors(_facts()) == []


@pytest.mark.parametrize(
    ("kw", "needle"),
    [
        ({"chrome": None}, "Google Chrome"),
        ({"docker": None}, "docker saknas"),
        ({"cloud_key": False}, "SD_CLOUD_API_KEY= i filen .env"),
        ({"nvidia_smi": None}, "nvidia-smi"),
        ({"busy": [14550, 8080]}, "Port 14550, 8080"),
    ],
)
def test_preflight_names_each_missing_part(kw: dict[str, object], needle: str) -> None:
    errors = perf.preflight_errors(_facts(**kw))
    assert len(errors) == 1
    assert needle in errors[0]


def test_the_cloud_key_is_found_in_env_without_being_shown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SD_CLOUD_API_KEY", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr(perf, "busy_ports", lambda *_a: [])
    assert perf.gather_preflight(tmp_path, {}).cloud_key is False
    (tmp_path / ".env").write_text("SD_CLOUD_API_KEY=\n", encoding="utf-8")
    assert perf.gather_preflight(tmp_path, {}).cloud_key is False
    (tmp_path / ".env").write_text("SD_CLOUD_API_KEY=sk-test\n", encoding="utf-8")
    facts = perf.gather_preflight(tmp_path, {})
    assert facts.cloud_key is True
    assert "sk-test" not in repr(facts)


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
    assert "avbröt P6 efter 1 av 5 STOPP" in notes[0]


def test_on_the_ground_nothing_is_pressed() -> None:
    clock, acks = FakeClock(), []
    pilot = FakePilot(clock, acks, airborne=False)
    latencies, notes = perf.measure_stops(
        pilot, _visible(clock, acks), clock_ns=clock.now, sleep=clock.sleep
    )
    assert latencies == [] and pilot.calls == []
    assert "inte i luften" in notes[0]
    assert "avbröt P6 efter 0 av 5 STOPP" in notes[0]


# --- the browser's result -------------------------------------------------------------------------


def _ping(t: int, offset: int) -> dict[str, int]:
    return {
        "t_send_ns": t,
        "t_recv_ns": t + 2_000_000,
        "server_ns": t,
        "server_mono_ns": t + 1_000_000 + offset,
    }


def _browser_data() -> dict[str, object]:
    """A good perf-browser.json: 15 fps video, 5 Hz telemetry 50 ms on the way."""
    offset = -5_000_000_000  # server monotonic = page clock - 5 s
    base = 10_000_000_000
    start = [_ping(base + i * 10_000_000, offset) for i in range(10)]
    end = [_ping(base + 200_000_000_000 + i * 10_000_000, offset) for i in range(10)]
    arrivals: list[list[int]] = []
    for i in range(100):
        boot_ms = 5_000 + i * 200  # 5 Hz
        page_ns = boot_ms * 1_000_000 - offset + 50_000_000  # 50 ms on the way
        arrivals.append([boot_ms, page_ns])
    return {
        "video": {"frames": 450, "seconds": 30, "fps": 15.0, "mode": "webrtc"},
        "telemetry": {"arrivals": arrivals, "pings_start": start, "pings_end": end},
        "errors": ["first", 3],
    }


def test_browser_result_reads_video_and_telemetry() -> None:
    res = perf.browser_result(_browser_data())
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


def test_the_browser_notes_tell_the_cases_apart() -> None:
    missing = perf.browser_note("missing")
    assert "perf-browser.json saknas" in missing and "avbröts" not in missing
    assert "avbröts innan den blev klar" in perf.browser_note("missing", aborted=True)
    assert "gick inte att läsa" in perf.browser_note("broken")
    never = perf.browser_note("not_started", aborted=True)
    assert "startade inte" in never and "saknas" not in never
    assert perf.browser_result(None, "not_started", aborted=True).notes == [never]


def test_read_browser_file_states(tmp_path: Path) -> None:
    path = tmp_path / "perf-browser.json"
    assert perf.read_browser_file(path) == (None, "missing")
    path.write_text("{not json", encoding="utf-8")
    assert perf.read_browser_file(path) == (None, "broken")
    path.write_text("[1, 2]", encoding="utf-8")
    assert perf.read_browser_file(path) == (None, "broken")
    path.write_text(json.dumps({"video": {}}), encoding="utf-8")
    assert perf.read_browser_file(path) == ({"video": {}}, "read")


# --- a flight that breaks off still reads the browser and the chat ---------------------------


class _FakeStack:
    def __init__(self, root: Path) -> None:
        self.repo = root
        self.run_dir = root / "run"
        self.world = None
        self.ports = {"ground_panel": 18080}
        self.login_url = "http://127.0.0.1:18080/#login=code"

    def up(self) -> None:
        pass

    def down(self) -> None:
        pass

    def mask(self, text: str) -> str:
        return text


class _FlightPilot:
    """Airborne at once; the flight itself breaks off with ``fail`` (or ends fine)."""

    def __init__(self, fail: BaseException | None, *, airborne_fails: bool = False) -> None:
        self.fail = fail
        self.airborne_fails = airborne_fails
        self.chat_reads = 0

    def fly_demo_until_airborne(self, _timeout_s: float) -> None:
        if self.airborne_fails and self.fail is not None:
            raise self.fail

    def wait_flight_done(self, _timeout_s: float) -> None:
        if self.fail is not None:
            raise self.fail

    def chat_latencies_s(self, _stack: object = None) -> list[float]:
        self.chat_reads += 1
        return [2.5, 3.5]

    def close(self) -> None:
        pass


def _fly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pilot: _FlightPilot,
    browser_file: str | None,
    *,
    eval_: bool = False,
) -> tuple[perf.LiveRun, BaseException | None]:
    """fly_and_measure with every live part faked; the browser 'writes' ``browser_file``."""
    stack = _FakeStack(tmp_path)
    started: list[str] = []

    def fake_start(self: perf.BrowserRun, base_url: str, code: str) -> None:
        started.append(base_url)
        if browser_file is not None:
            self.result.parent.mkdir(parents=True, exist_ok=True)
            self.result.write_text(browser_file, encoding="utf-8")

    monkeypatch.setattr(perf, "build_panel", lambda *_a: None)
    monkeypatch.setattr(perf, "start_mediamtx", lambda *_a: None)
    monkeypatch.setattr(perf, "Pilot", lambda _stack: pilot)
    monkeypatch.setattr(perf.BrowserRun, "start", fake_start)
    monkeypatch.setattr(perf, "measure_stops", lambda *_a, **_k: ([0.3], []))
    monkeypatch.setattr(
        perf, "run_cloud_perf", lambda *_a: perf_report.ModelResult(900.0, 1500.0, "r", True)
    )
    live = perf.LiveRun()
    live.timeline.mark("start")
    error: BaseException | None = None
    try:
        perf.fly_and_measure(
            perf.PerfArgs(eval=eval_),
            stack,  # pyright: ignore[reportArgumentType]
            live,
            [],
        )
    except BaseException as exc:  # the test looks at what came out
        error = exc
    return live, error


def _metrics(live: perf.LiveRun) -> dict[str, perf_report.Metric]:
    res = perf.browser_result(live.browser, live.browser_state, aborted=live.aborted)
    m = perf_report.Measurements(
        video_fps=res.video_fps,
        video_detail=res.video_detail,
        telemetry=res.telemetry,
        chat_latencies_s=live.chat,
        model=live.model,
        notes=live.notes + res.notes,
    )
    return {x.id: x for x in perf_report.evaluate(m)}


def test_a_broken_off_flight_still_yields_p2_p7_and_p8(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pilot = _FlightPilot(VehicleError("flygningen blev inte klar inom 900 s"))
    live, error = _fly(monkeypatch, tmp_path, pilot, json.dumps(_browser_data()), eval_=True)
    assert isinstance(error, VehicleError)  # the flight's own error still ends the run
    assert live.aborted and live.browser_state == "read"
    assert live.stops == [0.3]
    assert live.chat == [2.5, 3.5] and pilot.chat_reads == 1
    assert live.model is not None and live.model.p50_ms == 900.0
    metrics = _metrics(live)
    assert metrics["P2"].value == 15.0
    assert metrics["P7"].value is not None
    assert metrics["P8"].value is not None
    notes = live.notes + perf.browser_result(live.browser, live.browser_state).notes
    assert not any("resultatfil" in n for n in notes)


def test_a_broken_off_flight_without_the_file_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    live, error = _fly(monkeypatch, tmp_path, _FlightPilot(InfraError("borta")), None)
    assert isinstance(error, InfraError)
    assert live.browser_state == "missing"
    note = perf.browser_result(live.browser, live.browser_state, aborted=live.aborted).notes[0]
    assert "saknas" in note and "avbröts innan den blev klar" in note
    assert live.chat == [2.5, 3.5]


def test_a_flight_that_never_got_airborne_never_started_the_browser(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pilot = _FlightPilot(VehicleError("lyfte inte"), airborne_fails=True)
    live, error = _fly(monkeypatch, tmp_path, pilot, json.dumps(_browser_data()))
    assert isinstance(error, VehicleError)
    assert live.browser_state == "not_started" and live.browser is None
    note = perf.browser_result(live.browser, live.browser_state, aborted=live.aborted).notes[0]
    assert "startade inte" in note
    assert pilot.chat_reads == 1  # the chat is still read


def test_a_finished_flight_reads_everything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pilot = _FlightPilot(None)
    live, error = _fly(monkeypatch, tmp_path, pilot, json.dumps(_browser_data()))
    assert error is None and not live.aborted
    assert live.browser_state == "read" and live.chat == [2.5, 3.5]
    assert live.model is None  # eval=0


def test_a_failing_read_is_a_note_not_a_new_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pilot = _FlightPilot(VehicleError("flygningen"))

    def broken_chat(_stack: object = None) -> list[float]:
        raise InfraError("Databasen svarar inte på 127.0.0.1:5432")

    pilot.chat_latencies_s = broken_chat  # type: ignore[method-assign]
    live, error = _fly(monkeypatch, tmp_path, pilot, json.dumps(_browser_data()))
    assert isinstance(error, VehicleError)  # the flight's error, not the database's
    assert live.browser_state == "read"
    assert any(n.startswith("P8:") and "Databasen" in n for n in live.notes)


def test_a_stop_request_collects_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pilot = _FlightPilot(StopRequestedError())
    live, error = _fly(monkeypatch, tmp_path, pilot, json.dumps(_browser_data()))
    assert isinstance(error, StopRequestedError)
    assert pilot.chat_reads == 0 and live.browser is None


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
        json.dumps({"latency_p50_ms": 2100, "latency_p95_ms": 7300.5, "ok": True}),
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
    # 0 ms or none when no call got a model answer (the cloud down, all errors).
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps({"latency_p50_ms": 0, "latency_p95_ms": 0, "ok": False}), encoding="utf-8"
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
