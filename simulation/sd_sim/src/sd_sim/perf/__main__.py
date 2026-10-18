"""Performance run: ``python -m sd_sim.perf [runs=1] [gui=true] [eval=1]`` (``just perf``).

Each run starts the demo as ``just demo`` does (Gazebo, SITL, database, ground station,
onboard computer, the local model, camera video), flies the SC01 demo with a test rig
pilot and measures P0-P8 from the spec while it flies: cold start, Gazebo's real time
factor, video frames in Google Chrome, memory in WSL, Windows and the graphics card,
STOP latency, telemetry latency and the chat's answer time. Then, with ``eval=1``, the
AI's model evaluation runs against the same llama-server while Gazebo keeps the graphics
card busy. The report goes to ``data/reports/perf-<time>.json`` (and ``.md``).

Exit codes: 0 every limit held, 1 a limit failed or the run broke, 2 wrong arguments or
a missing prerequisite (nothing was started).

The cloud model is off for the whole run: only the local model is measured. The login
code for the browser is passed in the environment only and never printed.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any, Protocol

from sd_sim import _args, config, paths, ports, procs, services
from sd_sim import report as sim_report
from sd_sim.errors import InfraError, StopRequestedError, VehicleError
from sd_sim.launch import Stack, read_env_value
from sd_sim.perf import report as perf_report
from sd_sim.perf import samplers, stats, tlogack
from sd_sim.perf.pilot import Pilot, not_brake
from sd_sim.util import busy_ports, new_run_dir, say

ALLOWED_KEYS = ("runs", "gui", "eval")
INSTANCE = 0  # video and MediaMTX exist on instance 0 only
STOP_TRIES = 5
ACK_WITHIN_S = 3.0  # P6: an ACK later than this counts as missing
FLUSH_WAIT_S = 2.0  # linksim flushes radio.tlog once a second
SETTLE_S = 3.0
MODE_TIMEOUT_S = 30.0
AIRBORNE_TIMEOUT_S = 600.0
FLIGHT_TIMEOUT_S = 900.0
BROWSER_TIMEOUT_S = 600.0
EVAL_TIMEOUT_S = 5400.0
PANEL_BUILD_TIMEOUT_S = 600.0
COMPOSE_TIMEOUT_S = 300.0
VIDEO_S = 30
TELEMETRY_S = 120
STOP_MODE = "BRAKE"
# The ports the stack binds on instance 0 (the launcher checks its own again in up()).
STACK_PORTS = (
    "mavtap_in",
    "radio_mavlink_in",
    "video_raw_udp",
    "linksim_sdlink",
    "linksim_control",
    "harness_mavlink",
    "ground_panel",
    "ground_commander",
    "ground_sdlink",
    "ground_mavlink",
)
FALLBACK_HINT = (
    "Reservplan: kör just perf gui=false (Gazebo utan fönster). Hjälper inte det, se "
    "installationsguidens avsnitt om prestanda."
)
EVAL_REPORT_PREFIX = "Rapport: "


# --- arguments ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerfArgs:
    runs: int = 1
    gui: bool = True
    eval: bool = True


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _flag(key: str, raw: str) -> bool:
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    _args.fail(f"sd_sim.perf: {key}= ska vara true eller false (eller 1/0), inte '{raw}'.")


def parse_args(argv: Sequence[str]) -> PerfArgs:
    values = _args.parse(argv, ALLOWED_KEYS, "sd_sim.perf")
    raw_runs = values.get("runs", "1")
    if not raw_runs.isdigit() or not 1 <= int(raw_runs) <= 99:
        _args.fail(f"sd_sim.perf: runs= ska vara 1-99, inte '{raw_runs}'.")
    return PerfArgs(
        runs=int(raw_runs),
        gui=_flag("gui", values.get("gui", "true")),
        eval=_flag("eval", values.get("eval", "1")),
    )


def settings_for(
    args: PerfArgs, cfg: config.SimConfig, *, cloud: bool, test_rig: bool
) -> dict[str, object]:
    return {
        "physics": cfg.physics,
        "gui": cfg.gui,
        "brain": cfg.brain,
        "cloud": cloud,
        "test_rig": test_rig,
        "link_profile": cfg.link_profile,
        "eval": args.eval,
        "runs": args.runs,
    }


def fallback_for(args: PerfArgs) -> str | None:
    """Fallback step 1 of the plan is Gazebo without a window."""
    return None if args.gui else "gui=false"


def perf_config(args: PerfArgs) -> config.SimConfig:
    """The demo's settings (the public home, as ``just demo``), Gazebo and the local model."""
    values = {
        "physics": "gz",
        "gui": "true" if args.gui else "false",
        "video": "camera",
        "brain": "llm_local",
        "instance": str(INSTANCE),
    }
    cfg = config.load(values, demo=True)
    cfg.open_browser = False
    return cfg


# --- preflight ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreflightFacts:
    chrome: str | None  # google-chrome on PATH
    docker: str | None
    docker_runtimes: str  # `docker info --format '{{json .Runtimes}}'`, "" when it failed
    llm_up: bool  # llama-server already answers (then the graphics card works)
    model_file: Path
    nvidia_smi: str | None
    busy: list[int]


def preflight_errors(facts: PreflightFacts) -> list[str]:
    """Swedish errors for every missing prerequisite (empty = ready to start)."""
    errors: list[str] = []
    if facts.chrome is None:
        errors.append(
            "Google Chrome saknas i WSL (google-chrome). Playwrights egen Chromium kan inte "
            "spela H.264. Kör bash tools/setup/wsl-setup.sh igen."
        )
    if facts.docker is None:
        errors.append("docker saknas. Kör bash tools/bootstrap.sh.")
    elif not facts.llm_up and "nvidia" not in facts.docker_runtimes:
        errors.append(
            "Docker har ingen nvidia-runtime, så den lokala modellen får inget grafikkort. "
            "Installera NVIDIA Container Toolkit (bash tools/setup/wsl-setup.sh) och starta "
            "om Docker."
        )
    if facts.nvidia_smi is None:
        errors.append(
            "nvidia-smi saknas i WSL. Installera NVIDIA:s Windows-drivrutin med WSL-stöd "
            "och starta om WSL (wsl --shutdown)."
        )
    if not facts.model_file.is_file():
        errors.append(f"Modellfilen models/{facts.model_file.name} saknas. Kör just models-fetch.")
    if facts.busy:
        ports_text = ", ".join(str(p) for p in facts.busy)
        errors.append(f"Port {ports_text} är upptagen. Kör just stop och försök igen.")
    return errors


def model_file(repo: Path) -> Path:
    lock = tomllib.loads((repo / "software/brain/runtime.lock.toml").read_text(encoding="utf-8"))
    return repo / "models" / str(lock["model"]["file"])


def _run_text(argv: list[str], timeout: float = 20.0) -> str:
    try:
        res = subprocess.run(  # noqa: S603 - fixed argv, no shell
            argv, capture_output=True, text=True, check=False, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return res.stdout if res.returncode == 0 else ""


def gather_preflight(repo: Path, stack_ports: dict[str, int]) -> PreflightFacts:
    docker = shutil.which("docker")
    runtimes = _run_text([docker, "info", "--format", "{{json .Runtimes}}"]) if docker else ""
    return PreflightFacts(
        chrome=shutil.which("google-chrome"),
        docker=docker,
        docker_runtimes=runtimes,
        llm_up=services.tcp_open(stack_ports["llm_http"]),
        model_file=model_file(repo),
        nvidia_smi=shutil.which("nvidia-smi"),
        busy=busy_ports(stack_ports, list(STACK_PORTS)),
    )


# --- time marks ----------------------------------------------------------------------------


@dataclass
class Timeline:
    """Monotonic marks of one run: start, stack ready, airborne, flight done."""

    clock: Callable[[], float] = time.monotonic
    marks: dict[str, float] = field(default_factory=dict[str, float])

    def mark(self, name: str) -> float:
        self.marks[name] = self.clock()
        return self.marks[name]

    def span(self, start: str, end: str) -> float | None:
        if start not in self.marks or end not in self.marks:
            return None
        return self.marks[end] - self.marks[start]

    @property
    def cold_start_s(self) -> float | None:
        return self.span("start", "ready")

    def flight_window(self) -> tuple[float, float] | None:
        """Airborne to done; a flight that never finished ends where the live part ended."""
        if "airborne" not in self.marks:
            return None
        for name in ("done", "end"):
            if name in self.marks:
                return self.marks["airborne"], self.marks[name]
        return self.marks["airborne"], self.clock()


# --- P6: STOP latency -------------------------------------------------------------------------


class StopPilot(Protocol):
    def press_stop(self) -> int: ...
    def press_resume(self) -> int: ...
    def mode(self) -> str | None: ...
    def airborne(self) -> bool: ...
    def wait_mode(self, pred: Callable[[str | None], bool], timeout_s: float) -> Any: ...


def measure_stops(
    pilot: StopPilot,
    read_acks: Callable[[], list[int]],
    *,
    tries: int = STOP_TRIES,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
    settle_s: float = SETTLE_S,
    mode_timeout_s: float = MODE_TIMEOUT_S,
) -> tuple[list[float], list[str]]:
    """Press STOP ``tries`` times in the air: each latency is the press (host monotonic)
    to the vehicle's first accepted DO_SET_MODE ACK in radio.tlog. A STOP without an ACK
    within 3 s is a note and no latency (so a missing ACK makes P6 fail on missing
    data rather than look fast). A drone that does not fly on after Resume ends P6."""
    latencies: list[float] = []
    notes: list[str] = []
    wait_ns = int((ACK_WITHIN_S + FLUSH_WAIT_S) * 1e9)
    for n in range(1, tries + 1):
        try:
            pilot.wait_mode(not_brake, mode_timeout_s)
        except VehicleError as exc:
            notes.append(f"STOPP {n}: inget flygläge utom {STOP_MODE} ({exc}); avbröt P6.")
            break
        if not pilot.airborne():
            notes.append(f"STOPP {n}: drönaren var inte i luften; avbröt P6.")
            break
        try:
            t0 = pilot.press_stop()
        except VehicleError as exc:
            notes.append(f"STOPP {n}: knappen gick inte fram ({exc}); avbröt P6.")
            break
        ack: int | None = None
        while ack is None and clock_ns() - t0 < wait_ns:
            ack = tlogack.first_after(read_acks(), t0, within_s=ACK_WITHIN_S)
            if ack is None:
                sleep(0.25)
        if ack is None:
            notes.append(f"STOPP {n}: inget COMMAND_ACK (DO_SET_MODE) inom {ACK_WITHIN_S:.0f} s.")
        else:
            latencies.append((ack - t0) / 1e9)
        try:
            pilot.press_resume()
            pilot.wait_mode(not_brake, mode_timeout_s)
        except VehicleError as exc:
            notes.append(f"STOPP {n}: drönaren lämnade inte {STOP_MODE} efter Fortsätt ({exc}).")
            break
        sleep(settle_s)
    return latencies, notes


# --- browser and model results ----------------------------------------------------------------


@dataclass
class BrowserResult:
    video_fps: float | None = None
    video_detail: str = ""
    telemetry: stats.TelemetryResult | None = None
    notes: list[str] = field(default_factory=list[str])


def _pings(rows: object) -> list[stats.Ping]:
    out: list[stats.Ping] = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            out.append(
                stats.Ping(
                    t_send_ns=int(row["t_send_ns"]),
                    t_recv_ns=int(row["t_recv_ns"]),
                    server_ns=int(row["server_ns"]),
                    server_mono_ns=int(row["server_mono_ns"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def browser_result(data: dict[str, Any] | None) -> BrowserResult:
    """The Playwright perf spec's JSON (P2 and P7); missing parts stay None."""
    res = BrowserResult()
    if data is None:
        res.notes.append("Webbläsarmätningen gav ingen resultatfil.")
        return res
    res.notes.extend(f"webbläsaren: {e}" for e in data.get("errors") or [] if isinstance(e, str))
    video = data.get("video")
    if isinstance(video, dict):
        fps = video.get("fps")
        if isinstance(fps, int | float) and math.isfinite(fps) and video.get("seconds"):
            res.video_fps = float(fps)
        res.video_detail = (
            f"{video.get('frames')} bilder på {video.get('seconds')} s, läge {video.get('mode')}"
        )
    tel = data.get("telemetry")
    if isinstance(tel, dict):
        arrivals: list[tuple[int, int]] = []
        for row in tel.get("arrivals") or []:
            with contextlib.suppress(TypeError, ValueError, IndexError):
                arrivals.append((int(row[0]), int(row[1])))
        res.telemetry = stats.telemetry_result(
            arrivals, _pings(tel.get("pings_start")), _pings(tel.get("pings_end"))
        )
    return res


def eval_report_path(output: str, repo: Path) -> Path | None:
    """The report file ``sd-brain eval`` names on its ``Rapport: <path>`` line."""
    for line in reversed(output.splitlines()):
        if line.startswith(EVAL_REPORT_PREFIX):
            path = Path(line.removeprefix(EVAL_REPORT_PREFIX).strip())
            return path if path.is_absolute() else repo / path
    return None


def model_result(path: Path | None, repo: Path) -> perf_report.ModelResult:
    """The model evaluation's latencies; a missing or broken report has no values."""
    if path is None or not path.is_file():
        return perf_report.ModelResult(None, None, "", False)
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        data = {}

    def ms(key: str) -> float | None:
        # sd-brain writes 0 when the model never answered; that is no data, not a fast pass.
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            return None
        return float(value)

    name = str(path.relative_to(repo)) if path.is_relative_to(repo) else path.name
    return perf_report.ModelResult(
        ms("latency_p50_ms"), ms("latency_p95_ms"), name, bool(data.get("passed"))
    )


def login_code(link: str) -> str:
    """The code from the login link (``...#login=<code>``); never printed."""
    _, sep, code = link.partition("#login=")
    if not sep or not code:
        raise InfraError("Inloggningslänken saknar #login=.")
    return code


def summary_lines(metrics: Sequence[perf_report.Metric]) -> list[str]:
    lines = [f"{'Mätpunkt':<8} {'Värde':>12}  {'Gräns':<22} Resultat"]
    for m in metrics:
        value = "-" if m.value is None else f"{m.value:.2f} {m.unit}".strip()
        verdict = {True: "OK", False: "FEL", None: "info"}[m.passed]
        lines.append(f"{m.id:<8} {value:>12}  {m.limit_sv:<22} {verdict}")
        if m.passed is False and m.detail:
            lines.append(f"{'':<8} {m.detail}")
    return lines


def run_exit_code(codes: Sequence[int]) -> int:
    return max(codes, default=1)


# --- live parts (not unit-tested) ------------------------------------------------------------


def _stamp(now: datetime) -> str:
    return now.strftime("%Y%m%dT%H%M%SZ")


def _iso(now: datetime) -> str:
    return now.isoformat(timespec="seconds").replace("+00:00", "Z")


def _git_dirty(repo: Path) -> bool:
    git = shutil.which("git")
    if git is None:
        return False
    return bool(_run_text([git, "-C", str(repo), "status", "--porcelain"]).strip())


def _logged(
    argv: list[str], log: Path, *, cwd: Path, timeout: float, env: dict[str, str] | None = None
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as out:
        try:
            res = subprocess.run(  # noqa: S603 - fixed argv, no shell
                argv,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return 124
    return res.returncode


def build_panel(repo: Path, logs: Path) -> None:
    say("==> panelen (pnpm build)")
    pnpm = shutil.which("pnpm") or "pnpm"
    argv = [pnpm, "--filter", "@searchdrone/panel", "build"]
    if _logged(argv, logs / "panel-build.log", cwd=repo, timeout=PANEL_BUILD_TIMEOUT_S) != 0:
        raise InfraError(f"Panelen gick inte att bygga. Se {logs / 'panel-build.log'}.")


def start_mediamtx(repo: Path, logs: Path) -> None:
    say("==> MediaMTX (docker compose up mediamtx)")
    data = read_env_value(repo / ".env", "SD_DATA_DIR")
    if not data:
        raise InfraError("SD_DATA_DIR saknas i .env. Kör just setup-config.")
    (Path(data).expanduser() / "video").mkdir(parents=True, exist_ok=True)
    docker = shutil.which("docker") or "docker"
    argv = [docker, "compose", "--env-file", ".env", "up", "-d", "--wait", "mediamtx"]
    if _logged(argv, logs / "mediamtx-up.log", cwd=repo, timeout=COMPOSE_TIMEOUT_S) != 0:
        raise InfraError(f"MediaMTX startade inte. Se {logs / 'mediamtx-up.log'}.")


class BrowserRun:
    """The Playwright perf spec in Google Chrome, in the background."""

    def __init__(self, repo: Path, run_dir: Path) -> None:
        self.repo = repo
        self.result = run_dir / "perf-browser.json"
        self.log = run_dir / "logs" / "playwright-perf.log"
        self.popen: subprocess.Popen[bytes] | None = None

    def start(self, base_url: str, code: str) -> None:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(
            SD_PERF_BASE_URL=base_url,
            SD_PERF_LOGIN_CODE=code,
            SD_PERF_RESULT=str(self.result),
            SD_PERF_VIDEO_S=str(VIDEO_S),
            SD_PERF_TELEMETRY_S=str(TELEMETRY_S),
        )
        pnpm = shutil.which("pnpm") or "pnpm"
        argv = [pnpm, "--filter", "@searchdrone/panel", "exec", "playwright", "test"]
        argv += ["-c", "playwright.perf.config.ts"]
        with self.log.open("ab") as out:
            self.popen = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                argv,
                cwd=self.repo,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    def wait(self, timeout_s: float) -> dict[str, Any] | None:
        if self.popen is not None:
            try:
                self.popen.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                say("Webbläsarmätningen blev inte klar i tid; avbryter den.")
            self.stop()
        try:
            data: Any = json.loads(self.result.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def stop(self) -> None:
        if self.popen is not None and self.popen.poll() is None:
            procs.stop_pid(self.popen.pid, self.popen)


def stop_brain(stack: Stack) -> None:
    """sd-brain stops so it does not share llama-server with the evaluation."""
    for proc in stack.group.procs:
        if proc.name == "sd-brain" and proc.alive():
            procs.stop_pid(proc.pid, proc.popen)


def run_eval(repo: Path, logs: Path, gpu: str) -> perf_report.ModelResult:
    say("==> modellens utvärdering (sd-brain eval, 3 varv; Gazebo kör vidare)")
    uv = shutil.which("uv") or "uv"
    argv = [uv, "run", "--package", "sd-brain", "sd-brain", "eval", "--backend", "llm_local"]
    argv += ["--repeat", "3", "--gpu-name", gpu, "--report-dir", "data/reports"]
    env = dict(os.environ)
    env["SD_BRAIN__CLOUD__MODEL"] = ""
    log = logs / "brain-eval.log"
    _logged(argv, log, cwd=repo, timeout=EVAL_TIMEOUT_S, env=env)
    output = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    return model_result(eval_report_path(output, repo), repo)


@contextlib.contextmanager
def _stopping(steps: list[Callable[[], None]]) -> Iterator[None]:
    """Run every clean-up step, newest first, whatever happened; a second Ctrl-C must
    not interrupt the clean-up itself."""
    try:
        yield
    finally:
        old_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        old_term = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            for step in reversed(steps):
                with contextlib.suppress(Exception):
                    step()
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)


def _acks_reader(tlog: Path) -> Callable[[], list[int]]:
    idx = tlog.with_suffix(".idx.jsonl")

    def read() -> list[int]:
        try:
            return tlogack.stop_acks(tlog, idx)
        except OSError:
            return []

    return read


@dataclass
class LiveRun:
    """What the live part of one run measured (partly filled when it broke off)."""

    timeline: Timeline = field(default_factory=Timeline)
    gz: samplers.GzStatsSampler | None = None
    stops: list[float] = field(default_factory=list[float])
    chat: list[float] = field(default_factory=list[float])
    browser: dict[str, Any] | None = None
    model: perf_report.ModelResult | None = None
    notes: list[str] = field(default_factory=list[str])


def fly_and_measure(
    args: PerfArgs, stack: Stack, live: LiveRun, cleanup: list[Callable[[], None]], gpu: str
) -> None:
    repo, run_dir = stack.repo, stack.run_dir
    logs = run_dir / "logs"
    build_panel(repo, logs)
    start_mediamtx(repo, logs)
    cleanup.append(stack.down)
    stack.up()
    live.timeline.mark("ready")
    say(f"Stacken är redo efter {live.timeline.cold_start_s:.0f} s.")
    if stack.world is not None:
        gz = samplers.GzStatsSampler(stack.world.name, INSTANCE, run_dir / "gz-stats.log")
        live.gz = gz
        cleanup.append(gz.stop)
        gz.start()
    else:
        live.notes.append("Gazebo kördes inte; P1 saknar mätvärden.")
    pilot = Pilot(stack)
    cleanup.append(pilot.close)
    pilot.fly_demo_until_airborne(AIRBORNE_TIMEOUT_S)
    live.timeline.mark("airborne")
    say("==> i luften; webbläsarmätningen (video och telemetri) startar")
    browser = BrowserRun(repo, run_dir)
    cleanup.append(browser.stop)
    base_url = f"http://127.0.0.1:{stack.ports['ground_panel']}"
    browser.start(base_url, login_code(stack.login_url))
    say(f"==> P6: STOPP {STOP_TRIES} gånger")
    live.stops, stop_notes = measure_stops(pilot, _acks_reader(run_dir / "radio.tlog"))
    live.notes.extend(stop_notes)
    say("==> väntar på att flygningen blir klar")
    pilot.wait_flight_done(FLIGHT_TIMEOUT_S)
    live.timeline.mark("done")
    live.browser = browser.wait(BROWSER_TIMEOUT_S)
    live.chat = pilot.chat_latencies_s(stack)
    if args.eval:
        stop_brain(stack)
        live.model = run_eval(repo, logs, gpu)


def run_once(args: PerfArgs, index: int, repo: Path = paths.REPO) -> int:
    started = datetime.now(UTC)
    say(f"=== Prestandamätning, varv {index} av {args.runs} ===")
    cfg = perf_config(args)
    stopped = procs.stop_recorded(paths.instance_dir(INSTANCE, repo) / "pids.json")
    if stopped:
        say("Stoppade kvarlämnade program: " + ", ".join(stopped))
    errors = preflight_errors(gather_preflight(repo, ports.for_instance(INSTANCE)))
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 2
    run_dir = new_run_dir(INSTANCE, "perf", repo)
    stack = Stack(cfg, run_dir, services=True, test_rig=True, cloud=False, repo=repo)

    gpu = samplers.gpu_name()
    ram_gb = samplers.ram_total_gb()
    win_ram_gb = samplers.windows_ram_total_gb()
    warnings = perf_report.config_warnings(
        repo / "config" / "sim.toml", gpu, win_ram_gb if win_ram_gb is not None else ram_gb
    )
    for warning in warnings:
        say(f"\033[33m{warning}\033[0m")

    live = LiveRun()
    ram = samplers.RamSampler()
    vram = samplers.VramSampler(run_dir / "vram.csv")
    win = samplers.WindowsWatch(
        run_dir / "windows-watch.csv",
        run_dir / "windows-watch.stop",
        repo / "tools" / "setup" / "windows-check.ps1",
    )
    cleanup: list[Callable[[], None]] = [ram.stop, vram.stop, win.stop]
    with _stopping(cleanup):
        live.timeline.mark("start")
        for sampler in (ram, vram, win):
            sampler.start()
        try:
            fly_and_measure(args, stack, live, cleanup, gpu)
        except StopRequestedError:
            raise  # Ctrl-C or `just stop`: tear down, no report, no further runs
        except InfraError as exc:
            # The report still shows what was measured; the missing parts fail.
            live.notes.append(f"Körningen avbröts: {stack.mask(str(exc)).splitlines()[0]}")
            say(stack.mask(str(exc)))
        except Exception as exc:  # the pilot's HTTP and timeout errors too
            live.notes.append(f"Körningen avbröts: {type(exc).__name__}: {stack.mask(str(exc))}")
            say(live.notes[-1])
        finally:
            live.timeline.mark("end")
    if args.eval and live.model is None:
        live.model = perf_report.ModelResult(None, None, "", False)  # never ran: no data
    if not args.eval:
        live.notes.append("eval=0: modellens svarstid mättes inte.")
    window = live.timeline.flight_window()
    rtf = live.gz.between(*window) if live.gz is not None and window is not None else []
    browser_res = browser_result(live.browser)
    measurements = perf_report.Measurements(
        cold_start_s=live.timeline.cold_start_s,
        rtf=rtf,
        video_fps=browser_res.video_fps,
        video_detail=browser_res.video_detail,
        wsl_used_mb=[mb for _, mb in ram.samples],
        win_available_mb=win.available_mb if win.available else None,
        vram_used_mb=vram.samples,
        stop_latencies_s=live.stops,
        telemetry=browser_res.telemetry,
        chat_latencies_s=live.chat,
        model=live.model,
        notes=live.notes + browser_res.notes,
    )
    return finish(
        args,
        cfg,
        measurements,
        repo=repo,
        gpu=gpu,
        ram_gb=ram_gb,
        win_ram_gb=win_ram_gb,
        warnings=warnings,
        started=started,
    )


def finish(
    args: PerfArgs,
    cfg: config.SimConfig,
    measurements: perf_report.Measurements,
    *,
    repo: Path,
    gpu: str,
    ram_gb: float,
    win_ram_gb: float | None,
    warnings: list[str],
    started: datetime,
) -> int:
    metrics = perf_report.evaluate(measurements)
    finished = datetime.now(UTC)
    report = perf_report.build_report(
        metrics,
        git_sha=sim_report.git_sha() or "",
        dirty=_git_dirty(repo),
        gpu_name=gpu,
        ram_total_gb=ram_gb,
        windows_ram_total_gb=win_ram_gb,
        settings=settings_for(args, cfg, cloud=False, test_rig=True),
        warnings=warnings + measurements.notes,
        fallback=fallback_for(args),
        started_utc=_iso(started),
        finished_utc=_iso(finished),
    )
    json_path, _ = perf_report.write_report(report, repo / "data" / "reports", _stamp(started))
    for line in summary_lines(metrics):
        say(line)
    for note in measurements.notes:
        say(f"Not: {note}")
    passed = bool(report.get("passed"))
    say(f"Rapport: {json_path.relative_to(repo) if json_path.is_relative_to(repo) else json_path}")
    if passed:
        say("Alla gränser höll.")
        return 0
    say("Minst en gräns höll inte.")
    if args.gui:
        say(FALLBACK_HINT)
    return 1


def _raise_stop(_sig: int, _frame: FrameType | None) -> None:
    if not procs.defer_stop():
        raise StopRequestedError


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    old_term = signal.signal(signal.SIGTERM, _raise_stop)
    codes: list[int] = []
    try:
        for index in range(1, args.runs + 1):
            code = run_once(args, index)
            codes.append(code)
            if code == 2:
                break
    except config.DemoHomeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (StopRequestedError, KeyboardInterrupt):
        say("Prestandamätningen avbröts.")
        return 1
    except InfraError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, old_term)
    return run_exit_code(codes)


if __name__ == "__main__":
    sys.exit(main())
