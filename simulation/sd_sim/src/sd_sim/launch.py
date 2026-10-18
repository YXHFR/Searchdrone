"""Simulation launcher: ``python -m sd_sim.launch {up|down|status} [--demo] [key=value ...]``.

``up`` starts the simulator stack for one instance in the foreground, waits for each
step to be healthy before the next and stops everything again on Ctrl-C. ``down``
stops what an earlier ``up`` started. Parts of the stack that are not built yet are
skipped with a message.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from pathlib import Path
from types import FrameType

from sd_sim import _args, config, gazebo, objects, paths, ports, procs, services, world
from sd_sim.config import SimConfig
from sd_sim.errors import InfraError, StopRequestedError
from sd_sim.sitl import SitlLayout, SitlOptions, build_sitl_argv, write_qgc_params
from sd_sim.util import LOOPBACK_OPENER, busy_ports, new_run_dir, say, udp_port_taken

ALLOWED_KEYS = (
    "physics",
    "gui",
    "video",
    "brain",
    "script",
    "instance",
    "link_profile",
    "qgc",
    "record",
    "lan",
    "private_db",
    "dry",
    "objects",
)
COMMANDS = ("up", "down", "status")
FLAGS = ("--demo", "--with-db")
# The database and the ground station start BEFORE SITL, so that they stop after it
# (processes stop in reverse start order and the ground station's heartbeat must end
# last); the onboard computer and the AI start after SITL.

HEALTH_LINK_S = 10.0
HEALTH_DB_S = 60.0
HEALTH_GROUND_S = 30.0
HEALTH_ONBOARD_S = 30.0
HEALTH_BRAIN_S = 30.0
HEALTH_LLM_S = 120.0
DEMO_SCRIPT = "simulation/scenarios/scripts/SC01.yaml"
HEALTH_HEARTBEAT_S = 60.0
HEALTH_READY_S = 60.0
HEALTH_VIDEO_S = 15.0
VIDEO_VEHICLE_ID = "sim-01"
VIDEO_SECRET = "mediamtx_publish"  # noqa: S105 - a file name, not a secret
VIDEO_PASSWORD_KEY = "SD_GROUND__VIDEO_PUBLISH_PASSWORD"  # noqa: S105 - a variable name
VIDEO_API_PASSWORD_KEY = "SD_GROUND__VIDEO_API_PASSWORD"  # noqa: S105 - a variable name
VIDEO_API_USER = "ground"
PLACEHOLDERS = {"", "change_me", "change-me", "changeme"}


def read_env_value(path: Path, key: str) -> str:
    """One value from ``.env`` (the process environment wins), without reading the rest."""
    if os.environ.get(key):
        return os.environ[key]
    if not path.is_file():
        return ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        name, sep, value = raw.strip().partition("=")
        if sep and name.strip() == key:
            return value.strip().strip('"').strip("'")
    return ""


def write_secret(directory: Path, name: str, value: str) -> Path:
    """Write a secret file with mode 0600 inside a 0700 folder."""
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    path = directory / name
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        out.write(value)
    path.chmod(0o600)
    return path


def check_secret_modes(directory: Path) -> None:
    for path in directory.iterdir():
        if path.stat().st_mode & 0o777 != 0o600:
            raise InfraError(f"{path.name} i hemlighetsmappen har fel rättigheter (ska vara 0600)")


class Stack:
    """The running simulator for one instance."""

    def __init__(
        self,
        cfg: SimConfig,
        run_dir: Path,
        *,
        observer: bool = True,
        repo: Path = paths.REPO,
        param_layers: tuple[Path, ...] = (),
        services: bool = False,
        test_rig: bool = False,
    ) -> None:
        self.cfg = cfg
        # services: also start the database, sd-ground, sd-onboard and sd-brain (`just
        # sim`, `just demo` and the end-to-end test); without, SITL and its links only.
        self.services = services
        # test_rig: the ground station lets CLI tokens confirm (the end-to-end test's
        # pilot) and the launcher writes a pilot and an observer token.
        self.test_rig = test_rig
        self.param_layers = param_layers  # a test's own layers, loaded last
        self.run_dir = run_dir
        self.repo = repo
        self.observer = observer
        self.ports = ports.for_instance(cfg.instance)
        self.inst_dir = paths.instance_dir(cfg.instance, repo)
        self.secrets_dir = self.inst_dir / "secrets"
        self.group = procs.ProcessGroup(self.inst_dir / "pids.json", run_dir / "logs")
        self.linksim_token = ""
        self.world: gazebo.World | None = None
        self.placed: list[objects.WorldObject] = []
        self.login_url = ""
        self._ground_env: dict[str, str] = {}
        # How each service was started (argv, environment), for a test's restart.
        self._specs: dict[str, tuple[list[str], dict[str, str]]] = {}

    # --- helpers -------------------------------------------------------------------

    def _fail(self, step: str, proc: procs.Proc | None, hint: str) -> InfraError:
        tail = ""
        if proc is not None and proc.log.is_file():
            lines = proc.log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
            tail = "\n".join(self.mask(line) for line in lines)
        return InfraError(f"Steget '{step}' blev inte friskt. {hint}\n{tail}".rstrip())

    def mask(self, text: str) -> str:
        if self.linksim_token:
            text = text.replace(self.linksim_token, "***")
        return text

    def _wait(self, step: str, proc: procs.Proc | None, timeout: float, check, hint: str) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if proc is not None and not proc.alive():
                raise self._fail(step, proc, "Processen avslutades.")
            if check():
                return
            time.sleep(0.25)
        raise self._fail(step, proc, hint)

    def linksim_get(self, path: str) -> dict[str, object]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.ports['linksim_control']}{path}",
            headers={"Authorization": f"Bearer {self.linksim_token}"},
        )
        with LOOPBACK_OPENER.open(req, timeout=2) as resp:
            return json.loads(resp.read())

    def _linksim_healthy(self) -> bool:
        try:
            return self.linksim_get("/health").get("ok") is True
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def harness_state(self) -> dict[str, object]:
        try:
            return json.loads((self.inst_dir / "harness_state.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return {}

    # --- steps ---------------------------------------------------------------------

    def check_ports_free(self) -> None:
        names = [
            "mavtap_in",
            "radio_mavlink_in",
            "video_raw_udp",
            "linksim_sdlink",
            "linksim_control",
        ]
        if self.observer:  # otherwise the caller's own harness already holds the port
            names.append("harness_mavlink")
        busy = busy_ports(self.ports, names)
        if busy:
            ports_text = ", ".join(str(p) for p in busy)
            raise InfraError(f"Port {ports_text} är upptagen. Kör just stop och försök igen.")

    def make_secrets(self) -> None:
        self.linksim_token = secrets.token_urlsafe(32)
        write_secret(self.secrets_dir, "linksim_token", self.linksim_token)
        check_secret_modes(self.secrets_dir)

    def renders_world(self) -> bool:
        """Only the Searchdrone world is rendered here, and only with Gazebo physics."""
        return (
            self.cfg.physics == "gz"
            and self.cfg.vehicle == "sd_quad14"
            and self.cfg.world == world.WORLD_NAME
        )

    def world_objects(self) -> list[objects.WorldObject]:
        """Load and check ``objects=`` before anything starts; refuse it where it cannot apply."""
        if not self.cfg.objects:
            return []
        if not self.renders_world():
            raise InfraError(
                f"objects= kräver physics=gz, vehicle=sd_quad14 och världen {world.WORLD_NAME}."
            )
        path = (self.repo / self.cfg.objects).resolve()
        if not path.is_relative_to(self.repo.resolve()):
            raise InfraError(f"objects={self.cfg.objects}: filen ska ligga i repot")
        try:
            placed: list[objects.WorldObject] = list(objects.load(path))
            # The same checks the renderer makes, so a bad file stops here, not in Gazebo.
            objects.check(placed, world.GROUND_SIZE_M / 2)
            for obj in placed:
                world.checked_fragment(obj)
        except objects.ObjectError as exc:
            raise InfraError(f"objects={self.cfg.objects}: {exc}") from exc
        return placed

    def start_gazebo(self) -> None:
        if self.renders_world():
            placed = self.placed
            world.render_world(self.cfg.world, self.cfg.instance, self.repo, placed)
            if placed:
                say(f"{len(placed)} objekt i världen, facit i build/sim/worlds/")
        self.world = gazebo.resolve_world(self.cfg, self.repo)
        if self.world.note:
            say(self.world.note)
        argv = gazebo.gz_sim_argv(self.world, gui=self.cfg.gui)
        env = gazebo.gz_env(self.cfg.instance, gui=self.cfg.gui)
        proc = self.group.start("gazebo", argv, env=env, cwd=self.run_dir)
        running = self.world
        self._wait(
            "gazebo",
            proc,
            60,
            lambda: gazebo.world_running(running, self.cfg.instance),
            "Gazebo svarar inte. Se reservvägarna i installationsguiden (gui=false).",
        )

    def start_link_processes(self) -> None:
        py = sys.executable
        inst = str(self.cfg.instance)
        env = procs.clean_env()
        if self.observer:
            self.group.start(
                "harness",
                [
                    py,
                    "-m",
                    "sd_sim.harness",
                    "observe",
                    "--instance",
                    inst,
                    "--state-file",
                    str(self.inst_dir / "harness_state.json"),
                    *(["--rc-port", str(self.ports["sitl_rc_in"])] if self.services else []),
                ],
                env=env,
            )
        self.group.start(
            "mavtap",
            [
                py,
                "-m",
                "sd_sim.mavtap",
                "--instance",
                inst,
                "--tlog",
                str(self.run_dir / "mavtap_onboard.tlog"),
            ],
            env=env,
        )
        linksim = self.group.start(
            "linksim",
            [
                py,
                "-m",
                "sd_sim.linksim",
                "--instance",
                inst,
                "--token-file",
                str(self.secrets_dir / "linksim_token"),
                "--radio-tlog",
                str(self.run_dir / "radio.tlog"),
                "--profile",
                self.cfg.link_profile,
            ],
            env=env,
        )
        self._wait(
            "linksim",
            linksim,
            HEALTH_LINK_S,
            lambda: (
                self._linksim_healthy()
                and udp_port_taken(self.ports["mavtap_in"])
                and (not self.observer or udp_port_taken(self.ports["harness_mavlink"]))
            ),
            "linksim eller mavtap svarar inte.",
        )

    def sitl_layout(self) -> SitlLayout:
        extra: tuple[Path, ...] = ()
        if self.cfg.qgc and self.cfg.instance == 0:
            extra = (write_qgc_params(self.inst_dir),)
        return SitlLayout(
            arducopter=paths.arducopter_bin(),
            ardupilot_home=paths.ardupilot_home(),
            repo=self.repo,
            extra_defaults=extra,
            test_layers=self.param_layers,
        )

    def sitl_argv(self) -> list[str]:
        layout = self.sitl_layout()
        opts = SitlOptions(
            instance=self.cfg.instance,
            physics=self.cfg.physics,
            vehicle=self.cfg.vehicle,
            speedup=self.cfg.speedup,
            home=self.cfg.home,
            qgc=self.cfg.qgc,
        )
        return build_sitl_argv(opts, self.ports, layout)

    def start_sitl(self) -> procs.Proc:
        argv = self.sitl_argv()
        if not Path(argv[0]).is_file():
            raise InfraError(f"{argv[0]} finns inte. Kör just sim-setup först.")
        work = self.run_dir / "sitl"
        work.mkdir(parents=True, exist_ok=True)
        if self.cfg.physics == "builtin" and self.cfg.vehicle == "sd_quad14":
            self.sitl_layout().stage_frame(work)
        return self.group.start("arducopter", argv, env=procs.clean_env(), cwd=work)

    def wait_vehicle_observed(self, sitl: procs.Proc) -> None:
        self._wait(
            "arducopter",
            sitl,
            HEALTH_HEARTBEAT_S,
            lambda: bool(self.harness_state().get("heartbeat")),
            "Ingen HEARTBEAT från flygkontrollern.",
        )
        self._wait(
            "arducopter",
            sitl,
            HEALTH_READY_S,
            lambda: bool(self.harness_state().get("ready")),
            "Flygkontrollen fick ingen position (EKF och GPS).",
        )

    def start_video_source(self) -> None:
        if self.cfg.instance != 0 or self.cfg.video == "off":
            return
        if self.cfg.video == "camera" and self.world is not None:
            gazebo.enable_camera(self.cfg.instance)
        elif self.cfg.video == "test":
            self.group.start(
                "testvideo",
                [
                    sys.executable,
                    "-m",
                    "sd_sim.testvideo",
                    "--port",
                    str(self.ports["video_raw_udp"]),
                ],
                env=procs.clean_env(),
            )

    def mediamtx_get(self, path: str) -> dict[str, object] | None:
        """MediaMTX's API (instance 0 only), as the user ``ground``; ``None`` when it does
        not answer. A refused password is an error, not "not running"."""
        password = read_env_value(self.repo / ".env", VIDEO_API_PASSWORD_KEY)
        token = base64.b64encode(f"{VIDEO_API_USER}:{password}".encode()).decode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.ports['mediamtx_api']}{path}",
            headers={"Authorization": f"Basic {token}"},
        )
        try:
            with LOOPBACK_OPENER.open(req, timeout=2) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == HTTPStatus.UNAUTHORIZED:
                raise InfraError(
                    f"MediaMTX nekade {VIDEO_API_PASSWORD_KEY}. Starta om med just video-up."
                ) from exc
            return None
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def _video_ready(self) -> bool:
        state = self.mediamtx_get(f"/v3/paths/get/drone/{VIDEO_VEHICLE_ID}/main")
        return state is not None and state.get("ready") is True

    def start_video_relay(self) -> None:
        """``video_relay`` publishes to MediaMTX (instance 0, when MediaMTX runs)."""
        if self.cfg.instance != 0 or self.cfg.video == "off":
            return
        if self.mediamtx_get("/v3/config/global/get") is None:
            say("hoppar över video_relay (MediaMTX kör inte; starta den med just video-up)")
            return
        password = read_env_value(self.repo / ".env", VIDEO_PASSWORD_KEY)
        if password.strip().lower() in PLACEHOLDERS:
            raise InfraError(f"{VIDEO_PASSWORD_KEY} saknas i .env. Kör just setup-config.")
        secret = write_secret(self.secrets_dir, VIDEO_SECRET, password)
        relay = self.group.start(
            "video_relay",
            [
                sys.executable,
                "-m",
                "sd_sim.video_relay",
                "--instance",
                str(self.cfg.instance),
                "--secret-file",
                str(secret),
                "--vehicle-id",
                VIDEO_VEHICLE_ID,
            ],
            env=procs.clean_env(),
        )
        self._wait(
            "video_relay",
            relay,
            HEALTH_VIDEO_S,
            self._video_ready,
            "MediaMTX fick ingen video. Kontrollera kameran/testbilden och lösenordet i .env.",
        )

    # --- Searchdrone services -------------------------------------------------------

    @property
    def video_on(self) -> bool:
        """Video for this instance: instance 0 only (one MediaMTX path per vehicle)."""
        return self.cfg.instance == 0 and self.cfg.video != "off"

    def stack_paths(self) -> services.StackPaths:
        # With video the ground station must use .env's data folder, where MediaMTX
        # records; otherwise each run keeps its own (and its own control socket).
        configured = read_env_value(self.repo / ".env", "SD_DATA_DIR")
        if not configured:
            raise InfraError("SD_DATA_DIR saknas i .env. Kör just setup-config.")
        base = Path(configured).expanduser()
        data = base if self.video_on else self.run_dir / "data"
        onboard = base / "sim" / f"i{self.cfg.instance}" / "onboard"
        return services.StackPaths(self.repo, self.run_dir, self.secrets_dir, data, onboard)

    def _env_secret(self, key: str) -> str:
        value = read_env_value(self.repo / ".env", key)
        if value.strip().lower() in PLACEHOLDERS:
            raise InfraError(f"{key} saknas i .env. Kör just setup-config.")
        return value

    def _compose(self, *args: str) -> bool:
        """``docker compose --env-file .env ...`` in the repository; False without Docker."""
        docker = shutil.which("docker")
        if docker is None:
            return False
        try:
            res = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [docker, "compose", "--env-file", ".env", *args],
                cwd=self.repo,
                capture_output=True,
                check=False,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            return False
        return res.returncode == 0

    def start_database(self) -> None:
        """Postgres (compose service ``db``) and a fresh database for this instance."""
        if not services.tcp_open(services.DB_PORT):
            say("startar databasen (docker compose up db)")
            if not self._compose("up", "-d", "--wait", "db"):
                raise InfraError(
                    f"Databasen svarar inte på 127.0.0.1:{services.DB_PORT}. Kör just db-up."
                )
        end = time.monotonic() + HEALTH_DB_S
        while not services.tcp_open(services.DB_PORT):
            if time.monotonic() > end:
                raise InfraError("Databasen blev inte redo inom 60 s. Se docker compose logs db.")
            time.sleep(0.5)
        name = services.database_name(self.cfg.instance)
        services.recreate_database(self._env_secret("SD_GROUND__DB_PASSWORD"), name)

    def provision(self) -> None:
        """Schema, the vehicle, the commander and link tokens (and the test rig's tokens)."""
        sp = self.stack_paths()
        for folder in (sp.data_dir, sp.onboard_dir):
            folder.mkdir(parents=True, exist_ok=True, mode=0o700)
            folder.chmod(0o700)
        write_secret(self.secrets_dir, "brain_token", secrets.token_urlsafe(32))
        env = self.ground_process_env()
        cli = services.ground_cli
        cli(self.repo, env, "db", "setup")
        cli(
            self.repo,
            env,
            "vehicle",
            "add",
            services.VEHICLE_ID,
            "--kind",
            "sim",
            "--sysid",
            "1",
            "--name",
            "Sim 1",
            "--if-missing",
        )
        cli(
            self.repo,
            env,
            "link-token",
            "create",
            "--vehicle",
            services.VEHICLE_ID,
            "--write-onboard",
            str(sp.link_token),
        )
        if self.test_rig:
            cli(
                self.repo,
                env,
                "token",
                "create",
                "--name",
                "harness",
                "--role",
                "pilot",
                "--expires",
                "24h",
                stdout_file=self.secrets_dir / "pilot_token",
            )
            cli(
                self.repo,
                env,
                "token",
                "create",
                "--name",
                "observer",
                "--role",
                "observer",
                "--expires",
                "24h",
                stdout_file=self.secrets_dir / "observer_token",
            )
        if self.cfg.lan:
            tls = self.repo / "config" / "local" / "tls"
            if not (tls / "server.crt").is_file():
                cli(self.repo, env, "init", "--lan")
        check_secret_modes(self.secrets_dir)

    def ground_process_env(self) -> dict[str, str]:
        if not self._ground_env:
            self._ground_env = procs.clean_env(
                services.ground_env(
                    self.ports,
                    self.stack_paths(),
                    self.cfg.instance,
                    video=self.video_on and self._mediamtx_running(),
                    lan=self.cfg.lan,
                    test_rig=self.test_rig,
                )
            )
        return self._ground_env

    def _mediamtx_running(self) -> bool:
        return self.mediamtx_get("/v3/config/global/get") is not None

    def health(self) -> services.Health | None:
        return services.read_health(self.ports["ground_panel"])

    def _health_ok(self, *parts: str) -> bool:
        h = self.health()
        return h is not None and h.ok(*parts)

    def _start_service(self, name: str, argv: list[str], env: dict[str, str]) -> procs.Proc:
        self._specs[name] = (argv, env)
        return self.group.start(name, argv, env=env, cwd=self.repo)

    def kill_service(self, name: str) -> None:
        """A test's fault: the service dies at once (SIGKILL), without a goodbye."""
        self.group.kill(name)

    def restart_service(self, name: str) -> procs.Proc:
        """A test's restart of a service with the same settings, in its old stop order."""
        argv, env = self._specs[name]
        return self.group.restart(name, argv, env=env, cwd=self.repo)

    def start_ground(self) -> procs.Proc:
        env = self.ground_process_env()
        proc = self._start_service("sd-ground", [sys.executable, "-m", "sd_ground", "serve"], env)
        self._wait(
            "sd-ground",
            proc,
            HEALTH_GROUND_S,
            lambda: self._health_ok("db"),
            "Markstationen svarar inte på /api/v1/health. Se loggen.",
        )
        return proc

    def wait_ground_sees_vehicle(self, ground: procs.Proc) -> None:
        self._wait(
            "sd-ground",
            ground,
            HEALTH_HEARTBEAT_S,
            lambda: self._health_ok("mavlink"),
            "Markstationen ser ingen HEARTBEAT från flygkontrollern (linksim).",
        )

    def start_onboard(self) -> None:
        env = procs.clean_env(services.onboard_env(self.ports, self.stack_paths()))
        proc = self._start_service("sd-onboard", [sys.executable, "-m", "sd_onboard"], env)
        self._wait(
            "sd-onboard",
            proc,
            HEALTH_ONBOARD_S,
            lambda: self._health_ok("sdlink", "onboard"),
            "Ombordsdatorn kopplade inte upp SD-link eller syns inte på MAVLink.",
        )

    def brain_script(self) -> Path | None:
        if self.cfg.brain != "scripted":
            return None
        script = self.cfg.script or DEMO_SCRIPT
        path = Path(script)
        return path if path.is_absolute() else self.repo / path

    def start_brain(self) -> None:
        kind = self.cfg.brain
        if kind == "none":
            say("hoppar över sd-brain (brain=none)")
            return
        timeout = HEALTH_BRAIN_S
        if kind == "llm_local":
            timeout = HEALTH_LLM_S
            if not services.tcp_open(self.ports["llm_http"]):
                say("llama-server kör inte; försöker starta den (docker compose --profile ai)")
                if not self._compose("--profile", "ai", "up", "-d", "--wait", "llm"):
                    say("llama-server startade inte; AI:n får använda molnet eller kommandoläget")
        env = procs.clean_env(
            services.brain_env(self.ports, self.stack_paths(), kind, self.brain_script())
        )
        config_file = self.repo / "config" / "brain.toml"
        if not config_file.is_file():
            config_file = self.repo / "config" / "brain.example.toml"
        proc = self._start_service(
            "sd-brain", [sys.executable, "-m", "sd_brain", "run", "--config", str(config_file)], env
        )
        self._wait(
            "sd-brain",
            proc,
            timeout,
            lambda: self._health_ok("brain"),
            "AI-befälhavaren kopplade inte upp mot markstationen.",
        )

    def wait_ready(self) -> None:
        """``GET /api/v1/health``: ``ready = true``, and every started part ``ok``."""
        wanted = ["db", "mavlink", "sdlink", "onboard"]
        if self.cfg.brain != "none":
            wanted.append("brain")

        def ready() -> bool:
            h = self.health()
            return h is not None and h.ready and h.ok(*wanted)

        self._wait("hälsa", None, HEALTH_READY_S, ready, "Markstationen blev inte redo.")

    def make_login_link(self) -> str:
        res = services.ground_cli(self.repo, self.ground_process_env(), "login-link")
        for line in reversed(res.stdout.splitlines()):
            if "#login=" in line:
                return line.split()[-1]
        raise InfraError("sd-ground login-link gav ingen länk.")

    def up(self) -> None:
        os.umask(0o077)
        self.placed = self.world_objects()
        if not self.renders_world():
            world.clear_truth(self.cfg.instance, self.repo)  # no world, so no facit
        self.check_ports_free()
        self.make_secrets()
        self.inst_dir.mkdir(parents=True, exist_ok=True)
        (self.inst_dir / "ports.json").write_text(json.dumps(self.ports, indent=2), "utf-8")
        if self.cfg.physics == "gz":
            say("==> Gazebo")
            self.start_gazebo()
        say("==> testrigg, mavtap och linksim")
        self.start_link_processes()
        ground: procs.Proc | None = None
        if self.services:
            say("==> databasen")
            self.start_database()
            self.provision()
            say("==> sd-ground")
            ground = self.start_ground()
        say("==> arducopter (SITL)")
        sitl = self.start_sitl()
        if self.observer:
            self.wait_vehicle_observed(sitl)
        if ground is not None:
            self.wait_ground_sees_vehicle(ground)
            say("==> sd-onboard")
            self.start_onboard()
            say("==> sd-brain")
            self.start_brain()
        self.start_video_source()
        self.start_video_relay()
        if ground is not None:
            self.wait_ready()
            self.login_url = self.make_login_link()

    def down(self) -> None:
        self.group.stop_all()


_stopping = False


def _raise_stop(_sig: int, _frame: FrameType | None) -> None:
    # Only the first stop raises: a second signal (timeout and uv both pass one on) must
    # not land in the clean-up and leave the processes running.
    global _stopping
    if _stopping:
        return
    if not procs.defer_stop():
        _stopping = True
        raise StopRequestedError


def _run_foreground(stack: Stack) -> int:
    global _stopping
    # The handlers raise, so a stop also interrupts a health wait during start-up.
    signal.signal(signal.SIGTERM, _raise_stop)
    signal.signal(signal.SIGINT, _raise_stop)
    try:
        started = time.monotonic()
        stack.up()
        say(f"Klart. Simulatorn kör (instans {stack.cfg.instance}). Loggar: {stack.run_dir}")
        if stack.login_url:
            took = time.monotonic() - started
            say(f"Allt är redo efter {took:.0f} s. Öppna panelen: {stack.login_url}")
            if stack.cfg.open_browser and stack.cfg.gui:
                open_browser(stack.login_url)
        say("Stoppa med Ctrl-C eller just stop.")
        while True:
            for proc in stack.group.procs:
                if not proc.alive():
                    say(f"{proc.name} avslutades oväntat. Se {proc.log}.")
                    return 1
            time.sleep(0.5)
    except StopRequestedError:
        say("Stoppar simulatorn.")
        return 0
    except InfraError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        # A second Ctrl-C must not interrupt the clean-up itself.
        _stopping = True
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        stack.down()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        _args.fail(f"sd_sim.launch: ange ett av {', '.join(COMMANDS)}.")
    command, rest = argv[0], argv[1:]
    flags = [a for a in rest if a.startswith("--")]
    for flag in flags:
        if flag not in FLAGS:
            _args.fail(f"sd_sim.launch: okänd flagga '{flag}'. Inställningar skrivs nyckel=värde.")
    values = _args.parse([a for a in rest if not a.startswith("--")], ALLOWED_KEYS, "sd_sim.launch")
    if "record" in values and "--demo" not in flags:
        _args.fail("sd_sim.launch: record= fungerar bara med just demo.")
    demo = "--demo" in flags
    dry = values.pop("dry", "") == "1" or os.environ.get("SD_SIM__DRY_RUN") == "1"
    values.pop("record", None)
    try:
        cfg = config.load(values, demo=demo)
    except config.DemoHomeError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except config.ConfigError as exc:
        _args.fail(f"sd_sim.launch: {exc}")
    if dry:
        print(json.dumps({"command": command, "flags": flags, "args": rest}))
        return 0
    if command == "down":
        stopped = procs.stop_recorded(paths.instance_dir(cfg.instance) / "pids.json")
        say("Stoppade: " + (", ".join(stopped) if stopped else "inget kördes"))
        return 0
    if command == "status":
        return _status(cfg.instance)
    if cfg.personal_home:
        say(f"\033[33m{config.PERSONAL_HOME_WARNING}\033[0m")
    if shutil.which("ss") is None:
        say("Varning: ss saknas, portkontrollen hoppas över (installera iproute2).")
    stack = Stack(cfg, new_run_dir(cfg.instance, "demo" if demo else "sim"), services=True)
    return _run_foreground(stack)


def open_browser(url: str) -> None:
    """Open the login link in the owner's browser (WSL: Windows' browser); best effort."""
    for opener in ("wslview", "xdg-open"):
        exe = shutil.which(opener)
        if exe is None:
            continue
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.Popen(  # noqa: S603 - a fixed opener and our own loopback URL
                [exe, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return


def _status(instance: int) -> int:
    rows = procs.read_recorded(paths.instance_dir(instance) / "pids.json").get("procs", [])
    if not rows:
        say(f"Instans {instance}: inget kör.")
        return 0
    for row in rows:
        alive = Path(f"/proc/{row['pid']}").exists()
        say(f"{row['name']:<12} pid {row['pid']:<8} {'kör' if alive else 'stoppad'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
