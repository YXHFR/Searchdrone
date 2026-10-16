"""Simulation launcher: ``python -m sd_sim.launch {up|down|status} [--demo] [key=value ...]``.

``up`` starts the simulator stack for one instance in the foreground, waits for each
step to be healthy before the next and stops everything again on Ctrl-C. ``down``
stops what an earlier ``up`` started. Parts of the stack that are not built yet are
skipped with a message.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import FrameType

from sd_sim import _args, config, gazebo, paths, ports, procs
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
)
COMMANDS = ("up", "down", "status")
FLAGS = ("--demo", "--with-db")
# Parts of the stack that later work adds; until then `up` says it skips them.
# The database and the ground station start BEFORE SITL, so that they stop after it
# (processes stop in reverse start order and the ground station's heartbeat must end
# last); the rest start after SITL.
NOT_BUILT_BEFORE_SITL = ("postgres", "sd-ground")
NOT_BUILT_AFTER_SITL = ("sd-onboard", "sd-brain", "video_relay")

HEALTH_LINK_S = 10.0
HEALTH_HEARTBEAT_S = 60.0
HEALTH_READY_S = 60.0


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
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.repo = repo
        self.observer = observer
        self.ports = ports.for_instance(cfg.instance)
        self.inst_dir = paths.instance_dir(cfg.instance, repo)
        self.secrets_dir = self.inst_dir / "secrets"
        self.group = procs.ProcessGroup(self.inst_dir / "pids.json", run_dir / "logs")
        self.linksim_token = ""
        self.world: gazebo.World | None = None

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

    def start_gazebo(self) -> None:
        self.world = gazebo.resolve_world(self.cfg, self.repo)
        if self.world.note:
            say(self.world.note)
        argv = gazebo.gz_sim_argv(self.world, gui=self.cfg.gui)
        env = gazebo.gz_env(self.cfg.instance, gui=self.cfg.gui)
        proc = self.group.start("gazebo", argv, env=env, cwd=self.run_dir)
        world = self.world
        self._wait(
            "gazebo",
            proc,
            60,
            lambda: gazebo.world_running(world, self.cfg.instance),
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

    def sitl_argv(self) -> list[str]:
        extra: tuple[Path, ...] = ()
        if self.cfg.qgc and self.cfg.instance == 0:
            extra = (write_qgc_params(self.inst_dir),)
        layout = SitlLayout(
            arducopter=paths.arducopter_bin(),
            ardupilot_home=paths.ardupilot_home(),
            repo=self.repo,
            extra_defaults=extra,
        )
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
                gazebo.test_video_argv(self.ports["video_raw_udp"]),
                env=procs.clean_env(),
            )

    def up(self) -> None:
        os.umask(0o077)
        self.check_ports_free()
        self.make_secrets()
        self.inst_dir.mkdir(parents=True, exist_ok=True)
        (self.inst_dir / "ports.json").write_text(json.dumps(self.ports, indent=2), "utf-8")
        if self.cfg.physics == "gz":
            say("==> Gazebo")
            self.start_gazebo()
        say("==> testrigg, mavtap och linksim")
        self.start_link_processes()
        for part in NOT_BUILT_BEFORE_SITL:
            say(f"hoppar över {part} (inte byggt än)")
        say("==> arducopter (SITL)")
        sitl = self.start_sitl()
        if self.observer:
            self.wait_vehicle_observed(sitl)
        for part in NOT_BUILT_AFTER_SITL:
            say(f"hoppar över {part} (inte byggt än)")
        self.start_video_source()

    def down(self) -> None:
        self.group.stop_all()


def _raise_stop(_sig: int, _frame: FrameType | None) -> None:
    if not procs.defer_stop():
        raise StopRequestedError


def _run_foreground(stack: Stack) -> int:
    # The handlers raise, so a stop also interrupts a health wait during start-up.
    signal.signal(signal.SIGTERM, _raise_stop)
    signal.signal(signal.SIGINT, _raise_stop)
    try:
        stack.up()
        say(f"Klart. Simulatorn kör (instans {stack.cfg.instance}). Loggar: {stack.run_dir}")
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
    stack = Stack(cfg, new_run_dir(cfg.instance))
    return _run_foreground(stack)


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
