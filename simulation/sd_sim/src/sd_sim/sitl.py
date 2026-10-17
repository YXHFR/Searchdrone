"""The ArduPilot SITL command line, built by one pure function.

Every serial port SITL uses is a ``udpclient`` to 127.0.0.1, so SITL opens no TCP port
for them (its default for serial0 is a TCP listener on all addresses that also makes
SITL wait for a connection). In ArduPilot 4.7.1 serial3 and serial4 are the simulated
GPS receivers and there is no serial9, so those are never overridden. The optional
QGroundControl link therefore uses serial5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sd_sim.errors import InfraError

# ArduPilot's public SITL site (CMAC). Never a private address.
CMAC_HOME = "-35.363261,149.165230,584.0,353"
QGC_SERIAL = "--serial5"
# The project's parameter layers, loaded after every generated layer: the safety
# parameters (failsafes, fence, battery) and the sim-only parameters. Both are required:
# a missing layer stops the start instead of silently flying without it.
SAFETY_PARAM_FILE = "searchdrone-safety.parm"
SIM_PARAM_FILE = "sd_quad14-sim.parm"
PROJECT_PARAM_FILES = (SAFETY_PARAM_FILE, SIM_PARAM_FILE)
# SITL strips the leading "/" from a frame file path and opens it relative to its working
# directory, so the launcher copies the frame file there and passes the bare name.
FRAME_FILE = "sd_quad14.json"


@dataclass(frozen=True)
class SitlLayout:
    """Files the command line points at. The project's parameter layers are required."""

    arducopter: Path
    ardupilot_home: Path
    repo: Path
    extra_defaults: tuple[Path, ...] = field(default_factory=tuple)
    # Per-test layers (``simulation/sitl/params/tests/<test>.parm``), loaded last.
    test_layers: tuple[Path, ...] = field(default_factory=tuple)

    @property
    def frame_json(self) -> Path:
        return self.repo / "simulation" / "sitl" / "frames" / FRAME_FILE

    def stage_frame(self, workdir: Path) -> Path:
        """Copy the builtin-physics frame file into SITL's working directory."""
        if not self.frame_json.is_file():
            raise InfraError(f"{self.frame_json} saknas: modellen sd_quad14 kan inte startas.")
        workdir.mkdir(parents=True, exist_ok=True)
        target = workdir / FRAME_FILE
        target.write_bytes(self.frame_json.read_bytes())
        return target

    def defaults(self, physics: str) -> list[Path]:
        """Parameter layers, later files win: ArduPilot's sim defaults, Gazebo's (gz
        only), extras (e.g. the QGC port), then the safety parameters and the sim-only
        parameters. The safety layer comes after every generated layer, so nothing
        generated at run time can override a failsafe."""
        layers = [self.ardupilot_home / "Tools" / "autotest" / "default_params" / "copter.parm"]
        if physics == "gz":
            layers.append(
                self.ardupilot_home / "Tools" / "autotest" / "default_params" / "gazebo-iris.parm"
            )
        layers.extend(self.extra_defaults)
        layers.extend(self.project_layers())
        layers.extend(self.checked_test_layers())
        return layers

    def checked_test_layers(self) -> list[Path]:
        """A test's own parameter layer must live in ``simulation/sitl/params/tests/``
        (where ``tools/params_check.py`` checks it), and exist."""
        tests = (self.repo / "simulation" / "sitl" / "params" / "tests").resolve()
        out = []
        for path in self.test_layers:
            p = path.resolve()
            if p.parent != tests or not p.is_file():
                raise InfraError(f"Testets parameterlager måste ligga i {tests}: {path}")
            out.append(p)
        return out

    def project_layers(self) -> list[Path]:
        """The project's own parameter files in load order (later wins). Every one must
        exist; a missing layer raises instead of being skipped."""
        params = self.repo / "simulation" / "sitl" / "params"
        layers = [params / name for name in PROJECT_PARAM_FILES]
        missing = [p.name for p in layers if not p.is_file()]
        if missing:
            raise InfraError(f"Parameterlager saknas i {params}: {', '.join(missing)}")
        return layers


@dataclass(frozen=True)
class SitlOptions:
    instance: int = 0
    physics: str = "builtin"  # "builtin" | "gz"
    vehicle: str = "sd_quad14"  # "sd_quad14" | "stock" (ArduPilot's own quad / iris)
    speedup: int = 1
    home: str = CMAC_HOME
    qgc: bool = False


def build_sitl_argv(opts: SitlOptions, ports: dict[str, int], layout: SitlLayout) -> list[str]:
    if opts.physics not in ("builtin", "gz"):
        raise ValueError(f"okänd fysik {opts.physics!r}")
    argv = [
        str(layout.arducopter),
        "-I",
        str(opts.instance),
        "--speedup",
        str(opts.speedup),
        "--home",
        opts.home,
        "--wipe",
    ]
    if opts.physics == "gz":
        argv += ["--model", "JSON", "--sim-address", "127.0.0.1"]
    elif opts.vehicle == "sd_quad14":
        argv += ["--model", f"quad:{FRAME_FILE}"]
    elif opts.vehicle == "stock":
        argv += ["--model", "quad"]
    else:
        raise ValueError(f"okänt fordon {opts.vehicle!r}")
    argv += ["--defaults", ",".join(str(p) for p in layout.defaults(opts.physics))]
    argv += [
        "--serial0",
        f"udpclient:127.0.0.1:{ports['harness_mavlink']}",
        "--serial1",
        f"udpclient:127.0.0.1:{ports['mavtap_in']}",
        "--serial2",
        f"udpclient:127.0.0.1:{ports['radio_mavlink_in']}",
    ]
    if opts.qgc and opts.instance == 0:
        argv += [QGC_SERIAL, f"udpclient:127.0.0.1:{ports['qgc_mavlink']}"]
    return argv


def write_qgc_params(directory: Path) -> Path:
    """Sim-only parameter layer that makes serial5 a MAVLink 2 port for QGroundControl."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "qgc.parm"
    path.write_text("SERIAL5_PROTOCOL 2\n", encoding="utf-8")
    return path
