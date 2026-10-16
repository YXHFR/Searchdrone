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

# ArduPilot's public SITL site (CMAC). Never a private address.
CMAC_HOME = "-35.363261,149.165230,584.0,353"
QGC_SERIAL = "--serial5"
# The project's parameter layers, loaded after every generated layer: the safety
# parameters (failsafes, fence, battery) and the sim-only parameters.
SAFETY_PARAM_FILE = "searchdrone-safety.parm"
PROJECT_PARAM_FILES = (SAFETY_PARAM_FILE, "sd_quad14-sim.parm")


@dataclass(frozen=True)
class SitlLayout:
    """Files the command line points at. Missing optional files are simply left out."""

    arducopter: Path
    ardupilot_home: Path
    repo: Path
    extra_defaults: tuple[Path, ...] = field(default_factory=tuple)

    @property
    def frame_json(self) -> Path:
        return self.repo / "simulation" / "sitl" / "frames" / "sd_quad14.json"

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
        return layers

    def project_layers(self) -> list[Path]:
        """The project's own parameter files that exist, in load order (later wins)."""
        params = self.repo / "simulation" / "sitl" / "params"
        return [params / name for name in PROJECT_PARAM_FILES if (params / name).is_file()]


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
    elif opts.vehicle == "sd_quad14" and layout.frame_json.is_file():
        argv += ["--model", f"quad:{layout.frame_json}"]
    else:
        argv += ["--model", "quad"]
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
