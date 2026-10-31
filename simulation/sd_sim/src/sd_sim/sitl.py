"""The ArduPilot SITL command line, built by one pure function.

Every serial port SITL uses is a ``udpclient`` to 127.0.0.1, so SITL opens no TCP port
for them (its default for serial0 is a TCP listener on all addresses that also makes
SITL wait for a connection). In ArduPilot 4.7.1 serial3 and serial4 are the simulated
GPS receivers and there is no serial9, so those are never overridden. The optional
QGroundControl link therefore uses serial5.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from sd_sim import fpv7
from sd_sim.errors import InfraError

# ArduPilot's public SITL site (CMAC). Never a private address.
CMAC_HOME = "-35.363261,149.165230,584.0,353"
QGC_SERIAL = "--serial5"
# The project's parameter layers, loaded after every generated layer: the safety
# parameters (failsafes, fence, battery) and the vehicle's sim-only parameters. Both are
# required: a missing layer stops the start instead of silently flying without it.
SAFETY_PARAM_FILE = "searchdrone-safety.parm"
SIM_PARAM_FILE = "sd_quad14-sim.parm"
PROJECT_PARAM_FILES = (SAFETY_PARAM_FILE, SIM_PARAM_FILE)
# The sim-only layer per vehicle; ArduPilot's stock quad keeps sd_quad14's.
SIM_PARAM_FILES = {
    "sd_quad14": SIM_PARAM_FILE,
    "stock": SIM_PARAM_FILE,
    fpv7.MODEL_NAME: f"{fpv7.MODEL_NAME}-sim.parm",
}
# SITL strips the leading "/" from a frame file path and opens it relative to its working
# directory, so the launcher copies the frame file there and passes the bare name.
# sd_quad14's frame file is committed; sd_fpv7's is written from its CAD numbers.
FRAME_FILE = "sd_quad14.json"
FPV7_FRAME_FILE = f"{fpv7.MODEL_NAME}.json"
# SITL's builtin frame per vehicle: sd_quad14 flies ArduPilot's "quad" (FRAME_TYPE 0 from
# copter.parm), sd_fpv7 an X ("x", FRAME_TYPE 1 in its parameter layer).
BUILTIN_MODELS = {"sd_quad14": f"quad:{FRAME_FILE}", fpv7.MODEL_NAME: f"x:{FPV7_FRAME_FILE}"}
# The flight controller's Lua scripts (SCR_ENABLE 1 in the safety layer). SITL runs every
# script in ``scripts/`` under its working directory; on a real flight controller they go
# to the SD card's ``APM/scripts/``. A missing script stops the start.
LUA_SCRIPTS = ("sd_gcs_loiter_cap.lua",)
# SCR_LD_CHECKSUM / SCR_RUN_CHECKSUM keep any other script set from arming: ArduPilot XORs
# the CRC32 of every loaded (running) script file and compares the low 23 bits with the
# parameter (AP_Scripting::arming_checks, Copter-4.7.1). Its CRC32 is the plain table
# loop from 0 with no final inversion (crc_crc32 in AP_Math/crc.cpp, through
# AP_Filesystem::crc32), not zlib's.
SCRIPT_CHECKSUM_MASK = 0x007FFFFF


def script_crc32(data: bytes) -> int:
    """ArduPilot's ``crc_crc32(0, data)``: zlib's CRC32 without its two inversions."""
    return zlib.crc32(data, 0xFFFFFFFF) ^ 0xFFFFFFFF


def scripts_checksum(files: Iterable[Path]) -> int:
    """The value SCR_LD_CHECKSUM and SCR_RUN_CHECKSUM must hold when exactly ``files``
    are loaded and running."""
    out = 0
    for path in files:
        out ^= script_crc32(path.read_bytes())
    return out & SCRIPT_CHECKSUM_MASK


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

    def stage_frame(self, workdir: Path, vehicle: str = "sd_quad14") -> Path:
        """Put the builtin-physics frame file into SITL's working directory."""
        workdir.mkdir(parents=True, exist_ok=True)
        if vehicle == fpv7.MODEL_NAME:
            target = workdir / FPV7_FRAME_FILE
            target.write_text(json.dumps(fpv7.frame_json(), indent=2) + "\n", encoding="utf-8")
            return target
        if not self.frame_json.is_file():
            raise InfraError(f"{self.frame_json} saknas: modellen sd_quad14 kan inte startas.")
        target = workdir / FRAME_FILE
        target.write_bytes(self.frame_json.read_bytes())
        return target

    def stage_scripts(self, workdir: Path) -> list[Path]:
        """Put the project's Lua scripts into ``scripts/`` under SITL's working directory."""
        source = self.repo / "simulation" / "sitl" / "scripts"
        target = workdir / "scripts"
        target.mkdir(parents=True, exist_ok=True)
        out = []
        for name in LUA_SCRIPTS:
            if not (source / name).is_file():
                raise InfraError(f"{source / name} saknas: flygkontrollens Lua-skript.")
            (target / name).write_bytes((source / name).read_bytes())
            out.append(target / name)
        return out

    def defaults(self, physics: str, vehicle: str = "sd_quad14") -> list[Path]:
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
        layers.extend(self.project_layers(vehicle))
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

    def project_layers(self, vehicle: str = "sd_quad14") -> list[Path]:
        """The project's own parameter files in load order (later wins): the safety layer,
        then the vehicle's sim-only layer. Every one must exist; a missing layer raises
        instead of being skipped."""
        params = self.repo / "simulation" / "sitl" / "params"
        layers = [params / SAFETY_PARAM_FILE, params / SIM_PARAM_FILES[vehicle]]
        missing = [p.name for p in layers if not p.is_file()]
        if missing:
            raise InfraError(f"Parameterlager saknas i {params}: {', '.join(missing)}")
        return layers


@dataclass(frozen=True)
class SitlOptions:
    instance: int = 0
    physics: str = "builtin"  # "builtin" | "gz"
    vehicle: str = "sd_quad14"  # "sd_quad14" | "sd_fpv7" | "stock" (ArduPilot's quad / iris)
    speedup: int = 1
    home: str = CMAC_HOME
    qgc: bool = False


def build_sitl_argv(opts: SitlOptions, ports: dict[str, int], layout: SitlLayout) -> list[str]:
    if opts.physics not in ("builtin", "gz"):
        raise ValueError(f"okänd fysik {opts.physics!r}")
    if opts.vehicle not in SIM_PARAM_FILES:
        raise ValueError(f"okänt fordon {opts.vehicle!r}")
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
    elif opts.vehicle == "stock":
        argv += ["--model", "quad"]
    else:
        argv += ["--model", BUILTIN_MODELS[opts.vehicle]]
    layers = layout.defaults(opts.physics, opts.vehicle)
    argv += ["--defaults", ",".join(str(p) for p in layers)]
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
