"""Simulator settings: ``config/sim.example.toml`` < ``config/sim.toml`` < ``SD_SIM__<KEY>``
< ``key=value`` on the command line.

Demo mode reads no local settings at all and refuses to start when a personal home
position is configured anywhere, so a recording can never show the owner's location.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sd_sim import paths, site
from sd_sim.detector_choice import DETECTORS
from sd_sim.sitl import CMAC_HOME

# Global illumination for the cameras: off, or Gazebo's voxel cone tracing
# (``sd_sim.site_world.GI_VCT``).
GI_MODES = ("off", "vct")

DEMO_HOME_REFUSED = (
    "Demo kräver en offentlig startplats (CMAC eller en offentlig världs hem). "
    "Ta bort home_override eller kör just sim."
)
# `just sim` uses the public terrain world unless a world is set. `just demo` and the
# test suites keep the flat world at CMAC: the scripted commander's scenarios (SC01) have
# pins, charters and heights for that world only.
LAUNCH_WORLD = "sd_hellasgarden"
TEST_WORLD = "sd_lake_forest"
DEMO_LOCAL_SITE_REFUSED = "Demo kräver en offentlig värld. En lokal plats är ditt eget område."
SCRIPTED_SITE_REFUSED = (
    "brain=scripted fungerar bara i den platta testvärlden sd_lake_forest: manusen har "
    "nålar, ramar och höjder för den."
)
GI_SITE_ONLY = (
    "gi=vct (global belysning för kameran) fungerar bara med physics=gz i en platsvärld, "
    "t.ex. world=sd_hellasgarden."
)
PERSONAL_HOME_WARNING = "Personlig startplats används. Spela inte in demo med den."
_HOME = re.compile(r"^-?\d+(\.\d+)?,-?\d+(\.\d+)?,-?\d+(\.\d+)?,-?\d+(\.\d+)?$")
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off", "")


class DemoHomeError(Exception):
    """A personal home position is configured while starting the demo (exit 3)."""


class ConfigError(Exception):
    """A setting has an invalid value (exit 2)."""


# Searchdrone's own models (Gazebo and builtin physics), then ArduPilot's stock quad.
VEHICLES = ("sd_quad14", "sd_fpv7", "stock")
VEHICLE_ALIASES = {"quad14": "sd_quad14", "fpv7": "sd_fpv7"}


@dataclass
class SimConfig:
    instance: int = 0
    physics: str = "gz"
    gui: bool = True
    video: str = "camera"
    world: str = "sd_lake_forest"
    objects: str = ""  # YAML list of sd_sim.objects boxes, relative to the repository
    vehicle: str = "sd_quad14"
    speedup: int = 1
    qgc: bool = False
    link_profile: str = "perfect"
    brain: str = "cloud"
    script: str = ""
    lan: bool = False
    private_db: bool = False
    open_browser: bool = True
    home: str = CMAC_HOME
    personal_home: bool = False
    gi: str = "off"  # "off" | "vct": global illumination for the cameras (sd_sim.site_world)
    detector: str = "oracle"  # "oracle" | "model": the person detector (sd_sim.detector_choice)
    model: str = ""  # detector=model: the manifest (.json); else SD_DETECTOR_MODEL


def _bool(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ConfigError(f"{key}= ska vara true eller false, inte {value!r}")


def _choice(key: str, value: Any, allowed: tuple[str, ...]) -> str:
    text = str(value).strip()
    if text not in allowed:
        raise ConfigError(f"{key}= ska vara ett av {', '.join(allowed)}, inte {value!r}")
    return text


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _has_home(data: dict[str, Any]) -> bool:
    return bool(str(data.get("sim", {}).get("home_override", "")).strip())


def stale_local_warnings(repo: Path = paths.REPO) -> list[str]:
    """Settings in ``config/sim.toml`` that newer versions changed. ``just setup-config``
    never overwrites an existing file, so an old copy keeps pinning what it pinned."""
    example = _read_toml(repo / "config" / "sim.example.toml")
    local = _read_toml(repo / "config" / "sim.toml")
    warnings = []
    if local.get("brain", {}).get("kind") == "llm_local":
        warnings.append(
            'config/sim.toml har brain.kind = "llm_local", som inte finns längre '
            '(cloud används). Byt till kind = "cloud" eller ta bort raden.'
        )
    world = local.get("sim", {}).get("world")
    example_world = example.get("sim", {}).get("world")
    if world is not None and world != example_world:
        default = f'"{example_world}"' if example_world else f"ingen (just sim kör {LAUNCH_WORLD})"
        warnings.append(
            f'config/sim.toml sätter world = "{world}", exemplet sätter {default}. '
            "Ta bort raden om du inte valt världen med flit."
        )
    return warnings


def load(
    args: dict[str, str],
    *,
    demo: bool = False,
    repo: Path = paths.REPO,
    environ: dict[str, str] | None = None,
    default_world: str = TEST_WORLD,
) -> SimConfig:
    """Merge the layers; a world built from a site brings its own home (``Site.home_arg``)
    unless a personal home is set outside the demo."""
    env = dict(os.environ if environ is None else environ)
    example = _read_toml(repo / "config" / "sim.example.toml")
    local = _read_toml(repo / "config" / "sim.toml")
    local_dir = _read_toml(repo / "config" / "local" / "sim.toml")
    env_home = any(k.startswith("SD_SIM__HOME") and v.strip() for k, v in env.items())
    if demo and (_has_home(local) or _has_home(local_dir) or env_home):
        raise DemoHomeError(DEMO_HOME_REFUSED)

    merged: dict[str, Any] = {}
    for layer in (example, {} if demo else local):
        merged.update(layer.get("sim", {}))
        if "brain" in layer:
            merged["brain"] = layer["brain"].get("kind", merged.get("brain"))
            merged["script"] = layer["brain"].get("script", merged.get("script", ""))
        if "open" in layer:
            merged["open_browser"] = layer["open"].get("browser", merged.get("open_browser"))
    for key, value in env.items():
        if key.startswith("SD_SIM__") and key != "SD_SIM__HEADLESS":
            merged[key.removeprefix("SD_SIM__").lower()] = value
    headless = _bool("SD_SIM__HEADLESS", env.get("SD_SIM__HEADLESS", ""))
    if headless:
        merged["physics"] = "builtin"
        merged["gui"] = False
    merged.update(args)

    cfg = SimConfig()
    try:
        cfg.instance = int(merged.get("instance", 0))
        cfg.speedup = int(merged.get("speedup", 1))
    except ValueError as exc:
        raise ConfigError("instance= och speedup= ska vara heltal") from exc
    if not 0 <= cfg.instance <= 7:
        raise ConfigError("instance= ska vara 0-7")
    if cfg.speedup != 1:
        raise ConfigError("speedup ska vara 1 (simulatorn körs i realtid)")
    cfg.physics = _choice("physics", merged.get("physics", "gz"), ("gz", "builtin"))
    cfg.gui = _bool("gui", merged.get("gui", True)) and cfg.physics == "gz"
    cfg.video = _choice("video", merged.get("video", "camera"), ("camera", "test", "off"))
    if cfg.video == "camera" and cfg.physics != "gz":
        cfg.video = "test"
    cfg.world = str(merged.get("world", default_world)).strip() or default_world
    cfg.objects = str(merged.get("objects", "")).strip()
    vehicle = str(merged.get("vehicle", cfg.vehicle)).strip()
    cfg.vehicle = _choice("vehicle", VEHICLE_ALIASES.get(vehicle, vehicle), VEHICLES)
    cfg.qgc = _bool("qgc", merged.get("qgc", False))
    cfg.link_profile = _choice(
        "link_profile", merged.get("link_profile", "perfect"), ("perfect", "field_typical", "dead")
    )
    brain = merged.get("brain", "cloud")
    if brain == "llm_local":  # old sim.toml from before the local model was removed
        brain = "cloud"
    cfg.brain = _choice("brain", brain, ("cloud", "scripted", "grammar", "none"))
    cfg.script = str(merged.get("script", ""))
    cfg.lan = _bool("lan", merged.get("lan", False))
    cfg.private_db = _bool("private_db", merged.get("private_db", False))
    cfg.open_browser = _bool("open.browser", merged.get("open_browser", True))
    cfg.gi = _choice("gi", merged.get("gi", "off"), GI_MODES)
    cfg.detector = _choice("detector", merged.get("detector", "oracle"), DETECTORS)
    cfg.model = str(merged.get("model", "")).strip()
    home = str(merged.get("home_override", "")).strip()
    if home and not demo:
        if not _HOME.match(home):
            raise ConfigError("home_override ska vara lat,lon,alt,heading")
        cfg.home = home
        cfg.personal_home = True
    sites = site.world_sites(repo)
    if cfg.world in sites:
        if cfg.brain == "scripted":
            raise ConfigError(SCRIPTED_SITE_REFUSED)
        try:
            _, public = site.folder_for(sites[cfg.world], repo)
            if demo and not public:
                raise DemoHomeError(DEMO_LOCAL_SITE_REFUSED)
            if not cfg.personal_home:
                cfg.home = site.load(sites[cfg.world], repo).home_arg
                # A local site is the owner's own area: the same care as a personal home.
                cfg.personal_home = not public
        except site.SiteError as exc:
            raise ConfigError(str(exc)) from exc
    if cfg.gi != "off" and (cfg.world not in sites or cfg.physics != "gz"):
        raise ConfigError(GI_SITE_ONLY)
    return cfg
