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

from sd_sim import paths
from sd_sim.sitl import CMAC_HOME

DEMO_HOME_REFUSED = (
    "Demo kräver den offentliga startplatsen CMAC. Ta bort home_override eller kör just sim."
)
PERSONAL_HOME_WARNING = "Personlig startplats används. Spela inte in demo med den."
_HOME = re.compile(r"^-?\d+(\.\d+)?,-?\d+(\.\d+)?,-?\d+(\.\d+)?,-?\d+(\.\d+)?$")
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off", "")


class DemoHomeError(Exception):
    """A personal home position is configured while starting the demo (exit 3)."""


class ConfigError(Exception):
    """A setting has an invalid value (exit 2)."""


@dataclass
class SimConfig:
    instance: int = 0
    physics: str = "gz"
    gui: bool = True
    video: str = "camera"
    world: str = "sd_lake_forest"
    vehicle: str = "sd_quad14"
    speedup: int = 1
    qgc: bool = False
    link_profile: str = "perfect"
    brain: str = "llm_local"
    script: str = ""
    lan: bool = False
    private_db: bool = False
    open_browser: bool = True
    home: str = CMAC_HOME
    personal_home: bool = False


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


def load(
    args: dict[str, str],
    *,
    demo: bool = False,
    repo: Path = paths.REPO,
    environ: dict[str, str] | None = None,
) -> SimConfig:
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
    cfg.world = str(merged.get("world", cfg.world))
    cfg.vehicle = _choice("vehicle", merged.get("vehicle", cfg.vehicle), ("sd_quad14", "stock"))
    cfg.qgc = _bool("qgc", merged.get("qgc", False))
    cfg.link_profile = _choice(
        "link_profile", merged.get("link_profile", "perfect"), ("perfect", "field_typical", "dead")
    )
    cfg.brain = _choice(
        "brain", merged.get("brain", "llm_local"), ("llm_local", "scripted", "grammar", "none")
    )
    cfg.script = str(merged.get("script", ""))
    cfg.lan = _bool("lan", merged.get("lan", False))
    cfg.private_db = _bool("private_db", merged.get("private_db", False))
    cfg.open_browser = _bool("open.browser", merged.get("open_browser", True))
    home = str(merged.get("home_override", "")).strip()
    if home and not demo:
        if not _HOME.match(home):
            raise ConfigError("home_override ska vara lat,lon,alt,heading")
        cfg.home = home
        cfg.personal_home = True
    return cfg
