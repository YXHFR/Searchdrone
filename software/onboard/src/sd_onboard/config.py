"""Configuration for ``sd-onboard`` (``config/onboard.toml``) and the start checks.

Load order: code defaults < ``config/onboard.example.toml`` < ``config/onboard.toml`` <
environment variables ``SD_ONBOARD__<KEY>`` (in a section: ``SD_ONBOARD__<SECTION>__<KEY>``).
Unknown keys are errors. ``${SD_DATA_DIR}`` in a path is the data directory outside the
repository. There is no limit in here: the hard limits are constants in ``sd_safety``.

The link token is read only from ``link_token_file``, never from the TOML files, the
environment or the command line.
"""

from __future__ import annotations

import os
import re
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sd_link.client import is_safe_url
from sd_mav.connection import BROADCAST_SYSID, GCS_COMPID
from sd_onboard.io.params import effective_params
from sd_schemas.base import VEHICLE_ID_PATTERN

__all__ = [
    "MIN_TOKEN_BYTES",
    "ConfigError",
    "Loaded",
    "OnboardConfig",
    "TestHooks",
    "check",
    "load",
    "read_token",
]

MIN_TOKEN_BYTES = 32
_PREFIX = "SD_ONBOARD__"
_WINDOWS_DISK = Path("/mnt")


class ConfigError(Exception):
    """A configuration that sd-onboard refuses to start with (exit code 2)."""


class TestHooks(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    __test__ = False  # not a pytest class

    takeoff_delay_after_arm_s: float = Field(default=0.0, ge=0, le=30)


class OnboardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    vehicle_id: str = "sim-01"
    profile: Literal["sim", "bench", "field"] = "sim"
    mav_url: str = "udpin:127.0.0.1:14563"
    fc_sysid: int = 1
    own_compid: int = 191
    gcs_sysid: int = 250
    sdlink_url: str = "ws://127.0.0.1:7601/sdlink/v1/control"
    link_token_file: str = "${SD_DATA_DIR}/secrets/onboard_link_token"  # noqa: S105  # a path
    state_dir: str = ""
    allow_companion_arm: bool = True
    expected_param_files: list[str] = Field(
        default_factory=lambda: ["simulation/sitl/params/searchdrone-safety.parm"]
    )
    test_hooks: TestHooks = Field(default_factory=TestHooks)


@dataclass(frozen=True)
class Loaded:
    config: OnboardConfig
    root: Path  # the repository root (config files and parameter files are under it)
    env: dict[str, str]

    def path(self, value: str) -> Path:
        data = self.env.get("SD_DATA_DIR") or str(Path.home() / ".local/share/searchdrone")
        expanded = value.replace("${SD_DATA_DIR}", data)
        p = Path(expanded).expanduser()
        return p if p.is_absolute() else self.root / p

    @property
    def token_file(self) -> Path:
        return self.path(self.config.link_token_file)

    @property
    def state_dir(self) -> Path:
        if self.config.state_dir:
            return self.path(self.config.state_dir)
        return self.path(f"${{SD_DATA_DIR}}/onboard/{self.config.vehicle_id}")

    def param_files(self) -> list[Path]:
        return [self.path(f) for f in self.config.expected_param_files]


def _merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def _env_layer(env: dict[str, str]) -> dict[str, Any]:
    layer: dict[str, Any] = {}
    for name, raw in env.items():
        if not name.startswith(_PREFIX):
            continue
        parts = name.removeprefix(_PREFIX).lower().split("__")
        value: Any = raw
        if parts[-1] in ("fc_sysid", "own_compid", "gcs_sysid"):
            value = int(raw)
        elif parts[-1] == "allow_companion_arm":
            value = raw.strip().lower() in ("1", "true", "yes")
        elif parts[-1] == "takeoff_delay_after_arm_s":
            value = float(raw)
        elif parts[-1] == "expected_param_files":
            value = [p for p in raw.split(",") if p]
        node = layer
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return layer


def load(root: Path, env: dict[str, str] | None = None) -> Loaded:
    env = dict(os.environ if env is None else env)
    data: dict[str, Any] = {}
    for name in ("onboard.example.toml", "onboard.toml"):
        path = root / "config" / name
        if path.is_file():
            data = _merge(data, tomllib.loads(path.read_text(encoding="utf-8")))
    data = _merge(data, _env_layer(env))
    try:
        cfg = OnboardConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Fel i config/onboard.toml / config error: {exc}") from exc
    return Loaded(config=cfg, root=root, env=env)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def check(loaded: Loaded) -> list[str]:
    """Every start check that fails, as text in Swedish and English; empty = start."""
    cfg, errors = loaded.config, []
    if cfg.profile != "sim":
        errors.append("Bara profilen 'sim' finns i v0.1 / only the 'sim' profile exists in v0.1.")
    if not re.match(VEHICLE_ID_PATTERN, cfg.vehicle_id):
        errors.append(f"Ogiltigt vehicle_id / invalid vehicle_id: {cfg.vehicle_id!r}.")
    if cfg.fc_sysid in (cfg.gcs_sysid, BROADCAST_SYSID) or not 1 <= cfg.fc_sysid <= 249:
        errors.append("fc_sysid måste vara 1..249 och inte markstationens / must be 1..249.")
    if cfg.own_compid == GCS_COMPID or not 1 <= cfg.own_compid <= 254:
        errors.append("own_compid måste vara 1..254 och inte 190 / must be 1..254, not 190.")
    if not is_safe_url(cfg.sdlink_url):
        errors.append(
            "sdlink_url måste vara wss:// eller ws:// till den egna datorn "
            "/ sdlink_url must be wss://, or ws:// to loopback."
        )
    errors += _token_problems(loaded)
    if _inside(loaded.state_dir, loaded.root):
        errors.append("state_dir ligger i repot; flytta den / state_dir is inside the repository.")
    hooks_set = cfg.test_hooks != TestHooks()
    if hooks_set and not (cfg.profile == "sim" and loaded.env.get("SD_TEST") == "1"):
        errors.append("[test_hooks] gäller bara i tester (SD_TEST=1) / test hooks need SD_TEST=1.")
    files = loaded.param_files()
    if not files:
        errors.append("expected_param_files får inte vara tom / no parameter file given.")
    missing = [p for p in files if not p.is_file()]
    for path in missing:
        errors.append(f"Parameterfilen saknas / parameter file missing: {path.name}.")
    if files and not missing:
        params = effective_params(files)
        if params.get("MAV_GCS_SYSID") != float(cfg.gcs_sysid):
            errors.append("gcs_sysid stämmer inte med MAV_GCS_SYSID / gcs_sysid differs.")
        if "FS_GCS_ENABLE" not in params:
            errors.append("Säkerhetsparametrarna saknas / the safety parameters are missing.")
    return errors


def _token_problems(loaded: Loaded) -> list[str]:
    path = loaded.token_file
    if _WINDOWS_DISK in path.parents or path == _WINDOWS_DISK:
        return [
            "Tokenfilen ligger på en Windows-disk; flytta den till Linux-filsystemet "
            "(t.ex. ~/.local/share/searchdrone) / the token file is on a Windows disk."
        ]
    if _inside(path, loaded.root):
        return ["Tokenfilen ligger i repot / the token file is inside the repository."]
    try:
        info = path.lstat()
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        return [f"Tokenfilen saknas / the token file is missing: {path}."]
    out = []
    if stat.S_ISLNK(info.st_mode):
        out.append("Tokenfilen får inte vara en symbolisk länk / no symbolic link.")
    elif info.st_uid != os.getuid():
        out.append("Tokenfilen ägs av en annan användare / owned by another user.")
    if stat.S_IMODE(info.st_mode) & 0o077:
        out.append("Tokenfilen får bara läsas av ägaren (chmod 600) / token file must be 600.")
    if len(token.encode()) < MIN_TOKEN_BYTES:
        out.append(f"Tokenen är kortare än {MIN_TOKEN_BYTES} byte / token too short.")
    return out


def read_token(loaded: Loaded) -> str:
    return loaded.token_file.read_text(encoding="utf-8").strip()
