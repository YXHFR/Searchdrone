"""Configuration for ``sd-ground`` (``config/ground.toml``) and its checks.

Load order: code defaults < ``config/ground.example.toml`` < ``config/ground.toml`` <
environment variables ``SD_GROUND__<SECTION>__<KEY>`` (top-level keys:
``SD_GROUND__<KEY>``). Unknown keys are errors, so a typo (or an invented switch such as
``auth.skip_login``) can never be silently ignored.

Secrets never live in the TOML files. The database and video passwords come from the
environment or the repository's ``.env``; the commander token from ``auth.brain_token_file``.
"""

from __future__ import annotations

import ipaddress
import os
import re
import stat
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sd_safety.constants import OPERATOR_ABSENT_S

__all__ = [
    "DEFAULT_LOGIN_CODE_TTL_S",
    "SECRET_ENV_KEYS",
    "ConfigError",
    "GroundConfig",
    "Loaded",
    "check",
    "load",
    "parse_bind",
    "read_env_file",
]

DEFAULT_LOGIN_CODE_TTL_S = 600
PLACEHOLDERS = {"", "change_me", "change-me", "changeme"}
# Secrets read from the environment / .env. They are not config keys.
SECRET_ENV_KEYS = (
    "SD_GROUND__DB_PASSWORD",
    "SD_GROUND__DB_APP_PASSWORD",
    "SD_GROUND__VIDEO_PUBLISH_PASSWORD",
)
_OTHER_ENV_KEYS = ("SD_UID", "SD_GID", "SD_DATA_DIR")
_PREFIX = "SD_GROUND__"
_TEST_BUILD = re.compile(r"""<meta\s+name=["']sd-build["']\s+content=["']test["']""", re.I)


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DbConfig(_Section):
    url: str = "postgresql+asyncpg://sd_app@127.0.0.1:5433/searchdrone"
    owner_url: str = "postgresql+asyncpg://sd_owner@127.0.0.1:5433/searchdrone"
    connect_timeout_s: float = Field(default=30.0, gt=0)


class HttpConfig(_Section):
    panel_bind: str = "127.0.0.1:8080"
    commander_bind: str = "127.0.0.1:7700"
    static_dir: str = "software/panel/dist"


class LanConfig(_Section):
    enabled: bool = False
    bind: str = "0.0.0.0:8443"
    tls_dir: str = "config/local/tls"


class SdlinkConfig(_Section):
    bind: str = "127.0.0.1:7600"
    link_tokens_file: str = "secrets/link_tokens.toml"


class MavlinkConfig(_Section):
    listen: str = "udpin:127.0.0.1:14560"
    gcs_sysid: int = Field(default=250, ge=1, le=255)
    gcs_compid: int = Field(default=190, ge=1, le=255)
    fc_endpoint: str = ""


class GateConfig(_Section):
    # Owner decision: no pilot in the panel for 30 s in an active mission gives the
    # alarm and a hold. Fixed: the keys stay readable, but only these values load; a
    # settings file with another time or "off" / "return_home" is refused at start, with
    # the field named (the guard cannot be switched off).
    operator_absent_s: float = Field(
        default=OPERATOR_ABSENT_S, ge=OPERATOR_ABSENT_S, le=OPERATOR_ABSENT_S
    )
    operator_absent_action: Literal["hold"] = "hold"


class AuthConfig(_Section):
    cli_may_confirm: bool = False  # only the test rig sets true
    brain_token_file: str = "secrets/brain_token"  # noqa: S105 - a path, not a secret
    login_code_ttl_s: int = Field(default=DEFAULT_LOGIN_CODE_TTL_S, ge=1, le=3600)


class CharterConfig(_Section):
    max_area_km2: float = Field(default=4.0, gt=0)


class VideoConfig(_Section):
    mediamtx_api: str = "http://127.0.0.1:9997"
    mediamtx_whep: str = "http://127.0.0.1:8889"
    mediamtx_hls: str = "http://127.0.0.1:8888"


class MapConfig(_Section):
    # The offline map pack <data_dir>/tiles/<pack>.pmtiles, made by `just maps-fetch`.
    pack: str = Field(default="demo", pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")  # tiles.PACK_NAME
    zoom: float = Field(default=15.0, ge=0, le=22)


class HealthConfig(_Section):
    internet_probe: bool = False


class SimConfig(_Section):
    dataflash_dir: str = ""


class RemoteConfig(_Section):
    enabled: bool = False  # reserved; true is refused in v0.1.0


class GroundConfig(_Section):
    profile: str = "sim"
    data_dir: str = "~/.local/share/searchdrone"
    default_vehicle_id: str = Field(default="sim-01", pattern=r"^[a-z0-9][a-z0-9-]{2,31}$")
    db: DbConfig = DbConfig()
    http: HttpConfig = HttpConfig()
    lan: LanConfig = LanConfig()
    sdlink: SdlinkConfig = SdlinkConfig()
    mavlink: MavlinkConfig = MavlinkConfig()
    gate: GateConfig = GateConfig()
    auth: AuthConfig = AuthConfig()
    charter: CharterConfig = CharterConfig()
    video: VideoConfig = VideoConfig()
    map: MapConfig = MapConfig()
    health: HealthConfig = HealthConfig()
    sim: SimConfig = SimConfig()
    remote: RemoteConfig = RemoteConfig()


class ConfigError(Exception):
    """The configuration cannot be loaded. ``errors`` are Swedish lines for the operator."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class Loaded:
    """A loaded configuration plus where it came from."""

    config: GroundConfig
    root: Path
    env: dict[str, str] = field(repr=False)  # secrets: never printed

    def path(self, value: str) -> Path:
        """A config path: ``~`` expanded, relative paths taken from the repository root."""
        p = Path(value).expanduser()
        return p if p.is_absolute() else self.root / p

    @property
    def data_dir(self) -> Path:
        return self.path(self.config.data_dir)

    def secret(self, key: str) -> str:
        return self.env.get(key, "")


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.partition("=")
        if sep:
            values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def _merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)  # pyright: ignore[reportUnknownArgumentType]
        else:
            out[key] = value
    return out


def _read_toml(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        errors.append(f"{path.name} går inte att läsa: {exc}")
        return {}


def _env_overrides(environ: dict[str, str], errors: list[str]) -> dict[str, Any]:
    sections = set(GroundConfig.model_fields)
    out: dict[str, Any] = {}
    for name, value in environ.items():
        if not name.startswith(_PREFIX) or name in SECRET_ENV_KEYS:
            continue
        parts = name[len(_PREFIX) :].lower().split("__")
        if len(parts) == 1 and parts[0] in sections:
            out[parts[0]] = value
        elif len(parts) == 2 and parts[0] in sections:
            out.setdefault(parts[0], {})[parts[1]] = value
        else:
            errors.append(f"okänd miljövariabel {name}")
    return out


def load(
    root: Path | None = None,
    environ: dict[str, str] | None = None,
    *,
    config_file: Path | None = None,
) -> Loaded:
    """Load the configuration. Raises ``ConfigError`` on unknown keys or bad values."""
    root = (root or Path.cwd()).resolve()
    process_env = dict(os.environ) if environ is None else dict(environ)
    errors: list[str] = []
    data = _read_toml(root / "config" / "ground.example.toml", errors)
    data = _merge(data, _read_toml(config_file or root / "config" / "ground.toml", errors))
    data = _merge(data, _env_overrides(process_env, errors))
    config = GroundConfig()
    try:
        config = GroundConfig.model_validate(data)
    except ValidationError as exc:
        for err in exc.errors():
            where = ".".join(str(p) for p in err["loc"])
            if err["type"] == "extra_forbidden":
                errors.append(f"okänd inställning {where}")
            else:
                errors.append(f"fel värde för {where}: {err['msg']}")
    if errors:
        raise ConfigError(errors)
    env = read_env_file(root / ".env")
    env.update({k: v for k, v in process_env.items() if k in SECRET_ENV_KEYS + _OTHER_ENV_KEYS})
    return Loaded(config=config, root=root, env=env)


def parse_bind(bind: str) -> tuple[str, int]:
    host, sep, port = bind.rpartition(":")
    if not sep or not port.isdigit():
        raise ValueError(f"ogiltig adress {bind!r} (väntade värd:port)")
    return host.strip("[]"), int(port)


def _is_loopback(bind: str) -> bool:
    try:
        host, _ = parse_bind(bind)
    except ValueError:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _private_mode(path: Path) -> bool:
    return stat.S_IMODE(path.stat().st_mode) & 0o077 == 0


def check(loaded: Loaded) -> list[str]:
    """Everything that makes ``config check`` (and ``serve``) exit with code 1."""
    cfg, errors = loaded.config, []
    for name, bind in (
        ("http.panel_bind", cfg.http.panel_bind),
        ("http.commander_bind", cfg.http.commander_bind),
        ("sdlink.bind", cfg.sdlink.bind),
    ):
        if not _is_loopback(bind):
            errors.append(f"{name} = {bind} är inte loopback. Bara 127.0.0.1 är tillåtet.")
    if cfg.profile != "sim":
        errors.append(f"profile = {cfg.profile!r}: bara profilen sim finns i v0.1.0.")
    if cfg.remote.enabled:
        errors.append("Fjärråtkomst finns inte i v0.1.0 (remote.enabled = true).")
    tls = loaded.path(cfg.lan.tls_dir)
    if cfg.lan.enabled and not ((tls / "server.crt").is_file() and (tls / "server.key").is_file()):
        errors.append("LAN-läget saknar certifikat. Kör: sd-ground init --lan")
    if not cfg.auth.cli_may_confirm and cfg.auth.login_code_ttl_s != DEFAULT_LOGIN_CODE_TTL_S:
        errors.append("auth.login_code_ttl_s får bara ändras i testriggen (cli_may_confirm).")
    env_file = loaded.root / ".env"
    for key in SECRET_ENV_KEYS:
        if loaded.secret(key).strip().lower() in PLACEHOLDERS:
            errors.append(f"{key} saknas eller är en platshållare i .env. Kör: just setup-config")
    sd_data_dir = loaded.secret("SD_DATA_DIR")
    if sd_data_dir and Path(sd_data_dir).expanduser().resolve() != loaded.data_dir.resolve():
        errors.append("SD_DATA_DIR i .env skiljer sig från data_dir i config/ground.toml.")
    if env_file.is_file() and not _private_mode(env_file):
        errors.append(".env kan läsas av andra. Kör: chmod 600 .env")
    errors += _check_private_dir(loaded.root / "secrets", "*")
    errors += _check_private_dir(tls, "*.key")
    token = loaded.path(cfg.auth.brain_token_file)
    if not token.is_file():
        errors.append(
            f"Befälhavarens token saknas ({cfg.auth.brain_token_file}). Kör: sd-ground init"
        )
    elif not _private_mode(token):
        errors.append(f"{cfg.auth.brain_token_file} kan läsas av andra. Kör: chmod 600 på filen")
    try:
        limit_caps(loaded)
    except ConfigError as exc:
        errors += exc.errors
    index = loaded.path(cfg.http.static_dir) / "index.html"
    if (
        cfg.profile != "sim"
        and index.is_file()
        and _TEST_BUILD.search(index.read_text(encoding="utf-8", errors="replace"))
    ):
        errors.append("Panelen är en testbygge (sd-build=test) och får bara köras i profilen sim.")
    return errors


def _check_private_dir(directory: Path, pattern: str) -> list[str]:
    if not directory.is_dir():
        return []
    errors = []
    if stat.S_IMODE(directory.stat().st_mode) != 0o700:
        errors.append(f"{directory.name}/ måste ha rättigheterna 700. Kör: chmod 700 {directory}")
    for item in sorted(directory.glob(pattern)):
        if item.is_file() and not _private_mode(item):
            errors.append(f"{directory.name}/{item.name} kan läsas av andra. Kör: chmod 600")
    return errors


def limit_caps(loaded: Loaded) -> Any:
    """The outer cap for per-mission limits from ``config/local/limits.toml`` (owner's file).

    Missing file or keys: the standard caps (300 m, 25 m/s, 2 000 m). A bad file stops the
    server rather than silently allowing more.
    """
    from sd_schemas.safety import LimitCaps

    path = loaded.root / "config" / "local" / "limits.toml"
    if not path.is_file():
        return LimitCaps()
    try:
        return LimitCaps.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ConfigError([f"config/local/limits.toml går inte att använda: {exc}"]) from exc
