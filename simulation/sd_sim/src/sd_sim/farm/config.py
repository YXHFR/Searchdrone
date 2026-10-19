"""The farm's settings: ``config/simfarm.example.toml`` < ``config/simfarm.toml`` < ``key=value``.

``config/simfarm.toml`` is gitignored and is the only place for the R2 key; the
repository has only the example, with the key fields empty. Without a key the farm keeps
everything in ``build/simfarm/``.
"""

from __future__ import annotations

import stat
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from sd_sim import paths, site
from sd_sim.farm.host import Limits
from sd_sim.farm.store import R2Config

EXAMPLE = "config/simfarm.example.toml"
LOCAL = "config/simfarm.toml"
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


class FarmConfigError(ValueError):
    """A farm setting has a bad value."""


@dataclass(frozen=True)
class FarmConfig:
    commander: str
    worlds: tuple[str, ...]
    limits: Limits
    r2: R2Config | None
    max_failures: int = 3


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise FarmConfigError(f"{path}: {exc}") from exc


def _bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise FarmConfigError(f"{key} ska vara true eller false")


def _limits(table: dict[str, Any]) -> Limits:
    values: dict[str, Any] = {}
    for f in fields(Limits):
        if f.name not in table:
            continue
        raw = table[f.name]
        try:
            if f.type in ("bool", bool):
                values[f.name] = _bool(raw, f.name)
            elif f.type in ("int", int):
                values[f.name] = int(raw)
            else:
                values[f.name] = float(raw)
        except (TypeError, ValueError) as exc:
            raise FarmConfigError(f"farm.{f.name}: ogiltigt värde {raw!r}") from exc
    limits = Limits(**values)
    if not 0.05 <= limits.max_duty <= 1.0:
        raise FarmConfigError("farm.max_duty ska ligga mellan 0.05 och 1")
    if not 0 <= limits.nice <= 19:
        raise FarmConfigError("farm.nice ska ligga mellan 0 och 19")
    return limits


def _max_failures(raw: Any) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise FarmConfigError(f"farm.max_failures: ogiltigt värde {raw!r}") from exc
    if value < 1:
        raise FarmConfigError("farm.max_failures ska vara minst 1")
    return value


def refuse_photo_worlds(worlds: list[str], repo: Path = paths.REPO) -> None:
    """With an R2 key the farm uploads its pictures: never of a world draped with the
    owner's orthophoto, which must stay on the owner's computer."""
    draped = site.photo_worlds(worlds, repo)
    if draped:
        raise FarmConfigError(
            f"{', '.join(draped)} har ett ortofoto i build/ och simfarmen laddar upp till R2: "
            "ta bort världen ur farm.worlds, eller R2-nyckeln, eller fotot"
        )


def load(overrides: dict[str, str] | None = None, repo: Path = paths.REPO) -> FarmConfig:
    merged: dict[str, Any] = {"farm": {}, "r2": {}}
    for path in (repo / EXAMPLE, repo / LOCAL):
        data = _read(path)
        for section in ("farm", "r2"):
            merged[section].update(data.get(section, {}))
    farm = merged["farm"]
    for key, value in (overrides or {}).items():
        farm[key] = value
    commander = str(farm.get("commander", "script"))
    if commander not in ("script", "sd-brain"):
        raise FarmConfigError("commander ska vara script eller sd-brain")
    worlds = farm.get("worlds", ["sd_lake_forest"])
    if isinstance(worlds, str):
        worlds = [w for w in worlds.split(",") if w]
    if not worlds:
        raise FarmConfigError("farm.worlds får inte vara tom")
    try:
        r2 = R2Config.from_table(merged["r2"])
    except ValueError as exc:
        raise FarmConfigError(str(exc)) from exc
    local = repo / LOCAL
    if r2 is not None and local.is_file() and local.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise FarmConfigError(
            f"{LOCAL} har R2-nyckeln och kan läsas av andra: kör chmod 600 {LOCAL}"
        )
    if r2 is not None:
        refuse_photo_worlds([str(w) for w in worlds], repo)
    return FarmConfig(
        commander=commander,
        worlds=tuple(str(w) for w in worlds),
        limits=_limits(farm),
        r2=r2,
        max_failures=_max_failures(farm.get("max_failures", 3)),
    )
