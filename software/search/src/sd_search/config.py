"""Configuration for ``sd-search`` (``config/search.toml``) and the cloud key.

Load order: code defaults < ``config/search.example.toml`` < ``config/search.toml``.
Unknown keys are errors. The key comes from ``SD_CLOUD_API_KEY`` (the environment, else the
repository's ``.env``) and is never printed. ``provider.data_collection`` can only be
``deny``: no call goes to a provider that may store or train on the crops.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sd_brain.config import CloudConfig, ProviderConfig

__all__ = [
    "CloudMatchConfig",
    "MotionConfig",
    "SearchConfig",
    "SearchConfigError",
    "cloud_key",
    "load",
]

KEY_ENV = "SD_CLOUD_API_KEY"


class SearchConfigError(Exception):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MatchProvider(_Section):
    data_collection: Literal["deny"] = "deny"
    require_parameters: bool = True
    zdr: bool = True  # off only when the config says so, never from the environment


class CloudMatchConfig(_Section):
    enabled: bool = True
    base_url: str = "https://openrouter.ai/api/v1"
    fast_model: str = Field(default="anthropic/claude-haiku-5.5", min_length=1, max_length=120)
    timeout_s: float = Field(default=8.0, gt=0, le=30)
    max_concurrent: int = Field(default=4, ge=1, le=8)
    queue_max: int = Field(default=8, ge=1, le=64)
    retry_after_s: float = Field(default=60.0, ge=1, le=3600)
    provider: MatchProvider = MatchProvider()

    def runtime_config(self) -> CloudConfig:
        """The brain runtime's settings for the fast tier: no reasoning, no fallback."""
        return CloudConfig(
            base_url=self.base_url,
            model=self.fast_model,
            fallback_model="",
            reasoning_effort="",
            provider=ProviderConfig(
                data_collection=self.provider.data_collection,
                require_parameters=self.provider.require_parameters,
                zdr=self.provider.zdr,
            ),
            call_timeout_s=self.timeout_s,
            retry_after_s=self.retry_after_s,
        )


class MotionConfig(_Section):
    enabled: bool = True  # false: the map as if the person stood still (the old map)


class SearchConfig(_Section):
    cloud: CloudMatchConfig = CloudMatchConfig()
    motion: MotionConfig = MotionConfig()


def _merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)  # pyright: ignore[reportUnknownArgumentType]
        else:
            out[key] = value
    return out


def load(root: Path) -> SearchConfig:
    data: dict[str, Any] = {}
    errors: list[str] = []
    for name in ("search.example.toml", "search.toml"):
        path = root / "config" / name
        if not path.is_file():
            continue
        try:
            data = _merge(data, tomllib.loads(path.read_text(encoding="utf-8")))
        except tomllib.TOMLDecodeError as exc:
            errors.append(f"config/{name} går inte att läsa: {exc}")
    try:
        cfg = SearchConfig.model_validate(data)
    except ValidationError as exc:
        for err in exc.errors():
            where = ".".join(str(p) for p in err["loc"])
            if err["type"] == "extra_forbidden":
                errors.append(f"okänd inställning {where}")
            else:
                errors.append(f"fel värde för {where}: {err['msg']}")
        cfg = SearchConfig()
    if not errors:
        try:
            cfg.cloud.runtime_config()
        except ValidationError as exc:
            errors.append(f"[cloud] går inte att använda: {exc.errors()[0]['msg']}")
    if errors:
        raise SearchConfigError(errors)
    return cfg


def cloud_key(root: Path, environ: dict[str, str] | None = None) -> str | None:
    """The cloud key, or None (no cloud matching). Never logged."""
    env = os.environ if environ is None else environ
    value = env.get(KEY_ENV, "").strip()
    if not value:
        path = root / ".env"
        if path.is_file():
            for raw in path.read_text(encoding="utf-8").splitlines():
                name, sep, rest = raw.strip().partition("=")
                if sep and name.strip() == KEY_ENV:
                    value = rest.strip().strip('"').strip("'")
    return value or None
