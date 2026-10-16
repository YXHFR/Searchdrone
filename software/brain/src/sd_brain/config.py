"""``BrainConfig``: read from ``config/brain.toml`` with ``SD_BRAIN__…`` overrides.

Environment variables named ``SD_BRAIN__<SECTION>__<KEY>`` (``__`` between levels)
override the file, e.g. ``SD_BRAIN__LLM__ROOT_URL``. ``.env`` is never read here; the
secrets are read by :mod:`sd_brain.secrets` under their own names.
"""

from __future__ import annotations

import ipaddress
import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sd_brain.constants import LLM_CALL_TIMEOUT_S, STOP_CHECK_TARGET_S

__all__ = [
    "BackendConfig",
    "BehaviourConfig",
    "BrainConfig",
    "CloudConfig",
    "ConfigError",
    "DecideConfig",
    "DecisionEngineConfig",
    "GroundConfig",
    "LlmConfig",
    "load_config",
]

ENV_PREFIX = "SD_BRAIN__"


class ConfigError(ValueError):
    """The configuration cannot be used. The message is Swedish, for the operator."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _is_loopback(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class GroundConfig(_Section):
    commander_url: str = "http://127.0.0.1:7700/commander/v1"
    vehicle_id: str = Field(default="sim-01", pattern=r"^[a-z0-9][a-z0-9-]{2,31}$")

    @field_validator("commander_url")
    @classmethod
    def _loopback(cls, value: str) -> str:
        if not _is_loopback(value) or urlsplit(value).scheme != "http":
            raise ValueError("commander_url måste vara http på 127.0.0.1 (befälhavar-API:t)")
        return value.rstrip("/")


class BackendConfig(_Section):
    kind: Literal["llm_local", "scripted", "grammar"] = "llm_local"
    fallback: Literal["grammar"] = "grammar"
    scripted_file: str = ""


class LlmConfig(_Section):
    """The local model server (llama-server) on loopback."""

    root_url: str = "http://127.0.0.1:8091"
    model: str = "qwen3-vl-4b-instruct-q4_k_m"
    call_timeout_s: float = Field(default=LLM_CALL_TIMEOUT_S, gt=0, le=LLM_CALL_TIMEOUT_S)
    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=0.8, gt=0, le=1)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    seed: int | None = 42
    context_tokens: int = Field(default=8192, ge=1024)
    prompt_budget_tokens: int = Field(default=6000, ge=512)
    health_interval_s: float = Field(default=2.0, gt=0)
    health_timeout_s: float = Field(default=1.0, gt=0)

    @field_validator("root_url")
    @classmethod
    def _loopback(cls, value: str) -> str:
        if not _is_loopback(value):
            raise ValueError("llm.root_url måste ligga på 127.0.0.1 (llama-server)")
        return value.rstrip("/")

    @property
    def base_url(self) -> str:
        return self.root_url + "/v1"


class CloudConfig(_Section):
    """The cloud model (any OpenAI-compatible API, default OpenRouter).

    It answers first whenever a key (``SD_CLOUD_API_KEY``) and a model are set; the local
    model takes over in the same turn when it fails or is slow.
    """

    base_url: str = "https://openrouter.ai/api/v1"
    model: str = ""
    call_timeout_s: float = Field(default=LLM_CALL_TIMEOUT_S, gt=0, le=LLM_CALL_TIMEOUT_S)
    retry_after_s: float = Field(default=60.0, ge=0)

    @field_validator("base_url")
    @classmethod
    def _https(cls, value: str) -> str:
        if urlsplit(value).scheme != "https" and not _is_loopback(value):
            raise ValueError("cloud.base_url måste använda https")
        return value.rstrip("/")


class DecisionEngineConfig(_Section):
    base_url: str
    model: str = ""

    @field_validator("base_url")
    @classmethod
    def _https(cls, value: str) -> str:
        if urlsplit(value).scheme != "https" and not _is_loopback(value):
            raise ValueError("base_url för en beslutsmotor måste använda https")
        return value.rstrip("/")


class DecideConfig(_Section):
    """The decision interface (``sd_brain.decide``): best available engine first."""

    engines: list[Literal["jev", "openai_decisions", "local"]] = Field(
        default_factory=lambda: ["jev", "openai_decisions", "local"]
    )
    stop_check: bool = True
    stop_check_timeout_s: float = Field(default=STOP_CHECK_TARGET_S, gt=0, le=5)
    stop_threshold: float = Field(default=0.5, gt=0, lt=1)
    jev: DecisionEngineConfig = Field(
        default_factory=lambda: DecisionEngineConfig(
            base_url="https://api.typesafe.ai", model="jev-latest"
        )
    )
    openai_decisions: DecisionEngineConfig = Field(
        default_factory=lambda: DecisionEngineConfig(
            base_url="https://api.openai.com", model="gpt-6-luna"
        )
    )

    @field_validator("engines")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("decide.engines får inte innehålla dubbletter")
        return value


class BehaviourConfig(_Section):
    max_read_tool_rounds: int = Field(default=4, ge=0, le=8)
    max_repair_rounds: int = Field(default=1, ge=0, le=2)
    max_actions_per_turn: Literal[1] = 1
    idle_airborne_tick_s: float = Field(default=60, gt=0)
    language: Literal["sv"] = "sv"
    takeoff_policy: Literal["request_then_clearance", "clearance_only"] = "request_then_clearance"
    rule_takeoff: bool = True


class BrainConfig(_Section):
    profile: Literal["sim", "bench", "field"] = "sim"
    token_file: str = "secrets/brain_token"  # noqa: S105 - a path, not a secret
    ground: GroundConfig = Field(default_factory=GroundConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    cloud: CloudConfig = Field(default_factory=CloudConfig)
    decide: DecideConfig = Field(default_factory=DecideConfig)
    behaviour: BehaviourConfig = Field(default_factory=BehaviourConfig)

    @model_validator(mode="after")
    def _scripted_only_in_sim(self) -> BrainConfig:
        if self.backend.kind == "scripted" and self.profile != "sim":
            raise ValueError("ett förskrivet manus (scripted) får bara användas i profilen sim")
        return self


def _parse_env_value(raw: str) -> Any:
    text = raw.strip()
    if text[:1] in ("[", "{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return raw
    return raw


def _apply_env(data: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    out = json.loads(json.dumps(data))
    for name, raw in sorted(env.items()):
        if not name.startswith(ENV_PREFIX):
            continue
        path = [p.lower() for p in name[len(ENV_PREFIX) :].split("__") if p]
        if not path:
            continue
        node = out
        for part in path[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[path[-1]] = _parse_env_value(raw)
    return out


def load_config(path: str | Path | None, env: Mapping[str, str] | None = None) -> BrainConfig:
    """Read the TOML file (``None`` = defaults only) and apply ``SD_BRAIN__`` overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"Konfigurationen {p} saknas. Kör: just setup-config")
        try:
            data = tomllib.loads(p.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"Konfigurationen {p} går inte att läsa: {exc}") from None
    merged = _apply_env(data, env or {})
    try:
        return BrainConfig.model_validate(merged)
    except ValueError as exc:
        raise ConfigError(f"Fel i konfigurationen: {exc}") from None
