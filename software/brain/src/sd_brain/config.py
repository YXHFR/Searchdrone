"""``BrainConfig``: read from ``config/brain.toml`` with ``SD_BRAIN__…`` overrides.

Environment variables named ``SD_BRAIN__<SECTION>__<KEY>`` (``__`` between levels)
override the file, e.g. ``SD_BRAIN__CLOUD__MODEL``. ``.env`` is never read here; the
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
    "ProviderConfig",
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
    kind: Literal["cloud", "scripted", "grammar"] = "cloud"
    fallback: Literal["grammar"] = "grammar"
    scripted_file: str = ""


class LlmConfig(_Section):
    """How a turn talks to the language model (any backend that uses one)."""

    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=0.8, gt=0, le=1)
    # The cloud model reasons before it answers and the reasoning counts here: at 512 a
    # turn after STOP and Resume ran out before its tool call. Answers are 100-300.
    max_tokens: int = Field(default=4096, ge=1, le=4096)
    seed: int | None = 42
    context_tokens: int = Field(default=16384, ge=1024)  # prompt budget + max_tokens fit
    prompt_budget_tokens: int = Field(default=6000, ge=512)


ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


class ProviderConfig(_Section):
    """OpenRouter's provider routing (``provider`` in the request).

    ``data_collection = "deny"`` keeps the calls away from providers that may store or
    train on the data. ``require_parameters`` routes only to providers that support every
    parameter sent (tools, reasoning). ``zdr`` narrows it to zero-data-retention
    endpoints. Other OpenAI-compatible APIs ignore or refuse the field: turn it off with
    ``send = false``.
    """

    send: bool = True
    data_collection: Literal["allow", "deny"] = "deny"
    require_parameters: bool = True
    zdr: bool = False
    only: list[str] = Field(default_factory=list[str])
    order: list[str] = Field(default_factory=list[str])

    def body(self) -> dict[str, Any] | None:
        if not self.send:
            return None
        out: dict[str, Any] = {
            "data_collection": self.data_collection,
            "require_parameters": self.require_parameters,
        }
        if self.zdr:
            out["zdr"] = True
        if self.only:
            out["only"] = list(self.only)
        if self.order:
            out["order"] = list(self.order)
        return out


class CloudConfig(_Section):
    """The cloud model (any OpenAI-compatible API, default OpenRouter).

    It answers whenever a key (``SD_CLOUD_API_KEY``) and a model are set. When the main
    model fails, ``fallback_model`` is asked in the same call; when neither answers in
    time, the turn goes to the command mode.
    """

    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "z-ai/glm-5.3-flash"
    fallback_model: str = "anthropic/claude-haiku-5.5"
    reasoning_effort: ReasoningEffort | Literal[""] = "low"
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    call_timeout_s: float = Field(default=LLM_CALL_TIMEOUT_S, gt=0, le=LLM_CALL_TIMEOUT_S)
    retry_after_s: float = Field(default=60.0, ge=0)

    @field_validator("base_url")
    @classmethod
    def _https(cls, value: str) -> str:
        if urlsplit(value).scheme != "https" and not _is_loopback(value):
            raise ValueError("cloud.base_url måste använda https")
        return value.rstrip("/")

    def extra_body(self) -> dict[str, Any]:
        """The OpenRouter fields added to every chat request."""
        out: dict[str, Any] = {}
        if self.reasoning_effort:
            out["reasoning"] = {"effort": self.reasoning_effort, "exclude": True}
        provider = self.provider.body()
        if provider is not None:
            out["provider"] = provider
        return out


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

    engines: list[Literal["jev", "openai_decisions"]] = Field(
        default_factory=lambda: ["jev", "openai_decisions"]
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
    takeoff_policy: Literal["request_then_clearance"] = "request_then_clearance"
    rule_takeoff: bool = True


# Keys of the removed local model that older config/brain.toml files still carry. They
# are dropped on load so an existing file keeps working; nothing reads them.
_REMOVED_LLM_KEYS = ("root_url", "model", "call_timeout_s", "health_interval_s", "health_timeout_s")


def _drop_local_model(data: dict[str, Any]) -> dict[str, Any]:
    llm = data.get("llm")
    if isinstance(llm, dict):
        for key in _REMOVED_LLM_KEYS:
            llm.pop(key, None)
    backend = data.get("backend")
    if isinstance(backend, dict) and backend.get("kind") == "llm_local":
        backend["kind"] = "cloud"
    decide = data.get("decide")
    if isinstance(decide, dict) and isinstance(decide.get("engines"), list):
        decide["engines"] = [e for e in decide["engines"] if e != "local"]
    return data


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
    merged = _drop_local_model(_apply_env(data, env or {}))
    try:
        return BrainConfig.model_validate(merged)
    except ValueError as exc:
        raise ConfigError(f"Fel i konfigurationen: {exc}") from None
