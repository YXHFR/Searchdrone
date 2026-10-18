"""Configuration and secrets (05 §3.3, §3.6; AC-B01)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from sd_brain.config import BrainConfig, ConfigError, load_config
from sd_brain.secrets import (
    SecretError,
    brain_env,
    check_secret,
    install_log_masking,
    load_secrets,
    mask,
)

ROOT = Path(__file__).resolve().parents[4]
GOOD = "k" * 40


def _token(tmp_path: Path, value: str = "t" * 43, mode: int = 0o600) -> Path:
    path = tmp_path / "brain_token"
    path.write_text(value + "\n")
    path.chmod(mode)
    return path


def test_example_config_is_valid() -> None:
    cfg = load_config(ROOT / "config/brain.example.toml", {})
    assert cfg.backend.kind == "cloud"
    assert urlsplit(cfg.cloud.base_url).hostname == "openrouter.ai"
    assert cfg.cloud.model == "z-ai/glm-5.3-flash"
    assert cfg.cloud.fallback_model == "anthropic/claude-haiku-5.5"
    assert cfg.cloud.reasoning_effort == "low"
    assert cfg.cloud.provider.data_collection == "deny"
    assert cfg.cloud.extra_body() == {
        "reasoning": {"effort": "low", "exclude": True},
        "provider": {"data_collection": "deny", "require_parameters": True},
    }
    assert cfg.decide.stop_check_timeout_s == 0.5


def test_config_with_dotenv_secrets(tmp_path: Path) -> None:
    token = _token(tmp_path)
    env = {
        "SD_BRAIN__TOKEN_FILE": str(token),
        "SD_CLOUD_API_KEY": GOOD,
        "SD_BRAIN__CLOUD__REASONING_EFFORT": "none",
        "SD_BRAIN__CLOUD__PROVIDER__ZDR": "true",
        "SD_BRAIN__DECIDE__ENGINES": '["jev"]',
    }
    cfg = load_config(None, env)
    assert cfg.cloud.reasoning_effort == "none" and cfg.cloud.provider.zdr is True
    assert cfg.decide.engines == ["jev"]
    secrets = load_secrets(cfg.token_file, env, tmp_path / "none.env")
    assert secrets.cloud_api_key == GOOD
    assert secrets.jev_api_key is None


def test_config_without_any_key_starts(tmp_path: Path) -> None:
    secrets = load_secrets(_token(tmp_path), {}, tmp_path / "none.env")
    assert secrets.cloud_api_key is None and secrets.values() == ["t" * 43]


def test_old_local_model_config_still_loads(tmp_path: Path) -> None:
    old = tmp_path / "brain.toml"
    old.write_text(
        '[backend]\nkind = "llm_local"\n'
        '[llm]\nroot_url = "http://127.0.0.1:8091"\nmodel = "q"\nhealth_interval_s = 2\n'
        "temperature = 0.3\n"
        '[decide]\nengines = ["jev", "openai_decisions", "local"]\n'
    )
    cfg = load_config(old, {})
    assert cfg.backend.kind == "cloud" and cfg.llm.temperature == 0.3
    assert cfg.decide.engines == ["jev", "openai_decisions"]


def test_provider_can_be_turned_off_and_narrowed() -> None:
    off = BrainConfig.model_validate(
        {"cloud": {"reasoning_effort": "", "provider": {"send": False}}}
    )
    assert off.cloud.extra_body() == {}
    narrow = BrainConfig.model_validate(
        {"cloud": {"provider": {"only": ["deepinfra"], "order": ["deepinfra"], "zdr": True}}}
    )
    assert narrow.cloud.extra_body()["provider"] == {
        "data_collection": "deny",
        "require_parameters": True,
        "zdr": True,
        "only": ["deepinfra"],
        "order": ["deepinfra"],
    }


@pytest.mark.parametrize(
    "env",
    [
        {"SD_BRAIN__GROUND__COMMANDER_URL": "http://192.0.2.5:7700/commander/v1"},
        {"SD_BRAIN__CLOUD__REASONING_EFFORT": "huge"},
        {"SD_BRAIN__CLOUD__PROVIDER__DATA_COLLECTION": "maybe"},
        {"SD_BRAIN__CLOUD__BASE_URL": "http://openrouter.ai/api/v1"},
        {"SD_BRAIN__BACKEND__KIND": "scripted", "SD_BRAIN__PROFILE": "field"},
        {"SD_BRAIN__UNKNOWN": "1"},
    ],
)
def test_config_refuses(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        load_config(None, env)


def test_config_file_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="saknas"):
        load_config(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("profile = [")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_scripted_allowed_in_sim() -> None:
    assert BrainConfig.model_validate({"backend": {"kind": "scripted"}}).profile == "sim"


@pytest.mark.parametrize("value", ["", "  ", "CHANGE_ME", "change-me", "short"])
def test_check_secret_refuses(value: str) -> None:
    with pytest.raises(SecretError):
        check_secret("befälhavartokenet", value)


def test_load_secrets_refuses_weak_token_and_open_file(tmp_path: Path) -> None:
    with pytest.raises(SecretError, match="saknas"):
        load_secrets(tmp_path / "nope", {"SD_CLOUD_API_KEY": GOOD})
    with pytest.raises(SecretError):
        load_secrets(_token(tmp_path, "CHANGE_ME"), {"SD_CLOUD_API_KEY": GOOD})
    if os.name == "posix":
        with pytest.raises(SecretError, match="600"):
            load_secrets(_token(tmp_path, mode=0o644), {"SD_CLOUD_API_KEY": GOOD})
    with pytest.raises(SecretError):
        load_secrets(_token(tmp_path), {"SD_CLOUD_API_KEY": "CHANGE_ME"}, tmp_path / "x")
    with pytest.raises(SecretError):
        load_secrets(_token(tmp_path), {"SD_JEV_API_KEY": "change_me"}, tmp_path / "x")


def test_brain_env_reads_only_its_own_keys(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        f"SD_GROUND__DB_PASSWORD=secret-db\nSD_LLM_API_KEY={GOOD}\nSD_CLOUD_API_KEY='{GOOD}x'\n"
    )
    env = brain_env({"SD_JEV_API_KEY": "j" * 40}, dotenv)
    assert set(env) == {"SD_CLOUD_API_KEY", "SD_JEV_API_KEY"}  # the old local key is not read
    assert env["SD_CLOUD_API_KEY"] == GOOD + "x"
    assert "secret-db" not in env.values()
    # the environment wins over the file
    assert brain_env({"SD_CLOUD_API_KEY": "e" * 40}, dotenv)["SD_CLOUD_API_KEY"] == "e" * 40


def test_secrets_are_masked_in_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    token = "t" * 43
    secrets = load_secrets(_token(tmp_path, token), {"SD_CLOUD_API_KEY": GOOD}, tmp_path / "x")
    assert token not in repr(secrets)
    logger = logging.getLogger("sd_brain.test_mask")
    install_log_masking(secrets, logger)
    with caplog.at_level(logging.INFO, logger="sd_brain.test_mask"):
        logger.info("token %s key %s", token, GOOD)
        logger.info("Authorization: Bearer %s", "z" * 30)
    text = caplog.text
    assert token not in text
    assert GOOD not in text
    assert "z" * 30 not in text
    assert mask(f"a {GOOD} b", [GOOD]) == "a *** b"


def test_tracebacks_are_masked(tmp_path: Path) -> None:
    secrets = load_secrets(_token(tmp_path, "t" * 43), {"SD_CLOUD_API_KEY": GOOD}, tmp_path / "x")
    logger = logging.getLogger("sd_brain.test_mask_exc")
    stream = __import__("io").StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    install_log_masking(secrets, logger)
    try:
        raise RuntimeError(f"failed with {GOOD}")
    except RuntimeError:
        logger.exception("oops", stack_info=True)
    assert GOOD not in stream.getvalue() and "failed with ***" in stream.getvalue()
