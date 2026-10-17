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
    assert cfg.backend.kind == "llm_local"
    assert cfg.llm.base_url == "http://127.0.0.1:8091/v1"
    assert urlsplit(cfg.cloud.base_url).hostname == "openrouter.ai"
    assert cfg.decide.stop_check_timeout_s == 0.5


def test_config_with_dotenv_secrets(tmp_path: Path) -> None:
    token = _token(tmp_path)
    env = {
        "SD_BRAIN__TOKEN_FILE": str(token),
        "SD_LLM_API_KEY": GOOD,
        "SD_BRAIN__LLM__ROOT_URL": "http://127.0.0.1:9999",
        "SD_BRAIN__DECIDE__ENGINES": '["local"]',
    }
    cfg = load_config(None, env)
    assert cfg.llm.root_url == "http://127.0.0.1:9999"
    assert cfg.decide.engines == ["local"]
    secrets = load_secrets(cfg.token_file, env, tmp_path / "none.env")
    assert secrets.llm_api_key == GOOD
    assert secrets.cloud_api_key is None


@pytest.mark.parametrize(
    "env",
    [
        {"SD_BRAIN__GROUND__COMMANDER_URL": "http://192.0.2.5:7700/commander/v1"},
        {"SD_BRAIN__LLM__ROOT_URL": "http://198.51.100.2:8091"},
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
        check_secret("SD_LLM_API_KEY", value)


def test_load_secrets_refuses_weak_token_and_open_file(tmp_path: Path) -> None:
    with pytest.raises(SecretError, match="saknas"):
        load_secrets(tmp_path / "nope", {"SD_LLM_API_KEY": GOOD})
    with pytest.raises(SecretError):
        load_secrets(_token(tmp_path, "CHANGE_ME"), {"SD_LLM_API_KEY": GOOD})
    if os.name == "posix":
        with pytest.raises(SecretError, match="600"):
            load_secrets(_token(tmp_path, mode=0o644), {"SD_LLM_API_KEY": GOOD})
    with pytest.raises(SecretError):
        load_secrets(_token(tmp_path), {"SD_LLM_API_KEY": "CHANGE_ME"}, tmp_path / "x")
    with pytest.raises(SecretError):
        load_secrets(
            _token(tmp_path), {"SD_LLM_API_KEY": GOOD, "SD_CLOUD_API_KEY": "change_me"}, None
        )


def test_brain_env_reads_only_its_own_keys(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        f"SD_GROUND__DB_PASSWORD=secret-db\nSD_LLM_API_KEY={GOOD}\nSD_CLOUD_API_KEY='{GOOD}x'\n"
    )
    env = brain_env({"SD_JEV_API_KEY": "j" * 40}, dotenv)
    assert set(env) == {"SD_LLM_API_KEY", "SD_CLOUD_API_KEY", "SD_JEV_API_KEY"}
    assert env["SD_CLOUD_API_KEY"] == GOOD + "x"
    assert "secret-db" not in env.values()
    # the environment wins over the file
    assert brain_env({"SD_LLM_API_KEY": "e" * 40}, dotenv)["SD_LLM_API_KEY"] == "e" * 40


def test_secrets_are_masked_in_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    token = "t" * 43
    secrets = load_secrets(_token(tmp_path, token), {"SD_LLM_API_KEY": GOOD}, tmp_path / "x")
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
    secrets = load_secrets(_token(tmp_path, "t" * 43), {"SD_LLM_API_KEY": GOOD}, tmp_path / "x")
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
