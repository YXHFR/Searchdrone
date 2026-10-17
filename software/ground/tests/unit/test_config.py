"""``config check`` and the config loader (04 T23 without video, D1, D4, D5, C9)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sd_ground.config import ConfigError, check, load

# Low-entropy stand-ins, built at run time (never real secrets).
APP = "b" * 20
GOOD_ENV = "".join(
    f"{key}={value}\n"
    for key, value in (
        ("SD_GROUND__DB_PASSWORD", "a" * 20),
        ("SD_GROUND__DB_APP_PASSWORD", APP),
        ("SD_GROUND__VIDEO_PUBLISH_PASSWORD", "c" * 20),
        ("SD_GROUND__VIDEO_API_PASSWORD", "d" * 20),
    )
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "ground.toml").write_text(f'data_dir = "{tmp_path / "data"}"\n')
    env = tmp_path / ".env"
    env.write_text(GOOD_ENV + f"SD_DATA_DIR={tmp_path / 'data'}\n")
    env.chmod(0o600)
    secrets = tmp_path / "secrets"
    secrets.mkdir(mode=0o700)
    secrets.chmod(0o700)
    token = secrets / "brain_token"
    token.write_text("t" * 43)
    token.chmod(0o600)
    return tmp_path


def _errors(root: Path, extra_toml: str = "", environ: dict[str, str] | None = None) -> list[str]:
    if extra_toml:
        with (root / "config" / "ground.toml").open("a") as f:
            f.write(extra_toml)
    return check(load(root, environ or {}))


def test_good_config_has_no_errors(root: Path) -> None:
    assert _errors(root) == []


@pytest.mark.parametrize(
    ("toml", "needle"),
    [
        ('[http]\ncommander_bind = "0.0.0.0:7700"\n', "commander_bind"),
        ('[http]\npanel_bind = "192.168.1.5:8080"\n', "panel_bind"),
        ('[sdlink]\nbind = "0.0.0.0:7600"\n', "sdlink.bind"),
        ("[remote]\nenabled = true\n", "Fjärråtkomst finns inte i v0.1.0"),
        ('profile = "field"\n', "profilen sim"),
        ("[lan]\nenabled = true\n", "certifikat"),
        ("[auth]\nlogin_code_ttl_s = 5\n", "login_code_ttl_s"),
    ],
)
def test_refused_settings(root: Path, toml: str, needle: str) -> None:
    errors = _errors(root, toml)
    assert any(needle in e for e in errors), errors


def test_login_code_ttl_allowed_in_test_rig(root: Path) -> None:
    assert _errors(root, "[auth]\ncli_may_confirm = true\nlogin_code_ttl_s = 5\n") == []


def test_unknown_keys_stop_the_server(root: Path) -> None:
    (root / "config" / "ground.toml").write_text("[auth]\nskip_login = true\n")
    with pytest.raises(ConfigError) as exc:
        load(root, {})
    assert any("auth.skip_login" in e for e in exc.value.errors)
    with pytest.raises(ConfigError):
        load(root, {"SD_GROUND__AUTH__SKIP_LOGIN": "true"})


def test_env_overrides_toml(root: Path) -> None:
    loaded = load(root, {"SD_GROUND__AUTH__CLI_MAY_CONFIRM": "true", "SD_GROUND__PROFILE": "sim"})
    assert loaded.config.auth.cli_may_confirm is True


def test_brain_token_file_from_env(root: Path) -> None:
    other = root / "secrets" / "other_token"
    other.write_text("x")
    other.chmod(0o600)
    loaded = load(root, {"SD_GROUND__AUTH__BRAIN_TOKEN_FILE": str(other)})
    assert loaded.path(loaded.config.auth.brain_token_file) == other


@pytest.mark.parametrize("value", ["", "CHANGE_ME"])
def test_placeholder_secret(root: Path, value: str) -> None:
    (root / ".env").write_text(GOOD_ENV.replace(APP, value))
    assert any("SD_GROUND__DB_APP_PASSWORD" in e for e in _errors(root))


def test_data_dir_mismatch(root: Path) -> None:
    (root / ".env").write_text(GOOD_ENV + "SD_DATA_DIR=/somewhere/else\n")
    assert any("SD_DATA_DIR" in e for e in _errors(root))


@pytest.mark.parametrize(
    ("relative", "mode"),
    [(".env", 0o644), ("secrets/brain_token", 0o640), ("config/local/tls/server.key", 0o644)],
)
def test_loose_file_modes(root: Path, relative: str, mode: int) -> None:
    path = root / relative
    if relative.startswith("config/local/tls"):
        path.parent.mkdir(parents=True, mode=0o700)
        path.parent.chmod(0o700)
        path.write_text("key")
    path.chmod(mode)
    assert _errors(root), relative


def test_secrets_dir_must_be_700(root: Path) -> None:
    (root / "secrets").chmod(0o755)
    assert any("secrets/" in e for e in _errors(root))


def test_missing_brain_token(root: Path) -> None:
    (root / "secrets" / "brain_token").unlink()
    assert any("sd-ground init" in e for e in _errors(root))


def test_test_build_only_in_sim(root: Path) -> None:
    dist = root / "software" / "panel" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text('<meta name="sd-build" content="test">')
    assert _errors(root) == []
    assert any("sd-build=test" in e for e in _errors(root, 'profile = "bench"\n'))


def test_example_file_loads_without_errors() -> None:
    repo = Path(__file__).resolve().parents[4]
    loaded = load(repo, {}, config_file=repo / "config" / "ground.example.toml")
    assert loaded.config.http.commander_bind == "127.0.0.1:7700"
    assert loaded.config.remote.enabled is False


def test_no_setting_turns_off_login() -> None:
    from sd_ground.config import AuthConfig

    names = set(AuthConfig.model_fields)
    assert names == {"cli_may_confirm", "brain_token_file", "login_code_ttl_s"}
    assert not any(re.search(r"auto_?(approve|confirm|clear)|skip_|localhost", n) for n in names)


@pytest.mark.parametrize("value", ["", "CHANGE_ME", "change_me"])
def test_video_publish_password_placeholder(root: Path, value: str) -> None:
    """B1-09 point 6: an empty or CHANGE_ME publish password stops the ground station."""
    (root / ".env").write_text(GOOD_ENV.replace("c" * 20, value))
    assert any("SD_GROUND__VIDEO_PUBLISH_PASSWORD" in e for e in _errors(root))


@pytest.mark.parametrize(
    ("key", "url"),
    [
        ("mediamtx_whep", "http://0.0.0.0:8889"),
        ("mediamtx_api", "http://192.168.1.20:9997"),
        ("mediamtx_hls", "https://127.0.0.1:8888"),
        ("mediamtx_whep", "http://127.0.0.1:8889/../x"),
    ],
)
def test_video_server_only_on_loopback(root: Path, key: str, url: str) -> None:
    errors = _errors(root, f'[video]\n{key} = "{url}"\n')
    assert any(f"video.{key}" in e for e in errors)


def test_video_defaults_are_loopback(root: Path) -> None:
    assert not any(e.startswith("video.") for e in _errors(root))
