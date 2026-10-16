"""Start checks: every one stops sd-onboard with exit code 2; the token stays secret."""

from __future__ import annotations

import io
import logging
import os
import shutil
import tomllib
from pathlib import Path

import pytest

from sd_onboard import app
from sd_onboard.config import MIN_TOKEN_BYTES, check, load
from sd_onboard.logs import MASK, SecretFilter

REPO = Path(__file__).resolve().parents[4]
EXAMPLE = REPO / "config" / "onboard.example.toml"
PARM = REPO / "simulation/sitl/params/searchdrone-safety.parm"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A minimal repository root with the example config and the parameter file."""
    r = tmp_path / "repo"
    (r / "config").mkdir(parents=True)
    shutil.copy(EXAMPLE, r / "config" / "onboard.example.toml")
    (r / "simulation/sitl/params").mkdir(parents=True)
    shutil.copy(PARM, r / "simulation/sitl/params" / PARM.name)
    return r


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    data = tmp_path / "data"
    (data / "secrets").mkdir(parents=True)
    token = data / "secrets" / "onboard_link_token"
    token.write_text("t" * 43)
    token.chmod(0o600)
    return {"SD_DATA_DIR": str(data)}


def test_example_config_has_no_token() -> None:
    data = tomllib.loads(EXAMPLE.read_text())
    assert "token" not in data and "sdlink_token" not in data
    assert all("token" not in k or k == "link_token_file" for k in data)
    assert "${SD_DATA_DIR}" in data["link_token_file"]


def test_caps_not_configurable() -> None:
    import re

    keys = re.compile(r"max_alt|max_speed|max_range|battery_rtl")
    data = tomllib.loads(EXAMPLE.read_text())
    flat = list(data) + [k for v in data.values() if isinstance(v, dict) for k in v]
    assert not any(keys.search(k) for k in flat)
    from sd_onboard.config import OnboardConfig

    assert not any(keys.search(k) for k in OnboardConfig.model_fields)


def test_good_config_starts(root: Path, env: dict[str, str]) -> None:
    assert check(load(root, env)) == []


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("SD_ONBOARD__PROFILE", "field"),
        ("SD_ONBOARD__VEHICLE_ID", "Sim 01"),
        ("SD_ONBOARD__FC_SYSID", "250"),
        ("SD_ONBOARD__FC_SYSID", "255"),
        ("SD_ONBOARD__OWN_COMPID", "190"),
        ("SD_ONBOARD__TEST_HOOKS__TAKEOFF_DELAY_AFTER_ARM_S", "5"),
    ],
)
def test_startup_checks(root: Path, env: dict[str, str], key: str, value: str) -> None:
    assert check(load(root, {**env, key: value}))


def test_test_hooks_rejected_outside_test(root: Path, env: dict[str, str]) -> None:
    hook = {"SD_ONBOARD__TEST_HOOKS__TAKEOFF_DELAY_AFTER_ARM_S": "5"}
    assert check(load(root, {**env, **hook}))
    assert check(load(root, {**env, **hook, "SD_TEST": "1"})) == []


def test_token_file_mode_checked(root: Path, env: dict[str, str]) -> None:
    token = Path(env["SD_DATA_DIR"]) / "secrets" / "onboard_link_token"
    token.chmod(0o640)
    assert any("600" in e for e in check(load(root, env)))


def test_token_min_length(root: Path, env: dict[str, str]) -> None:
    token = Path(env["SD_DATA_DIR"]) / "secrets" / "onboard_link_token"
    token.write_text("x" * (MIN_TOKEN_BYTES - 1))
    assert check(load(root, env))
    token.write_text("x" * MIN_TOKEN_BYTES)
    assert check(load(root, env)) == []


def test_token_outside_repo(root: Path, env: dict[str, str]) -> None:
    inside = root / "secrets" / "onboard_link_token"
    inside.parent.mkdir()
    inside.write_text("x" * 40)
    inside.chmod(0o600)
    errors = check(load(root, {**env, "SD_ONBOARD__LINK_TOKEN_FILE": "secrets/onboard_link_token"}))
    assert any("repot" in e for e in errors)
    state_inside = {**env, "SD_ONBOARD__STATE_DIR": "state"}
    assert any("state_dir" in e for e in check(load(root, state_inside)))


def test_token_on_a_windows_disk_is_refused(root: Path, env: dict[str, str]) -> None:
    errors = check(load(root, {**env, "SD_ONBOARD__LINK_TOKEN_FILE": "/mnt/c/token"}))
    assert any("Windows" in e for e in errors)


def test_unknown_key_is_an_error(root: Path) -> None:
    (root / "config" / "onboard.toml").write_text("max_alt_agl_m = 300\n")
    assert app.main.__module__ == "sd_onboard.app"
    cwd = Path.cwd()
    os.chdir(root)
    try:
        assert app.main() == 2
    finally:
        os.chdir(cwd)


def test_main_exits_2_on_a_failed_check(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(root)
    monkeypatch.setenv("SD_DATA_DIR", str(root.parent / "nothing"))
    assert app.main() == 2


def test_token_masked_in_logs() -> None:
    secret = "s3cr3t-" + "x" * 40
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(SecretFilter([secret]))
    logger = logging.getLogger("sd_onboard.test")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        logger.info("dialling with %s", secret)
        logger.info("headers {'Authorization': 'Bearer abcdef123'}")
        logger.info("token=zzz999")
    finally:
        logger.removeHandler(handler)
    text = stream.getvalue()
    assert secret not in text and "abcdef123" not in text and "zzz999" not in text
    assert text.count(MASK) == 3
