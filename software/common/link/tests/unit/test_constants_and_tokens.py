"""Constants, close codes and link tokens."""

from __future__ import annotations

import hmac
import importlib
import stat
from pathlib import Path

import pytest

from sd_link import constants
from sd_link.close_codes import FATAL_CLOSE_CODES, CloseCode
from sd_link.tokens import (
    TOKEN_MIN_CHARS,
    TokenFileError,
    TokenStore,
    generate_token,
    hash_token,
    read_token_file,
    write_token_hash,
)


def test_close_codes_are_locked() -> None:
    assert {c.name: c.value for c in CloseCode} == {
        "GOING_AWAY": 1001,
        "POLICY_VIOLATION": 1008,
        "AUTH_FAILED": 4401,
        "FORBIDDEN": 4403,
        "BAD_HELLO": 4400,
        "REPLACED": 4409,
    }
    assert {4401, 4403, 4400} == set(FATAL_CLOSE_CODES)


def test_timing_constants() -> None:
    assert constants.SDLINK_RESEND_S == 1.0
    assert constants.SDLINK_MAX_RESENDS == 3
    assert constants.LINK_LOSS_DETECT_S == 3.0
    assert constants.LINK_LOST_S == 10.0
    assert constants.HELLO_TIMEOUT_S == 5.0
    assert constants.HEARTBEAT_INTERVAL_S == 1.0
    assert constants.MAX_MESSAGE_BYTES == 64 * 1024
    assert (constants.RECONNECT_INITIAL_S, constants.RECONNECT_MAX_S) == (0.5, 5.0)
    assert constants.AUTH_RETRY_S == 30.0
    assert constants.SDLINK_PATH == "/sdlink/v1/control"


def test_constants_agree_with_the_safety_package() -> None:
    try:
        safety = importlib.import_module("sd_safety.constants")
    except ModuleNotFoundError:
        pytest.skip("sd_safety.constants not built yet")
    for name in ("LINK_LOSS_DETECT_S", "LINK_LOST_S"):  # both packages must have these
        assert getattr(safety, name) == getattr(constants, name), name
    for name in ("SDLINK_RESEND_S", "SDLINK_MAX_RESENDS", "HEARTBEAT_INTERVAL_S"):
        if hasattr(safety, name):
            assert getattr(safety, name) == getattr(constants, name), name


def test_token_is_32_random_bytes_base64url() -> None:
    a, b = generate_token(), generate_token()
    assert a != b and len(a) == TOKEN_MIN_CHARS == 43
    assert set(a) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


def test_store_keeps_only_hash_and_compares_constant_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = generate_token()
    path = tmp_path / "secrets" / "link_tokens.toml"
    write_token_hash(path, "sim-01", hash_token(token))
    text = path.read_text()
    assert token not in text and hash_token(token) in text
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    write_token_hash(path, "sim-02", hash_token(generate_token()))
    store = TokenStore.load(path)
    assert "sim-01" in store and "sim-02" in store

    calls: list[tuple[str, str]] = []
    real = hmac.compare_digest

    def spy(a: str, b: str) -> bool:
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr("sd_link.tokens.hmac.compare_digest", spy)
    assert store.match(token) == "sim-01"
    assert len(calls) == 2  # every entry is compared, no early exit
    assert store.match(generate_token()) is None
    assert store.match("") is None and store.match(None) is None


@pytest.mark.parametrize(
    "content",
    ['"sim-01" = "abc"', '"Bad Name" = "' + "0" * 64 + '"', '"sim-01" = 5', "not toml ="],
)
def test_store_refuses_bad_files(tmp_path: Path, content: str) -> None:
    path = tmp_path / "link_tokens.toml"
    path.write_text(content)
    with pytest.raises(TokenFileError):
        TokenStore.load(path)


def test_missing_store_is_empty(tmp_path: Path) -> None:
    assert TokenStore.load(tmp_path / "none.toml").match(generate_token()) is None


def test_read_token_file_checks_mode_and_length(tmp_path: Path) -> None:
    path = tmp_path / "token"
    with pytest.raises(TokenFileError, match="missing"):
        read_token_file(path)
    token = generate_token()
    path.write_text(token + "\n")
    path.chmod(0o644)
    with pytest.raises(TokenFileError, match="600"):
        read_token_file(path)
    path.chmod(0o600)
    assert read_token_file(path) == token
    path.write_text("short")
    with pytest.raises(TokenFileError, match="short"):
        read_token_file(path)
