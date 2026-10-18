"""No token, login code, PIN or password reaches a log file (04 T25)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sd_ground import __main__ as cli
from sd_ground import app
from sd_ground.app import start_lines
from sd_ground.context import Ground
from sd_ground.logging import REDACTOR, get_logger, setup_logging

H = {"X-SD-Client": "panel", "Content-Type": "application/json"}
BRAIN = "brain-token-for-tests-0123456789abcdef"


@pytest.fixture
def log_file(tmp_path: Path) -> Iterator[Path]:
    path = tmp_path / "logs" / "sd-ground.jsonl"
    setup_logging(path, level=logging.DEBUG)
    yield path
    for handler in list(logging.getLogger().handlers):
        handler.close()
        logging.getLogger().removeHandler(handler)


def test_scenario_leaves_no_secret_in_logs(
    log_file: Path, panel: TestClient, ground: Ground, pilot_token: str
) -> None:
    secrets: list[str] = [BRAIN, pilot_token]
    code, url = ground.login_link()
    secrets.append(code)
    res = panel.post("/api/v1/session", json={"login_code": code}, headers=H)
    cookie = res.cookies["sd_session"]
    secrets.append(cookie)
    pair_code, pin, _ = ground.pair()
    secrets += [pair_code, pin]
    wrong = "000000" if pin != "000000" else "111111"
    for _ in range(5):
        panel.post("/api/v1/session", json={"pin": wrong}, headers=H)
    panel.get("/api/v1/health", headers={"Authorization": f"Bearer {BRAIN}"})
    log = get_logger("sd_ground.test")
    # Code that logs carelessly is still caught: by key and by known value.
    log.info("login.issued", url=url, token=cookie, password="hunter2-password")
    log.warning(f"something with {code} inside")
    REDACTOR.add(pilot_token)
    log.info("cli", detail=f"token {pilot_token}")
    for handler in logging.getLogger().handlers:
        handler.flush()
    text = log_file.read_text(encoding="utf-8")
    assert "login.issued" in text
    for secret in [*secrets, "hunter2-password"]:
        assert secret not in text, "a secret reached the log file"


def test_serve_for_a_launcher_prints_no_login_code_or_pin(
    make_ground: Callable[..., Ground], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The simulator logs sd-ground's output to a file that CI uploads: no code goes there."""
    g = make_ground(lan={"enabled": True})

    def no_code() -> object:
        raise AssertionError("a login code was created")

    monkeypatch.setattr(g.logins, "create_login_code", no_code)
    monkeypatch.setattr(g.logins, "create_pair", no_code)
    lines = start_lines(g, with_codes=False)
    assert lines and not any("#login=" in line or "PIN" in line for line in lines)
    assert "sd-ground login-link" in lines[0]


def test_serve_prints_link_and_pin_by_default(make_ground: Callable[..., Ground]) -> None:
    g = make_ground(lan={"enabled": True})
    link, pair = start_lines(g, with_codes=True)
    assert "#login=" in link and "PIN: " in pair


@pytest.mark.parametrize(("argv", "announce"), [([], True), (["--no-login-link"], False)])
def test_serve_flag(monkeypatch: pytest.MonkeyPatch, argv: list[str], announce: bool) -> None:
    seen: dict[str, bool] = {}

    def fake_serve(_loaded: object, *, announce_login: bool) -> int:
        seen["announce"] = announce_login
        return 0

    monkeypatch.setattr(cli, "_loaded", lambda: None)
    monkeypatch.setattr(app, "serve", fake_serve)
    assert cli.main(["serve", *argv]) == 0
    assert seen == {"announce": announce}
