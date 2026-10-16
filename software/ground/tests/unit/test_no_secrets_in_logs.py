"""No token, login code, PIN or password reaches a log file (04 T25)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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
