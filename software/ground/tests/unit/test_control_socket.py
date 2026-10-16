"""The CLI's Unix socket (04 T32 without the MAVLink part, C8)."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from sd_ground.context import Ground
from sd_ground.control_socket import ControlSocket, call, socket_path


@pytest.fixture
def control(tmp_path: Path, ground: Ground, emergency: object) -> Iterator[ControlSocket]:
    path = tmp_path / "run" / "sd-ground.sock"
    sock = ControlSocket(path, login_link=ground.login_link, pair=ground.pair, emergency=emergency)  # type: ignore[arg-type]
    sock.start()
    yield sock
    sock.stop()


def test_socket_modes(control: ControlSocket) -> None:
    assert stat.S_IMODE(control.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(control.path.parent.stat().st_mode) == 0o700


def test_socket_path_follows_runtime_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "xdg"))
    assert socket_path(tmp_path / "data") == tmp_path / "xdg" / "sd-ground.sock"
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    assert socket_path(tmp_path / "data") == tmp_path / "data" / "run" / "sd-ground.sock"


def test_login_link_code_works_once(control: ControlSocket, ground: Ground) -> None:
    reply = call(control.path, {"op": "login_link"})
    assert reply is not None and reply["ok"] is True
    code = str(reply["code"])
    assert str(reply["url"]).endswith(f"#login={code}")
    assert ground.logins.redeem("a", login_code=code, pin=None) == "local"
    with pytest.raises(Exception):  # noqa: B017 - LoginError, used once
        ground.logins.redeem("a", login_code=code, pin=None)


def test_pair_gives_pin(control: ControlSocket) -> None:
    reply = call(control.path, {"op": "pair"})
    assert reply is not None and reply["ok"] is True
    assert len(str(reply["pin"])) == 6 and str(reply["pin"]).isdigit()
    assert str(reply["url"]).startswith("https://")


def test_action_stop_reaches_emergency_path(control: ControlSocket, emergency: object) -> None:
    reply = call(control.path, {"op": "action", "vehicle_id": "sim-01", "action": "stop"})
    assert reply is not None and reply["ok"] is True and reply["action_id"]
    assert emergency.calls == [("stop", "sim-01", "operator:cli")]  # type: ignore[attr-defined]


def test_resume_is_not_offered(control: ControlSocket) -> None:
    assert call(control.path, {"op": "action", "vehicle_id": "sim-01", "action": "resume"}) == {
        "ok": False,
        "error": "UNSUPPORTED_ACTION",
    }


def test_unknown_op_and_long_line(control: ControlSocket) -> None:
    assert call(control.path, {"op": "nope"}) == {"ok": False, "error": "UNKNOWN_OP"}
    reply = call(control.path, {"op": "pair", "pad": "x" * 5000})
    assert reply == {"ok": False, "error": "LINE_TOO_LONG"}


def test_other_user_is_disconnected(tmp_path: Path, ground: Ground, emergency: object) -> None:
    sock = ControlSocket(
        tmp_path / "s" / "x.sock",
        login_link=ground.login_link,
        pair=ground.pair,
        emergency=emergency,  # pyright: ignore[reportArgumentType]
        peer_uid=lambda _c: os.getuid() + 1,  # type: ignore[arg-type]
    )
    sock.start()
    try:
        assert call(sock.path, {"op": "login_link"}) is None
    finally:
        sock.stop()


def test_cli_without_server_exits_2(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "SD_GROUND__DATA_DIR": str(tmp_path / "nowhere"),
        "XDG_RUNTIME_DIR": str(tmp_path),
    }
    for args in (["login-link"], ["pair"], ["action", "stop", "--vehicle", "sim-01"]):
        res = subprocess.run(
            [sys.executable, "-m", "sd_ground", *args],
            env=env,
            capture_output=True,
            text=True,
            cwd=tmp_path,
            check=False,
        )
        assert res.returncode == 2, res.stderr
        assert "sd-ground körs inte" in res.stderr


def test_garbage_never_stops_the_cli_stop_path(control: ControlSocket, emergency: object) -> None:
    import socket as raw

    for junk in (b"\xff\xfe\xfa\n", b"[1, 2]\n", b"\n", b'{"op": "action", "action": 5}\n'):
        with raw.socket(raw.AF_UNIX, raw.SOCK_STREAM) as s:
            s.connect(str(control.path))
            s.sendall(junk)
            assert b'"ok": false' in s.recv(4096)
    reply = call(control.path, {"op": "action", "vehicle_id": "sim-01", "action": "stop"})
    assert reply is not None and reply["ok"] is True
    assert emergency.calls[-1][0] == "stop"  # type: ignore[attr-defined]
