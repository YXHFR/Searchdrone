"""The Unix socket the ``sd-ground`` CLI uses to talk to the running server.

One JSON line in, one JSON line out. Only these exist: a new login link or PIN (for the
owner at the terminal; it creates a pilot login, so the AI may never use this socket: an
import contract guards against direct use, and the real boundary is the file mode plus the
peer's user id, which needs the AI to run as its own OS user in the field profile) and
STOP / RETURN HOME. Never Resume. The socket file is 600 in a 700 directory, and a peer
with another user id (``SO_PEERCRED``) is disconnected. One bad request never stops it.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import struct
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sd_ground.auth import Principal
from sd_ground.logging import get_logger
from sd_ground.sources import EmergencyPath, EmergencyUnavailableError

__all__ = ["MAX_LINE", "ControlSocket", "call", "socket_path"]

MAX_LINE = 4096
log = get_logger("sd_ground.control")
_CLI = Principal(role="pilot", issuer_ref="operator:cli", auth_mode="cli", via="local")


def socket_path(data_dir: Path) -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime) if runtime else data_dir / "run"
    return base / "sd-ground.sock"


def _peer_uid(conn: socket.socket) -> int:
    creds = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    _pid, uid, _gid = struct.unpack("3i", creds)
    return uid


class ControlSocket:
    def __init__(
        self,
        path: Path,
        *,
        login_link: Callable[[], tuple[str, str]],
        pair: Callable[[], tuple[str, str, str]],
        emergency: EmergencyPath,
        peer_uid: Callable[[socket.socket], int] = _peer_uid,
    ) -> None:
        self.path = path
        self._login_link = login_link
        self._pair = pair
        self._emergency = emergency
        self._peer_uid = peer_uid
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        directory = self.path.parent
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if directory != Path(os.environ.get("XDG_RUNTIME_DIR", "/nonexistent")):
            directory.chmod(0o700)
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)
        try:
            sock.bind(str(self.path))
        finally:
            os.umask(old)
        self.path.chmod(0o600)
        sock.listen(8)
        sock.settimeout(0.5)
        self._sock = sock
        self._thread = threading.Thread(target=self._serve, name="sd-ground-control", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._sock is not None:
            self._sock.close()
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()

    def _serve(self) -> None:
        if self._sock is None:
            return
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with conn:
                conn.settimeout(2.0)
                try:
                    if self._peer_uid(conn) != os.getuid():
                        continue
                    try:
                        reply = self._handle(self._read_line(conn))
                    except Exception as exc:  # one bad request never stops the CLI STOP path
                        log.error("control.request_failed", error=type(exc).__name__)
                        reply = {"ok": False, "error": "INTERNAL"}
                    conn.sendall((json.dumps(reply) + "\n").encode())
                except OSError:
                    continue

    @staticmethod
    def _read_line(conn: socket.socket) -> bytes | None:
        """One line; ``None`` when it is longer than ``MAX_LINE`` (the rest is drained)."""
        data = b""
        total = 0
        too_long = False
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            total += len(chunk)
            if not too_long:
                data += chunk
            if b"\n" in chunk or total > 16 * MAX_LINE:
                break
            if len(data) > MAX_LINE:
                too_long = True
        line = data.split(b"\n", 1)[0]
        return None if too_long or len(line) > MAX_LINE else line

    def _handle(self, line: bytes | None) -> dict[str, Any]:
        if line is None:
            return {"ok": False, "error": "LINE_TOO_LONG"}
        try:
            req = json.loads(line)
        except ValueError:  # bad JSON or not UTF-8
            return {"ok": False, "error": "BAD_JSON"}
        if not isinstance(req, dict):
            return {"ok": False, "error": "BAD_JSON"}
        op = req.get("op")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if op == "login_link":
            code, url = self._login_link()
            return {"ok": True, "url": url, "code": code}
        if op == "pair":
            code, pin, url = self._pair()
            return {"ok": True, "url": url, "code": code, "pin": pin}
        if op == "action":
            action = req.get("action")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if action not in ("stop", "return_home", "home"):
                return {"ok": False, "error": "UNSUPPORTED_ACTION"}
            try:
                result = self._emergency.dispatch(
                    "stop" if action == "stop" else "return_home",
                    str(req.get("vehicle_id", "")),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                    client_action_id=str(req.get("client_action_id") or _new_id()),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                    via="cli",
                    by=_CLI,
                )
            except EmergencyUnavailableError as exc:
                return {"ok": False, "error": exc.code, "detail_sv": exc.detail_sv}
            return {"ok": True, "action_id": str(result.action_id)}
        return {"ok": False, "error": "UNKNOWN_OP"}


def _new_id() -> str:
    from sd_schemas.base import new_id

    return str(new_id())


def call(path: Path, request: dict[str, Any], timeout: float = 5.0) -> dict[str, Any] | None:
    """Send one request to a running server. ``None`` when no server is running."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(path))
            sock.sendall((json.dumps(request) + "\n").encode())
            data = b""
            while b"\n" not in data:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
    except OSError:  # no server, a dead socket or a timeout (TimeoutError is an OSError)
        return None
    try:
        out = json.loads(data.split(b"\n", 1)[0] or b"null")
    except ValueError:
        return None
    return out if isinstance(out, dict) else None  # pyright: ignore[reportUnknownVariableType]
