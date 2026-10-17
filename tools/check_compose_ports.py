"""Keep every service in compose.yaml and the MediaMTX config on localhost.

Rules (each finding is printed as ``file: rule: message``, exit code 1):

compose.yaml
  * every ``ports:`` entry publishes on ``127.0.0.1`` (``127.0.0.1:<host>:<container>``
    or the long syntax with ``host_ip: 127.0.0.1``);
  * ``network_mode: host`` needs the comment ``# host-network: <reason>`` on the same
    line or the line above, and every ``MTX_*ADDRESS`` variable of that service must be
    ``127.0.0.1:<port>``; only ``MTX_WEBRTCLOCALUDPADDRESS`` (the WebRTC media port) may
    be ``${SD_VIDEO_BIND:-127.0.0.1}:<port>``.

config/mediamtx.yml
  * every key ending in ``Address`` is empty or starts with ``127.0.0.1:``;
  * ``rtmp``, ``srt`` and ``moq`` (Media over QUIC, on by default since 1.21 and bound
    to every interface) are ``no``;
  * ``CHANGE_ME`` appears only as an internal user's ``pass:`` value, and only when
    compose.yaml overrides that user's password with
    ``MTX_AUTHINTERNALUSERS_<n>_PASS: ${...:?...}``;
  * a user without a password may only ``read`` (the control API can rebind listeners,
    and publishing must need the password).

A missing file is skipped (the database and video tickets add them); a compose file
anywhere but the repository root is an error.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

COMPOSE = "compose.yaml"
MEDIAMTX = "config/mediamtx.yml"
LOCALHOST = "127.0.0.1"
HOST_NETWORK_RE = re.compile(r"^\s*network_mode:\s*[\"']?host[\"']?\s*(#.*)?$")
HOST_NETWORK_COMMENT_RE = re.compile(r"#\s*host-network:\s*\S")
MTX_ADDRESS_RE = re.compile(r"^MTX_\w*ADDRESS$")
MTX_ADDRESS_OK_RE = re.compile(r"^127\.0\.0\.1:\d{1,5}$")
MTX_LAN_ADDRESS_RE = re.compile(r"^\$\{SD_VIDEO_BIND:-127\.0\.0\.1\}:\d{1,5}$")
MTX_LAN_KEY = "MTX_WEBRTCLOCALUDPADDRESS"  # the WebRTC media port, the only one on the LAN
MTX_PASS_RE = re.compile(r"^MTX_AUTHINTERNALUSERS_(\d+)_PASS$")
REQUIRED_SECRET_RE = re.compile(r"^\$\{[A-Z0-9_]+:\?[^}]*\}$")
CHANGE_ME_PASS_RE = re.compile(r"^\s*-?\s*pass:\s*[\"']?CHANGE_ME[\"']?\s*(#.*)?$")


def port_ok(entry: Any) -> bool:
    if isinstance(entry, dict):
        return entry.get("host_ip") == LOCALHOST and "published" in entry
    if not isinstance(entry, str):
        return False  # a bare number publishes on every interface
    parts = entry.split(":")
    return len(parts) == 3 and parts[0] == LOCALHOST and all(parts[1:])


def environment(service: dict[str, Any]) -> dict[str, str]:
    env = service.get("environment") or {}
    if isinstance(env, list):
        pairs = (str(item).split("=", 1) for item in env)
        return {p[0]: (p[1] if len(p) > 1 else "") for p in pairs}
    return {str(k): "" if v is None else str(v) for k, v in env.items()}


def host_network_lines(text: str) -> Iterator[tuple[int, bool]]:
    """Yield (line number, has reason comment) for each ``network_mode: host`` line."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if HOST_NETWORK_RE.match(line):
            prev = lines[i - 1] if i > 0 else ""
            yield i + 1, bool(HOST_NETWORK_COMMENT_RE.search(line + " " + prev))


def check_compose(text: str) -> list[str]:
    errors: list[str] = []
    data = yaml.safe_load(text) or {}
    services: dict[str, Any] = data.get("services") or {}
    for name, svc in services.items():
        svc = svc or {}
        for entry in svc.get("ports") or []:
            if not port_ok(entry):
                errors.append(
                    f"{COMPOSE}: port: tjänsten {name} publicerar {entry!r} utanför 127.0.0.1"
                )
        if svc.get("network_mode") == "host":
            for key, value in environment(svc).items():
                if not MTX_ADDRESS_RE.match(key) or MTX_ADDRESS_OK_RE.match(value):
                    continue
                if key == MTX_LAN_KEY and MTX_LAN_ADDRESS_RE.match(value):
                    continue
                errors.append(
                    f"{COMPOSE}: host-network: tjänsten {name}: {key}={value!r} "
                    "ska vara 127.0.0.1:<port> (bara mediaporten får följa SD_VIDEO_BIND)"
                )
    for line, ok in host_network_lines(text):
        if not ok:
            errors.append(
                f"{COMPOSE}:{line}: host-network: network_mode: host saknar "
                "kommentaren '# host-network: <skäl>'"
            )
    return errors


def address_keys(node: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(node, dict):
        for k, v in node.items():
            key = f"{path}.{k}" if path else str(k)
            if str(k).endswith("Address"):
                yield key, v
            yield from address_keys(v, key)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from address_keys(v, f"{path}[{i}]")


def is_no(value: Any) -> bool:
    return value is False or (isinstance(value, str) and value.lower() in {"no", "false"})


def compose_mtx_passwords(compose_text: str | None) -> set[int]:
    """Indexes of the MediaMTX users whose password compose.yaml requires from .env."""
    if not compose_text:
        return set()
    data = yaml.safe_load(compose_text) or {}
    found: set[int] = set()
    for svc in (data.get("services") or {}).values():
        for key, value in environment(svc or {}).items():
            match = MTX_PASS_RE.match(key)
            if match and REQUIRED_SECRET_RE.match(value):
                found.add(int(match.group(1)))
    return found


def check_mediamtx(text: str, compose_text: str | None) -> list[str]:
    errors: list[str] = []
    data = yaml.safe_load(text) or {}
    for key, value in address_keys(data):
        if value in (None, "") or (isinstance(value, str) and value.startswith(f"{LOCALHOST}:")):
            continue
        errors.append(f"{MEDIAMTX}: address: {key}={value!r} ska börja med {LOCALHOST}:")
    for key in ("rtmp", "srt", "moq"):
        if not is_no(data.get(key)):
            errors.append(f"{MEDIAMTX}: protocol: {key} ska vara no")
    overrides = compose_mtx_passwords(compose_text)
    for n, line in enumerate(text.splitlines(), start=1):
        if "CHANGE_ME" in line.upper() and not CHANGE_ME_PASS_RE.match(line):
            errors.append(f"{MEDIAMTX}:{n}: change-me: CHANGE_ME är bara tillåtet i pass:")
    for i, user in enumerate(data.get("authInternalUsers") or []):
        user = user or {}
        name = user.get("user", f"#{i}")
        password = user.get("pass")
        actions = [str((p or {}).get("action")) for p in user.get("permissions") or []]
        if str(password).upper() == "CHANGE_ME" and i not in overrides:
            errors.append(
                f"{MEDIAMTX}: change-me: användaren {name}: compose.yaml måste sätta "
                f"MTX_AUTHINTERNALUSERS_{i}_PASS: ${{...:?...}}"
            )
        if password in (None, "") and any(a != "read" for a in actions):
            errors.append(
                f"{MEDIAMTX}: no-password: användaren {name} utan lösenord får bara read, "
                f"inte {', '.join(a for a in actions if a != 'read')}"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Keep compose.yaml and MediaMTX on localhost.")
    ap.add_argument("--root", type=Path, default=Path())
    args = ap.parse_args(argv)

    compose = args.root / COMPOSE
    # compose.yaml lives in the repository root. A compose file anywhere else would
    # escape this check (and the CI database step), so it is an error.
    strays = sorted(
        str(f.relative_to(args.root))
        for pattern in ("*compose*.yaml", "*compose*.yml")
        for f in args.root.rglob(pattern)
        if f != compose and not {"node_modules", ".venv", ".git"} & set(f.parts)
    )
    mediamtx = args.root / MEDIAMTX
    compose_text = compose.read_text(encoding="utf-8") if compose.is_file() else None
    errors: list[str] = []
    checked: list[str] = []
    if compose_text is not None:
        errors += check_compose(compose_text)
        checked.append(COMPOSE)
    if mediamtx.is_file():
        errors += check_mediamtx(mediamtx.read_text(encoding="utf-8"), compose_text)
        checked.append(MEDIAMTX)
    errors += [f"{s}: location: compose-filer hör hemma i {COMPOSE} i repots rot" for s in strays]
    for e in errors:
        print(e)
    if errors:
        return 1
    print(f"check_compose_ports: ok ({', '.join(checked) or 'inga filer än'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
