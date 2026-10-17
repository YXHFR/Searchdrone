"""Keep every service in compose.yaml and the MediaMTX config on localhost.

Rules (each finding is printed as ``file: rule: message``, exit code 1):

compose.yaml
  * every ``ports:`` entry publishes on ``127.0.0.1`` (``127.0.0.1:<host>:<container>``
    or the long syntax with ``host_ip: 127.0.0.1``);
  * ``network_mode: host`` needs the comment ``# host-network: <reason>`` on the same
    line or the line above, and every ``MTX_*ADDRESS`` variable of that service must
    default to ``127.0.0.1:<port>`` or ``${SD_VIDEO_BIND:-127.0.0.1}:<port>``.

config/mediamtx.yml
  * every key ending in ``Address`` is empty or starts with ``127.0.0.1:``;
  * ``rtmp``, ``srt`` and ``moq`` (Media over QUIC, on by default since 1.21 and bound
    to every interface) are ``no``;
  * ``CHANGE_ME`` appears only as the internal user's ``pass:`` value, and only when
    compose.yaml overrides it with ``MTX_AUTHINTERNALUSERS_0_PASS: ${...:?...}``.

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
MTX_ADDRESS_OK_RE = re.compile(r"^(127\.0\.0\.1|\$\{SD_VIDEO_BIND:-127\.0\.0\.1\}):\d{1,5}$")
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
                if MTX_ADDRESS_RE.match(key) and not MTX_ADDRESS_OK_RE.match(value):
                    errors.append(
                        f"{COMPOSE}: host-network: tjänsten {name}: {key}={value!r} "
                        "ska vara 127.0.0.1:<port> eller ${SD_VIDEO_BIND:-127.0.0.1}:<port>"
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


def compose_requires_mtx_pass(compose_text: str | None) -> bool:
    if not compose_text:
        return False
    data = yaml.safe_load(compose_text) or {}
    for svc in (data.get("services") or {}).values():
        value = environment(svc or {}).get("MTX_AUTHINTERNALUSERS_0_PASS")
        if value is not None and REQUIRED_SECRET_RE.match(value):
            return True
    return False


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
    override = compose_requires_mtx_pass(compose_text)
    for n, line in enumerate(text.splitlines(), start=1):
        if "CHANGE_ME" not in line.upper():
            continue
        if not CHANGE_ME_PASS_RE.match(line):
            errors.append(f"{MEDIAMTX}:{n}: change-me: CHANGE_ME är bara tillåtet i pass:")
        elif not override:
            errors.append(
                f"{MEDIAMTX}:{n}: change-me: compose.yaml måste sätta "
                "MTX_AUTHINTERNALUSERS_0_PASS: ${...:?...}"
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
