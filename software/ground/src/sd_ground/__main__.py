"""The ``sd-ground`` command line.

Exit codes: 0 ok, 1 refused (bad config, missing secret, wrong input), 2 not possible now
(no running server, or a part that is not built yet).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import ipaddress
import os
import re
import secrets
import socket
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

from sd_ground.auth import SessionRecord, hash_token, new_token, parse_duration, token_expiry
from sd_ground.config import SECRET_ENV_KEYS, ConfigError, Loaded, check, load, read_env_file
from sd_ground.control_socket import call, socket_path

__all__ = ["main"]

_VEHICLE = re.compile(r"^[a-z0-9][a-z0-9-]{2,31}$")
_TOKEN_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def _say(text: str, *, err: bool = False) -> None:
    print(text, file=sys.stderr if err else sys.stdout, flush=True)


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def _loaded() -> Loaded:
    try:
        return load()
    except ConfigError as exc:
        for line in exc.errors:
            _say(line, err=True)
        raise SystemExit(1) from None


# --- init ---------------------------------------------------------------------------


def _ensure_env(root: Path, data_dir: Path) -> list[str]:
    env_path = root / ".env"
    current = read_env_file(env_path)
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.is_file() else []
    wanted = {key: secrets.token_urlsafe(24) for key in SECRET_ENV_KEYS}
    wanted |= {"SD_UID": str(os.getuid()), "SD_GID": str(os.getgid()), "SD_DATA_DIR": str(data_dir)}
    filled = []
    out = []
    for line in lines:
        name = line.partition("=")[0].strip()
        if name in wanted and current.get(name, "").strip().lower() in (
            "",
            "change_me",
            "changeme",
        ):
            out.append(f"{name}={wanted[name]}")
            filled.append(name)
            current[name] = wanted[name]
        else:
            out.append(line)
    for name, value in wanted.items():
        if name not in current:
            out.append(f"{name}={value}")
            filled.append(name)
    _write_private(env_path, "\n".join(out) + "\n")
    return filled


def make_lan_cert(tls: Path) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    from sd_ground.context import lan_addresses

    key = ec.generate_private_key(ec.SECP256R1())
    names: list[x509.GeneralName] = []
    for host in lan_addresses():
        try:
            names.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            names.append(x509.DNSName(host))
    now = dt.datetime.now(dt.UTC)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"{socket.gethostname()}.local")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .sign(key, hashes.SHA256())
    )
    _private_dir(tls)
    _write_private(
        tls / "server.key",
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode(),
    )
    _write_private(tls / "server.crt", cert.public_bytes(serialization.Encoding.PEM).decode())


def cmd_init(args: argparse.Namespace) -> int:
    root = Path.cwd()
    local = root / "config" / "ground.toml"
    example = root / "config" / "ground.example.toml"
    if not local.exists() and example.is_file():
        _write_private(local, example.read_text(encoding="utf-8"))
        _say("skapade config/ground.toml")
    loaded = _loaded()
    for name in _ensure_env(root, loaded.data_dir):
        _say(f".env: fyllde i {name}")
    _private_dir(root / "secrets")
    for name in (loaded.config.auth.brain_token_file, loaded.config.auth.search_token_file):
        token_file = loaded.path(name)
        if not token_file.is_file():
            _private_dir(token_file.parent)
            _write_private(token_file, new_token() + "\n")
            _say(f"skapade {name}")
    _private_dir(loaded.data_dir)  # mission data and logs: only for the owner
    if args.lan:
        make_lan_cert(loaded.path(loaded.config.lan.tls_dir))
        _say(f"skapade certifikat i {loaded.config.lan.tls_dir}")
    _say("sd-ground init klar")
    return 0


# --- config, database, serve --------------------------------------------------------


def cmd_config_check(_: argparse.Namespace) -> int:
    errors = check(_loaded())
    for line in errors:
        _say(line, err=True)
    if not errors:
        _say("Konfigurationen är i ordning.")
    return 1 if errors else 0


def cmd_db_setup(_: argparse.Namespace) -> int:
    from sd_ground.db.setup import db_setup

    loaded = _loaded()
    owner, app = (
        loaded.secret("SD_GROUND__DB_PASSWORD"),
        loaded.secret("SD_GROUND__DB_APP_PASSWORD"),
    )
    if not owner or not app:
        _say("Lösenorden saknas i .env. Kör: just setup-config", err=True)
        return 1
    if db_setup(loaded.config.db.owner_url, owner, app):
        _say("skapade databasrollen sd_app")
    _say("Databasen är uppdaterad.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from sd_ground.app import serve

    return serve(_loaded(), announce_login=not args.no_login_link)


def _socket_call(request: dict[str, object]) -> dict[str, object] | None:
    loaded = _loaded()
    return call(socket_path(loaded.data_dir), request)


def cmd_login_link(args: argparse.Namespace) -> int:
    reply = _socket_call({"op": "login_link"})
    if reply is None:
        _say("sd-ground körs inte", err=True)
        return 2
    _say(str(reply["code"] if args.print_code else reply["url"]))
    return 0


def cmd_pair(_: argparse.Namespace) -> int:
    reply = _socket_call({"op": "pair"})
    if reply is None:
        _say("sd-ground körs inte", err=True)
        return 2
    _say(f"PIN: {reply['pin']}")
    _say(str(reply["url"]))
    try:
        import segno

        segno.make(str(reply["url"])).terminal(compact=True)
    except Exception:  # noqa: S110 - the QR code is a convenience
        pass
    return 0


def cmd_action(args: argparse.Namespace) -> int:
    reply = _socket_call({"op": "action", "vehicle_id": args.vehicle, "action": args.action})
    if reply is None:
        _say("sd-ground körs inte", err=True)
        return 2
    if not reply.get("ok"):
        _say(f"Gick inte: {reply.get('error')} {reply.get('detail_sv', '')}".strip(), err=True)
        return 2
    _say(f"skickat ({reply['action_id']})")
    return 0


# --- vehicles and tokens ------------------------------------------------------------


def _repo(loaded: Loaded):
    from sd_ground.db.engine import make_engine, with_password
    from sd_ground.db.repo import Repo

    password = loaded.secret("SD_GROUND__DB_APP_PASSWORD")
    if not password:
        raise SystemExit("SD_GROUND__DB_APP_PASSWORD saknas i .env. Kör: just setup-config")
    return Repo(make_engine(with_password(loaded.config.db.url, password)))


def cmd_vehicle_add(args: argparse.Namespace) -> int:
    if not _VEHICLE.match(args.id):
        _say("Ogiltigt fordons-id (t.ex. sim-01).", err=True)
        return 1

    async def run() -> int:
        repo = _repo(_loaded())
        try:
            added = await repo.add_vehicle(args.id, args.name or args.id, args.kind, args.sysid)
        finally:
            await repo.close()
        if not added and not args.if_missing:
            _say(f"Fordonet {args.id} finns redan.", err=True)
            return 1
        _say(f"fordonet {args.id} {'lades till' if added else 'fanns redan'}")
        return 0

    return asyncio.run(run())


def cmd_token_create(args: argparse.Namespace) -> int:
    if not _TOKEN_NAME.match(args.name):
        _say("Ogiltigt namn (små bokstäver, siffror, - och _).", err=True)
        return 1
    try:
        expires = token_expiry(
            dt.datetime.now(dt.UTC), parse_duration(args.expires) if args.expires else None
        )
    except ValueError as exc:
        _say(str(exc), err=True)
        return 1
    token = new_token()
    now = dt.datetime.now(dt.UTC)
    rec = SessionRecord(
        id="",
        kind="token",
        role=args.role,
        issuer_ref=f"operator:{args.name}" if args.role == "pilot" else f"observer:{args.name}",
        auth_mode="cli",
        token_sha256=hash_token(token),
        created_at=now,
        last_used_at=now,
        expires_at=expires,
        name=args.name,
    )

    async def run() -> int:
        repo = _repo(_loaded())
        try:
            if await repo.token_exists(args.name):
                _say(f"En token med namnet {args.name} finns redan. Ta bort den först.", err=True)
                return 1
            await repo.save_session(rec)
        finally:
            await repo.close()
        return 0

    code = asyncio.run(run())
    if code == 0:
        if sys.stdout.isatty():
            _say(f"Token (visas bara nu, gäller till {expires:%Y-%m-%d %H:%M} UTC):")
        _say(token)
    return code


def cmd_token_revoke(args: argparse.Namespace) -> int:
    async def run() -> bool:
        repo = _repo(_loaded())
        try:
            return await repo.revoke_token(args.name)
        finally:
            await repo.close()

    if not asyncio.run(run()):
        _say(f"Ingen aktiv token heter {args.name}.", err=True)
        return 1
    _say(f"token {args.name} borttagen (slutar gälla inom 10 s)")
    return 0


def cmd_link_token_create(args: argparse.Namespace) -> int:
    if not _VEHICLE.match(args.vehicle):
        _say("Ogiltigt fordons-id (t.ex. sim-01).", err=True)
        return 1
    loaded = _loaded()
    hashes_file = loaded.path(loaded.config.sdlink.link_tokens_file)
    current: dict[str, str] = {}
    if hashes_file.is_file():
        current = {
            k: str(v) for k, v in tomllib.loads(hashes_file.read_text(encoding="utf-8")).items()
        }
    onboard = Path(args.write_onboard) if args.write_onboard else None
    if (
        args.if_missing
        and args.vehicle in current
        and onboard is not None
        and onboard.is_file()
        and hash_token(onboard.read_text(encoding="utf-8").strip()) == current[args.vehicle]
    ):
        _say("länktoken finns redan")
        return 0
    token = new_token()
    current[args.vehicle] = hash_token(token)
    _private_dir(hashes_file.parent)
    _write_private(hashes_file, "".join(f'"{k}" = "{v}"\n' for k, v in sorted(current.items())))
    if onboard is not None:
        _write_private(onboard, token + "\n")
        _say(f"länktoken skriven till {onboard}")
    else:
        _say(token)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sd-ground", description="Searchdrone markstation")
    sub = p.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init", help="skapa lokal config, hemligheter och datamapp")
    init.add_argument("--lan", action="store_true", help="skapa även LAN-certifikatet")
    init.set_defaults(func=cmd_init)
    cfg = sub.add_parser("config", help="kontrollera konfigurationen")
    cfg_sub = cfg.add_subparsers(dest="sub", required=True)
    cfg_sub.add_parser("check").set_defaults(func=cmd_config_check)
    db = sub.add_parser("db", help="databasen")
    db_sub = db.add_subparsers(dest="sub", required=True)
    db_sub.add_parser("setup").set_defaults(func=cmd_db_setup)
    srv = sub.add_parser("serve", help="starta markstationen")
    srv.add_argument(
        "--no-login-link",
        action="store_true",
        help="skriv ingen inloggningskod eller PIN vid start (när utskriften loggas)",
    )
    srv.set_defaults(func=cmd_serve)
    ll = sub.add_parser("login-link", help="ny inloggningslänk för datorn")
    ll.add_argument("--print-code", action="store_true", help="skriv bara koden")
    ll.set_defaults(func=cmd_login_link)
    sub.add_parser("pair", help="PIN och QR-kod för mobilen").set_defaults(func=cmd_pair)
    act = sub.add_parser("action", help="STOPP eller FLYG HEM via den körande servern")
    act.add_argument("action", choices=["stop", "home"])
    act.add_argument("--vehicle", required=True)
    act.set_defaults(func=cmd_action)
    veh = sub.add_parser("vehicle", help="fordon")
    veh_sub = veh.add_subparsers(dest="sub", required=True)
    add = veh_sub.add_parser("add")
    add.add_argument("id")
    add.add_argument("--kind", choices=["sim", "real"], default="sim")
    add.add_argument("--sysid", type=int, default=1, choices=range(1, 250), metavar="1-249")
    add.add_argument("--name")
    add.add_argument("--if-missing", action="store_true")
    add.set_defaults(func=cmd_vehicle_add)
    tok = sub.add_parser("token", help="CLI-tokens för skript och tester")
    tok_sub = tok.add_subparsers(dest="sub", required=True)
    create = tok_sub.add_parser("create")
    create.add_argument("--name", required=True)
    create.add_argument("--role", choices=["pilot", "observer"], required=True)
    create.add_argument("--expires", help="t.ex. 30m, 24h, 7d (förval 24h, högst 30d)")
    create.set_defaults(func=cmd_token_create)
    revoke = tok_sub.add_parser("revoke")
    revoke.add_argument("--name", required=True)
    revoke.set_defaults(func=cmd_token_revoke)
    link = sub.add_parser("link-token", help="länktoken för SD-link")
    link_sub = link.add_subparsers(dest="sub", required=True)
    lc = link_sub.add_parser("create")
    lc.add_argument("--vehicle", required=True)
    lc.add_argument("--write-onboard")
    lc.add_argument("--if-missing", action="store_true")
    lc.set_defaults(func=cmd_link_token_create)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
