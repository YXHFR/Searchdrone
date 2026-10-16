"""``sd-ground db setup``: create the role ``sd_app`` if missing and migrate to head.

Runs as the owner role ``sd_owner``. ``CREATE ROLE`` gets a SCRAM verifier, never the
plain password, so a logged statement holds nothing that can log in.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import secrets
from pathlib import Path

import asyncpg
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import URL, make_url

__all__ = [
    "ALEMBIC_INI",
    "alembic_config",
    "db_setup",
    "head_revision",
    "scram_verifier",
    "wait_for_db",
]

ALEMBIC_INI = Path(__file__).resolve().parents[3] / "alembic.ini"


def scram_verifier(password: str, iterations: int = 4096) -> str:
    """SCRAM-SHA-256 verifier as Postgres stores it (RFC 5802/7677)."""
    salt = secrets.token_bytes(16)
    salted = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    client_key = hmac.digest(salted, b"Client Key", "sha256")
    stored_key = hashlib.sha256(client_key).digest()
    server_key = hmac.digest(salted, b"Server Key", "sha256")

    def b64(raw: bytes) -> str:
        return base64.b64encode(raw).decode()

    return f"SCRAM-SHA-256${iterations}:{b64(salt)}${b64(stored_key)}:{b64(server_key)}"


def alembic_config(owner_url: URL) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.attributes["url"] = owner_url
    return cfg


def head_revision() -> str:
    head = ScriptDirectory.from_config(Config(str(ALEMBIC_INI))).get_current_head()
    if head is None:
        raise RuntimeError("no Alembic head revision")
    return head


def _asyncpg_args(url: URL) -> dict[str, object]:
    return {
        "host": url.host,
        "port": url.port,
        "user": url.username,
        "password": url.password,
        "database": url.database,
        "timeout": 5,
    }


async def wait_for_db(url: URL, timeout_s: float) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        try:
            conn = await asyncpg.connect(**_asyncpg_args(url))  # type: ignore[arg-type]
            await conn.close()
            return True
        except (OSError, asyncpg.PostgresError, TimeoutError):
            if loop.time() >= deadline:
                return False
            await asyncio.sleep(1.0)


async def _ensure_app_role(owner_url: URL, app_password: str) -> bool:
    conn = await asyncpg.connect(**_asyncpg_args(owner_url))  # type: ignore[arg-type]
    try:
        if await conn.fetchval("SELECT 1 FROM pg_roles WHERE rolname = 'sd_app'"):
            return False
        quoted = await conn.fetchval("SELECT quote_literal($1)", scram_verifier(app_password))
        await conn.execute(
            "CREATE ROLE sd_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS "
            f"NOREPLICATION PASSWORD {quoted}"
        )
        return True
    finally:
        await conn.close()


def db_setup(
    owner_url: str, owner_password: str, app_password: str, *, wait_s: float = 30.0
) -> bool:
    """Wait for the database, create ``sd_app`` if missing, migrate. True if the role was made."""
    url = make_url(owner_url).set(password=owner_password)
    if not asyncio.run(wait_for_db(url, wait_s)):
        raise SystemExit("Databasen svarar inte. Kör: just db-up")
    created = asyncio.run(_ensure_app_role(url, app_password))
    command.upgrade(alembic_config(url), "head")
    return created
