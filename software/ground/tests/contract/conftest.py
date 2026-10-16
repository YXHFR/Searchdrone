"""Database fixtures for contract tests (needs `just db-up`).

Tests never touch the database ``searchdrone``: each module gets its own
``searchdrone_test_<pid>_<n>`` database, created as ``sd_owner`` and dropped afterwards.
Passwords come from the environment or, as `just` never exports ``.env``, from the
repository's ``.env`` file. Values are never printed.
"""

from __future__ import annotations

import asyncio
import itertools
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncpg
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import URL

ROOT = Path(__file__).resolve().parents[4]
GROUND = ROOT / "software" / "ground"
HOST, PORT = "127.0.0.1", 5433
TEST_PREFIX = "searchdrone_test_"
_counter = itertools.count()


def env_value(key: str) -> str:
    value = os.environ.get(key, "")
    if value:
        return value
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, sep, rest = line.partition("=")
            if sep and name.strip() == key:
                return rest.strip().strip('"')
    pytest.fail(f"{key} saknas i miljön och i .env. Kör: just setup-config")


@dataclass(frozen=True)
class Db:
    name: str
    owner_password: str
    app_password: str

    def run(self, sql: str, *args: Any, role: str = "sd_owner") -> list[asyncpg.Record]:
        """Run one statement as ``role`` in its own connection and return the rows."""
        password = self.owner_password if role == "sd_owner" else self.app_password
        return asyncio.run(_fetch(self.name, role, password, sql, *args))

    def alembic(self, action: Callable[[Config, str], None], revision: str) -> None:
        cfg = Config(str(GROUND / "alembic.ini"))
        cfg.attributes["url"] = URL.create(
            "postgresql+asyncpg",
            username="sd_owner",
            password=self.owner_password,
            host=HOST,
            port=PORT,
            database=self.name,
        )
        action(cfg, revision)

    def upgrade(self, revision: str = "head") -> None:
        self.alembic(command.upgrade, revision)

    def downgrade(self, revision: str = "base") -> None:
        self.alembic(command.downgrade, revision)


async def _fetch(
    database: str, user: str, password: str, sql: str, *args: Any
) -> list[asyncpg.Record]:
    conn = await asyncpg.connect(
        host=HOST, port=PORT, user=user, password=password, database=database, timeout=5
    )
    try:
        return await conn.fetch(sql, *args)
    finally:
        await conn.close()


async def _admin(password: str, *statements: str) -> None:
    conn = await asyncpg.connect(
        host=HOST, port=PORT, user="sd_owner", password=password, database="postgres", timeout=5
    )
    try:
        for sql in statements:
            await conn.execute(sql)
    finally:
        await conn.close()


async def _ensure_app_role(owner_password: str, app_password: str) -> None:
    """Create sd_app like `sd-ground db setup` will: only if missing, never altered."""
    conn = await asyncpg.connect(
        host=HOST, port=PORT, user="sd_owner", password=owner_password, database="postgres"
    )
    try:
        if await conn.fetchval("SELECT 1 FROM pg_roles WHERE rolname = 'sd_app'"):
            return
        quoted = await conn.fetchval("SELECT quote_literal($1)", app_password)
        await conn.execute(
            "CREATE ROLE sd_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS "
            f"PASSWORD {quoted}"
        )
    finally:
        await conn.close()


def _quote_ident(name: str) -> str:
    if not name.startswith(TEST_PREFIX) or not name.replace("_", "").isalnum():
        raise ValueError(f"vägrar använda databasen {name!r} i tester")
    return f'"{name}"'


@pytest.fixture(scope="session")
def passwords() -> tuple[str, str]:
    owner = env_value("SD_GROUND__DB_PASSWORD")
    app = env_value("SD_GROUND__DB_APP_PASSWORD")
    try:
        asyncio.run(_ensure_app_role(owner, app))
    except OSError as exc:
        pytest.fail(f"Databasen svarar inte på {HOST}:{PORT} ({exc}). Kör: just db-up")
    return owner, app


@pytest.fixture
def fresh_db(passwords: tuple[str, str]) -> Iterator[Db]:
    """An empty database (PostGIS comes from the migration), dropped after the test."""
    owner, app = passwords
    name = f"{TEST_PREFIX}{os.getpid()}_{next(_counter)}"
    ident = _quote_ident(name)
    asyncio.run(
        _admin(owner, f"DROP DATABASE IF EXISTS {ident} WITH (FORCE)", f"CREATE DATABASE {ident}")
    )
    try:
        yield Db(name=name, owner_password=owner, app_password=app)
    finally:
        asyncio.run(_admin(owner, f"DROP DATABASE IF EXISTS {ident} WITH (FORCE)"))


@pytest.fixture
def migrated_db(fresh_db: Db) -> Db:
    fresh_db.upgrade()
    return fresh_db
