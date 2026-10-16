"""Alembic environment for the ground station database (async, asyncpg).

The URL comes from ``config.attributes["url"]`` when a caller (``sd-ground db setup``,
the test fixture) runs Alembic from Python. On the command line it is built from
``SD_GROUND__DB_HOST``/``_PORT``/``_NAME``/``_PASSWORD`` with the defaults of
``config/ground.example.toml``. Migrations always run as the owner role ``sd_owner``.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from alembic import context
from geoalchemy2 import alembic_helpers
from sqlalchemy import URL, Connection, pool
from sqlalchemy.ext.asyncio import async_engine_from_config

config = context.config

# Created by the postgis image (postgis_topology, postgis_tiger_geocoder); never ours.
_FOREIGN_SCHEMAS = {"tiger", "tiger_data", "topology"}
_FOREIGN_TABLES = {"spatial_ref_sys"}


def _url_from_env() -> URL:
    password = os.environ.get("SD_GROUND__DB_PASSWORD", "")
    if not password:
        raise SystemExit("SD_GROUND__DB_PASSWORD saknas i miljön (lösenordet finns i .env).")
    return URL.create(
        "postgresql+asyncpg",
        username="sd_owner",
        password=password,
        host=os.environ.get("SD_GROUND__DB_HOST", "127.0.0.1"),
        port=int(os.environ.get("SD_GROUND__DB_PORT", "5433")),
        database=os.environ.get("SD_GROUND__DB_NAME", "searchdrone"),
    )


def _url() -> URL:
    url = config.attributes.get("url")
    return url if isinstance(url, URL) else _url_from_env()


# Kept for `alembic revision --autogenerate` once sd_ground has models (later tickets).
def include_object(
    obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any
) -> bool:
    if type_ == "table" and (name in _FOREIGN_TABLES or obj.schema in _FOREIGN_SCHEMAS):
        return False
    return alembic_helpers.include_object(obj, name, type_, reflected, compare_to)


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=None,  # 0001_initial is hand-written (no models yet)
        include_object=include_object,
        render_item=alembic_helpers.render_item,
        transaction_per_migration=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = async_engine_from_config(
        {"sqlalchemy.url": _url()}, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    try:
        async with engine.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    raise SystemExit("Offline-läge stöds inte; migreringarna körs mot databasen.")

asyncio.run(run_async_migrations())
