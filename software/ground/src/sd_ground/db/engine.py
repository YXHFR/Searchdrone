"""The database connection pool (role ``sd_app``)."""

from __future__ import annotations

from sqlalchemy import URL, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

__all__ = ["make_engine", "with_password"]


def with_password(url: str, password: str) -> URL:
    """The URL from config with the password from ``.env`` (never stored in config)."""
    return make_url(url).set(password=password)


def make_engine(url: URL) -> AsyncEngine:
    return create_async_engine(
        url,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
        connect_args={"command_timeout": 5, "timeout": 5},
    )
