"""HTTP to the ground's loopback APIs (the commander API and the cloud budget).

The ground's server (uvicorn) closes a connection idle for 5 s, and httpx keeps one as
long: a request could go out on a connection the server is closing that very moment and
fail with no answer (``RemoteProtocolError``), which lost the AI's turn. Kept well below
the server's, a pooled connection is never one the server is about to close.
"""

from __future__ import annotations

from typing import Any, Final

import httpx

__all__ = ["GROUND_KEEPALIVE_S", "ground_http", "ground_limits"]

GROUND_KEEPALIVE_S: Final = 2.0


def ground_limits(keepalive_s: float = GROUND_KEEPALIVE_S) -> httpx.Limits:
    return httpx.Limits(keepalive_expiry=keepalive_s)


def ground_http(
    timeout: float = 5.0, *, keepalive_s: float = GROUND_KEEPALIVE_S, **kw: Any
) -> httpx.AsyncClient:
    """An HTTP client that drops idle connections before the ground's server does."""
    return httpx.AsyncClient(timeout=timeout, limits=ground_limits(keepalive_s), **kw)
