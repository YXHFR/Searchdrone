"""The ground's HTTP servers keep an idle connection for ``GROUND_HTTP_KEEPALIVE_S``.

The brain's clients drop theirs after at most half of it (``sd_brain.ground_http``), so
they never send on a connection the server is closing. Set on the ground, not left to
uvicorn's default, so a uvicorn upgrade cannot shrink the margin unseen.
"""

from __future__ import annotations

from fastapi import FastAPI

from sd_ground.app import _uvicorn  # pyright: ignore[reportPrivateUsage]
from sd_schemas.commander import GROUND_HTTP_KEEPALIVE_S


def test_the_servers_keep_an_idle_connection_for_the_agreed_time() -> None:
    server = _uvicorn(FastAPI(), "127.0.0.1:7700")
    assert server.config.timeout_keep_alive == GROUND_HTTP_KEEPALIVE_S == 5
