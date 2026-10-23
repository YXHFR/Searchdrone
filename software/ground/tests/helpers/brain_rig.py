"""The real ``sd-brain`` on a ``flow_rig`` ground station, for tests only.

``serve_commander`` serves the rig's commander API over real HTTP and WebSocket on a
loopback port (the rig must be built with ``http={"commander_bind": "127.0.0.1:<port>"}``
so the loopback guard accepts the host). ``run_brain`` starts ``sd-brain`` against it: its
own startup (registry check, active mission), event stream and turn loop, in the fixed
command mode (no language model in CI).
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator
from typing import Any, Literal

import uvicorn
from flow_rig import BRAIN, VID, Rig  # pyright: ignore[reportMissingImports]

from sd_brain.app import Brain, _startup, _stream, build
from sd_brain.config import BrainConfig
from sd_brain.decide.engines import EngineAnswer
from sd_brain.secrets import Secrets
from sd_ground.app import commander_asgi, create_commander_app
from sd_schemas.decide import ChoiceAnswer, DecideRequest, DecisionEngine
from sd_schemas.skills import SKILL_REGISTRY

__all__ = ["SaysStop", "loopback_listener", "run_brain", "serve_commander"]


def loopback_listener() -> socket.socket:
    """A listening socket on a free loopback port, held until uvicorn serves on it.

    Closing a probed port before the server binds it leaves a gap in which an outgoing
    connection (the rig's own link, database or MAVLink sockets) can get the same port
    from the ephemeral range, and the server then fails with "address already in use"."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    return sock


class SaysStop:
    """A decision engine that reads every line as STOP (stands in for Jev or OpenAI)."""

    name: DecisionEngine = "jev"
    location: Literal["cloud", "local"] = "cloud"
    configured: bool = True

    async def answer(self, req: DecideRequest, timeout_s: float) -> EngineAnswer:
        probs = {"none": 0.02, "stop": 0.95, "return_home": 0.03}
        return EngineAnswer(
            "jev", "cloud", {"intent": ChoiceAnswer(choice="stop", probabilities=probs)}, "m", 5
        )


@contextlib.asynccontextmanager
async def serve_commander(r: Rig, sock: socket.socket) -> AsyncIterator[None]:
    app = commander_asgi(r.g, create_commander_app(r.g))
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            lifespan="off",
            log_level="warning",
            timeout_graceful_shutdown=2,
        )
    )
    task = asyncio.ensure_future(server.serve(sockets=[sock]))
    try:
        await r.until(lambda: server.started, 10, "commander API up")
        yield
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)


@contextlib.asynccontextmanager
async def run_brain(r: Rig, port: int, **config: Any) -> AsyncIterator[Brain]:
    """``sd-brain`` on the served commander API, until it is ready on the stream."""
    cfg = BrainConfig.model_validate(
        {
            "backend": {"kind": "grammar"},
            "ground": {"commander_url": f"http://127.0.0.1:{port}/commander/v1", "vehicle_id": VID},
            "decide": {"engines": ["jev"]},
            **config,
        }
    )
    b = build(cfg, Secrets(BRAIN, "k" * 40), SKILL_REGISTRY)
    await _startup(b, SKILL_REGISTRY)  # the real registry check and active mission
    tasks = [asyncio.ensure_future(_stream(b)), asyncio.ensure_future(b.loop.run_forever())]
    try:
        await r.until(lambda: r.g.brain_state == "ready", 10, "sd-brain ready on the stream")
        yield b
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await b.ground.bye()
        await b.http.aclose()
