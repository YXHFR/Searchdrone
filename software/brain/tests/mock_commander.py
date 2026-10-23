"""A mock of the ground station's commander API (``/commander/v1``) for contract tests.

It speaks the real wire format over HTTP and WebSocket on 127.0.0.1, backed by
``FakeGround``. It stands in for the mission flow until the real endpoints exist.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from sd_brain.evals.fake_ground import FakeGround
from sd_schemas.charter import CharterProposal
from sd_schemas.commander import BrainMessage, DecisionLog, MissionMemoryUpdate, SkillRequest
from sd_schemas.decide import ChatIntentReport
from sd_schemas.search import SearchProfileProposal

TOKEN = "brain-token-for-contract-tests-0123456789"


class MockCommander:
    def __init__(self, fg: FakeGround, *, chat_intent_status: int = 202) -> None:
        self.fg = fg
        self.chat_intent_status = chat_intent_status
        self.epoch = "epoch-1"
        self.events: list[dict[str, Any]] = []
        self.hellos: list[dict[str, Any]] = []
        self.heartbeats: list[float] = []
        self.sockets: list[WebSocket] = []
        self.situation_calls = 0
        self.app = self._app()
        self.port = 0
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None

    def push(self, evt: dict[str, Any]) -> None:
        seq = len(self.events) + 1
        evt = {**evt, "seq": seq, "mission_id": self.fg.mission_id}
        self.events.append(evt)
        for ws in list(self.sockets):
            with contextlib.suppress(Exception):
                asyncio.get_running_loop().create_task(ws.send_json(evt))

    async def drop(self) -> None:
        for ws in list(self.sockets):
            with contextlib.suppress(Exception):
                await ws.close(1012)

    def _auth(self, request: Request) -> bool:
        return request.headers.get("authorization") == f"Bearer {TOKEN}"

    def _app(self) -> FastAPI:
        app = FastAPI()
        fg = self.fg
        base = "/commander/v1"

        @app.middleware("http")
        async def auth(request: Request, call_next: Any) -> Any:
            if not self._auth(request):
                return JSONResponse({"code": "UNAUTHORIZED"}, status_code=401)
            return await call_next(request)

        @app.get(base + "/skills")
        async def skills() -> Any:
            return [json.loads(s.model_dump_json()) for s in await fg.skills()]

        @app.get(base + "/missions/active")
        async def active(vehicle_id: str) -> Any:
            return {"mission_id": fg.mission_id}

        @app.get(base + "/missions/{mid}/situation")
        async def situation(mid: str) -> Any:
            self.situation_calls += 1
            return fg.sit

        @app.get(base + "/missions/{mid}/chat")
        async def chat(mid: str, after_seq: int = 0, limit: int = 50) -> Any:
            return [json.loads(m.model_dump_json()) for m in await fg.chat(mid, after_seq, limit)]

        @app.post(base + "/missions/{mid}/messages", status_code=201)
        async def messages(mid: str, request: Request) -> Any:
            msg = BrainMessage.model_validate(await request.json())
            return {"message_id": await fg.post_message(mid, msg)}

        @app.post(base + "/missions/{mid}/commands", status_code=201)
        async def commands(mid: str, request: Request) -> Any:
            req = SkillRequest.model_validate(await request.json())
            return json.loads((await fg.submit(mid, req)).model_dump_json())

        @app.post(base + "/missions/{mid}/estimate")
        async def estimate(mid: str, request: Request) -> Any:
            req = SkillRequest.model_validate(await request.json())
            return json.loads((await fg.estimate(mid, req)).model_dump_json())

        @app.post(base + "/missions/{mid}/profile-proposals")
        async def profile_proposals(mid: str, request: Request) -> Any:
            prop = SearchProfileProposal.model_validate(await request.json())
            pid = await fg.propose_search_profile(mid, prop)
            return JSONResponse({"proposal_id": pid}, status_code=201)

        @app.post(base + "/missions/{mid}/charter-proposals")
        async def proposals(mid: str, request: Request) -> Any:
            prop = CharterProposal.model_validate(await request.json())
            out = await fg.propose_charter(mid, prop)
            if out.ok:
                return JSONResponse(
                    {
                        "charter_id": str(out.proposal_id),
                        "approval_id": str(out.approval_id),
                        "validation": {"ok": True, "errors": []},
                    },
                    status_code=201,
                )
            errs = [e.model_dump(mode="json") for e in out.errors]
            return JSONResponse(
                {
                    "charter_id": None,
                    "approval_id": None,
                    "validation": {"ok": False, "errors": errs},
                },
                status_code=422,
            )

        @app.get(base + "/missions/{mid}/memory")
        async def memory(mid: str) -> Any:
            return json.loads((await fg.memory(mid)).model_dump_json())

        @app.put(base + "/missions/{mid}/memory", status_code=204)
        async def put_memory(mid: str, request: Request) -> Response:
            await fg.put_memory(mid, MissionMemoryUpdate.model_validate(await request.json()))
            return Response(status_code=204)

        @app.post(base + "/missions/{mid}/decisions", status_code=204)
        async def decisions(mid: str, request: Request) -> Response:
            await fg.post_decision(mid, DecisionLog.model_validate(await request.json()))
            return Response(status_code=204)

        @app.post(base + "/missions/{mid}/chat-intents", status_code=202)
        async def intents(mid: str, request: Request) -> Response:
            if self.chat_intent_status == 202:
                await fg.report_chat_intent(
                    mid, ChatIntentReport.model_validate(await request.json())
                )
            return Response(status_code=self.chat_intent_status)

        @app.websocket(base + "/ws")
        async def ws(websocket: WebSocket) -> None:
            await websocket.accept()
            if websocket.headers.get("authorization") != f"Bearer {TOKEN}":
                await websocket.close(4401)
                return
            if self.sockets:
                await websocket.close(4409)
                return
            self.sockets.append(websocket)
            try:
                while True:
                    msg = await websocket.receive_json()
                    kind = msg.get("type")
                    body = msg.get("body") or {}
                    if kind == "brain.hello":
                        self.hellos.append(body)
                        await websocket.send_json(
                            {
                                "type": "server.hello",
                                "seq": len(self.events),
                                "body": {
                                    "server_epoch": self.epoch,
                                    "seq": len(self.events),
                                    "profile": "sim",
                                },
                            }
                        )
                        since = body.get("since_seq")
                        if body.get("server_epoch") == self.epoch and isinstance(since, int):
                            for evt in self.events[since:]:
                                await websocket.send_json(evt)
                        elif body.get("server_epoch") is not None:
                            await websocket.send_json(
                                {
                                    "type": "resync_required",
                                    "seq": None,
                                    "body": {"server_epoch": self.epoch},
                                }
                            )
                    elif kind == "brain.heartbeat":
                        self.heartbeats.append(asyncio.get_running_loop().time())
            except WebSocketDisconnect:
                pass
            finally:
                self.sockets.remove(websocket)

        return app

    async def __aenter__(self) -> MockCommander:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        config = uvicorn.Config(
            self.app, host="127.0.0.1", port=self.port, log_level="warning", lifespan="off"
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        for _ in range(200):
            if self._server.started:
                break
            await asyncio.sleep(0.01)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        assert self._server is not None and self._task is not None
        self._server.should_exit = True
        await asyncio.wait_for(self._task, 5)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/commander/v1"
