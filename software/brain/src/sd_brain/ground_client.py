"""``GroundClient``: REST and WebSocket against the commander API on ``127.0.0.1:7700``.

The commander token goes only in the ``Authorization`` header, never in a URL. The
:class:`GroundApi` protocol is what the loop needs; ``FakeGround`` implements the same
protocol in memory for evaluations and tests.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

import httpx
import websockets
from pydantic import TypeAdapter, ValidationError

from sd_brain.charter_draft import CharterError, CharterProposalOutcome
from sd_schemas.api import ChatMessage
from sd_schemas.charter import CharterProposal
from sd_schemas.commander import (
    BrainMessage,
    DecisionLog,
    EstimateResult,
    MissionMemory,
    MissionMemoryUpdate,
    Situation,
    SkillRequest,
    SkillSubmitResult,
)
from sd_schemas.decide import ChatIntentReport
from sd_schemas.search import SearchProfileProposal
from sd_schemas.skills import SkillSpecPublic

__all__ = [
    "RECONNECT_DELAYS_S",
    "GroundApi",
    "GroundClient",
    "GroundError",
    "IntentOutcome",
    "SecondCommanderError",
    "StreamState",
]

log = logging.getLogger("sd_brain.ground")
RECONNECT_DELAYS_S = (1.0, 2.0, 5.0)
BYE_TIMEOUT_S = 2.0
_CHAT = TypeAdapter(list[ChatMessage])
_SKILLS = TypeAdapter(list[SkillSpecPublic])
IntentOutcome = Literal["acted", "refused"]


class GroundError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class SecondCommanderError(RuntimeError):
    """The server closed the stream with 4409: another commander is connected."""


class GroundApi(Protocol):
    async def skills(self) -> list[SkillSpecPublic]: ...
    async def active_mission(self, vehicle_id: str) -> str | None: ...
    async def situation(self, mission_id: str) -> Situation: ...
    async def chat(self, mission_id: str, after_seq: int, limit: int) -> list[ChatMessage]: ...
    async def post_message(self, mission_id: str, msg: BrainMessage) -> str: ...
    async def submit(self, mission_id: str, req: SkillRequest) -> SkillSubmitResult: ...
    async def estimate(self, mission_id: str, req: SkillRequest) -> EstimateResult: ...
    async def propose_charter(
        self, mission_id: str, proposal: CharterProposal
    ) -> CharterProposalOutcome: ...
    async def propose_search_profile(
        self, mission_id: str, proposal: SearchProfileProposal
    ) -> str: ...
    async def memory(self, mission_id: str) -> MissionMemory: ...
    async def put_memory(self, mission_id: str, mem: MissionMemoryUpdate) -> None: ...
    async def post_decision(self, mission_id: str, entry: DecisionLog) -> None: ...
    async def report_chat_intent(
        self, mission_id: str, report: ChatIntentReport
    ) -> IntentOutcome: ...


class StreamState:
    """Where the event stream stands, for resuming after a reconnect."""

    def __init__(self) -> None:
        self.server_epoch: str | None = None
        self.seq: int | None = None
        self.connected = False
        self.reconnects = 0


def _outcome(resp: httpx.Response) -> CharterProposalOutcome:
    data = resp.json()
    validation = data.get("validation") or {}
    errors = [CharterError.model_validate(e) for e in validation.get("errors") or []]
    if resp.status_code == 422 and not errors:
        for item in data.get("errors") or []:
            errors.append(
                CharterError(
                    code=str(item.get("code", "VALIDATION")),
                    detail_sv=str(item.get("detail_sv") or item.get("message_sv") or ""),
                    field=item.get("field"),
                    limit=item.get("limit"),
                )
            )
    ok = resp.status_code == 201 and bool(validation.get("ok", True)) and not errors
    return CharterProposalOutcome(
        ok=ok,
        proposal_id=data.get("charter_id") if ok else None,
        approval_id=data.get("approval_id") if ok else None,
        errors=errors,
    )


class GroundClient:
    def __init__(
        self, commander_url: str, token: str, client: httpx.AsyncClient | None = None
    ) -> None:
        self.base = commander_url.rstrip("/")
        self._token = token
        self._client = client or httpx.AsyncClient(timeout=5.0)
        self.stream = StreamState()
        self._ws: Any = None

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kw: Any) -> httpx.Response:
        try:
            resp = await self._client.request(method, self.base + path, headers=self._headers, **kw)
        except httpx.HTTPError as exc:
            raise GroundError(f"sd-ground svarar inte ({type(exc).__name__})") from None
        if resp.status_code == 401:
            raise GroundError("sd-ground godtar inte befälhavartokenet", 401)
        return resp

    async def _json(self, method: str, path: str, ok: tuple[int, ...], **kw: Any) -> Any:
        resp = await self._request(method, path, **kw)
        if resp.status_code not in ok:
            raise GroundError(f"{method} {path} gav HTTP {resp.status_code}", resp.status_code)
        return resp.json() if resp.content else None

    async def skills(self) -> list[SkillSpecPublic]:
        return _SKILLS.validate_python(await self._json("GET", "/skills", (200,)))

    async def active_mission(self, vehicle_id: str) -> str | None:
        resp = await self._request("GET", "/missions/active", params={"vehicle_id": vehicle_id})
        if resp.status_code == 204:
            return None
        if resp.status_code != 200:
            raise GroundError(f"GET /missions/active gav HTTP {resp.status_code}", resp.status_code)
        return str(resp.json()["mission_id"])

    async def situation(self, mission_id: str) -> Situation:
        data = await self._json("GET", f"/missions/{mission_id}/situation", (200,))
        return Situation.model_validate(data)

    async def chat(self, mission_id: str, after_seq: int, limit: int) -> list[ChatMessage]:
        data = await self._json(
            "GET",
            f"/missions/{mission_id}/chat",
            (200,),
            params={"after_seq": after_seq, "limit": limit},
        )
        return _CHAT.validate_python(data)

    async def post_message(self, mission_id: str, msg: BrainMessage) -> str:
        data = await self._json(
            "POST", f"/missions/{mission_id}/messages", (201,), json=msg.model_dump(mode="json")
        )
        return str(data["message_id"])

    async def submit(self, mission_id: str, req: SkillRequest) -> SkillSubmitResult:
        data = await self._json(
            "POST", f"/missions/{mission_id}/commands", (200, 201), json=req.model_dump(mode="json")
        )
        return SkillSubmitResult.model_validate(data)

    async def estimate(self, mission_id: str, req: SkillRequest) -> EstimateResult:
        data = await self._json(
            "POST", f"/missions/{mission_id}/estimate", (200,), json=req.model_dump(mode="json")
        )
        return EstimateResult.model_validate(data)

    async def propose_charter(
        self, mission_id: str, proposal: CharterProposal
    ) -> CharterProposalOutcome:
        resp = await self._request(
            "POST",
            f"/missions/{mission_id}/charter-proposals",
            json=proposal.model_dump(mode="json", exclude_none=True),
        )
        if resp.status_code not in (201, 422):
            raise GroundError(f"ramförslaget gav HTTP {resp.status_code}", resp.status_code)
        try:
            return _outcome(resp)
        except (ValueError, ValidationError, KeyError) as exc:
            raise GroundError(f"oväntat svar på ramförslaget: {type(exc).__name__}") from None

    async def propose_search_profile(self, mission_id: str, proposal: SearchProfileProposal) -> str:
        """Store the profile proposal for the pilot's card; returns its id."""
        data = await self._json(
            "POST",
            f"/missions/{mission_id}/profile-proposals",
            (201,),
            json=proposal.model_dump(mode="json"),
        )
        return str(data["proposal_id"])

    async def memory(self, mission_id: str) -> MissionMemory:
        data = await self._json("GET", f"/missions/{mission_id}/memory", (200,))
        return MissionMemory.model_validate(data)

    async def put_memory(self, mission_id: str, mem: MissionMemoryUpdate) -> None:
        await self._json(
            "PUT", f"/missions/{mission_id}/memory", (204,), json=mem.model_dump(mode="json")
        )

    async def post_decision(self, mission_id: str, entry: DecisionLog) -> None:
        await self._json(
            "POST",
            f"/missions/{mission_id}/decisions",
            (201, 204),
            json=entry.model_dump(mode="json"),
        )

    async def report_chat_intent(self, mission_id: str, report: ChatIntentReport) -> IntentOutcome:
        """``acted`` (202), or ``refused`` when the ground does not take it for that line
        (unknown, not a pilot's own, too old). Anything else is a ``GroundError``, also a
        422: the ground could not read the report (builds that do not match)."""
        resp = await self._request(
            "POST", f"/missions/{mission_id}/chat-intents", json=report.model_dump(mode="json")
        )
        if resp.status_code in (200, 202):
            return "acted"
        if resp.status_code in (403, 404, 409):
            return "refused"
        raise GroundError(f"chat-intents gav HTTP {resp.status_code}", resp.status_code)

    # --- WebSocket ------------------------------------------------------------------------

    def ws_url(self) -> str:
        parts = urlsplit(self.base)
        return f"ws://{parts.netloc}{parts.path}/ws"

    async def events(
        self,
        hello: Callable[[], dict[str, Any]],
        heartbeat: Callable[[], dict[str, Any]],
        *,
        on_resync: Callable[[], Awaitable[None]],
        on_connect: Callable[[bool], Awaitable[None]] | None = None,
        on_lost: Callable[[], None] | None = None,
        heartbeat_s: float = 1.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Events for ever; reconnects after 1, 2, 5, 5 … s with the last epoch and seq.
        ``on_lost`` runs each time the stream is down, before the wait to reconnect."""
        attempt = 0
        while True:
            try:
                async with websockets.connect(
                    self.ws_url(),
                    additional_headers=self._headers,
                    open_timeout=5,
                    proxy=None,  # loopback only: never send the token through a proxy
                ) as ws:
                    attempt = 0
                    async for evt in self._session(
                        ws, hello, heartbeat, on_resync, on_connect, heartbeat_s
                    ):
                        yield evt
            except websockets.ConnectionClosed as exc:
                code = exc.rcvd.code if exc.rcvd is not None else None
                if code == 4409:
                    raise SecondCommanderError("en annan befälhavare är redan ansluten") from None
                if code == 4401:
                    raise GroundError("sd-ground godtar inte befälhavartokenet", 401) from None
                log.warning("strömmen stängdes (%s); ansluter igen", code)
            except (OSError, websockets.InvalidHandshake, TimeoutError) as exc:
                log.warning("kan inte ansluta till sd-ground (%s)", type(exc).__name__)
            self.stream.connected = False
            if on_lost is not None:
                on_lost()
            delay = RECONNECT_DELAYS_S[min(attempt, len(RECONNECT_DELAYS_S) - 1)]
            attempt += 1
            await asyncio.sleep(delay)

    async def _session(
        self,
        ws: Any,
        hello: Callable[[], dict[str, Any]],
        heartbeat: Callable[[], dict[str, Any]],
        on_resync: Callable[[], Awaitable[None]],
        on_connect: Callable[[bool], Awaitable[None]] | None,
        heartbeat_s: float,
    ) -> AsyncIterator[dict[str, Any]]:
        self._ws = ws
        body = {**hello(), "server_epoch": self.stream.server_epoch, "since_seq": self.stream.seq}
        await ws.send(json.dumps({"type": "brain.hello", "body": body}))
        reconnect = self.stream.reconnects > 0 or self.stream.seq is not None
        self.stream.connected = True
        self.stream.reconnects += 1

        async def beat() -> None:
            while True:
                await ws.send(json.dumps({"type": "brain.heartbeat", "body": heartbeat()}))
                await asyncio.sleep(heartbeat_s)

        beater = asyncio.create_task(beat())
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = msg.get("type")
                body = msg.get("body") or {}
                if kind == "server.hello":
                    epoch = body.get("server_epoch")
                    if epoch != self.stream.server_epoch:
                        self.stream.seq = None
                    self.stream.server_epoch = epoch
                    if self.stream.seq is None:
                        self.stream.seq = body.get("seq")
                    if on_connect is not None:
                        await on_connect(reconnect)
                    yield {"type": "server.hello", "body": body}
                    continue
                if kind == "resync_required":
                    self.stream.server_epoch = body.get("server_epoch")
                    self.stream.seq = None
                    await on_resync()
                    continue
                if kind == "brain.ping":
                    await ws.send(json.dumps({"type": "brain.pong", "body": {}}))
                    continue
                seq = msg.get("seq")
                if isinstance(seq, int):
                    self.stream.seq = seq
                yield msg
        finally:
            beater.cancel()
            self._ws = None

    async def bye(self) -> None:
        """Say goodbye on the stream (SIGTERM). Best effort."""
        ws = self._ws
        if ws is not None:
            try:
                async with asyncio.timeout(BYE_TIMEOUT_S):  # never hold up a shutdown
                    await ws.send(json.dumps({"type": "brain.bye", "body": {}}))
                    await ws.close(1000)
            except (OSError, TimeoutError, websockets.ConnectionClosed):
                pass
