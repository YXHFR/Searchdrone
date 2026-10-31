"""sd-brain against the commander API's wire format (AC-B02, 05 §4.4).

Runs against a mock of ``/commander/v1`` until the mission-flow endpoints exist.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import typing
from typing import Any

import httpx
import pytest
from brain_helpers import REG, fixture
from mock_commander import TOKEN, MockCommander

from sd_brain.charter_draft import CharterDraft, to_proposal
from sd_brain.clock import utc_now
from sd_brain.evals.fake_ground import FakeGround
from sd_brain.ground_client import GroundClient, GroundError, SecondCommanderError
from sd_brain.loop import CommanderLoop
from sd_brain.memory import to_update
from sd_brain.turn import TurnDiagnostics
from sd_schemas.api import Backend
from sd_schemas.base import new_id
from sd_schemas.commander import BrainMessage, DecisionLog, SkillRequest
from sd_schemas.decide import ChatIntentReport

pytestmark = pytest.mark.contract


def _fg() -> FakeGround:
    fx = fixture("airborne_modeB.json")
    return FakeGround(fx["situation"], reg=REG, now=utc_now)


def _skill(skill: str = "hold", backend: Backend = "llm") -> SkillRequest:
    return SkillRequest(
        command_id=new_id(),
        skill=skill,
        params={},
        rationale_sv="x",
        in_reply_to=None,
        based_on=None,
        charter_version=1,
        backend=backend,
        ai_config_hash="h",  # type: ignore[arg-type]
    )


def test_commander_enums() -> None:
    """Every Literal value sd-brain can send is accepted by the shared wire models."""
    backends = typing.get_args(TurnDiagnostics.model_fields["backend"].annotation)
    for b in backends:
        SkillRequest.model_validate(_skill(backend=b).model_dump(mode="json"))  # pyright: ignore[reportArgumentType]
    errors = typing.get_args(typing.get_args(TurnDiagnostics.model_fields["error"].annotation)[0])
    for err in (*errors, None):
        DecisionLog(
            id=new_id(),
            turn_id=new_id(),
            trigger="operator_message",
            backend="rule",
            ai_config_hash="h",
            prompt_sha256="0" * 64,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0,
            rounds=0,
            repairs=0,
            tool_calls=[],
            command_id=None,
            gate_decision=None,
            error=err,
            prompt_json=None,
        )
    kinds = typing.get_args(BrainMessage.model_fields["kind"].annotation)
    assert set(kinds) == {"text", "question", "status", "proposal"}
    assert {"llm", "grammar", "scripted", "rule", "cloud"} == set(backends)


async def test_rest_roundtrip() -> None:
    fg = _fg()
    async with MockCommander(fg) as mock, httpx.AsyncClient() as http:
        g = GroundClient(mock.url, TOKEN, http)
        assert [s.name for s in await g.skills()] == [s.name for s in REG]
        mid = await g.active_mission("sim-01")
        assert mid is not None and mid == fg.mission_id
        s = await g.situation(mid)
        assert s.mission_status == "ACTIVE"
        await g.post_message(mid, BrainMessage(client_msg_id=new_id(), text="Hej."))
        assert (await g.chat(mid, 0, 10))[0].text == "Hej."
        res = await g.submit(mid, _skill())
        assert res.decision == "ALLOW"
        est = await g.estimate(mid, _skill())
        assert est.gate.decision == "ALLOW"
        draft = CharterDraft.model_validate(
            {
                "search_areas": [
                    {"kind": "rect", "center_pin_id": "P1", "width_m": 100, "height_m": 100}
                ],
                "reason_sv": "x",
            }
        )
        ok = await g.propose_charter(mid, to_proposal(draft, s))
        assert ok.ok and ok.approval_id is not None
        fg.responses = [
            {
                "match": {"skill": "propose_charter_change"},
                "result": {
                    "ok": False,
                    "errors": [
                        {
                            "code": "CHARTER_EXCEEDS_LIMITS",
                            "detail_sv": "För högt.",
                            "field": "alt_max_agl_m",
                            "limit": 120,
                        }
                    ],
                },
            }
        ]
        bad = await g.propose_charter(mid, to_proposal(draft, s))
        assert (
            not bad.ok
            and bad.errors[0].code == "CHARTER_EXCEEDS_LIMITS"
            and bad.errors[0].limit == 120
        )
        mem = await g.memory(mid)
        await g.put_memory(mid, to_update(mem.model_copy(update={"plan_note": "Plan"})))
        assert (await g.memory(mid)).plan_note == "Plan"
        report = ChatIntentReport(
            request_id=new_id(),
            message_id=new_id(),
            intent="stop",
            probability=0.9,
            engine="jev",
            location="cloud",
            latency_ms=12,
        )
        assert await g.report_chat_intent(mid, report) == "acted"
        for status in (403, 404, 409):
            mock.chat_intent_status = status
            assert await g.report_chat_intent(mid, report) == "refused"
        for status in (422, 503):  # 422: the ground could not read it (builds differ)
            mock.chat_intent_status = status
            with pytest.raises(GroundError):
                await g.report_chat_intent(mid, report)
        wrong = GroundClient(mock.url, "wrong-token-0123456789012345678901", http)
        with pytest.raises(GroundError) as exc:
            await wrong.skills()
        assert exc.value.status == 401


async def test_token_never_in_url() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=[])

    g = GroundClient(
        "http://127.0.0.1:7700/commander/v1",
        TOKEN,
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await g.skills()
    assert TOKEN not in str(seen[0].url) and seen[0].headers["authorization"] == f"Bearer {TOKEN}"
    assert g.ws_url() == "ws://127.0.0.1:7700/commander/v1/ws"


async def _consume(
    g: GroundClient,
    got: list[dict[str, Any]],
    resyncs: list[int],
    beat: Any = None,
    lost: list[int] | None = None,
) -> None:
    async def resync() -> None:
        resyncs.append(1)

    async for evt in g.events(
        lambda: {"version": "t"},
        beat or (lambda: {"status": "idle"}),
        on_resync=resync,
        on_lost=None if lost is None else lambda: lost.append(1),
        heartbeat_s=1.0,
    ):
        got.append(evt)


async def test_heartbeat_reconnect_and_resync() -> None:
    fg = _fg()
    async with MockCommander(fg) as mock, httpx.AsyncClient() as http:
        g = GroundClient(mock.url, TOKEN, http)
        got: list[dict[str, Any]] = []
        resyncs: list[int] = []
        lost: list[int] = []
        task = asyncio.create_task(_consume(g, got, resyncs, lost=lost))
        try:
            for _ in range(100):
                if mock.sockets:
                    break
                await asyncio.sleep(0.02)
            await asyncio.sleep(2.6)
            beats = mock.heartbeats
            assert len(beats) >= 3
            gaps = [b - a for a, b in itertools.pairwise(beats)]
            assert all(0.8 <= x <= 1.2 for x in gaps), gaps
            mock.push({"type": "event.skill", "body": {"skill": "goto", "status": "done"}})
            mock.push({"type": "event.skill", "body": {"skill": "hold", "status": "done"}})
            await asyncio.sleep(0.2)
            assert [e.get("seq") for e in got if e["type"] == "event.skill"] == [1, 2]
            # The stream drops; two events happen while it is down.
            await mock.drop()
            await asyncio.sleep(0.05)
            mock.push({"type": "event.skill", "body": {"skill": "x", "status": "done"}})
            mock.push({"type": "event.skill", "body": {"skill": "y", "status": "done"}})
            t0 = asyncio.get_running_loop().time()
            for _ in range(200):
                if (
                    len(mock.hellos) >= 2
                    and len([e for e in got if e["type"] == "event.skill"]) == 4
                ):
                    break
                await asyncio.sleep(0.02)
            assert asyncio.get_running_loop().time() - t0 < 2.0
            assert mock.hellos[1]["since_seq"] == 2 and mock.hellos[1]["server_epoch"] == "epoch-1"
            assert [e["seq"] for e in got if e["type"] == "event.skill"] == [1, 2, 3, 4]
            # A new server epoch: the client must fetch everything again.
            mock.epoch = "epoch-2"
            await mock.drop()
            for _ in range(200):
                if resyncs:
                    break
                await asyncio.sleep(0.02)
            assert resyncs == [1]
            assert lost == [1, 1]  # told each time the stream was down
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def test_second_commander_is_refused() -> None:
    fg = _fg()
    async with MockCommander(fg) as mock, httpx.AsyncClient() as http:
        first = asyncio.create_task(_consume(GroundClient(mock.url, TOKEN, http), [], []))
        for _ in range(100):
            if mock.sockets:
                break
            await asyncio.sleep(0.02)
        with pytest.raises(SecondCommanderError):
            await asyncio.wait_for(_consume(GroundClient(mock.url, TOKEN, http), [], []), 5)
        first.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await first


async def test_loop_over_http_queued_message_never_acts() -> None:
    """test_queued_messages_never_act over the wire, with the grammar backend as commander."""
    from sd_brain.backends import GrammarBackend
    from sd_brain.config import BrainConfig

    fg = _fg()
    async with MockCommander(fg) as mock, httpx.AsyncClient() as http:
        g = GroundClient(mock.url, TOKEN, http)
        loop = CommanderLoop(g, GrammarBackend(), GrammarBackend(), REG, BrainConfig(), "h")
        loop.mission_id = fg.mission_id
        evt = fg.operator_message("flyg till P1", delivery="queued")
        loop.handle_event(evt)
        await loop.settle()
        assert not fg.commands and loop.records[0].trigger == "recovered"
        live = fg.operator_message("flyg till P1")
        loop.handle_event(live)
        await loop.settle()
        assert [c.skill for c in fg.commands] == ["goto"]
        assert json.loads(fg.commands[0].model_dump_json())["params"]["alt_m"] == 40
        assert fg.decisions and all(d.ai_config_hash == "h" for d in fg.decisions)


async def test_stream_ignores_environment_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The commander token never goes to a proxy from the environment."""
    monkeypatch.setenv("http_proxy", "http://203.0.113.9:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://203.0.113.9:3128")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("NO_PROXY", "")
    fg = _fg()
    async with MockCommander(fg) as mock, httpx.AsyncClient() as http:
        task = asyncio.create_task(_consume(GroundClient(mock.url, TOKEN, http), [], []))
        try:
            for _ in range(150):
                if mock.hellos:
                    break
                await asyncio.sleep(0.02)
            assert mock.hellos, "the stream went somewhere else than loopback"
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
