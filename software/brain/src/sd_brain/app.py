"""Putting ``sd-brain`` together: configuration → runtimes, backends, loop, stream."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import time
from dataclasses import dataclass
from typing import Any

import httpx

from sd_brain import __version__
from sd_brain.ai_config import ai_config_hash
from sd_brain.backends import GrammarBackend, LlmCommander, ScriptedBackend
from sd_brain.backends.base import CommanderBackend
from sd_brain.config import BrainConfig
from sd_brain.constants import BRAIN_HEARTBEAT_S, CLOUD_RETRY_AFTER_S
from sd_brain.context import ContextBuilder
from sd_brain.decide import (
    DecisionService,
    JevEngine,
    LocalDecisionEngine,
    OpenAiDecisionsEngine,
    QuestionEngine,
)
from sd_brain.ground_client import GroundClient, GroundError
from sd_brain.health import HealthState
from sd_brain.loop import CommanderLoop
from sd_brain.runtime.base import ModelError, Sampling
from sd_brain.runtime.cloud import CloudModelRuntime
from sd_brain.runtime.fallback import BestAvailableRuntime
from sd_brain.runtime.local import LocalModelRuntime
from sd_brain.secrets import Secrets
from sd_schemas.skills import SKILL_REGISTRY, SkillRegistry, SkillSpecPublic

__all__ = ["Brain", "StartError", "build", "registry_mismatch", "run"]

log = logging.getLogger("sd_brain")


class StartError(RuntimeError):
    """``sd-brain`` must not start (exit code 1). The message is Swedish."""


@dataclass
class Brain:
    cfg: BrainConfig
    ground: GroundClient
    loop: CommanderLoop
    local: LocalModelRuntime
    runtime: BestAvailableRuntime
    health: HealthState
    http: httpx.AsyncClient
    cloud_http: httpx.AsyncClient | None


def sampling_of(cfg: BrainConfig) -> Sampling:
    return Sampling(
        temperature=cfg.llm.temperature,
        top_p=cfg.llm.top_p,
        max_tokens=cfg.llm.max_tokens,
        seed=cfg.llm.seed,
    )


def _engines(
    cfg: BrainConfig, secrets: Secrets, http: httpx.AsyncClient, cloud: httpx.AsyncClient | None
) -> list[QuestionEngine]:
    out: list[QuestionEngine] = []
    for name in cfg.decide.engines:
        if name == "local":
            out.append(
                LocalDecisionEngine(cfg.llm.base_url, cfg.llm.model, secrets.llm_api_key, http)
            )
        elif name == "jev":
            c = cfg.decide.jev
            out.append(JevEngine(c.base_url, c.model, secrets.jev_api_key, cloud))
        else:
            c = cfg.decide.openai_decisions
            out.append(OpenAiDecisionsEngine(c.base_url, c.model, secrets.openai_api_key, cloud))
    return out


def build(cfg: BrainConfig, secrets: Secrets, reg: SkillRegistry = SKILL_REGISTRY) -> Brain:
    http = httpx.AsyncClient(timeout=5.0, trust_env=False)  # loopback only, never a proxy
    cloud_http = (
        httpx.AsyncClient(timeout=20.0)
        if (secrets.cloud_api_key or secrets.jev_api_key or secrets.openai_api_key)
        else None
    )
    ground = GroundClient(cfg.ground.commander_url, secrets.brain_token, http)
    local = LocalModelRuntime(cfg.llm, secrets.llm_api_key, http)
    cloud = CloudModelRuntime(cfg.cloud, secrets.cloud_api_key, cloud_http)
    runtime = BestAvailableRuntime(
        cloud, local, retry_after_s=cfg.cloud.retry_after_s or CLOUD_RETRY_AFTER_S
    )
    sampling = sampling_of(cfg)
    ai_hash = ai_config_hash(
        reg, sampling, cloud_model=cfg.cloud.model if cloud.configured else None
    )
    context = ContextBuilder(
        reg,
        takeoff_policy=cfg.behaviour.takeoff_policy,
        prompt_budget_tokens=cfg.llm.prompt_budget_tokens,
    )
    backend: CommanderBackend
    if cfg.backend.kind == "scripted":
        if not cfg.backend.scripted_file:
            raise StartError("backend.kind = scripted kräver backend.scripted_file (manuset)")
        backend = ScriptedBackend.from_file(cfg.backend.scripted_file)
    elif cfg.backend.kind == "grammar":
        backend = GrammarBackend()
    else:
        backend = LlmCommander(
            runtime,
            context,
            reg,
            cfg.behaviour,
            sampling,
            context_tokens=cfg.llm.context_tokens,
        )
    health = HealthState(time.monotonic, has_probe=cfg.backend.kind == "llm_local")
    loop_ref: list[CommanderLoop] = []

    async def on_answer(req: Any, outcomes: Any) -> None:
        if loop_ref:
            await loop_ref[0].log_decide(req, outcomes)

    # Late answers are only logged; keep them short so the local model's single slot
    # is free for the turn.
    decide = DecisionService(
        _engines(cfg, secrets, http, cloud_http), on_answer=on_answer, late_answer_s=1.0
    )
    loop = CommanderLoop(
        ground,
        backend,
        GrammarBackend(),
        reg,
        cfg,
        ai_hash,
        health=health,
        decide=decide,
        engine=runtime.engine if cfg.backend.kind == "llm_local" else None,
    )
    loop_ref.append(loop)
    return Brain(cfg, ground, loop, local, runtime, health, http, cloud_http)


def registry_mismatch(local: SkillRegistry, remote: list[SkillSpecPublic]) -> list[str]:
    """Names whose name, status, stage or parameter schema differ between the processes."""
    mine = {s.name: s for s in local.public()}
    theirs = {s.name: s for s in remote}
    out = sorted(set(mine) ^ set(theirs))
    for name in set(mine) & set(theirs):
        a, b = mine[name], theirs[name]
        if (a.status, a.stage, a.params_schema) != (b.status, b.stage, b.params_schema):
            out.append(name)
    return sorted(out)


async def _startup(b: Brain, reg: SkillRegistry) -> None:
    try:
        remote = await b.ground.skills()
    except GroundError as exc:
        raise StartError(f"sd-ground svarar inte: {exc}") from None
    if registry_mismatch(reg, remote):
        raise StartError("registret i sd-ground och sd-brain skiljer sig")
    if b.cfg.backend.kind == "llm_local":
        try:
            await b.local.check_key()
        except ModelError as exc:
            if "nyckel" in str(exc):
                raise StartError(str(exc)) from None
            log.warning("den lokala modellen svarar inte än (%s); kommandoläget gäller", exc)
            for _ in range(3):
                b.health.probe(False)
    mission = await b.ground.active_mission(b.cfg.ground.vehicle_id)
    await b.loop.set_mission(mission)


async def _health_probe(b: Brain) -> None:
    while True:
        await asyncio.sleep(b.cfg.llm.health_interval_s)
        b.health.probe(await b.local.health())
        if b.health.may_recover():
            ok = True
            if b.health.needs_test_call():
                try:
                    await b.local.chat(
                        [{"role": "user", "content": "ok"}],
                        [],
                        b.cfg.llm.call_timeout_s,
                        Sampling(max_tokens=1),
                    )
                except ModelError:
                    ok = False
            if ok:
                await b.loop.health_recovered()


async def _scripted_recovery(b: Brain) -> None:
    while True:
        await asyncio.sleep(1)
        if b.health.may_recover():
            await b.loop.health_recovered()


async def _idle_ticker(b: Brain) -> None:
    while True:
        await asyncio.sleep(1)
        b.loop.idle_tick()


async def _stream(b: Brain) -> None:
    async def resync() -> None:
        mission = await b.ground.active_mission(b.cfg.ground.vehicle_id)
        await b.loop.set_mission(mission)

    def hello() -> dict[str, Any]:
        return {
            "version": __version__,
            "backend": b.loop.heartbeat()["backend"],
            "model": b.runtime.model_id,
            "ai_config_hash": b.loop.ai_hash,
        }

    async for evt in b.ground.events(
        hello,
        b.loop.heartbeat,
        on_resync=resync,
        on_connect=b.loop.connected,
        heartbeat_s=BRAIN_HEARTBEAT_S,
    ):
        if evt.get("type") == "server.hello":
            profile = (evt.get("body") or {}).get("profile")
            if b.cfg.backend.kind == "scripted" and profile != "sim":
                raise StartError(
                    "manuset (scripted) får bara köras mot en sd-ground i profilen sim"
                )
            continue
        b.loop.handle_event(evt)


async def run(cfg: BrainConfig, secrets: Secrets, reg: SkillRegistry = SKILL_REGISTRY) -> None:
    b = build(cfg, secrets, reg)
    await _startup(b, reg)
    tasks = [
        asyncio.create_task(_stream(b)),
        asyncio.create_task(b.loop.run_forever()),
        asyncio.create_task(_idle_ticker(b)),
        asyncio.create_task(
            _health_probe(b) if cfg.backend.kind == "llm_local" else _scripted_recovery(b)
        ),
    ]
    stop = asyncio.Event()
    with contextlib.suppress(NotImplementedError):
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
        asyncio.get_running_loop().add_signal_handler(signal.SIGINT, stop.set)
    stopper = asyncio.create_task(stop.wait())
    try:
        done, _ = await asyncio.wait([*tasks, stopper], return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task is not stopper and task.exception() is not None:
                raise task.exception()  # type: ignore[misc]
    finally:
        for task in tasks:
            task.cancel()
        await b.ground.bye()
        await b.http.aclose()
        if b.cloud_http is not None:
            await b.cloud_http.aclose()
