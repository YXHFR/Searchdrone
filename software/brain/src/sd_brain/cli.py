"""``sd-brain run | eval | check-model | print-prompt | config-check``."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sd_brain.ai_config import ai_config_hash, model_sha256
from sd_brain.config import BrainConfig, ConfigError, load_config
from sd_brain.secrets import SecretError, brain_env, install_log_masking, load_secrets

__all__ = ["main"]

log = logging.getLogger("sd_brain")


def _out(text: str) -> None:
    sys.stdout.write(text + "\n")


def _err(text: str) -> None:
    sys.stderr.write(text + "\n")


def _config(args: argparse.Namespace) -> BrainConfig:
    return load_config(args.config, os.environ)


def _git_sha(root: Path) -> str | None:
    head = root / ".git" / "HEAD"
    if not head.is_file():
        return None
    ref = head.read_text(encoding="utf-8").strip()
    if ref.startswith("ref: "):
        target = root / ".git" / ref[5:]
        if target.is_file():
            return target.read_text(encoding="utf-8").strip()
        packed = root / ".git" / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(ref[5:]):
                    return line.split()[0]
        return None
    return ref


def _ram_total_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return round(int(line.split()[1]) / 1024 / 1024, 1)
    return None


def cmd_config_check(args: argparse.Namespace) -> int:
    cfg = _config(args)
    _out(f"Konfigurationen är giltig (backend {cfg.backend.kind}, profil {cfg.profile}).")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from sd_brain.app import StartError, run
    from sd_brain.backends.scripted import ScriptError
    from sd_brain.ground_client import GroundError, SecondCommanderError

    cfg = _config(args)
    secrets = load_secrets(cfg.token_file, os.environ)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    install_log_masking(secrets)
    try:
        asyncio.run(run(cfg, secrets))
    except ScriptError as exc:
        _err(str(exc))
        return 2
    except (StartError, SecondCommanderError, GroundError) as exc:
        _err(str(exc))
        return 1
    return 0


async def _eval(args: argparse.Namespace, paths: list[Path], cfg: BrainConfig) -> dict[str, Any]:
    import httpx

    from sd_brain.evals.runner import run_scenario, summarize
    from sd_brain.runtime.base import Sampling
    from sd_brain.runtime.local import LocalModelRuntime

    runs: list[list[Any]] = []
    client: httpx.AsyncClient | None = None
    factory: Callable[[dict[str, Any]], Any] | None = None
    if args.backend == "llm_local":
        secrets_key = brain_env().get("SD_LLM_API_KEY", "")
        client = httpx.AsyncClient(trust_env=False)
        local = LocalModelRuntime(cfg.llm, secrets_key, client)

        def make(_: dict[str, Any]) -> Any:
            return local

        factory = make

    seeds = [42, 43, 44][: max(1, args.repeat)]
    try:
        for seed in seeds:
            results = []
            for p in paths:
                r = await run_scenario(
                    p,
                    runtime_factory=factory,
                    golden_dir=Path(args.golden) if args.golden else None,
                    backend="replay" if args.backend == "replay" else "llm",
                    sampling=Sampling(temperature=0.2, top_p=0.8, max_tokens=512, seed=seed),
                )
                results.append(r)
                _out(f"{r.id}: {'ok' if r.ok else 'FEL'} {'; '.join(r.failures)[:300]}")
            runs.append(results)
    finally:
        if client is not None:
            await client.aclose()
    return summarize(runs)


def cmd_eval(args: argparse.Namespace) -> int:
    from sd_brain.evals.registry import foundation_registry
    from sd_brain.runtime.base import Sampling

    paths = sorted(Path(args.scenarios).glob("*.yaml"))
    if args.only:
        paths = [p for p in paths if any(o in p.name for o in args.only.split(","))]
    cfg = load_config(args.config if Path(args.config).is_file() else None, os.environ)
    report = asyncio.run(_eval(args, paths, cfg))
    if args.backend == "llm_local":
        h = ai_config_hash(foundation_registry(), Sampling())
        report.update(
            backend="llm_local",
            ai_config_hash=h,
            model_sha256=model_sha256(),
            gpu_name=args.gpu_name or None,
            ram_total_gb=_ram_total_gb(),
            git_sha=_git_sha(Path.cwd()),
        )
        out_dir = Path(args.report_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        path = out_dir / f"eval-grunden-{stamp}-{h[:8]}.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _out(f"Rapport: {path}")
    _out(json.dumps({k: v for k, v in report.items() if k != "failures"}, ensure_ascii=False))
    if args.backend == "replay":
        return 0 if not report["failures"] else 1
    return 0 if report["passed"] else 1


def cmd_check_model(args: argparse.Namespace) -> int:
    import httpx

    from sd_brain.checkmodel import check_model, write_report
    from sd_brain.runtime.base import ModelError
    from sd_brain.runtime.local import LocalModelRuntime
    from sd_brain.secrets import check_secret

    try:
        key = check_secret("SD_LLM_API_KEY", brain_env().get("SD_LLM_API_KEY"))
    except SecretError as exc:
        _err(str(exc))
        return 2
    cfg = load_config(args.config if Path(args.config).is_file() else None, os.environ)

    async def go() -> int:
        async with httpx.AsyncClient(trust_env=False) as client:
            rt = LocalModelRuntime(cfg.llm, key, client)
            code, report = await check_model(
                rt,
                duration_s=args.duration_s,
                interval_s=args.interval_s,
                vram_log=Path(args.vram_log) if args.vram_log else None,
                image=args.image,
            )
            write_report(Path(args.report_dir), report)
            _out(json.dumps(report))
            return code

    try:
        return asyncio.run(go())
    except ModelError as exc:
        _err(str(exc))
        return 1


def cmd_print_prompt(args: argparse.Namespace) -> int:
    import httpx

    from sd_brain.context import ContextBuilder
    from sd_brain.ground_client import GroundClient
    from sd_brain.tools import exposed_tools
    from sd_brain.turn import TurnInput
    from sd_schemas.base import new_id
    from sd_schemas.skills import SKILL_REGISTRY

    cfg = _config(args)
    secrets = load_secrets(cfg.token_file, os.environ)

    async def go() -> int:
        async with httpx.AsyncClient(trust_env=False) as client:
            g = GroundClient(cfg.ground.commander_url, secrets.brain_token, client)
            s = await g.situation(args.mission)
            chat = await g.chat(args.mission, 0, 200)
            mem = await g.memory(args.mission)
            now = dt.datetime.now(dt.UTC)
            ctx = ContextBuilder(
                SKILL_REGISTRY,
                takeoff_policy=cfg.behaviour.takeoff_policy,
                prompt_budget_tokens=cfg.llm.prompt_budget_tokens,
            )
            turn = TurnInput(
                turn_id=new_id(),
                mission_id=s.mission_id,
                trigger="operator_message",
                events=[],
                situation=s,
                chat_tail=chat[-12:],
                memory=mem,
                tools=exposed_tools(SKILL_REGISTRY, s, mem, now),
                now=now,
            )
            for m in ctx.build(turn):
                _out(f"--- {m['role']} ---\n{m['content']}")
            _out("--- tools ---\n" + ", ".join(t.name for t in turn.tools))
            return 0

    return asyncio.run(go())


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sd-brain", description="Searchdrone AI-befälhavare")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "config-check", "print-prompt"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default="config/brain.toml")
        if name == "print-prompt":
            sp.add_argument("--mission", required=True)
    ev = sub.add_parser("eval")
    ev.add_argument("--backend", choices=["replay", "llm_local"], default="replay")
    ev.add_argument("--scenarios", default="software/brain/src/sd_brain/evals/scenarios")
    ev.add_argument("--golden", default="software/brain/tests/golden")
    ev.add_argument("--repeat", type=int, default=1)
    ev.add_argument("--report-dir", default="data/reports")
    ev.add_argument("--config", default="config/brain.toml")
    ev.add_argument("--gpu-name", default="")
    ev.add_argument("--only", default="")
    cm = sub.add_parser("check-model")
    cm.add_argument("--duration-s", type=float, default=300)
    cm.add_argument("--interval-s", type=float, default=10)
    cm.add_argument("--report-dir", default="data/reports")
    cm.add_argument("--vram-log", default="")
    cm.add_argument("--image", default="")
    cm.add_argument("--config", default="config/brain.toml")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    handlers = {
        "run": cmd_run,
        "config-check": cmd_config_check,
        "eval": cmd_eval,
        "check-model": cmd_check_model,
        "print-prompt": cmd_print_prompt,
    }
    try:
        return handlers[args.cmd](args)
    except (ConfigError, SecretError) as exc:
        _err(str(exc))
        return 1
