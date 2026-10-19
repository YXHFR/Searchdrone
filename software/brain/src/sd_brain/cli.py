"""``sd-brain run | eval | scores | bank | perf-cloud | print-prompt | config-check``."""

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

from sd_brain.ai_config import ai_config_hash
from sd_brain.config import BrainConfig, ConfigError, load_config
from sd_brain.secrets import (
    SecretError,
    Secrets,
    brain_env,
    install_log_masking,
    load_secrets,
    optional_secret,
)

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
    if warning := cfg.zdr_warning():
        _err(warning)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from sd_brain.app import StartError, run
    from sd_brain.backends.scripted import ScriptError
    from sd_brain.ground_client import GroundError, SecondCommanderError

    cfg = _config(args)
    secrets = load_secrets(cfg.token_file, os.environ)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    install_log_masking(secrets)
    if warning := cfg.zdr_warning():
        log.warning(warning)
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
    from sd_brain.runtime.cloud import CloudModelRuntime

    runs: list[list[Any]] = []
    client: httpx.AsyncClient | None = None
    factory: Callable[[dict[str, Any]], Any] | None = None
    if args.backend == "cloud":
        key = _cloud_key()
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
        install_log_masking(Secrets(brain_token="", cloud_api_key=key))
        client = httpx.AsyncClient()
        cloud = CloudModelRuntime(cfg.cloud, key, client)

        def make(_: dict[str, Any]) -> Any:
            return cloud

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


def _cloud_key() -> str:
    key = optional_secret("SD_CLOUD_API_KEY", brain_env().get("SD_CLOUD_API_KEY"))
    if key is None:
        raise SecretError("SD_CLOUD_API_KEY saknas i .env; molnmodellen kan inte mätas")
    return key


def cmd_eval(args: argparse.Namespace) -> int:
    from sd_brain.runtime.base import Sampling
    from sd_schemas.skills import SKILL_REGISTRY

    if args.action == "grade":
        return cmd_eval_grade(args)
    if args.backend == "sim":
        return cmd_eval_sim(args)

    paths = sorted(Path(args.scenarios).glob("*.yaml"))
    if args.only:
        paths = [p for p in paths if any(o in p.name for o in args.only.split(","))]
    cfg = load_config(args.config if Path(args.config).is_file() else None, os.environ)
    report = asyncio.run(_eval(args, paths, cfg))
    if args.backend == "replay" and args.golden:
        from sd_brain.simeval.golden import load_fixture

        sims = [load_fixture(p) for p in sorted((Path(args.golden) / "sim").glob("SIM-*.json"))]
        report.update(asyncio.run(_replay_sim(sims, report["failures"])))
        report["passed"] = bool(report["passed"] and not report["sim_failed"])
    if args.backend == "cloud":
        h = ai_config_hash(SKILL_REGISTRY, Sampling(), cloud=cfg.cloud)
        report.update(
            backend="cloud",
            ai_config_hash=h,
            model=cfg.cloud.model,
            fallback_model=cfg.cloud.fallback_model,
            reasoning_effort=cfg.cloud.reasoning_effort,
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


async def _replay_sim(
    fixtures: list[dict[str, Any]], failures: dict[str, list[str]]
) -> dict[str, Any]:
    """Fly the simulator replay fixtures again (``tests/golden/sim``)."""
    from sd_brain.simeval.golden import replay_golden

    for fx in fixtures:
        bad = await replay_golden(fx)
        _out(f"{fx['id']}: {'ok' if not bad else 'FEL'} {'; '.join(bad)[:300]}")
        if bad:
            failures[str(fx["id"])] = bad
    sim_failed = sum(1 for fx in fixtures if str(fx["id"]) in failures)
    return {"sim_replays": len(fixtures), "sim_failed": sim_failed}


def cmd_eval_grade(args: argparse.Namespace) -> int:
    """Grade one simulator-farm round; writes ``grade.json`` in the round folder."""
    from sd_brain.simeval.farm import RoundError, grade_round

    if not args.round:
        _err("ange rundans katalog: sd-brain eval grade --round <katalog>")
        return 2
    try:
        grade = grade_round(Path(args.round))
    except RoundError as exc:
        _err(str(exc))
        return 1
    _out(json.dumps({k: v for k, v in grade.items() if k != "runs"}, ensure_ascii=False))
    return 0


async def _eval_sim(args: argparse.Namespace, cfg: BrainConfig) -> list[Any]:
    import httpx

    from sd_brain.runtime.base import Sampling
    from sd_brain.runtime.cloud import CloudModelRuntime
    from sd_brain.simeval.bank import bank_from_env, bank_name, example_record, worth_keeping
    from sd_brain.simeval.budget import CostLedger, MeteredRuntime
    from sd_brain.simeval.decider import make_decision_maker
    from sd_brain.simeval.runner import fly
    from sd_brain.simeval.scenario import make_scenario
    from sd_schemas.skills import SKILL_REGISTRY

    key = optional_secret("SD_CLOUD_API_KEY", brain_env().get("SD_CLOUD_API_KEY"))
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    install_log_masking(Secrets(brain_token="", cloud_api_key=key))
    ledger = CostLedger(Path(args.ledger), cap_sek=args.cap_sek)
    sampling = Sampling(temperature=0.2, top_p=0.8, max_tokens=512)
    h = ai_config_hash(SKILL_REGISTRY, sampling, cloud=cfg.cloud)
    bank = bank_from_env(os.environ, Path(args.bank_dir)) if args.bank_dir else None
    seed = args.seed if args.seed is not None else int(dt.datetime.now(dt.UTC).timestamp())
    scores_dir = Path(args.scores_dir) / ledger.day()
    results = []
    async with httpx.AsyncClient() as client:
        meter = None
        if key is not None and cfg.cloud.model:
            cloud = CloudModelRuntime(cfg.cloud, key, client)
            meter = MeteredRuntime(cloud, ledger, run_cap_sek=args.max_cost_sek)
        for i in range(args.flights):
            scenario = make_scenario(seed + i)
            decider = make_decision_maker(meter, cfg.behaviour, sampling=sampling)
            before = meter.cost_sek if meter is not None else 0.0
            res = await fly(
                scenario,
                decider,
                model=cfg.cloud.model,
                ai_config_hash=h,
                cost_before_sek=before,
                meter=meter,
                max_turns=args.max_turns,
            )
            sc = res.score
            scores_dir.mkdir(parents=True, exist_ok=True)
            path = scores_dir / f"{sc.flight_id}.json"
            path.write_text(sc.model_dump_json(indent=2) + "\n", encoding="utf-8")
            kept = ""
            if bank is not None and worth_keeping(sc):
                record = example_record(sc, res.records, {"seed": scenario.seed}, res.replies)
                kept = " sparad: " + bank.put(bank_name(sc), record)
            _out(
                f"flygning {i + 1}/{args.flights} seed {scenario.seed}: {sc.decider}"
                f"{'' if sc.exam else ' (inget prov: ' + str(res.switch_reason) + ')'} "
                f"betyg {sc.total:.0f} {'godkänd' if sc.passed else 'underkänd'}"
                f"{'' if sc.safety.ok else ' SÄKERHETSFEL'} "
                f"hittad {'ja' if sc.found else 'nej'} täckning {sc.coverage:.0%} "
                f"frågor {sc.pilot_questions} kostnad {_cost_text(sc)}{kept}"
            )
            results.append(sc)
    return results


def _cost_text(sc: Any) -> str:
    return f"{sc.cost_sek:.2f} kr" if sc.cost_measured else "ej uppmätt"


def cmd_eval_sim(args: argparse.Namespace) -> int:
    """Random search flights with an answer key; the cloud model decides under the cap."""
    from sd_brain.simeval.budget import CostLedger

    cfg = load_config(args.config if Path(args.config).is_file() else None, os.environ)
    if args.model:
        cfg.cloud = cfg.cloud.model_copy(update={"model": args.model})
    scores = asyncio.run(_eval_sim(args, cfg))
    exams = [s for s in scores if s.exam]
    n = len(scores) or 1
    ledger = CostLedger(Path(args.ledger), cap_sek=args.cap_sek)
    measured = all(s.cost_measured for s in scores) and bool(scores)
    cost = round(sum(s.cost_sek for s in scores), 4) if measured else None
    report = {
        "backend": "sim",
        "model": cfg.cloud.model,
        "flights": len(scores),
        "exams": len(exams),
        "mean_total": round(sum(s.total for s in scores) / n, 1),
        "pass_rate": round(sum(s.passed for s in scores) / n, 2),
        "found_rate": round(sum(s.found for s in scores) / n, 2),
        "safety_fails": sum(not s.safety.ok for s in scores),
        # Only what the provider priced; null ("ej uppmätt") when any call went unpriced.
        "cost_sek": cost,
        "cost_per_exam_sek": round(sum(s.cost_sek for s in exams) / len(exams), 4)
        if exams and measured
        else None,
        "cost_measured": measured,
        "fallback_reasons": sorted({str(s.fallback_reason) for s in scores if s.fallback_reason}),
        "spent_today_sek": round(ledger.spent_sek(), 4),
        "cap_sek": args.cap_sek,
        "git_sha": _git_sha(Path.cwd()),
    }
    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"eval-sim-{stamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _out(json.dumps(report, ensure_ascii=False))
    _out(f"Rapport: {path}")
    return 0 if report["safety_fails"] == 0 else 1


def cmd_scores(args: argparse.Namespace) -> int:
    from sd_brain.simeval.trend import load_scores, trend_rows, trend_text

    rows = trend_rows(load_scores(Path(args.scores_dir)))
    _out(json.dumps(rows, ensure_ascii=False, indent=2) if args.json else trend_text(rows))
    return 0


def cmd_bank(args: argparse.Namespace) -> int:
    """``bank export-golden``: a good exam to a replay fixture; ``bank failures``: the list."""
    import gzip

    from sd_brain.simeval.golden import GoldenError, export_golden
    from sd_brain.simeval.trend import failure_rows, failure_text, load_scores

    if args.bank_cmd == "failures":
        rows = failure_rows(load_scores(Path(args.scores_dir)))
        _out(json.dumps(rows, ensure_ascii=False, indent=2) if args.json else failure_text(rows))
        return 0
    src = Path(args.flight)
    try:
        raw = gzip.decompress(src.read_bytes()) if src.suffix == ".gz" else src.read_bytes()
        fixture = export_golden(json.loads(raw), args.name, {**os.environ, **brain_env()})
    except (OSError, ValueError, KeyError) as exc:
        _err(str(exc) if isinstance(exc, GoldenError) else f"kan inte läsa {src}: {exc!r}")
        return 1
    out = Path(args.out) / f"{args.name}.json"
    if out.exists() and not args.replace:
        _err(f"{out} finns redan (--replace skriver över)")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _out(f"Replay-fixtur: {out} (seed {fixture['seed']}, {len(fixture['replies'])} svar)")
    return 0


def cmd_perf_cloud(args: argparse.Namespace) -> int:
    """Time ``runs`` small chat calls to the cloud model; skip with exit 0 without a key."""
    import httpx

    from sd_brain.perf import measure_cloud, skip_line

    cfg = load_config(args.config if Path(args.config).is_file() else None, os.environ)
    key = optional_secret("SD_CLOUD_API_KEY", brain_env().get("SD_CLOUD_API_KEY"))
    if key is None or not cfg.cloud.model:
        _out(skip_line())
        return 0
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    install_log_masking(Secrets(brain_token="", cloud_api_key=key))

    async def go() -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            return await measure_cloud(cfg.cloud, key, client, runs=args.runs)

    report = asyncio.run(go())
    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"perf-cloud-{stamp}.json"
    report["git_sha"] = _git_sha(Path.cwd())
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _out(json.dumps(report, ensure_ascii=False))
    _out(f"Rapport: {path}")
    return 0 if report["ok"] else 1


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
    ev.add_argument("action", nargs="?", choices=["grade"], help="grade: a farm round")
    ev.add_argument("--round", default="", help="the round folder (eval grade)")
    ev.add_argument("--backend", choices=["replay", "cloud", "sim"], default="replay")
    ev.add_argument("--scenarios", default="software/brain/src/sd_brain/evals/scenarios")
    ev.add_argument("--golden", default="software/brain/tests/golden")
    ev.add_argument("--repeat", type=int, default=1)
    ev.add_argument("--report-dir", default="data/reports")
    ev.add_argument("--config", default="config/brain.toml")
    ev.add_argument("--only", default="")
    # --backend sim
    ev.add_argument("--flights", type=int, default=3)
    ev.add_argument("--seed", type=int, default=None)
    ev.add_argument("--model", default="")
    ev.add_argument("--max-turns", type=int, default=40)
    ev.add_argument("--max-cost-sek", type=float, default=None)
    ev.add_argument("--cap-sek", type=float, default=30.0)
    ev.add_argument("--ledger", default="data/brain-eval/cost-ledger.json")
    ev.add_argument("--scores-dir", default="data/brain-scores")
    ev.add_argument("--bank-dir", default="data/example-bank")
    sc = sub.add_parser("scores")
    sc.add_argument("--scores-dir", default="data/brain-scores")
    sc.add_argument("--json", action="store_true")
    bk = sub.add_parser("bank")
    bsub = bk.add_subparsers(dest="bank_cmd", required=True)
    ex = bsub.add_parser("export-golden")
    ex.add_argument("flight", help="a good-<id>.json.gz from the local example bank")
    ex.add_argument("--name", required=True, help="SIM-NN")
    ex.add_argument("--out", default="software/brain/tests/golden/sim")
    ex.add_argument("--replace", action="store_true")
    fl = bsub.add_parser("failures")
    fl.add_argument("--scores-dir", default="data/brain-scores")
    fl.add_argument("--json", action="store_true")
    pc = sub.add_parser("perf-cloud")
    pc.add_argument("--runs", type=int, default=10)
    pc.add_argument("--report-dir", default="data/reports")
    pc.add_argument("--config", default="config/brain.toml")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    handlers = {
        "run": cmd_run,
        "config-check": cmd_config_check,
        "eval": cmd_eval,
        "perf-cloud": cmd_perf_cloud,
        "scores": cmd_scores,
        "bank": cmd_bank,
        "print-prompt": cmd_print_prompt,
    }
    try:
        return handlers[args.cmd](args)
    except (ConfigError, SecretError) as exc:
        _err(str(exc))
        return 1
