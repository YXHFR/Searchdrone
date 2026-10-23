"""``sd-search run``: the search service against the ground station's search API.

With ``SD_CLOUD_API_KEY`` set and ``[cloud] enabled``, the fast cloud model assesses the
candidates after the code; each call is booked against the ground station's cloud
budget with the search token. Without a key only the code assesses.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
from pathlib import Path

import httpx

from sd_brain.runtime.budget import HttpBudgetClient
from sd_brain.runtime.cloud import CloudModelRuntime
from sd_search.app import SearchService
from sd_search.client import GroundClient
from sd_search.config import SearchConfig, SearchConfigError, cloud_key, load
from sd_search.match.cascade import Cascade
from sd_search.match.engines import CloudMatchEngine
from sd_search.profile import FileProfileSource, GroundProfileSource, ProfileSource

__all__ = ["main"]

DEFAULT_URL = "http://127.0.0.1:7700"
DEFAULT_TOKEN_FILE = "secrets/search_token"  # noqa: S105 - a path, not a secret


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sd-search", description="Söktjänsten sd-search.")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="kör söktjänsten mot markstationen")
    run.add_argument(
        "--ground-url",
        default=os.environ.get("SD_SEARCH__GROUND_URL", DEFAULT_URL),
        help="markstationens loopback-lyssnare (befälhavarens)",
    )
    run.add_argument(
        "--token-file",
        default=os.environ.get("SD_SEARCH__TOKEN_FILE", DEFAULT_TOKEN_FILE),
        help="filen med söktjänstens token (just search-token)",
    )
    run.add_argument(
        "--profile-file",
        default=os.environ.get("SD_SEARCH__PROFILE_FILE") or None,
        help="en godkänd sökprofil från fil (bara simulatorn, SD_SIM=1)",
    )
    run.add_argument("--poll-s", type=float, default=0.25)
    run.add_argument("--root", default=".", help="repots rot (config/search.toml, .env)")
    return p


def build_cascade(
    cfg: SearchConfig, key: str | None, ground_url: str, token: str
) -> tuple[Cascade, list[httpx.AsyncClient | HttpBudgetClient]]:
    """The cascade with the fast cloud engine, or without one (no key, or turned off)."""
    if not cfg.cloud.enabled or key is None:
        logging.getLogger("sd_search").info("molnmatchningen är av; bara koden bedömer")
        return Cascade(None), []
    http = httpx.AsyncClient()
    budget = HttpBudgetClient(ground_url, token)
    rt = CloudModelRuntime(cfg.cloud.runtime_config(), key, http, budget=budget)
    engine = CloudMatchEngine(rt, timeout_s=cfg.cloud.timeout_s)
    cascade = Cascade(
        engine,
        max_concurrent=cfg.cloud.max_concurrent,
        queue_max=cfg.cloud.queue_max,
        retry_after_s=cfg.cloud.retry_after_s,
    )
    return cascade, [http, budget]


async def _run(args: argparse.Namespace, token: str, cfg: SearchConfig, key: str | None) -> int:
    client = GroundClient(args.ground_url, token)
    cascade, closers = build_cascade(cfg, key, args.ground_url, token)
    profiles: ProfileSource = (
        FileProfileSource(Path(args.profile_file))
        if args.profile_file
        else GroundProfileSource(client)
    )
    service = SearchService(client, profiles, cascade=cascade)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)
    try:
        await service.run(stop, poll_s=args.poll_s)
    finally:
        cascade.cancel()
        for c in closers:
            await c.aclose()
        await client.aclose()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    token = Path(args.token_file).read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit(f"{args.token_file} är tom. Kör: just search-token")
    root = Path(args.root)
    try:
        cfg = load(root)
    except SearchConfigError as exc:
        raise SystemExit("\n".join(exc.errors)) from None
    return asyncio.run(_run(args, token, cfg, cloud_key(root)))
