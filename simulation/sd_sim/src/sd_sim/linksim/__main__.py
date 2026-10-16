"""``python -m sd_sim.linksim --instance N --token-file F [--radio-tlog T] [--profile P]``.

Started by the launcher. The token is read from a file, never from the command line.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys
from pathlib import Path

from sd_sim import ports
from sd_sim.linksim.control import ControlApi
from sd_sim.linksim.core import LinkSim, load_profiles


async def run(args: argparse.Namespace) -> None:
    p = ports.for_instance(args.instance)
    profiles = load_profiles()
    token = args.token_file.read_text(encoding="utf-8").strip()
    sim = LinkSim(p, profiles[args.profile], seed=args.seed, radio_tlog=args.radio_tlog)
    api = ControlApi(sim, profiles, token, p["linksim_control"])
    await sim.start()
    await api.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    try:
        while not stop.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            sim.flush()
    finally:
        api.close()
        sim.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sd_sim.linksim")
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--radio-tlog", type=Path)
    parser.add_argument("--profile", default="perfect")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
