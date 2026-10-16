"""MAVLink tap between SITL serial1 and sd-onboard: ``python -m sd_sim.mavtap ...``.

SITL sends to ``mavtap_in``; the tap forwards to ``onboard_mavlink`` and sends the
replies back. It records every frame in both directions and never delays, drops or
changes anything, so it can never cause a fault itself. Breaking the onboard channel
is done by stopping the onboard process, not here.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys
from pathlib import Path

from sd_sim import ports
from sd_sim.tlog import TlogWriter
from sd_sim.udprelay import UdpRelay


async def run(instance: int, tlog: Path) -> None:
    p = ports.for_instance(instance)
    relay = UdpRelay(
        name="mavtap",
        listen_port=p["mavtap_in"],
        target_port=p["onboard_mavlink"],
        tap=TlogWriter(tlog),
    )
    await relay.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    try:
        while not stop.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            if relay.tap is not None:
                relay.tap.flush()
    finally:
        relay.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sd_sim.mavtap")
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--tlog", type=Path, required=True)
    args = parser.parse_args(argv)
    asyncio.run(run(args.instance, args.tlog))
    return 0


if __name__ == "__main__":
    sys.exit(main())
