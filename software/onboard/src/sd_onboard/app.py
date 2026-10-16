"""``sd-onboard``: start checks, the MAVLink channel, the executor and SD-link.

Exit code 2 (with the reason in Swedish and English) when a start check fails. One
asyncio loop: the executor ticks at 10 Hz, SD-link reconnects by itself, SIGINT and
SIGTERM stop everything cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys
from pathlib import Path
from typing import Any

from sd_mav.connection import Role, connect
from sd_mav.vehicle import SendOrigin, Vehicle
from sd_obs.clock import MonotonicClock
from sd_onboard.config import ConfigError, Loaded, check, load, read_token
from sd_onboard.executor.executor import Executor, ExecutorOptions
from sd_onboard.io.params import effective_params
from sd_onboard.io.persistence import StateStore
from sd_onboard.link_client import OnboardLink
from sd_onboard.logs import setup_logging

__all__ = ["main", "serve"]

log = logging.getLogger("sd_onboard")


async def serve(loaded: Loaded, stop: asyncio.Event | None = None) -> int:
    cfg = loaded.config
    clock = MonotonicClock()
    token = read_token(loaded)
    setup_logging(loaded.state_dir, secrets=(token,))
    stop = stop or asyncio.Event()
    executor: Executor | None = None

    def on_event(kind: str, details: dict[str, Any]) -> None:
        if executor is not None:
            executor.on_vehicle_event(kind, details)

    def heartbeat_allowed() -> bool:
        return executor is not None and executor.loop_alive()

    def gate(origin: SendOrigin) -> str | None:
        return "starting" if executor is None else executor.send_gate(origin)

    conn = connect(
        cfg.mav_url, role=Role.ONBOARD, source_system=cfg.fc_sysid, source_component=cfg.own_compid
    )
    vehicle = Vehicle(
        conn,
        role=Role.ONBOARD,
        clock=clock,
        fc_sysid=cfg.fc_sysid,
        on_event=on_event,
        heartbeat_allowed=heartbeat_allowed,
        send_gate=gate,
    )
    link = OnboardLink(url=cfg.sdlink_url, token=token, vehicle_id=cfg.vehicle_id, clock=clock)
    executor = Executor(
        vehicle,
        clock,
        link,
        ExecutorOptions(
            vehicle_id=cfg.vehicle_id,
            allow_companion_arm=cfg.allow_companion_arm,
            takeoff_delay_after_arm_s=cfg.test_hooks.takeoff_delay_after_arm_s,
            expected_params=effective_params(loaded.param_files()),
        ),
        StateStore(loaded.state_dir, clock),
    )
    link.executor = executor
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, stop.set)
    await vehicle.start()
    tasks = [
        asyncio.ensure_future(executor.run(stop)),
        asyncio.ensure_future(link.run(stop)),
    ]
    try:
        await stop.wait()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await executor.close()
        await vehicle.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    root = Path.cwd()
    try:
        loaded = load(root)
    except ConfigError as exc:
        print(exc, file=sys.stderr)  # noqa: T201
        return 2
    errors = check(loaded)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)  # noqa: T201
        return 2
    return asyncio.run(serve(loaded))
