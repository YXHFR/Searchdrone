"""``ThreadSafeMav`` hands the mission's ``flight.tlog`` to the vehicle channel, and is
harmless with a vehicle that cannot record (the test doubles)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sd_ground.mav.thread import ThreadSafeMav


def _mav(vehicle: Any) -> ThreadSafeMav:
    loop = asyncio.new_event_loop()
    loop.close()  # the tlog calls never touch the loop
    return ThreadSafeMav(SimpleNamespace(vehicle=vehicle), loop)  # pyright: ignore[reportArgumentType]


def test_start_and_stop_reach_the_vehicle(tmp_path: Path) -> None:
    calls: list[Any] = []
    vehicle = SimpleNamespace(
        start_tlog=lambda path: calls.append(("start", path)) or True,
        stop_tlog=lambda: calls.append(("stop", None)),
    )
    mav = _mav(vehicle)
    assert mav.start_tlog(tmp_path / "flight.tlog") is True
    mav.stop_tlog()
    assert calls == [("start", tmp_path / "flight.tlog"), ("stop", None)]


def test_a_vehicle_without_a_log_is_skipped(tmp_path: Path) -> None:
    mav = _mav(SimpleNamespace())
    assert mav.start_tlog(tmp_path / "flight.tlog") is False
    mav.stop_tlog()  # nothing to stop, no error
