"""Helpers for the STOP / FLY HOME scenarios: acknowledgements, modes, standing still,
landing. They read the scenario's own recording; nothing here writes to the vehicle.

``AckLog`` records every ``COMMAND_ACK`` the flight controller sends on a MAVLink
channel (ground 250/190, or the onboard channel through mavtap), with the host time.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from flow_rig import VID  # pyright: ignore[reportMissingImports]
from scenario_rig import Scenario  # pyright: ignore[reportMissingImports]

from sd_mav import CopterMode
from sd_safety.targets import distance_m
from sd_sim import expect

__all__ = ["Ack", "AckLog", "ack_log", "fallback_proven", "wait_mode_after"]

DO_SET_MODE = 176
ACCEPTED = 0


@dataclass(frozen=True)
class Ack:
    t: float
    command: int
    result: int


@dataclass
class AckLog:
    acks: list[Ack] = field(default_factory=list[Ack])

    def first(self, command: int, after: float, result: int | None = ACCEPTED) -> Ack | None:
        return next(
            (
                a
                for a in self.acks
                if a.command == command and a.t >= after and (result is None or a.result == result)
            ),
            None,
        )

    def refused(self, command: int, after: float, before: float) -> bool:
        return any(
            a.command == command and after <= a.t <= before and a.result != ACCEPTED
            for a in self.acks
        )


class _Recorder:
    """A waiter that never completes: its predicate sees every message from the FC."""

    __slots__ = ("future", "predicate")

    def __init__(self, log: AckLog) -> None:
        self.future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

        def predicate(msg: Any) -> bool:
            if msg.get_type() == "COMMAND_ACK":
                log.acks.append(Ack(time.monotonic(), int(msg.command), int(msg.result)))
            return False

        self.predicate: Callable[[Any], bool] = predicate


async def _keep(vehicle: Any, log: AckLog) -> None:
    rec = _Recorder(log)
    while True:
        waiters = vehicle._waiters  # pyright: ignore[reportPrivateUsage]
        if rec not in waiters:
            waiters.append(rec)
        await asyncio.sleep(0.2)


def ack_log(sc: Scenario, vehicle: Any) -> AckLog:
    """Record the flight controller's acknowledgements on ``vehicle``'s channel."""
    log = AckLog()
    sc._tasks.append(asyncio.ensure_future(_keep(vehicle, log)))  # pyright: ignore[reportPrivateUsage]
    return log


def fallback_proven(log: AckLog, after: float, before: float) -> bool:
    """A fallback mode counts only after a refused DO_SET_MODE (07's proof rule)."""
    return log.refused(DO_SET_MODE, after, before)


async def wait_mode_after(
    sc: Scenario, modes: Iterable[CopterMode], t0: float, limit_s: float
) -> tuple[float, int]:
    """Host seconds from ``t0`` until a HEARTBEAT shows one of ``modes``; and that mode."""
    wanted = {int(m) for m in modes}
    end = t0 + limit_s
    while time.monotonic() <= end + 0.5:
        for s in sc.tl.samples:
            if s.t >= t0 and s.mode in wanted:
                return s.t - t0, int(s.mode)
        await asyncio.sleep(0.02)
    raise AssertionError(
        f"inget av lägena {sorted(wanted)} inom {limit_s} s (läge {sc.tel.fc_mode})"
    )


async def stop_ack_and_mode(
    sc: Scenario,
    log: AckLog,
    t0: float,
    first: CopterMode,
    chain: tuple[CopterMode, ...],
    prefix: str,
) -> int:
    """L-STOP-ACK and L-STOP-HB for an operator's STOP or FLY HOME pressed at ``t0``;
    a later step of ``chain`` only with a refused DO_SET_MODE before it."""
    hb_s, mode = await wait_mode_after(sc, chain, t0, expect.L_STOP_HB_S)
    ack = log.first(DO_SET_MODE, t0)
    ack_s = None if ack is None else round(ack.t - t0, 3)
    sc.criterion(
        f"{prefix}.ack",
        ack_s is not None and ack_s <= expect.L_STOP_ACK_S,
        "COMMAND_ACK(DO_SET_MODE, ACCEPTED) (L-STOP-ACK)",
        measured=ack_s,
        limit=expect.L_STOP_ACK_S,
        unit="s",
        clock="host",
    )
    proven = mode == int(first) or fallback_proven(log, t0, t0 + hb_s)
    sc.criterion(
        f"{prefix}.hb",
        hb_s <= expect.L_STOP_HB_S and proven,
        f"HEARTBEAT visar {CopterMode(mode).name} (L-STOP-HB)",
        measured=round(hb_s, 3),
        limit=expect.L_STOP_HB_S,
        unit="s",
        clock="host",
        mode=CopterMode(mode).name,
    )
    return mode


async def still_and_hold(sc: Scenario, t0: float, prefix: str, hold: bool = True) -> None:
    """L-STOP-STILL from ``t0``, then (``hold``) L-STOP-HOLD."""
    # Still in every direction: a STOP during the climb also has to stop the climb.
    still_s = await sc.until(
        lambda: sc.tel.groundspeed_ms < expect.STILL_MS and abs(sc.tel.climb_ms) < expect.STILL_MS,
        expect.L_STOP_STILL_S + 1.0,
        "står still",
    )
    still_s = time.monotonic() - t0
    sc.criterion(
        f"{prefix}.still",
        still_s <= expect.L_STOP_STILL_S,
        "fart < 0,5 m/s (L-STOP-STILL)",
        measured=round(still_s, 2),
        limit=expect.L_STOP_STILL_S,
        unit="s",
    )
    if hold:
        h, v = await sc.drift(expect.L_STOP_HOLD_S)
        sc.criterion(
            f"{prefix}.hold",
            h <= expect.L_HOLD_H_M and v <= expect.L_HOLD_V_M,
            "står kvar (L-STOP-HOLD)",
            measured=[round(h, 2), round(v, 2)],
            limit=[expect.L_HOLD_H_M, expect.L_HOLD_V_M],
            unit="m",
        )


async def lands_home(sc: Scenario, t0: float, prefix: str) -> None:
    """L-LAND: landed and disarmed within L_LAND_S of ``t0``, within L_LAND_M of home."""

    def down() -> bool:
        t = sc.tel
        return not t.armed and t.landed_state == "on_ground"

    await sc.until(down, expect.L_LAND_S + 5.0, "landad och desarmerad")
    took = time.monotonic() - t0
    d = distance_m(sc.r.home[0], sc.r.home[1], sc.tel.lat, sc.tel.lon)
    sc.criterion(
        f"{prefix}.land",
        took <= expect.L_LAND_S and d <= expect.L_LAND_M,
        "landad nära hem (L-LAND)",
        measured=[round(took, 1), round(d, 2)],
        limit=[expect.L_LAND_S, expect.L_LAND_M],
    )


async def snapshot(sc: Scenario) -> dict[str, Any]:
    res = await sc.r.panel.get(f"/api/v1/vehicles/{VID}/state")
    assert res.status_code == 200, res.text
    return dict(res.json().get("snapshot") or {})


async def mission(sc: Scenario) -> dict[str, Any]:
    return dict((await sc.r.mission(sc.mission_id))["mission"])


async def system_status(sc: Scenario) -> dict[str, Any]:
    res = await sc.r.panel.get("/api/v1/system/status")
    assert res.status_code == 200, res.text
    return dict(res.json())


def moves_since(sc: Scenario, index: int, keep_mode: int | None = None) -> list[Any]:
    """Steering or mode changes from 1/191 or 250/190 since ``tl.sent[index]``; the
    ground re-sending ``keep_mode`` (STOP's repeat) does not count."""
    out = []
    for m in sc.steering_since(index, ("onboard", "ground")):
        if m.sender == "ground" and m.mode is not None and m.mode == keep_mode:
            continue
        out.append(m)
    return out
