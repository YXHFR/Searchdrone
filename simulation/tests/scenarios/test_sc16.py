"""SC16: the flight controller's fence as the last guard.

The harness flies the vehicle past the gate on purpose (GUIDED with a velocity target
straight to the flight controller): a: 8 m/s east out of the charter's area; b: 3 m/s up
past the fence's height. The flight controller sees the breach and takes its fence action
(SmartRTL, else RTL, else LAND) within 2 s; it never gets far past the fence; the onboard
computer lets go and sends nothing that steers; the ground logs the flight controller's
text; the vehicle lands at home. INV1, INV3 and INV4 do not apply (the harness breaks them
on purpose); INV5 allows the harness's GUIDED flight.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from rig_faults import (  # pyright: ignore[reportMissingImports]
    GUIDED_VELOCITY,
    ground_vehicle_events,
    harness_guided_velocity,
)
from scenario_rig import Scenario, scenario  # pyright: ignore[reportMissingImports]

from sd_mav.failsafe_signals import PATTERNS
from sd_safety.constants import FENCE_POLYGON_MARGIN_M
from sd_safety.targets import Plane, polygon_from_geojson
from sd_schemas.executor import ExecutorState
from sd_sim import expect

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.script("SC16")]

FENCE_ACTION_S = 2.0  # breach -> the fence action's mode (sim)
OUTSIDE_FENCE_M = 20.0  # never further past the fence polygon (a)
ABOVE_FENCE_M = 5.0  # never higher above FENCE_ALT_MAX (b)
FLY_S = 60.0
EAST_MS = 8.0
UP_MS = 3.0
FENCE_TEXT = PATTERNS["fence_breach"]


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param("a", marks=pytest.mark.budget_s(240), id="a"),
        pytest.param("b", marks=pytest.mark.budget_s(280), id="b"),
    ],
)
async def test_sc16(sim: Any, migrated_db: Any, tmp_path: Path, variant: str) -> None:
    async with scenario(f"test_sc16[{variant}]", sim, migrated_db, tmp_path, "SC16") as sc:
        await _run(sc, variant)


async def _run(sc: Scenario, variant: str) -> None:
    fc_texts = ground_vehicle_events(sc)
    await sc.arrange_airborne()
    situation = await sc.situation()
    charter = situation["charter"]
    area = charter["area"]
    buffer_m = float(charter["geofence_buffer_m"])
    plane = Plane(sc.r.home[0], sc.r.home[1])
    poly = polygon_from_geojson(area, plane)
    fence_m = buffer_m + FENCE_POLYGON_MARGIN_M
    alt_max = expect.value("FENCE_ALT_MAX")
    chain = {int(m) for m in expect.fs_chain("FENCE_ACTION")}

    def past_fence(s: Any) -> float:
        """How far a sample is past the flight controller's fence (negative: inside)."""
        if variant == "b":
            return s.alt_m - alt_max
        xy = plane.xy(s.lat, s.lon)
        out = poly.boundary_distance(xy) if not poly.contains(xy) else -poly.boundary_distance(xy)
        return out - fence_m

    first = len(sc.tl.samples)
    sent_before = len(sc.tl.sent)
    if variant == "a":
        left = await harness_guided_velocity(sc, 0.0, EAST_MS, 0.0, FLY_S)
    else:
        left = await harness_guided_velocity(sc, 0.0, 0.0, -UP_MS, FLY_S)
    await sc.until(lambda: sc.mode() in chain, 5, "staketets åtgärd")
    samples = sc.tl.samples[first:]
    # The breach: the first sample past the fence, or the flight controller's own report
    # of it (it may hold the vehicle right at the fence height before it breaches).
    crossed = [s.t for s in samples if past_fence(s) > 0]
    reported = [t for t, _, text in fc_texts if t >= samples[0].t and FENCE_TEXT.search(text)]
    breach = min([*crossed[:1], *reported[:1]], default=None)
    # The mode can show before the breach text arrives: the delay may be negative.
    action = next((s for s in samples if s.mode in chain), None)
    sc.criterion(
        "SC16.1",
        breach is not None and action is not None and action.t - breach <= FENCE_ACTION_S,
        "staketbrott ger FENCE_ACTION:s läge inom 2 s",
        measured=None if breach is None or action is None else round(action.t - breach, 2),
        limit=FENCE_ACTION_S,
        mode=None if action is None else action.mode,
        left_guided_s=round(left - samples[0].t, 1) if samples else None,
        worst_m=round(max((past_fence(s) for s in samples), default=0.0), 1),
        modes=sorted({s.mode for s in samples if s.mode is not None}),
        top_m=round(max((s.alt_m for s in samples), default=0.0), 1),
        fc_texts=[(round(t - samples[0].t, 1), text) for t, _, text in fc_texts][-6:],
    )
    await sc.until(lambda: sc.mode() in chain, 5, "staketets åtgärd")
    await sc.until(
        lambda: (
            sc.onboard.ex.snapshot().executor_state in (ExecutorState.HELD, ExecutorState.FAILSAFE)
        ),
        3,
        "ombord släpper",
    )
    held = sc.onboard.ex.snapshot().executor_state
    started = time.monotonic()
    await sc.until(lambda: not sc.tel.armed, expect.L_LAND_S, "landad")
    worst = max(past_fence(s) for s in sc.tl.samples[first:])
    limit = OUTSIDE_FENCE_M if variant == "a" else ABOVE_FENCE_M
    sc.criterion(
        "SC16.2",
        worst <= limit,
        "aldrig långt förbi staketet" if variant == "a" else "aldrig över FENCE_ALT_MAX + 5 m",
        measured=round(worst, 1),
        limit=limit,
        unit="m",
    )
    steering = [m for m in sc.steering_since(sent_before, ("onboard",)) if m.t >= (breach or 0)]
    logged = [
        p
        for _, kind, p in sc._bus  # pyright: ignore[reportPrivateUsage]
        if kind == "fc.statustext" and FENCE_TEXT.search(str(p.get("text", "")))
    ]
    seen = [x for x in fc_texts if FENCE_TEXT.search(str(x[2]))]
    sc.criterion(
        "SC16.3",
        not steering,
        "ombord släpper och styr inget",
        executor=str(held),
        steering=len(steering),
    )
    sc.criterion(
        "SC16.3b",
        bool(seen) and bool(logged),
        "sd-ground loggar staketbrottet som fc.statustext",
        texts=[x[2] for x in seen],
        logged=len(logged),
    )
    sc.criterion(
        "SC16.4",
        sc.from_home_m() <= expect.L_LAND_M,
        "landar hemma (L-LAND)",
        measured=round(sc.from_home_m(), 1),
        limit=expect.L_LAND_M,
        land_s=round(time.monotonic() - started, 1),
    )
    await sc.finish(("INV1", "INV3", "INV4"), harness_extra=GUIDED_VELOCITY)
