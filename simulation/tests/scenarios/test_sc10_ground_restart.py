"""SC10a: the ground station restarts mid-search, on the demo's stack as processes.

30 s into the AI's search ``sd-ground`` is killed (SIGKILL, no goodbye) and started again
at once with the same settings. Its GCS heartbeat (250/190) is back before the flight
controller's GCS failsafe could act, so the drone never turns home or lands; the onboard
computer keeps flying the search it has (SD-link is down for less than its grace time);
the restarted ground station is healthy within 15 s and shows the same mission, charter,
chat and running order; the search is not sent again, it finishes, and the AI flies home.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_rig import (  # pyright: ignore[reportMissingImports]
    ALT_M,
    MIN_COVERAGE,
    SPACING_M,
    airborne,
    box_enu,
    demo_stack,
    pilot,
)

from sd_geo.coverage import coverage_fraction
from sd_sim.launch import Stack
from sd_sim.report import measure

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.brain]
__all__ = ["demo_stack"]

FS_GCS_TIMEOUT_S = 20.0  # the safety parameter file; this test never sets it
HEALTHY_AFTER_RESTART_S = 15.0
FAILSAFE_MODES = frozenset({"RTL", "SMART_RTL", "LAND"})


def gcs_heartbeats_after(stack: Stack, since_ns: int) -> list[int]:
    """Host times (ns) of the ground's HEARTBEATs to the drone in linksim's radio log."""
    out: list[int] = []
    index = stack.run_dir / "radio.idx.jsonl"
    for line in index.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if (
            row["dir"] == "to_vehicle"
            and row["sysid"] == 250
            and row["msgid"] == 0
            and row["mono_ns"] > since_ns
        ):
            out.append(int(row["mono_ns"]))
    return out


@pytest.mark.budget_s(720)
async def test_sc10a_ground_restart_mid_search(demo_stack: Stack) -> None:
    stack = demo_stack
    async with pilot(stack) as d:
        await airborne(d)
        survey = await d.wait_command("survey_sector", "running", 120)
        t_survey = time.monotonic()
        await asyncio.sleep(30.0)
        mission = await d.get(f"/api/v1/missions/{d.mid}")
        chat = await d.get(f"/api/v1/missions/{d.mid}/chat")
        before_chat = [m["id"] for m in (chat["items"] if isinstance(chat, dict) else chat)]

        killed_ns = time.monotonic_ns()
        stack.kill_service("sd-ground")
        await asyncio.to_thread(stack.restart_service, "sd-ground")
        restarted = time.monotonic()

        # 1: the GCS heartbeat is back in time (linksim writes its radio log once a second).
        beats: list[int] = []
        while not beats and (time.monotonic_ns() - killed_ns) / 1e9 < FS_GCS_TIMEOUT_S:
            await asyncio.sleep(0.5)
            beats = gcs_heartbeats_after(stack, killed_ns)
        assert beats, "ingen HEARTBEAT från 250 efter omstarten"
        back_s = (beats[0] - killed_ns) / 1e9
        measure("HEARTBEAT från markstationen efter omstarten", back_s, FS_GCS_TIMEOUT_S - 2.0)
        assert back_s <= FS_GCS_TIMEOUT_S - 2.0
        # 4 (health): healthy within 15 s of the restart.
        while True:
            h = await asyncio.to_thread(stack.health)
            if h is not None and h.ready and h.ok("db", "mavlink", "sdlink", "onboard"):
                break
            assert time.monotonic() - restarted < HEALTHY_AFTER_RESTART_S, (
                f"markstationen inte frisk {HEALTHY_AFTER_RESTART_S} s efter omstart: {h}"
            )
            await asyncio.sleep(0.2)

        # 4: the same mission, charter, chat and running order, with the same ids.
        await d.until(lambda: bool(d.tel), 10, "telemetri efter omstarten")
        again = await d.get(f"/api/v1/missions/{d.mid}")
        assert again["mission"]["id"] == mission["mission"]["id"]
        assert again["charter"]["id"] == mission["charter"]["id"]
        assert again["mission"]["status"] == "ACTIVE", again["mission"]
        chat = await d.get(f"/api/v1/missions/{d.mid}/chat")
        after_chat = [m["id"] for m in (chat["items"] if isinstance(chat, dict) else chat)]
        assert after_chat[: len(before_chat)] == before_chat
        running = [c for c in await d.commands() if c["status"] in ("accepted", "running")]
        assert [c["id"] for c in running] == [survey["id"]], running

        # 3 and 5: the search goes on, is not sent again, and finishes with its coverage.
        await d.wait_command("survey_sector", "done", 300)
        # 2: no GCS failsafe from the kill to the end of the search (the track has the
        # test rig's own channel while the ground station was away).
        modes = {s.mode for s in d.track if s.t >= killed_ns / 1e9}
        assert not modes & FAILSAFE_MODES, modes
        surveys = [c for c in await d.commands() if c["skill"] == "survey_sector"]
        assert [c["id"] for c in surveys] == [survey["id"]]
        flown = [(*d.plane.xy(s.lat, s.lon), s.alt_m) for s in d.track if s.t >= t_survey]
        assert coverage_fraction(box_enu(), flown, ALT_M, SPACING_M) >= MIN_COVERAGE
        await d.wait_command("return_home", "done", 300)
        await d.until(lambda: not d.tel["armed"], 60, "desarmerad efter landningen")
        assert d.from_home_m() <= 3.0
