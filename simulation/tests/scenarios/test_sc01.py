"""SC01, the demo, end to end: the whole stack as ``just demo`` starts it, as processes.

The launcher starts Postgres (a fresh ``searchdrone_i<N>``), ``sd-ground``, SITL with
builtin physics, linksim, ``sd-onboard``, ``sd-brain`` with the scripted commander and
SC01's script and, on instance 0, the test picture through ``video_relay`` into MediaMTX
(``just video-up``). The test is the pilot: a CLI token on the panel API over real HTTP
and the live WebSocket (which keeps the operator present), never anything in-process.

The steps are the demo's D1-D10: the request in the chat, the AI's charter card,
"Godkänn ram" (mode B), "Fritt runt drönaren", the AI's take-off to 40 m by its start
rule, the flight to P1, "Gå upp till 200 meter." refused by the gate, the lawnmower
search (>= 95 % of the box), the flight home, landing within 3 m and the automatic close
10 s after disarming. Afterwards the database (one mission, vehicle and charter across
every table, telemetry >= 2 Hz without gaps, a gate decision for every order) and the
mission folder (flight log, DataFlash log, events, mission export and the video).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_rig import (  # pyright: ignore[reportMissingImports]
    ALT_M,
    CLOSE_LIMIT_S,
    DEMO_LIMIT_S,
    FIXTURE,
    MIN_COVERAGE,
    SPACING_M,
    box_enu,
    check_database,
    check_folder,
    demo_stack,
    is_status,
    pilot,
    still_for,
)

from sd_geo.coverage import coverage_fraction
from sd_safety.targets import distance_m
from sd_sim.invariants import check_all
from sd_sim.launch import Stack
from sd_sim.services import VEHICLE_ID

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.brain, pytest.mark.e2e]
__all__ = ["demo_stack"]

D3 = (
    "Flyg till P1 och sök av en ruta på 100 gånger 100 meter där, på 40 meters höjd. "
    "Landa hemma sen."
)
D7 = "Gå upp till 200 meter."


# --- the demo ---------------------------------------------------------------------------


@pytest.mark.budget_s(720)
async def test_demo_flight(demo_stack: Stack) -> None:
    stack = demo_stack
    # D1: everything reports ready.
    health = stack.health()
    assert health is not None and health.ready, health
    async with pilot(stack) as d:
        # D2-D3: the mission, the request in the chat, the AI's charter card within 2 s.
        res = await d.post(f"/api/v1/vehicles/{VEHICLE_ID}/missions/open", {})
        assert res.status_code == 200, res.text
        d.mid = str(res.json()["id"])
        await d.say(D3)
        cards: list[dict[str, Any]] = []

        async def card() -> bool:
            cards[:] = [
                a
                for a in await d.get(f"/api/v1/missions/{d.mid}/approvals", status="pending")
                if a["kind"] == "charter"
            ]
            return bool(cards)

        took = await d.until(card, 10, "AI:ns ramkort")
        assert took <= 2.0 + 1.0, f"ramkortet kom efter {took:.1f} s"  # + one poll round
        # The card and the mission's new state are written a moment apart.
        await d.until(is_status(d, "CHARTER_PROPOSED"), 2, "uppdraget CHARTER_PROPOSED")

        # D4: "Godkänn ram" in mode B.
        res = await d.post(
            f"/api/v1/approvals/{cards[0]['id']}/decision",
            {
                "decision": "approve",
                "subject_version": cards[0]["subject_version"],
                "autonomy_mode": "WITHIN_CHARTER",
            },
        )
        assert res.status_code == 200, res.text
        await d.until(is_status(d, "READY"), 90, "uppdraget READY")
        # The onboard computer reads the parameters and the fence back after the charter
        # is installed (a few seconds); a start asked for before that is refused.
        await d.until(
            lambda: (d._live.get("snapshot") or {}).get("preflight_ok") is True,
            90,
            "ombordsdatorns förflygningskontroll",
        )

        # D5: "Fritt runt drönaren"; the AI takes off by its start rule.
        res = await d.post(f"/api/v1/missions/{d.mid}/clearance", {})
        assert res.status_code == 200, res.text
        takeoff = await d.wait_command("takeoff", "accepted", 30, or_later=("running", "done"))
        t_takeoff = time.monotonic()
        await d.wait_command("takeoff", "done", 60)
        assert time.monotonic() - t_takeoff <= 30.0 + 2.0
        assert abs(d.tel["alt_rel_home_m"] - ALT_M) <= 2.0
        assert takeoff["params"]["alt_m"] == ALT_M

        # D6: at P1 and standing still for 2 s.
        p1 = next(p for p in FIXTURE["pins"] if p["id"] == "P1")
        p1_ll = d.plane.latlon(float(p1["east_m"]), float(p1["north_m"]))
        await d.wait_command("goto", "done", 90)

        # D8 starts (after the AI's pause at P1); D7: "Gå upp till 200 meter." is refused.
        survey = await d.wait_command("survey_sector", "running", 30, or_later=("done",))
        t_survey = time.monotonic()
        arrived = [
            s
            for s in d.track
            if s.t < t_survey and distance_m(*p1_ll, s.lat, s.lon) <= 5.0 and s.groundspeed_ms < 0.5
        ]
        assert still_for(arrived, 2.0), "inte stilla inom 5 m från P1 i 2 s"
        await asyncio.sleep(4.0)  # past the gate's one-AI-order-per-3-s limit
        await d.say(D7)

        async def refused() -> dict[str, Any] | None:
            for c in await d.commands():
                if c["skill"] == "goto" and c["params"].get("alt_m") == 200:
                    return c
            return None

        await d.until(refused, 15, "AI:ns goto på 200 m")
        high = await refused()
        assert high is not None and high["status"] == "rejected_by_gate", high
        gates = await d.gate(high["id"])
        assert [g["decision"] for g in gates] == ["REJECT"], gates
        assert "ALT_ABOVE_CHARTER" in gates[0]["reason_codes"], gates
        assert gates[0]["message_sv"], gates

        await d.wait_command("survey_sector", "done", 300)
        t_done = time.monotonic()
        sweep = [s for s in d.track if t_survey <= s.t <= t_done and s.armed]
        off = [round(s.alt_m, 1) for s in sweep if abs(s.alt_m - ALT_M) > 2.0]
        assert not off, f"höjden under sökningen utanför 40 ± 2 m: {off[:20]}"
        box = box_enu()
        flown = [(*d.plane.xy(s.lat, s.lon), s.alt_m) for s in sweep]
        got = coverage_fraction(box, flown, ALT_M, SPACING_M)
        assert got >= MIN_COVERAGE, f"täckning {got:.3f}"
        survey_gates = await d.gate(survey["id"])
        assert [g["decision"] for g in survey_gates if g["side"] == "ground"] == ["ALLOW"]

        # D9: home, landed within 3 m, disarmed, closed by itself 10 s later.
        await d.wait_command("return_home", "done", 300)
        await d.until(lambda: not d.tel["armed"], 60, "desarmerad efter landningen")
        t_disarmed = time.monotonic()
        assert d.from_home_m() <= 3.0
        assert t_disarmed - t_takeoff <= DEMO_LIMIT_S
        await d.until(is_status(d, "CLOSED"), CLOSE_LIMIT_S, "uppdraget CLOSED av sig självt")

        # States, clicks and chat.
        events = list(await d.get(f"/api/v1/missions/{d.mid}/events", limit="1000"))
        states = [
            e["payload"].get("to")
            for e in events
            if e["type"] == "mission.state_changed" and e["payload"].get("to")
        ]
        tail = [s for s in states if s in ("RETURNING", "LANDED", "CLOSED")]
        assert tail[-3:] == ["RETURNING", "LANDED", "CLOSED"], states
        closed = next(e for e in events if e["payload"].get("to") == "CLOSED")
        assert closed["payload"].get("closed_reason", "auto") == "auto", closed
        clicks = [
            e
            for e in events
            if e["type"] in ("charter.approved", "clearance.confirmed")
            and str(e["payload"].get("by", "")).startswith("operator:")
        ]
        assert sorted(e["type"] for e in clicks) == ["charter.approved", "clearance.confirmed"]
        cmds = await d.commands()
        assert [c["skill"] for c in cmds if c["issuer_kind"] == "operator"] == []
        rejected = [c for c in cmds if c["status"] == "rejected_by_gate"]
        assert [c["id"] for c in rejected] == [high["id"]], "förloppets enda grindnekande"
        chat = await d.get(f"/api/v1/missions/{d.mid}/chat")
        lines = list(chat["items"] if isinstance(chat, dict) else chat)
        assert any(m["text"].startswith("Uppdraget är klart.") for m in lines), lines
        for i, m in enumerate(lines):
            if m["role"] == "operator":
                assert any(n["role"] in ("brain", "system") for n in lines[i + 1 :]), m

        # D10: the mission over the API, and the database and folder behind it.
        detail = await d.get(f"/api/v1/missions/{d.mid}")
        assert detail["mission"]["status"] == "CLOSED"
        assert detail["charter"] is not None
        for path in ("charters", "chat", "commands", "gate_decisions", "events", "track"):
            res = await d.panel.get(f"/api/v1/missions/{d.mid}/{path}")
            assert res.status_code == 200, f"{path}: {res.status_code}"
        charters = list(await d.get(f"/api/v1/missions/{d.mid}/charters"))
        all_gates = {c["id"]: await d.gate(c["id"]) for c in cmds}
        broken = check_all(d.track, charters, cmds, all_gates, events)
        assert not broken, broken

    approved = await check_database(stack, d.mid)
    assert approved["approved_by"].startswith("operator:")
    assert approved["autonomy_mode"] == "WITHIN_CHARTER"
    assert approved["approved_at"] is not None and approved["fence_verified_at"] is not None
    flight_s = t_disarmed - t_takeoff
    not_checked = check_folder(stack, d.mid, flight_s)
    stack.run_dir.joinpath("result.json").write_text(
        json.dumps(
            {
                "scenario": "SC01",
                "passed": True,
                "duration_s": round(flight_s),
                "coverage": round(got, 3),
                "not_built": ", ".join(not_checked) or "-",
                "finished_at": dt.datetime.now(dt.UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
