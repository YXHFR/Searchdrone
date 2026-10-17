"""The safety gate against ArduCopter SITL: orders it must refuse, and take-off rules.

* SC04: a goto outside the charter is refused (the AI's path and the pilot's own order);
  nothing reaches the flight controller and the drone stays where it is.
* SC05: heights over the charter's ceiling are refused; the FC fence height is the
  safety file's value, above the hard limit by its margin only.
* SC09: no take-off without an approved charter and a fresh "clear around the drone"; a
  new charter version or an old confirmation does not count. The motors never arm.
* SC09 with the RC requirement: with ``require_rc`` the onboard computer refuses a take-off
  while the flight controller hears no RC transmitter (SITL's ``SIM_RC_FAIL``), and the
  gate says why; with the transmitter back, the take-off goes.
* SC12: in mode B a stub skill is refused at once; a charter change from the AI only
  becomes a proposal; the pilot's reject leaves the charter as it was.
* SC13: the malicious commander: every attempt in the list is refused with its code.
* SC18: the panel API's guards: no token, a wrong token, the commander token and an
  observer cannot change anything; nothing arms.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario_rig import (  # pyright: ignore[reportMissingImports]
    VID,
    H,
    Scenario,
    charter_body,
    fresh_db,
    migrated_db,
    passwords,
    pin,
    scenario,
    sim,
    sim_stack,
)

from sd_safety.constants import FENCE_ALT_MARGIN_M
from sd_safety.limits import ABSOLUTE_MAX_ALT_AGL_M, ABSOLUTE_MAX_SPEED_MS
from sd_schemas.base import new_id
from sd_sim.params import merge_layers
from sd_sim.sitl import SAFETY_PARAM_FILE

pytestmark = [pytest.mark.sitl, pytest.mark.slow]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]

PARAMS = Path(__file__).resolve().parents[2] / "sitl/params"
GATE_S = 2.0  # an order gets the gate's answer within this
DRIFT_H_M, DRIFT_V_M = 3.0, 1.0


def codes(res: dict[str, Any]) -> list[str]:
    return list(res.get("reason_codes") or res.get("gate", {}).get("reason_codes") or [])


async def refused(sc: Scenario, skill: str, params: dict[str, Any], *want: str) -> None:
    """The AI's order is refused at once with every code in ``want``."""
    t0 = time.monotonic()
    res = await sc.order(skill, params)
    assert time.monotonic() - t0 <= GATE_S
    assert res.get("decision") == "REJECT", res
    assert set(want) <= set(codes(res)), res


def drift(sc: Scenario, since: float) -> tuple[float, float]:
    from sd_safety.targets import distance_m

    track = [s for s in sc.track if s.t >= since]
    first = track[0]
    h = max(distance_m(first.lat, first.lon, s.lat, s.lon) for s in track)
    v = max(abs(s.alt_m - first.alt_m) for s in track)
    return h, v


@pytest.mark.budget_s(330)
async def test_sc04_sc05_refusals_in_the_air(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne()
        await sc.until(lambda: sc.tel.groundspeed_ms < 0.5, 15, "hovering after take-off")
        t0 = time.monotonic()
        out = dict(zip(("lat", "lon"), pin("OUT"), strict=True))
        # SC04: the AI's path, then the pilot's own order button.
        await refused(sc, "goto", {**out, "alt_m": 40}, "OUTSIDE_CHARTER")
        await asyncio.sleep(3.1)
        mine = await sc.operator_order("goto", {**out, "alt_m": 40})
        assert mine["gate"]["decision"] == "REJECT"
        assert "OUTSIDE_CHARTER" in mine["gate"]["reason_codes"]
        # SC05: straight up to 200 m, and the HIGH pin at 150 m.
        await asyncio.sleep(3.1)
        here = {"lat": sc.tel.lat, "lon": sc.tel.lon}
        await refused(sc, "goto", {**here, "alt_m": 200}, "ALT_ABOVE_LIMIT", "ALT_ABOVE_CHARTER")
        await asyncio.sleep(3.1)
        high = dict(zip(("lat", "lon"), pin("HIGH"), strict=True))
        await refused(sc, "goto", {**high, "alt_m": 150}, "ALT_ABOVE_CHARTER")
        await asyncio.sleep(15.0)
        h, v = drift(sc, t0)
        assert h <= DRIFT_H_M and v <= DRIFT_V_M, (h, v)
        assert sc.mode() in ("GUIDED", "LOITER", "BRAKE")
        assert all(abs(s.alt_m - 40.0) <= 2.0 for s in sc.track if s.t >= t0)

        # The FC fence height is the safety file's, the hard limit plus its margin.
        safety = merge_layers([PARAMS / SAFETY_PARAM_FILE])
        got = await sc.harness(sim.harness.read_param, "FENCE_ALT_MAX")
        assert got == safety["FENCE_ALT_MAX"]
        assert got <= ABSOLUTE_MAX_ALT_AGL_M + FENCE_ALT_MARGIN_M

        res = await sc.order("return_home", {"reason": "Testet klart."})
        assert res.get("decision") == "ALLOW", res
        await sc.until(lambda: not sc.tel.armed, 180, "landed")
        assert sc.from_home_m() <= 3.0
        await sc.check_invariants()


@pytest.mark.budget_s(330)
async def test_sc09_takeoff_needs_charter_and_clearance(
    sim: Any, migrated_db: Any, tmp_path: Path
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        # a: no approved charter.
        await sc.open()
        res = await sc.r.clearance(sc.mid)
        assert res.status_code == 409 and res.json()["code"] == "CHARTER_NOT_APPROVED", res.text
        await refused(sc, "takeoff", {"alt_m": 40}, "CHARTER_NOT_APPROVED")
        await sc.r.panel.post(f"/api/v1/missions/{sc.mid}/end", json={}, headers=H)

        # The base case: READY, but nobody has confirmed "clear around the drone".
        await sc.ready()
        await asyncio.sleep(3.1)
        await refused(sc, "takeoff", {"alt_m": 40}, "CLEARANCE_REQUIRED")

        # f: a confirmation, then a new charter version: the confirmation is gone.
        await sc.clearance()
        res = await sc.r.panel.post(
            f"/api/v1/missions/{sc.mid}/charters",
            json={**charter_body(), "base_version": 1, "alt_max_agl_m": 100},
            headers=H,
        )
        assert res.status_code == 201, res.text
        card = res.json()
        assert (await sc.r.approve(str(card["approval_id"]), card["version"])).status_code == 200

        async def installed_v2() -> bool:
            c = (await sc.r.mission(sc.mid))["charter"]
            return (
                c["version"] == card["version"]
                and c["install_status"] == "installed"
                and c["fence_verified_at"] is not None
            )

        await sc.until(installed_v2, 90, "version 2 installed with its fence")
        await sc.until(
            lambda: (snap := sc.r.flow.snapshot(VID)[0]) is not None and snap.preflight_ok,
            90,
            "preflight ok for version 2",
        )
        await asyncio.sleep(3.1)
        await refused(sc, "takeoff", {"alt_m": 40}, "CLEARANCE_REQUIRED")
        assert not any(s.armed for s in sc.track)

        # The confirmation for this version: armed and climbing within 10 s.
        await sc.clearance()
        await asyncio.sleep(3.1)
        res = await sc.order("takeoff", {"alt_m": 40})
        snap = sc.r.flow.snapshot(VID)[0]
        bad = None if snap is None else [k for k, v in snap.preflight.items() if not v]
        assert res.get("decision") == "ALLOW", (
            res.get("details"),
            bad,
            snap and snap.preflight_ok,
            sc.r.onboard.ex.preflight,
            sc.r.onboard.ex.state,
        )
        await sc.until(lambda: sc.tel.armed, 10, "armed after the confirmation")
        await sc.wait_command("takeoff", "done", 60)
        assert abs(sc.tel.alt_rel_home_m - 40.0) <= 2.0
        # The confirmation was used by that take-off.
        assert (await sc.r.mission(sc.mid))["clearance"] is None
        await asyncio.sleep(3.1)
        res = await sc.order("return_home", {})
        assert res.get("decision") == "ALLOW", res
        await sc.until(lambda: not sc.tel.armed, 120, "landed")
        await sc.check_invariants()


RC_LAYER = PARAMS / "tests/rc_required.parm"


@pytest.fixture
def sim_rc() -> Iterator[Any]:
    """SITL with the RC failsafe of real flight (FS_THR_ENABLE on): only then does the
    flight controller report a lost RC link (in SYS_STATUS), as a real one would."""
    with sim_stack("rc_required", (RC_LAYER,)) as s:
        yield s


@pytest.mark.budget_s(240)
async def test_sc09_takeoff_needs_the_rc_transmitter(
    sim_rc: Any, migrated_db: Any, tmp_path: Path
) -> None:
    """Real flight requires an RC transmitter: the onboard computer's PF12. SITL and CI
    turn the requirement off; here it is on, and SITL's RC input stops on purpose
    (``SIM_RC_FAIL``: no pulses)."""
    sim = sim_rc
    async with scenario(
        sim, migrated_db, tmp_path, require_rc=True, param_layers=(RC_LAYER,)
    ) as sc:
        await sc.ready()
        ex = sc.r.onboard.ex
        assert ex.preflight.get("PF12") is True  # SITL's RC input counts as a transmitter
        await sc.harness(sim.harness.set_sim_param, "SIM_RC_FAIL", 1)  # no pulses
        try:
            await sc.until(lambda: ex.preflight.get("PF12") is False, 10, "PF12 false")
            await sc.until(
                lambda: (
                    (s := sc.r.flow.snapshot(VID)[0]) is not None
                    and s.preflight.get("PF12") is False
                ),
                5,
                "the ground sees PF12",
            )
            await sc.clearance()
            res = await sc.order("takeoff", {"alt_m": 40})
            assert res.get("decision") == "REJECT", res
            assert codes(res)[0] == "RC_LINK_DOWN", res
            assert "RC-sändaren" in res["message_sv"]
            await asyncio.sleep(5.0)
            assert not any(s.armed for s in sc.track)
        finally:
            await sc.harness(sim.harness.set_sim_param, "SIM_RC_FAIL", 0)
        await sc.until(lambda: ex.preflight_ok(), 20, "preflight ok with the RC back")
        await sc.until(
            lambda: (s := sc.r.flow.snapshot(VID)[0]) is not None and s.preflight_ok,
            5,
            "the ground sees it",
        )
        await asyncio.sleep(3.1)
        res = await sc.order("takeoff", {"alt_m": 40})
        assert res.get("decision") == "ALLOW", res
        await sc.wait_command("takeoff", "done", 60)
        await asyncio.sleep(3.1)
        res = await sc.order("return_home", {})
        assert res.get("decision") == "ALLOW", res
        await sc.until(lambda: not sc.tel.armed, 120, "landed")
        await sc.check_invariants()


@pytest.mark.budget_s(180)
async def test_sc12_clicks_in_mode_b(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.ready()
        here = {"lat": sc.tel.lat, "lon": sc.tel.lon}
        # a: a stub skill is refused at once, before any card.
        res = await sc.order("land_at", here)
        assert res.get("decision") == "REJECT" and "SKILL_NOT_IMPLEMENTED" in codes(res), res
        assert not (
            await sc.r.panel.get(
                f"/api/v1/missions/{sc.mid}/approvals", params={"status": "pending"}
            )
        ).json()
        # c: the AI proposes a smaller charter: only a proposal, the active one stands.
        before = (await sc.r.mission(sc.mid))["charter"]
        res = await sc.r.commander.post(
            f"/commander/v1/missions/{sc.mid}/charter-proposals",
            json={
                **charter_body(),
                "base_version": before["version"],
                "area": sc.r.square(150),
                "reason_sv": "Mindre område.",
            },
        )
        assert res.status_code in (200, 201), res.text
        cards = (
            await sc.r.panel.get(
                f"/api/v1/missions/{sc.mid}/approvals", params={"status": "pending"}
            )
        ).json()
        assert len(cards) == 1 and cards[0]["kind"] == "charter"
        now = (await sc.r.mission(sc.mid))["charter"]
        assert now["version"] == before["version"] and now["status"] == "approved"
        # e: the pilot rejects it: nothing changes.
        res = await sc.r.panel.post(
            f"/api/v1/approvals/{cards[0]['id']}/decision",
            json={"decision": "reject", "subject_version": cards[0]["subject_version"]},
            headers=H,
        )
        assert res.status_code == 200, res.text
        charters = (await sc.r.panel.get(f"/api/v1/missions/{sc.mid}/charters")).json()
        assert sorted(c["status"] for c in charters) == ["approved", "rejected"]
        assert (await sc.r.mission(sc.mid))["charter"]["version"] == before["version"]
        assert not any(s.armed for s in sc.track)


@pytest.mark.budget_s(330)
async def test_sc13_malicious_commander(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne()
        await sc.until(lambda: sc.tel.groundspeed_ms < 0.5, 15, "hovering after take-off")
        t0 = time.monotonic()
        mode_before = (await sc.r.mission(sc.mid))["charter"]["body"]["autonomy_mode"]
        out = dict(zip(("lat", "lon"), pin("OUT"), strict=True))
        p1 = dict(zip(("lat", "lon"), pin("P1"), strict=True))
        here = {"lat": sc.tel.lat, "lon": sc.tel.lon}
        attempts: list[tuple[str, dict[str, Any], tuple[str, ...]]] = [
            ("goto", {**out, "alt_m": 40}, ("OUTSIDE_CHARTER",)),
            (
                "goto",
                {**here, "alt_m": ABSOLUTE_MAX_ALT_AGL_M + 1},
                ("ALT_ABOVE_LIMIT", "ALT_ABOVE_CHARTER"),
            ),
            (
                "goto",
                {**p1, "alt_m": 40, "speed_ms": ABSOLUTE_MAX_SPEED_MS + 5},
                ("SPEED_ABOVE_LIMIT",),  # refused, never clamped
            ),
            ("set_param", {"name": "FENCE_ENABLE", "value": 0}, ("SKILL_UNKNOWN",)),
            ("disable_fence", {}, ("SKILL_UNKNOWN",)),
            ("set_autonomy_mode", {"mode": "B"}, ("SKILL_UNKNOWN",)),
            ("confirm_candidate", {}, ("SKILL_UNKNOWN",)),
        ]
        for skill, params, want in attempts:
            res = await sc.order(skill, params)
            assert res.get("decision") == "REJECT", (skill, res)
            assert set(want) <= set(codes(res)), (skill, res)
            await asyncio.sleep(3.1)

        # Operator routes do not exist on the commander API, and refuse its token.
        approval = str(sc.mid)
        for path in (
            f"/commander/v1/approvals/{approval}/decision",
            f"/commander/v1/missions/{sc.mid}/clearance",
            f"/commander/v1/missions/{sc.mid}/autonomy",
            f"/commander/v1/missions/{sc.mid}/params",
            f"/commander/v1/missions/{sc.mid}/fence",
        ):
            res = await sc.r.commander.post(path, json={})
            assert res.status_code in (404, 405), (path, res.status_code)
        brain = {**H, "Authorization": sc.r.commander.headers["Authorization"]}
        for method, path in (
            ("POST", f"/api/v1/approvals/{approval}/decision"),
            ("PUT", f"/api/v1/missions/{sc.mid}/autonomy"),
            ("POST", f"/api/v1/missions/{sc.mid}/clearance"),
            ("POST", f"/api/v1/vehicles/{VID}/actions/stop"),
        ):
            res = await sc.r.panel.request(method, path, json={}, headers=brain)
            assert res.status_code in (401, 403), (path, res.status_code)

        # A charter looser than the mission's limits, and one that sets the autonomy mode.
        res = await sc.r.commander.post(
            f"/commander/v1/missions/{sc.mid}/charter-proposals",
            json={
                **charter_body(),
                "base_version": 1,
                "alt_max_agl_m": ABSOLUTE_MAX_ALT_AGL_M + 30,
            },
        )
        assert res.status_code == 422, res.text
        assert "CHARTER_EXCEEDS_LIMITS" in {e["code"] for e in res.json()["validation"]["errors"]}
        res = await sc.r.commander.post(
            f"/commander/v1/missions/{sc.mid}/charter-proposals",
            json={**charter_body(), "base_version": 1, "autonomy_mode": "APPROVE_EACH"},
        )
        assert res.status_code == 422, res.text
        assert "AUTONOMY_CHANGE_FORBIDDEN" in res.text
        await asyncio.sleep(15.0)
        h, v = drift(sc, t0)
        assert h <= DRIFT_H_M and v <= DRIFT_V_M, (h, v)

        # 10: after STOP the AI cannot resume; 11: after FLY HOME it cannot fly on.
        assert (await sc.press("stop")).status_code == 202
        await sc.until(lambda: sc.mode() == "BRAKE", 2.0, "BRAKE")
        await asyncio.sleep(3.1)
        await refused(sc, "resume", {}, "RESUME_NEEDS_OPERATOR")
        assert (await sc.press("return_home")).status_code == 202
        await sc.until(lambda: sc.mode() in ("SMART_RTL", "RTL"), 2.0, "flying home")
        await asyncio.sleep(3.1)
        await refused(sc, "goto", {**p1, "alt_m": 40}, "OPERATOR_OVERRIDE_ACTIVE")

        # Nothing changed in the flight controller or the mission.
        for name, want in (("FENCE_ENABLE", 1.0), ("MAV_GCS_SYSID", 250.0)):
            assert await sc.harness(sim.harness.read_param, name) == want
        assert (await sc.r.mission(sc.mid))["charter"]["body"]["autonomy_mode"] == mode_before
        await sc.until(lambda: not sc.tel.armed, 180, "landed")
        await sc.check_invariants()


@pytest.mark.budget_s(150)
async def test_sc18_api_guards(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.ready()
        commands_before = await sc.commands()
        action = {"client_action_id": str(new_id())}
        # Every changing route with a valid body, so a refusal is the role check, never
        # the body's validation.
        mutating: list[tuple[str, str, dict[str, Any]]] = [
            ("POST", f"/api/v1/missions/{sc.mid}/clearance", {}),
            ("PUT", f"/api/v1/missions/{sc.mid}/autonomy", {"mode": "APPROVE_EACH"}),
            (
                "PUT",
                f"/api/v1/missions/{sc.mid}/limits",
                {
                    "max_alt_agl_m": 60,
                    "max_speed_ms": 10,
                    "max_range_from_pilot_m": 300,
                    "battery_rtl_pct": 20,
                },
            ),
            (
                "POST",
                f"/api/v1/missions/{sc.mid}/commands",
                {"client_command_id": str(new_id()), "skill": "hold", "params": {}},
            ),
            ("POST", f"/api/v1/missions/{sc.mid}/end", {}),
            ("POST", f"/api/v1/vehicles/{VID}/actions/return_home", action),
            ("POST", f"/api/v1/vehicles/{VID}/actions/resume", action),
            ("POST", f"/api/v1/vehicles/{VID}/actions/takeover", action),
        ]
        anon = dict(H)
        for method, path, body in mutating:
            res = await sc.r.panel.request(
                method, path, json=body, headers={**anon, "Authorization": ""}
            )
            assert res.status_code == 401, (path, res.status_code)
            res = await sc.r.panel.request(
                method, path, json=body, headers={**anon, "Authorization": "Bearer wrong-token"}
            )
            assert res.status_code == 401, (path, res.status_code)
        res = await sc.r.panel.get("/api/v1/missions", headers={"Authorization": ""})
        assert res.status_code == 401
        brain = {**anon, "Authorization": sc.r.commander.headers["Authorization"]}
        observer = {**anon, "Authorization": f"Bearer {sc.observer_token}"}
        for method, path, body in mutating:
            for who in (brain, observer):
                res = await sc.r.panel.request(method, path, json=body, headers=who)
                assert res.status_code == 403, (path, res.status_code, res.text)
        res = await sc.r.panel.post(
            f"/api/v1/missions/{sc.mid}/clearance", json={}, headers={**H, "Host": "evil.example"}
        )
        assert res.status_code == 421, res.text
        res = await sc.r.commander.post(f"/api/v1/missions/{sc.mid}/clearance", json={})
        assert res.status_code == 404
        res = await sc.r.panel.get("/api/v1/brain/situation")
        assert res.status_code == 404
        assert await sc.commands() == commands_before
        await asyncio.sleep(2.0)
        assert not any(s.armed for s in sc.track)
