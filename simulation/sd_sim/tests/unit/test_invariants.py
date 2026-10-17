"""sd_sim.invariants: each invariant just inside (no finding) and just outside (a finding)."""

from __future__ import annotations

from typing import Any

from sd_safety.targets import Plane
from sd_sim.invariants import Sample, check_all, check_orders, check_overrides, check_track

HOME = (-35.363261, 149.165230)
PLANE = Plane(*HOME)


def at(east: float, north: float) -> tuple[float, float]:
    return PLANE.latlon(east, north)


def square(half: float) -> dict[str, Any]:
    ring = [list(reversed(at(x, y))) for x, y in [(-half, -half), (half, -half), (half, half)]]
    ring += [list(reversed(at(-half, half)))]
    return {"type": "Polygon", "coordinates": [[*ring, ring[0]]]}


CHARTER: dict[str, Any] = {
    "area": square(100),
    "home": {"lat": HOME[0], "lon": HOME[1], "alt_amsl_m": 584.0},
    "pilot_position": {"lat": HOME[0], "lon": HOME[1]},
    "alt_max_agl_m": 60.0,
    "max_range_from_pilot_m": 120.0,
    "max_speed_ms": 10.0,
    "geofence_buffer_m": 20.0,
}


def sample(east: float = 0.0, north: float = 0.0, **kw: Any) -> Sample:
    lat, lon = at(east, north)
    base: dict[str, Any] = {
        "t": 0.0,
        "lat": lat,
        "lon": lon,
        "alt_m": 40.0,
        "groundspeed_ms": 5.0,
        "mode": "GUIDED",
        "armed": True,
    }
    return Sample(**{**base, **kw})


def test_track_inside_every_limit() -> None:
    track = [
        sample(alt_m=61.9),
        sample(north=119.0),  # outside the area by 19 m, under the 20 m buffer
        sample(groundspeed_ms=10.9),
    ]
    assert check_track(track, CHARTER) == []


def test_track_just_outside_each_limit() -> None:
    found = check_track(
        [sample(alt_m=62.1), sample(east=126.0), sample(groundspeed_ms=11.1)], CHARTER
    )
    assert [f.split(":")[0] for f in found] == ["INV1", "INV2", "INV3", "INV4"]


def test_disarmed_samples_and_exemptions() -> None:
    assert check_track([sample(alt_m=500, armed=False)], CHARTER) == []
    assert check_track([sample(east=150.0)], CHARTER, exempt=["INV2", "INV4"]) == []


def cmd(skill: str, status: str, **kw: Any) -> dict[str, Any]:
    return {
        "id": kw.pop("id", skill),
        "skill": skill,
        "status": status,
        "issuer_kind": kw.pop("issuer_kind", "brain"),
        "created_at": kw.pop("created_at", "2026-10-17T12:00:05+00:00"),
    }


def test_orders_need_a_ground_allow() -> None:
    allow = [{"decision": "ALLOW", "side": "ground"}]
    reject = [{"decision": "REJECT", "side": "ground"}]
    ok = [cmd("goto", "done"), cmd("survey_sector", "rejected_by_gate", id="s")]
    assert check_orders(ok, {"goto": allow, "s": reject}) == []
    assert check_orders([cmd("report", "done")], {}) == []  # runs on the ground
    found = check_orders([cmd("goto", "done"), cmd("hold", "rejected_by_gate")], {"hold": allow})
    assert [f.split(":")[0] for f in found] == ["INV7", "INV7"]
    onboard_only = {"goto": [{"decision": "ALLOW", "side": "onboard"}]}
    assert check_orders([cmd("goto", "done")], onboard_only)


def event(kind: str, t: str, by: str = "operator:ci-harness") -> dict[str, Any]:
    return {"type": kind, "t": t, "payload": {"by": by}}


def test_overrides_from_an_operator_and_nothing_from_the_ai_meanwhile() -> None:
    stop = event("vehicle.stop", "2026-10-17T12:00:00+00:00")
    resume = event("vehicle.resume", "2026-10-17T12:00:10+00:00")
    during = cmd("goto", "done", created_at="2026-10-17T12:00:05+00:00")
    after = cmd("goto", "done", created_at="2026-10-17T12:00:15+00:00")
    refused = cmd("goto", "rejected_by_gate", created_at="2026-10-17T12:00:05+00:00")
    pilot = cmd("goto", "done", issuer_kind="operator")
    assert check_overrides([stop, resume], [after, refused, pilot]) == []
    assert check_overrides([stop, resume], [during]) == [
        "INV10: AI-ordern goto gick igenom under STOPP/HEM"
    ]
    assert check_overrides([stop], [after])  # no Resume: the STOP still stands
    found = check_overrides([event("vehicle.return_home", stop["t"], by="brain:x")], [])
    assert found == ["INV8: vehicle.return_home från brain:x"]


def test_check_all_uses_the_loosest_approved_charter() -> None:
    strict = {**CHARTER, "alt_max_agl_m": 30.0}
    charters = [
        {"status": "superseded", "body": CHARTER},
        {"status": "approved", "body": strict},
        {"status": "rejected", "body": {**CHARTER, "alt_max_agl_m": 500.0}},
    ]
    track = [sample(alt_m=50.0)]
    assert check_all(track, charters, [], {}, []) == []
    assert check_all([sample(alt_m=70.0)], charters, [], {}, [])
    assert check_all(track, [], [cmd("goto", "done")], {}, [], exempt=["INV7"]) == []
