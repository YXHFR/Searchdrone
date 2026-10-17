"""sd_sim.invariants: each invariant passes a clean timeline and fails a broken one."""

from __future__ import annotations

from typing import Any

import pytest

from sd_safety.targets import Plane
from sd_sim import invariants as inv
from sd_sim.invariants import Charter, Sample, Sent, Timeline

HOME = (-35.363261, 149.165230)
PLANE = Plane(*HOME)


def _area(half: float) -> dict[str, Any]:
    ring = []
    for x, y in [(-half, -half), (half, -half), (half, half), (-half, half), (-half, -half)]:
        lat, lon = PLANE.latlon(x, y)
        ring.append([lon, lat])
    return {"type": "Polygon", "coordinates": [ring]}


def _charter(**kw: Any) -> Charter:
    base: dict[str, Any] = {
        "since": 0.0,
        "area": _area(150),
        "pilot": HOME,
        "alt_max_m": 120.0,
        "speed_max_ms": 15.0,
        "range_m": 500.0,
        "buffer_m": 20.0,
        "autonomy_mode": "WITHIN_CHARTER",
        "approved_by": "operator:ci-harness",
    }
    return Charter(**{**base, **kw})


def _at(t: float, east: float = 0, north: float = 0, alt: float = 40, gs: float = 5) -> Sample:
    lat, lon = PLANE.latlon(east, north)
    return Sample(t, lat, lon, alt, gs, inv.GUIDED, True)


def _clean() -> Timeline:
    tl = Timeline(charters=[_charter()], clearances=[1.0])
    tl.samples = [_at(t, north=t) for t in range(2, 60)]
    tl.sent = [
        Sent(1.5, "onboard", "COMMAND_LONG", command=inv.CMD_ARM_DISARM, params=(1, 0)),
        Sent(3.0, "onboard", "SET_POSITION_TARGET_GLOBAL_INT", alt_m=40, armed=True),
        Sent(4.0, "ground", "COMMAND_LONG", command=inv.CMD_SET_MESSAGE_INTERVAL),
        Sent(0.5, "ground", "MISSION_COUNT", mission_type=1),
    ]
    tl.commands = [{"id": "c1", "skill": "goto", "status": "done"}]
    tl.accepted = {"c1": 0.0}
    tl.gates = [{"command_id": "c1", "side": "ground", "decision": "ALLOW"}]
    return tl


def _ids(results: list[inv.Result]) -> list[str]:
    return [r.id for r in inv.failed(results)]


def test_clean_timeline_passes() -> None:
    results = inv.check_all(_clean())
    assert [r.id for r in results] == list(inv.ALL)
    assert _ids(results) == []


def test_height_speed_distance_and_area() -> None:
    tl = _clean()
    tl.samples.append(_at(61, north=100, alt=122.5))  # 120 + 2 m
    tl.samples.append(_at(62, north=100, gs=16.5))  # 15 + 1 m/s
    tl.samples.append(_at(63, east=171))  # 21 m outside, buffer 20
    assert _ids(inv.check_all(tl)) == ["INV1", "INV3", "INV4"]
    far = Timeline(charters=[_charter(area=_area(800))], samples=[_at(1, north=506)])
    assert _ids(inv.check_all(far)) == ["INV2"]
    tight = _clean()
    tight.inside_m = 0.0
    tight.samples.append(_at(64, east=151))
    assert "INV4" in _ids(inv.check_all(tight))


def test_samples_on_the_ground_or_before_a_charter_do_not_count() -> None:
    tl = Timeline(charters=[_charter(since=10.0)])
    tl.samples = [_at(5, alt=300), Sample(20, *HOME, 300, 0, inv.LOITER, False)]
    assert _ids(inv.check_all(tl)) == []


@pytest.mark.parametrize(
    ("sent", "why"),
    [
        (Sent(5, "onboard", "PARAM_SET", param_id="FENCE_ENABLE"), "skrev FENCE_ENABLE"),
        (Sent(5, "onboard", "PARAM_SET", param_id="WP_SPD"), "onboard: PARAM_SET"),
        (Sent(5, "ground", "PARAM_SET", param_id="FS_GCS_ENABLE"), "skrev FS_GCS_ENABLE"),
        (Sent(5, "ground", "PARAM_SET", param_id="FENCE_RADIUS", armed=True), "ground: PARAM"),
        (Sent(5, "ground", "MISSION_COUNT", mission_type=1, armed=True), "MISSION_COUNT"),
        (Sent(5, "ground", "COMMAND_LONG", command=176, params=(1, 4)), "176"),
        (Sent(5, "ground", "COMMAND_LONG", command=400, params=(0, 21196)), "400"),
        (Sent(5, "harness", "PARAM_SET", param_id="FENCE_RADIUS"), "harness: PARAM_SET"),
        (Sent(5, "harness", "COMMAND_LONG", command=176, params=(1, 4)), "harness"),
        (Sent(5, "qgc", "HEARTBEAT"), None),  # reading is allowed for anyone
        (Sent(5, "qgc", "COMMAND_LONG", command=176), "okänd avsändare"),
        (Sent(5, "onboard", "COMMAND_LONG", command=400, params=(0, 0), alt_m=12), "i luften"),
        (Sent(5, "onboard", "MISSION_ITEM_INT", mission_type=1), "mission_type 1"),
    ],
)
def test_who_may_write_what(sent: Sent, why: str | None) -> None:
    tl = _clean()
    tl.sent.append(sent)
    result = inv.inv5(tl)
    if why is None:
        assert result.passed, result
    else:
        assert not result.passed and why in result.detail, result


def test_allowed_writes() -> None:
    tl = _clean()
    tl.sent += [
        Sent(5, "ground", "COMMAND_LONG", command=176, params=(1, inv.BRAKE)),
        Sent(5, "ground", "COMMAND_LONG", command=400, params=(0, 0), landed=True),
        Sent(5, "ground", "PARAM_SET", param_id="FENCE_ALT_MAX"),
        Sent(5, "ground", "MANUAL_CONTROL", armed=True),
        Sent(5, "harness", "PARAM_SET", param_id="SIM_RC_FAIL"),
        Sent(5, "harness", "COMMAND_LONG", command=inv.CMD_BATTERY_RESET, params=(1, 26)),
    ]
    assert inv.inv5(tl).passed
    tl.sent.append(Sent(6, "harness", "SET_POSITION_TARGET_LOCAL_NED"))
    assert not inv.inv5(tl).passed
    assert inv.inv5(tl, harness_extra=["SET_POSITION_TARGET_LOCAL_NED"]).passed


def test_arming_needs_a_charter_and_its_own_clearance() -> None:
    tl = _clean()
    tl.clearances = []
    assert _ids(inv.check_all(tl)) == ["INV6"]
    twice = _clean()
    twice.sent.append(
        Sent(30, "onboard", "COMMAND_LONG", command=inv.CMD_ARM_DISARM, params=(1, 0))
    )
    assert "INV6" in _ids(inv.check_all(twice))  # one clearance, two arms
    uninstalled = _clean()
    uninstalled.charters = [_charter(install_status="pending")]
    assert "INV6" in _ids(inv.check_all(uninstalled))


def test_every_order_that_ran_has_a_ground_allow() -> None:
    tl = _clean()
    tl.gates = []
    assert _ids(inv.check_all(tl)) == ["INV7"]
    refused = _clean()
    refused.accepted = {}
    refused.gates = []
    assert _ids(inv.check_all(refused)) == []
    mismatch = _clean()
    mismatch.events = [(5.0, "gate.mismatch", {})]
    assert _ids(inv.check_all(mismatch)) == ["INV7"]
    assert _ids(inv.check_all(mismatch, allow_mismatch=True)) == []
    ground_skill = _clean()
    ground_skill.commands.append({"id": "r", "skill": "report", "status": "done"})
    ground_skill.accepted["r"] = 0.0
    assert _ids(inv.check_all(ground_skill)) == []


def test_stop_and_home_only_from_the_operator() -> None:
    tl = _clean()
    tl.events = [(10.0, "vehicle.stop", {"by": "brain:ground", "via": "chat"})]
    assert "INV8" in _ids(inv.check_all(tl))
    tl.events = [(10.0, "vehicle.stop", {"by": "system:operator_absent", "via": "system"})]
    assert "INV8" not in _ids(inv.check_all(tl))


def test_unknown_order_status() -> None:
    tl = _clean()
    tl.commands.append({"id": "x", "skill": "hold", "status": "teleported"})
    assert _ids(inv.check_all(tl)) == ["INV9"]


def _stopped(*extra: Sent, release: float | None = None) -> Timeline:
    tl = _clean()
    tl.events = [(30.0, "vehicle.stop", {"by": "operator:ci-harness", "via": "button"})]
    if release is not None:
        tl.events.append((release, "vehicle.resume", {"by": "operator:ci-harness"}))
    tl.sent.append(Sent(30.1, "ground", "COMMAND_LONG", command=176, params=(1, inv.BRAKE)))
    tl.sent += extra
    return tl


def test_nothing_fights_the_operators_stop() -> None:
    assert inv.inv10(_stopped()).passed
    steer = Sent(35, "onboard", "SET_POSITION_TARGET_GLOBAL_INT")
    assert not inv.inv10(_stopped(steer)).passed
    assert inv.inv10(_stopped(steer, release=34.0)).passed  # after Resume
    home = Sent(33.5, "ground", "COMMAND_LONG", command=176, params=(1, inv.SMART_RTL))
    assert inv.inv10(_stopped(home, release=34.5)).passed  # FLY HOME, logged just after
    resend = Sent(32, "ground", "COMMAND_LONG", command=176, params=(1, inv.BRAKE))
    assert inv.inv10(_stopped(resend)).passed
    fallback = Sent(32, "ground", "COMMAND_LONG", command=176, params=(1, inv.LOITER))
    assert inv.inv10(_stopped(fallback)).passed
    late = Sent(40, "ground", "COMMAND_LONG", command=176, params=(1, inv.LOITER))
    assert not inv.inv10(_stopped(late)).passed
    other = Sent(32, "onboard", "COMMAND_LONG", command=176, params=(1, inv.GUIDED))
    assert not inv.inv10(_stopped(other)).passed


def test_nothing_fights_a_failsafe() -> None:
    tl = _clean()
    lat, lon = PLANE.latlon(0, 50)
    tl.samples.append(Sample(70, lat, lon, 40, 3, inv.SMART_RTL, True))  # nobody asked
    tl.sent.append(Sent(75, "onboard", "SET_POSITION_TARGET_GLOBAL_INT"))
    assert not inv.inv10(tl).passed
    ours = _clean()
    ours.sent.append(Sent(69, "onboard", "COMMAND_LONG", command=176, params=(1, inv.SMART_RTL)))
    ours.samples.append(Sample(70, lat, lon, 40, 3, inv.SMART_RTL, True))
    ours.sent.append(Sent(75, "onboard", "SET_POSITION_TARGET_GLOBAL_INT"))
    assert inv.inv10(ours).passed  # our own return home, not a failsafe


def test_autonomy_only_by_the_operator() -> None:
    tl = _clean()
    tl.charters.append(_charter(since=50, autonomy_mode="APPROVE_EACH", approved_by="brain:x"))
    assert _ids(inv.check_all(tl)) == ["INV11"]
    tl.charters[-1] = _charter(since=50, autonomy_mode="APPROVE_EACH")
    assert _ids(inv.check_all(tl)) == []


def test_exemptions_are_named() -> None:
    assert [r.id for r in inv.check_all(_clean(), ["INV1", "INV9"])] == [
        i for i in inv.ALL if i not in ("INV1", "INV9")
    ]
    with pytest.raises(ValueError, match="okända"):
        inv.check_all(_clean(), ["INV12"])
