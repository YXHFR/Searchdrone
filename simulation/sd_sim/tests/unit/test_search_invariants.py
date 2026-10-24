"""The search scenarios' invariants E1, E2 and E8 catch what they name."""

from __future__ import annotations

import datetime as dt

import pytest

from sd_safety.targets import Plane
from sd_sim.invariants import Sample
from sd_sim.search_invariants import (
    CandidateWatch,
    SeenKeepout,
    keepout_distances,
    keepout_never_entered,
    no_confirm_without_human,
)

T0 = dt.datetime(2026, 10, 23, 10, 0, tzinfo=dt.UTC)
HOME = (-35.363261, 149.165230)  # ArduPilot's SITL site CMAC


def at(s: float) -> dt.datetime:
    return T0 + dt.timedelta(seconds=s)


def rev(state: str, s: float, cid: str = "c1", n: int = 2) -> dict[str, object]:
    return {"candidate_id": cid, "rev": n, "state": state, "created_at": at(s)}


def dec(decision: str, s: float, cid: str = "c1", ref: str = "operator:ci", session: str = "s1"):
    return {"candidate_id": cid, "decision": decision, "operator_ref": ref,
            "session_id": session, "created_at": at(s)}  # fmt: skip


VIEW = {"detection_id": "d1", "session_id": "s1", "created_at": at(1)}
OBSERVED = {"c1": ["d1"]}


def test_a_person_confirming_after_a_view_is_fine() -> None:
    assert no_confirm_without_human([rev("ready", 0, n=1), rev("confirmed", 3)],
                                    [dec("confirm", 2)], [VIEW], OBSERVED) == []  # fmt: skip


def test_confirmed_without_a_decision_breaks_e1() -> None:
    broken = no_confirm_without_human([rev("confirmed", 3)], [], [VIEW], OBSERVED)
    assert broken and "utan en persons beslut" in broken[0]


def test_a_decision_after_the_state_does_not_count() -> None:
    assert no_confirm_without_human([rev("rejected", 3)], [dec("reject", 5)], [], OBSERVED)


def test_confirming_without_a_large_view_breaks_e1() -> None:
    late = {**VIEW, "created_at": at(9)}
    other = {**VIEW, "session_id": "s2"}
    for views in ([], [late], [other]):
        broken = no_confirm_without_human([rev("confirmed", 3)], [dec("confirm", 2)], views,
                                          OBSERVED)  # fmt: skip
        assert any("stor vy" in b for b in broken), views


def test_a_decision_without_an_operator_breaks_e1() -> None:
    broken = no_confirm_without_human([], [dec("reject", 1, ref="service:search")], [], {})
    assert broken and "utan en person" in broken[0]


def test_a_vanished_or_older_candidate_breaks_e2() -> None:
    w = CandidateWatch()
    w.look({"a": 1, "b": 2})
    w.look({"a": 2, "b": 2, "c": 1})
    assert w.broken == []
    w.look({"a": 1, "c": 1})
    assert len(w.broken) == 2


def _track(points: list[tuple[float, float, float]], mode: str = "GUIDED") -> list[Sample]:
    plane = Plane(*HOME)
    return [Sample(t, *plane.latlon(e, n), 40.0, 5.0, mode, True) for t, e, n in points]


def _cyl(since: float = 0.0, until: float = float("inf")) -> SeenKeepout:
    lat, lon = Plane(*HOME).latlon(100.0, 0.0)
    return SeenKeepout("k1", lat, lon, 30.0, since, until)


def test_flying_around_a_cylinder_is_fine() -> None:
    track = _track([(1, 0, 0), (2, 60, 0), (3, 100, 40), (4, 140, 0)])
    assert keepout_never_entered([_cyl()], track) == []
    assert keepout_distances([_cyl()], track) is not None


def test_leaving_a_cylinder_made_around_the_drone_is_fine_but_coming_back_is_not() -> None:
    leave = _track([(1, 95, 0), (2, 80, 0), (3, 40, 0)])
    assert keepout_never_entered([_cyl()], leave) == []
    back = _track([(1, 95, 0), (2, 40, 0), (3, 90, 0)])
    broken = keepout_never_entered([_cyl()], back)
    assert broken and broken[0].startswith("E8")
    assert (keepout_distances([_cyl()], back) or 0.0) < 0


def test_the_way_home_goes_past_cylinders() -> None:
    track = _track([(1, 95, 0), (2, 40, 0)]) + _track([(3, 90, 0), (4, 140, 0)], "SMART_RTL")
    assert keepout_never_entered([_cyl()], track) == []
    searching = _track([(1, 95, 0), (2, 40, 0), (3, 90, 0), (4, 140, 0)])
    assert keepout_never_entered([_cyl()], searching)


def test_searching_again_after_the_way_home_is_checked() -> None:
    track = (
        _track([(1, 95, 0), (2, 40, 0)]) + _track([(3, 40, 10)], "SMART_RTL") + _track([(4, 90, 0)])
    )
    broken = keepout_never_entered([_cyl()], track)
    assert broken and broken[0].startswith("E8")


def test_a_landing_after_stop_is_checked() -> None:
    home = _track([(1, 95, 0), (2, 40, 0)]) + _track([(3, 60, 0)], "SMART_RTL")
    assert keepout_never_entered([_cyl()], home + _track([(4, 90, 0)], "LAND")) == []
    stop = _track([(1, 95, 0), (2, 40, 0)]) + _track([(3, 90, 0)], "LAND")
    broken = keepout_never_entered([_cyl()], stop)
    assert broken and broken[0].startswith("E8")


def test_the_distance_counts_from_clearly_outside() -> None:
    leave = _track([(1, 95, 0), (2, 70, 0), (3, 60, 0), (4, 20, 0)])
    assert keepout_distances([_cyl()], leave) == pytest.approx(10.0, abs=0.5)


def test_a_cylinder_counts_only_while_it_is_in_force() -> None:
    track = _track([(1, 40, 0), (2, 100, 0), (3, 40, 0)])
    assert keepout_never_entered([_cyl(since=2.5)], track) == []
    assert keepout_never_entered([_cyl(until=0.5)], track) == []
    assert keepout_never_entered([_cyl()], track)
