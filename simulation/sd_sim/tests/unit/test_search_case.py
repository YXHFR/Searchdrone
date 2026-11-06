"""A broken search run's numbers (``search_case.json``) read back and run again give what
the run found."""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import pytest

from sd_safety.targets import Plane
from sd_sim.invariants import Sample
from sd_sim.search_case import CASE, load_case, main, replay, write_case
from sd_sim.search_invariants import SeenKeepout, keepout_never_entered, no_confirm_without_human

HOME = (-35.363261, 149.165230)  # ArduPilot's SITL site CMAC
T0 = dt.datetime(2026, 11, 6, 10, 0, tzinfo=dt.UTC)
PLANE = Plane(*HOME)


def _cylinder(**kw: object) -> SeenKeepout:
    lat, lon = PLANE.latlon(100.0, 0.0)
    k = SeenKeepout("k1", lat, lon, 30.0, 10.0)
    return SeenKeepout(**{**k.__dict__, **kw})  # type: ignore[arg-type]


def _straight_through(speed: float = 5.0) -> list[Sample]:
    """East along y = 0 at ``speed``, through the cylinder long after it came."""
    return [
        Sample(float(t), *PLANE.latlon(speed * t, 0.0), 40.0, speed, "GUIDED", True)
        for t in range(40)
    ]


def _e1() -> dict[str, object]:
    """A confirmation without a large view in the same session (E1)."""
    return {
        "revisions": [
            {"candidate_id": "c1", "rev": 2, "state": "confirmed",
             "created_at": T0 + dt.timedelta(seconds=3)},
        ],
        "decisions": [
            {"candidate_id": "c1", "decision": "confirm", "operator_ref": "operator:ci",
             "session_id": "s1", "created_at": T0 + dt.timedelta(seconds=2)},
        ],
        "views": [],
        "observed": {"c1": ["d1"]},
    }  # fmt: skip


def test_an_e8_case_runs_again_to_the_same_break(tmp_path: Path) -> None:
    track, keepouts = _straight_through(), [_cylinder()]
    broken = keepout_never_entered(keepouts, track)
    assert broken and broken[0].startswith("E8")
    path = write_case(tmp_path, scenario="SK06", seed=25, broken=broken, track=track,
                      keepouts=keepouts, params={"with_map": True})  # fmt: skip
    assert path == tmp_path / CASE
    case = load_case(path)
    assert (case.scenario, case.seed, case.broken) == ("SK06", 25, broken)
    assert case.params == {"with_map": True}
    assert replay(case) == broken


def test_a_cylinder_s_open_end_and_its_first_shape_come_back(tmp_path: Path) -> None:
    open_end = _cylinder()
    later = _cylinder(keepout_id="k1", since=12.5, until=20.25, came=10.0, passable=True)
    write_case(tmp_path, scenario="SK03", seed=2, broken=[], track=[], keepouts=[open_end, later])
    got = load_case(tmp_path / CASE).keepouts
    assert math.isinf(got[0].until) and got[0].came is None and not got[0].passable
    assert (got[1].since, got[1].until, got[1].came, got[1].passable) == (12.5, 20.25, 10.0, True)
    assert got[1].arrival == 10.0


def test_e1_rows_with_their_times_run_again(tmp_path: Path) -> None:
    e1 = _e1()
    broken = no_confirm_without_human(e1["revisions"], e1["decisions"], e1["views"],  # type: ignore[arg-type]
                                      e1["observed"])  # type: ignore[arg-type]  # fmt: skip
    assert broken
    write_case(tmp_path, scenario="SK01", seed=1, broken=broken, track=[], keepouts=[], e1=e1)
    assert replay(load_case(tmp_path / CASE)) == broken


def test_a_case_holds_numbers_only_and_stays_small(tmp_path: Path) -> None:
    """A 30 minute flight at 10 Hz stays well under the few MB an artifact may take."""
    track = [
        Sample(i / 10.0, *PLANE.latlon(0.5 * i, 0.3 * i), 40.123456, 5.4321, "AUTO", True)
        for i in range(30 * 60 * 10)
    ]
    path = write_case(tmp_path, scenario="SK06", seed=1, broken=["E8: x"], track=track,
                      keepouts=[_cylinder()])  # fmt: skip
    assert path.stat().st_size < 1_500_000
    body = json.loads(path.read_text(encoding="utf-8"))
    assert set(body) == {"version", "scenario", "seed", "broken", "params", "keepouts",
                         "track_columns", "track", "e1"}  # fmt: skip
    assert body["track"][1] == [0.1, *[round(x, 7) for x in PLANE.latlon(0.5, 0.3)], 40.12,
                                5.43, "AUTO", True]  # fmt: skip


def test_another_version_is_refused(tmp_path: Path) -> None:
    path = write_case(tmp_path, scenario="SK01", seed=1, broken=[], track=[], keepouts=[])
    body = json.loads(path.read_text(encoding="utf-8"))
    body["version"] = 99
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="version 99"):
        load_case(path)


def test_the_cli_shows_ci_s_and_today_s_breaks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    track, keepouts = _straight_through(), [_cylinder()]
    broken = keepout_never_entered(keepouts, track)
    bad = write_case(tmp_path / "bad", scenario="SK06", seed=25, broken=broken, track=track,
                     keepouts=keepouts)  # fmt: skip
    fine = write_case(tmp_path / "fine", scenario="SK06", seed=26, broken=["E8: gammalt"],
                      track=track[:5], keepouts=keepouts)  # fmt: skip
    assert main([str(fine)]) == 0
    assert "i CI: E8: gammalt" in capsys.readouterr().out
    assert main([str(fine), str(bad)]) == 1
    out = capsys.readouterr().out
    assert "inga brott" in out and f"nu:   {broken[0]}" in out
