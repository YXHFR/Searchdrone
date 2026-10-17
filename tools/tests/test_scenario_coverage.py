"""Every scenario SC00-SC19 has a SITL test: a ``test_scNN...`` function under
``simulation/tests/scenarios/``, or one of the named tests elsewhere that fly it."""

from __future__ import annotations

import re
from pathlib import Path

import gen_traceability as gt

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = [f"SC{n:02d}" for n in range(20)]
# Scenarios flown by the emergency-path and onboard SITL tests (same rig, own files).
ELSEWHERE = {
    "SC10": ["test_ground_kill_gcs_failsafe", "test_hung_ground_station_lets_the_gcs_failsafe_act"],
    "SC15": ["test_stop_bypasses_onboard", "test_home_bypasses_onboard"],
    "SC17": ["test_onboard_kill_in_guided", "test_watchdog_respects_stop"],
}
NAME = re.compile(r"^test_((?:sc\d{2}_?)+)")


def test_every_scenario_has_a_sitl_test() -> None:
    tests = gt.test_functions(ROOT, ("simulation/tests",))
    named: set[str] = set()
    for name, files in tests.items():
        m = NAME.match(name)
        if m and any(f.startswith("simulation/tests/scenarios/") for f in files):
            named.update(f"SC{n}" for n in re.findall(r"sc(\d{2})", m.group(1)))
    for sc, others in ELSEWHERE.items():
        assert all(o in tests for o in others), (sc, [o for o in others if o not in tests])
        named.add(sc)
    assert [sc for sc in SCENARIOS if sc not in named] == []
