"""B1-14: a skill that is not built never passes the gate: any issuer, phase or side.

The stage 2 skills (detections, following, follow me / find me) exist in the registry only
as stubs. However an order for them is phrased and whoever sends it, the gate refuses it
with ``SKILL_NOT_IMPLEMENTED``.
"""

from __future__ import annotations

import pytest
from gate_fixtures import REGISTRY, ctx, req

from sd_safety import evaluate

STAGE2 = ("inspect_detection", "follow_track", "stop_follow", "follow_me", "find_me")
STUBS = sorted(s.name for s in REGISTRY.as_mapping().values() if s.status == "stub")
ISSUERS = {"brain": "brain:ground", "operator": "operator:demo"}


def test_stage2_skills_are_stubs() -> None:
    assert set(STAGE2) <= set(STUBS)


@pytest.mark.parametrize("skill", STUBS)
@pytest.mark.parametrize("issuer", sorted(ISSUERS))
@pytest.mark.parametrize("phase", ["submit", "approve"])
@pytest.mark.parametrize("side", ["ground", "onboard"])
def test_stub_never_allowed(skill: str, issuer: str, phase: str, side: str) -> None:
    for params in ({}, {"lat": -35.3632, "lon": 149.1652, "distance_m": 10, "alt_m": 30}):
        r = req(skill, params, issuer_kind=issuer, issuer_ref=ISSUERS[issuer], phase=phase)
        d = evaluate(r, ctx(side=side))
        assert d.decision == "REJECT", (skill, issuer, phase, side, params)
        assert "SKILL_NOT_IMPLEMENTED" in d.reason_codes
