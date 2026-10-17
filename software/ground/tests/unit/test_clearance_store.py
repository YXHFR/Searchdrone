"""``Clearances``: the active "clear around the drone" per vehicle, in memory."""

from __future__ import annotations

from sd_ground.services.clearance import Clearances
from sd_safety.constants import CLEARANCE_VALID_S
from sd_schemas.base import new_id

VALID_MS = CLEARANCE_VALID_S * 1000


def _confirm(store: Clearances, vehicle_id: str, now_ms: int) -> str:
    cid = str(new_id())
    store.confirm(
        clearance_id=cid,
        origin="local",
        mission_id=str(new_id()),
        vehicle_id=vehicle_id,
        charter_id=str(new_id()),
        charter_version=1,
        now_ms=now_ms,
    )
    return cid


def test_expired_takes_out_only_what_ran_out() -> None:
    store = Clearances()
    old = _confirm(store, "sim-01", now_ms=1_000)
    new = _confirm(store, "sim-02", now_ms=60_000)
    assert store.expired(1_000 + VALID_MS) == []  # exactly 120 s still counts
    gone = store.expired(1_000 + VALID_MS + 1)
    assert [(vid, a.clearance_id) for vid, a in gone] == [("sim-01", old)]
    assert store.get("sim-01") is None
    still = store.get("sim-02")
    assert still is not None and still.clearance_id == new
    assert store.expired(1_000 + VALID_MS + 1) == []  # reported once


def test_a_new_confirmation_replaces_the_old_one() -> None:
    store = Clearances()
    _confirm(store, "sim-01", now_ms=0)
    newer = _confirm(store, "sim-01", now_ms=10_000)
    assert [vid for vid, _ in store.expired(VALID_MS + 1)] == []  # the old one is gone
    active = store.consume("sim-01")
    assert active is not None and active.clearance_id == newer
    assert store.consume("sim-01") is None and store.drop("sim-01") is None
