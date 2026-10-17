"""``OverrideState`` on its own: action numbers, the Hello's latch and who took over."""

from __future__ import annotations

import pytest

from sd_onboard.executor.override import OVERRIDE_KINDS, OverrideState
from sd_schemas.base import new_id
from sd_schemas.link import LatchRef, ManualTakeover, OperatorOverrideNotice


def _notice(kind: str, seq: int) -> OperatorOverrideNotice:
    return OperatorOverrideNotice.model_validate(
        {
            "override_id": new_id(),
            "kind": kind,
            "action_seq": seq,
            "issuer_kind": "operator",
            "operator_id": "operator:pilot",
        }
    )


def _takeover(seq: int, *, active: bool = True) -> ManualTakeover:
    return ManualTakeover(
        takeover_id=new_id(), active=active, action_seq=seq, operator_id="operator:pilot"
    )


def test_takeover_older_than_the_newest_action_is_ignored() -> None:
    over = OverrideState()
    assert over.apply_notice(_notice("return_home", 5)) is not None
    assert over.apply_takeover(_takeover(4)) is False
    assert over.manual is False and over.last_action_seq == 5
    assert over.apply_takeover(_takeover(5)) is False  # the same number is not newer
    assert over.apply_takeover(_takeover(6)) is True
    assert over.manual is True and over.last_action_seq == 6
    assert over.apply_takeover(_takeover(7, active=False)) is True  # hand-back
    assert over.manual is False


def test_hello_reports_the_latch_until_release_or_landing() -> None:
    over = OverrideState()
    assert over.latch_ref() is None
    over.apply_notice(_notice("stop", 3))
    assert over.latch_ref() == LatchRef(action="stop", action_seq=3)
    over.released()
    assert over.latch_ref() is None
    over.apply_notice(_notice("battery_rtl", 4))
    assert over.latch_ref() == LatchRef(action="battery_rtl", action_seq=4)
    over.landed()
    assert over.latch_ref() is None and not over.battery_rtl_active


def test_mode_change_without_a_notice_is_external() -> None:
    assert OverrideState().held_by_for_mode_change() == "external"


@pytest.mark.parametrize("kind", sorted(OVERRIDE_KINDS))
def test_mode_change_after_a_notice_is_held_by_its_kind(kind: str) -> None:
    over = OverrideState()
    over.apply_notice(_notice(kind, 1))
    assert over.held_by_for_mode_change() == OVERRIDE_KINDS[kind].held_by
    over.released()
    assert over.held_by_for_mode_change() == "external"
