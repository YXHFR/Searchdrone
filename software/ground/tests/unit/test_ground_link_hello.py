"""The onboard computer's Hello: action numbers and the operator latch are reconciled."""

from __future__ import annotations

from typing import Any

from sd_ground.link import GroundLink
from sd_ground.services.latches import ADOPTED_BY, Latches
from sd_link.tokens import TokenStore, generate_token, hash_token
from sd_schemas.executor import ExecutorState
from sd_schemas.link import SCHEMA_MAJOR, Hello, LatchRef
from sd_testing import FakeClock

VID = "sim-01"


def make() -> tuple[GroundLink, Latches, list[tuple[str, dict[str, Any]]]]:
    clock = FakeClock()
    latches = Latches(clock)
    events: list[tuple[str, dict[str, Any]]] = []
    link = GroundLink(
        clock=clock,
        latches=latches,
        tokens=TokenStore({VID: hash_token(generate_token())}),
        vehicle_ids=[VID],
        bind="127.0.0.1:0",
        on_event=lambda t, _sev, _vid, payload: events.append((t, payload)),
    )
    return link, latches, events


def hello(latch: LatchRef | None, last_seq: int) -> Hello:
    return Hello(
        schema_major=SCHEMA_MAJOR,
        boot_id=7,
        executor_state=ExecutorState.HELD,
        charter_id=None,
        charter_version=None,
        fc_firmware=None,
        operator_latch=latch,
        last_action_seq=last_seq,
    )


async def test_onboard_only_latch_is_adopted() -> None:
    link, latches, events = make()
    far = 10**15  # above any number the ground has handed out
    ack = await link._on_hello(VID, hello(LatchRef(action="stop", action_seq=far), far))  # pyright: ignore[reportPrivateUsage]
    latch = latches.get(VID)
    assert latch is not None and latch.action == "stop" and latch.action_seq == far
    assert latch.set_by == ADOPTED_BY
    assert ack.operator_latch == LatchRef(action="stop", action_seq=far)
    assert [t for t, _ in events] == ["alarm.LATCH_ADOPTED"]
    # The next number the ground hands out is above the onboard one.
    assert latches.next_seq(VID) > far


async def test_no_latch_raises_the_sequence_only() -> None:
    link, latches, events = make()
    far = 10**15
    ack = await link._on_hello(VID, hello(None, far))  # pyright: ignore[reportPrivateUsage]
    assert latches.get(VID) is None and ack.operator_latch is None and events == []
    assert latches.next_seq(VID) > far


async def test_the_grounds_latch_stays_when_both_have_one() -> None:
    link, latches, events = make()
    ours = latches.start(VID, "return_home", "operator:pilot")
    ack = await link._on_hello(  # pyright: ignore[reportPrivateUsage]
        VID, hello(LatchRef(action="stop", action_seq=ours.action_seq - 1), ours.action_seq - 1)
    )
    assert latches.get(VID) == ours
    assert ack.operator_latch == LatchRef(action="return_home", action_seq=ours.action_seq)
    assert events == []
