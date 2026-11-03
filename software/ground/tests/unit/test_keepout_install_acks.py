"""The keep-out list on board or not (``services/link_loss.py``): the ground waits for the
executor's answer to every ``KeepoutInstall``, sends the list again when it is refused or
not answered, and tells the operator."""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog

from sd_ground.services import link_loss
from sd_link.session import Expired
from sd_schemas.base import new_id
from sd_schemas.link import Ack, Envelope, Keepout, KeepoutInstall

VID = "sd-01"
EPOCH_MS = 1_790_000_000_000


@dataclass
class FakeLink:
    is_up: bool = True
    sent: list[tuple[uuid.UUID, KeepoutInstall]] = field(default_factory=list)

    def up(self, vehicle_id: str) -> bool:
        return self.is_up

    async def send_message(self, vehicle_id: str, message: Any, **_: Any) -> uuid.UUID | None:
        if not self.is_up:
            return None
        msg_id = uuid.UUID(str(new_id()))
        self.sent.append((msg_id, message))
        return msg_id


@dataclass
class FakeFlow:
    link: FakeLink = field(default_factory=FakeLink)
    link_watch: dict[str, link_loss.LinkWatch] = field(default_factory=dict)
    alerts: list[tuple[str, str, str]] = field(default_factory=list)
    mono_ms: int = 1_000

    def now(self) -> dt.datetime:
        return dt.datetime.fromtimestamp(EPOCH_MS / 1000, dt.UTC)

    def now_ms(self) -> int:
        return self.mono_ms

    def alert(self, code: str, severity: str, message_sv: str, **_: Any) -> None:
        self.alerts.append((code, severity, message_sv))


def keepout() -> Keepout:
    return Keepout(
        keepout_id=new_id(),
        lat=-35.36,
        lon=149.16,
        radius_m=20.0,
        source="confirmed",
        source_id=new_id(),
    )


async def send(flow: FakeFlow) -> tuple[uuid.UUID, KeepoutInstall]:
    w = link_loss.watch(flow, VID)  # type: ignore[arg-type]
    if not w.keepouts:
        w.keepouts = [keepout()]
    await link_loss._install(flow, None, w, resend=False, vehicle_id=VID)  # type: ignore[arg-type]
    return flow.link.sent[-1]


async def answer(flow: FakeFlow, msg_id: uuid.UUID, stage: str, *codes: str) -> None:
    ack = Ack(ref_id=msg_id, stage=stage, reason_codes=list(codes))  # type: ignore[arg-type]
    await link_loss.on_ack(flow, VID, ack)  # type: ignore[arg-type]


def codes(flow: FakeFlow) -> list[str]:
    return [code for code, _, _ in flow.alerts]


async def test_done_is_installed_and_quiet() -> None:
    flow = FakeFlow()
    msg_id, install = await send(flow)
    await answer(flow, msg_id, "received")  # the link's receipt: still waiting
    assert flow.link_watch[VID].pending is not None
    await answer(flow, msg_id, "done")
    w = flow.link_watch[VID]
    assert w.pending is None and w.installed_seq == install.install_seq
    assert flow.alerts == []
    await send(flow)  # unchanged: not sent again
    assert len(flow.link.sent) == 1


async def test_stale_refusal_alarms_and_resends_higher_until_done() -> None:
    flow = FakeFlow()
    onboard_seq = EPOCH_MS + 10 * link_loss.STALE_STEP_MS  # the ground's clock went back
    msg_id, install = await send(flow)
    seqs = [install.install_seq]
    while install.install_seq < onboard_seq:
        await answer(flow, msg_id, "rejected", "ONB_STALE_ORDER")
        assert flow.link_watch[VID].pending is None
        msg_id, install = await send(flow)  # the next poll sends it again
        assert install.install_seq > seqs[-1]
        seqs.append(install.install_seq)
    assert len(seqs) <= 6  # the jump doubles: an hour, two, four, ...
    # One red alarm for the whole episode, in Swedish, naming STOPP.
    assert codes(flow) == ["KEEPOUTS_NOT_INSTALLED"]
    _, severity, text = flow.alerts[0]
    assert severity == "critical" and "förbudszonerna" in text and "STOPP" in text
    await answer(flow, msg_id, "done")
    assert codes(flow) == ["KEEPOUTS_NOT_INSTALLED", "KEEPOUTS_INSTALLED"]
    assert flow.link_watch[VID].installed_seq == install.install_seq
    assert flow.link_watch[VID].refusals == 0


async def test_endless_stale_refusals_stop_at_the_cap_and_keep_the_alarm() -> None:
    """An onboard computer that refuses every number as stale: the jumps stop at
    ``INSTALL_SEQ_MAX`` (exact in JSON), the ground logs an error once per run of
    refusals, the red alarm stays."""
    flow = FakeFlow()
    msg_id, install = await send(flow)
    with structlog.testing.capture_logs() as logs:
        for _ in range(5_000):  # each jump is at most 2**20 hours
            await answer(flow, msg_id, "rejected", "ONB_STALE_ORDER")
            msg_id, install = await send(flow)
            assert install.install_seq <= link_loss.INSTALL_SEQ_MAX
            if install.install_seq == link_loss.INSTALL_SEQ_MAX:
                break
        for _ in range(3):  # still refused at the cap
            await answer(flow, msg_id, "rejected", "ONB_STALE_ORDER")
            msg_id, install = await send(flow)
    at_cap = [e for e in logs if e["event"] == "link_loss.install_seq_at_cap"]
    assert len(at_cap) == 1 and at_cap[0]["log_level"] == "error"
    assert install.install_seq == link_loss.INSTALL_SEQ_MAX
    assert flow.link_watch[VID].install_seq == link_loss.INSTALL_SEQ_MAX
    assert codes(flow) == ["KEEPOUTS_NOT_INSTALLED"]  # still on, once
    assert flow.link_watch[VID].not_installed
    await answer(flow, msg_id, "done")  # it takes the capped number: in again
    assert codes(flow) == ["KEEPOUTS_NOT_INSTALLED", "KEEPOUTS_INSTALLED"]
    with structlog.testing.capture_logs() as logs:  # a new run of refusals logs again
        flow.link_watch[VID].sent = None
        msg_id, install = await send(flow)
        await answer(flow, msg_id, "rejected", "ONB_STALE_ORDER")
    assert [e["event"] for e in logs].count("link_loss.install_seq_at_cap") == 1


async def test_back_in_is_info_as_on_the_panel() -> None:
    flow = FakeFlow()
    msg_id, _ = await send(flow)
    await answer(flow, msg_id, "failed", "ONB_INTERNAL")
    msg_id, _ = await send(flow)
    await answer(flow, msg_id, "done")
    assert flow.alerts[-1][:2] == ("KEEPOUTS_INSTALLED", "info")  # panel tone "info"


async def test_other_refusal_resends_without_a_jump() -> None:
    flow = FakeFlow()
    msg_id, first = await send(flow)
    await answer(flow, msg_id, "failed", "ONB_INTERNAL")
    assert codes(flow) == ["KEEPOUTS_NOT_INSTALLED"]
    _, again = await send(flow)
    assert first.install_seq < again.install_seq < first.install_seq + link_loss.STALE_STEP_MS


async def test_an_answer_to_an_older_list_is_ignored() -> None:
    flow = FakeFlow()
    old_id, _ = await send(flow)
    flow.link_watch[VID].keepouts = [keepout()]  # a new find: a new list replaces it
    new_id_, _ = await send(flow)
    await answer(flow, old_id, "rejected", "ONB_STALE_ORDER")
    assert flow.alerts == [] and flow.link_watch[VID].pending is not None
    await answer(flow, new_id_, "done")
    assert flow.alerts == []


async def test_no_answer_with_the_link_up_is_a_refusal() -> None:
    flow = FakeFlow()
    await send(flow)
    w = flow.link_watch[VID]
    flow.mono_ms += link_loss.INSTALL_ACK_WAIT_MS - 1
    link_loss._check_answer(flow, VID, w, flow.now_ms())  # type: ignore[arg-type]
    assert flow.alerts == []
    flow.link.is_up = False  # link down: the link-loss alarms speak, Hello resends
    flow.mono_ms += 10
    link_loss._check_answer(flow, VID, w, flow.now_ms())  # type: ignore[arg-type]
    assert flow.alerts == []
    flow.link.is_up = True
    link_loss._check_answer(flow, VID, w, flow.now_ms())  # type: ignore[arg-type]
    assert codes(flow) == ["KEEPOUTS_NOT_INSTALLED"]
    assert w.pending is None and w.sent is None


async def test_expired_list_is_sent_again() -> None:
    flow = FakeFlow()
    msg_id, install = await send(flow)
    env = Envelope(
        msg_id=msg_id,
        kind="KeepoutInstall",
        body=install.model_dump(mode="json"),
        sender="ground:main",
        vehicle_id=VID,
        mission_id=None,
        seq=1,
        sent_utc=flow.now(),
        ttl_ms=link_loss.KEEPOUT_TTL_MS,
        reliable=True,
        corr_id=None,
    )
    link_loss.on_expired(flow, VID, Expired(envelope=env, reason="LINK_NO_ACK"))  # type: ignore[arg-type]
    assert codes(flow) == ["KEEPOUTS_NOT_INSTALLED"]
    assert flow.link_watch[VID].sent is None
