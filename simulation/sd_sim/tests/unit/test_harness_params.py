"""Harness.read_all_params with a fake MAVLink connection (no SITL needed)."""

from __future__ import annotations

from collections import deque
from typing import Any

import pytest

from sd_sim import harness
from sd_sim.errors import InfraError


class Msg:
    def __init__(self, index: int, count: int, name: str, value: float, src: int = 1) -> None:
        self.param_index, self.param_count = index, count
        self.param_id, self.param_value = name.encode(), value
        self._src = src

    def get_type(self) -> str:
        return "PARAM_VALUE"

    def get_srcSystem(self) -> int:  # noqa: N802 - pymavlink's name
        return self._src

    def get_srcComponent(self) -> int:  # noqa: N802 - pymavlink's name
        return 1


class FakeMav:
    def __init__(self, conn: FakeConn) -> None:
        self.conn = conn

    def param_request_list_send(self, *_a: Any) -> None:
        self.conn.queue.extend(m for m in self.conn.all if m.param_index not in self.conn.drop)

    def param_request_read_send(self, _s: int, _c: int, _name: bytes, index: int) -> None:
        self.conn.reread.append(index)
        if self.conn.answer_rereads:
            self.conn.queue.append(self.conn.all[index])

    def heartbeat_send(self, *_a: Any) -> None:
        pass


class FakeConn:
    def __init__(self, count: int, drop: set[int], answer_rereads: bool = True) -> None:
        self.all = [Msg(i, count, f"P{i}", float(i)) for i in range(count)]
        self.drop, self.answer_rereads = drop, answer_rereads
        self.queue: deque[Msg] = deque([Msg(0, count, "OTHER", 9.0, src=2)])
        self.reread: list[int] = []
        self.mav = FakeMav(self)

    def recv_match(self, **_kw: Any) -> Msg | None:
        return self.queue.popleft() if self.queue else None


def make(conn: FakeConn) -> harness.Harness:
    h = harness.Harness.__new__(harness.Harness)
    h.state = harness.VehicleState()
    h._conn = conn
    h._next_heartbeat = 0.0
    h._streams_requested = True
    h.listeners = []
    return h


def test_reads_everything_and_rerequests_missing() -> None:
    conn = FakeConn(30, drop={3, 17})
    values = make(conn).read_all_params(timeout=10)
    assert len(values) == 30 and values["P17"] == 17.0
    assert sorted(conn.reread) == [3, 17]


def test_timeout_is_infra_error() -> None:
    conn = FakeConn(5, drop={2}, answer_rereads=False)
    with pytest.raises(InfraError, match="4 av 5"):
        make(conn).read_all_params(timeout=1.5)


def test_read_param_by_name() -> None:
    conn = FakeConn(5, set(), answer_rereads=False)

    def answer(*_a: Any) -> None:
        conn.queue.extend([Msg(1, 5, "OTHER", 9.0), Msg(2, 5, "P2", 2.0)])

    conn.mav.param_request_read_send = answer  # type: ignore[method-assign]
    assert make(conn).read_param("P2", timeout=3) == 2.0


def test_read_param_drops_a_stale_answer() -> None:
    """An answer that was already waiting (from an earlier request) does not count."""
    conn = FakeConn(5, set(), answer_rereads=False)
    conn.queue.append(Msg(2, 5, "P2", 2.0))
    with pytest.raises(InfraError, match="P2"):
        make(conn).read_param("P2", timeout=0.5)


def test_read_param_timeout_is_infra_error() -> None:
    with pytest.raises(InfraError, match="MISSING"):
        make(FakeConn(5, set(), answer_rereads=False)).read_param("MISSING", timeout=0.5)


class Position:
    relative_alt, lat, lon, vx, vy = 20_000, -353632610, 1491652300, 300, 400

    def get_type(self) -> str:
        return "GLOBAL_POSITION_INT"

    def get_srcSystem(self) -> int:  # noqa: N802 - pymavlink's name
        return 1

    def get_srcComponent(self) -> int:  # noqa: N802 - pymavlink's name
        return 1


def test_position_speed_and_listeners() -> None:
    h = make(FakeConn(1, set()))
    seen: list[str] = []
    h.listeners.append(lambda kind, _s: seen.append(kind))
    h._apply(Position())
    assert h.state.rel_alt_m == 20.0
    assert (h.state.lat, h.state.lon) == (-35.363261, 149.16523)
    assert h.state.ground_speed_ms == 5.0
    assert seen == ["GLOBAL_POSITION_INT"]
