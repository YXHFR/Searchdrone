"""``flight.tlog``: the ground station's MAVLink channel recorded per mission."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sd_ground.context import Ground
from sd_ground.services.files import close_files
from sd_ground.services.missions import TLOG_NAME, MissionFlow
from sd_schemas.api import Mission
from sd_schemas.base import new_id
from sd_schemas.mission import MissionState
from sd_schemas.safety import SafetyLimits

NOW = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.UTC)


@dataclass
class StubMav:
    """Records the calls; ``ok`` is whether the transport can record (a fake cannot)."""

    ok: bool = True
    started: list[Path] = field(default_factory=list[Path])
    stopped: int = 0

    def start_tlog(self, path: Path) -> bool:
        if self.ok:
            self.started.append(path)
            path.write_bytes(b"\x00" * 8)
        return self.ok

    def stop_tlog(self) -> None:
        self.stopped += 1


def mission(status: MissionState, *, activated: bool = True) -> Mission:
    return Mission(
        id=new_id(),
        vehicle_id="sim-01",
        title="test",
        status=status,
        paused_by=None,
        returning_cause=None,
        state_seq=1,
        limits=SafetyLimits(),
        active_charter_id=None,
        autonomy_mode=None,
        ai_enabled=True,
        created_by="operator:demo",
        created_at=NOW,
        activated_at=NOW if activated else None,
        landed_at=None,
        closed_at=None,
        closed_reason=None,
        notes=None,
    )


def flow(ground: Ground, mav: StubMav | None) -> MissionFlow:
    repo: Any = None
    return MissionFlow(ground, repo, mav=mav)  # pyright: ignore[reportArgumentType]


def test_records_from_the_first_activation_once(ground: Ground) -> None:
    mav = StubMav()
    f = flow(ground, mav)
    m = mission(MissionState.ACTIVE)
    f._follow_tlog(m)  # pyright: ignore[reportPrivateUsage]
    f._follow_tlog(m)  # pyright: ignore[reportPrivateUsage]
    assert mav.started == [f.mission_dir(str(m.id)) / TLOG_NAME]
    assert f.stop_tlog(str(m.id)) and mav.stopped == 1
    assert not f.stop_tlog(str(m.id))  # only once


def test_not_before_activation_nor_after_closing(ground: Ground) -> None:
    mav = StubMav()
    f = flow(ground, mav)
    f._follow_tlog(mission(MissionState.READY, activated=False))  # pyright: ignore[reportPrivateUsage]
    f._follow_tlog(mission(MissionState.CLOSED))  # pyright: ignore[reportPrivateUsage]
    assert mav.started == []


def test_a_restarted_ground_station_records_again(ground: Ground) -> None:
    """After a restart the mission is still flying: the new process records again (the
    connection appends to the same file)."""
    mav = StubMav()
    m = mission(MissionState.RETURNING)
    flow(ground, mav)._follow_tlog(m)  # pyright: ignore[reportPrivateUsage]
    flow(ground, mav)._follow_tlog(m)  # pyright: ignore[reportPrivateUsage]
    assert len(mav.started) == 2 and mav.started[0] == mav.started[1]


def test_without_mav_or_a_recording_transport_nothing_happens(ground: Ground) -> None:
    m = mission(MissionState.ACTIVE)
    flow(ground, None)._follow_tlog(m)  # pyright: ignore[reportPrivateUsage]
    fake = StubMav(ok=False)
    f = flow(ground, fake)
    f._follow_tlog(m)  # pyright: ignore[reportPrivateUsage]
    assert not f.stop_tlog(str(m.id)) and fake.stopped == 0


@dataclass
class FakeRepo:
    files: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    async def mission_export(self, _mid: str) -> dict[str, Any]:
        return {}

    async def track(self, _mid: str) -> list[Any]:
        return []

    async def add_file(self, **row: Any) -> None:
        self.files.append(row)


async def test_close_files_stops_and_records_the_tlog(ground: Ground) -> None:
    mav = StubMav()
    repo = FakeRepo()
    f = MissionFlow(ground, repo, mav=mav)  # pyright: ignore[reportArgumentType]
    m = mission(MissionState.LANDED)
    f._follow_tlog(m)  # pyright: ignore[reportPrivateUsage]
    await close_files(f, m)
    assert mav.stopped == 1
    tlog = [r for r in repo.files if r["rel_path"] == TLOG_NAME]
    assert len(tlog) == 1 and tlog[0]["kind"] == "tlog"


def test_a_file_that_cannot_open_is_tried_once_and_never_raises(ground: Ground) -> None:
    @dataclass
    class Failing(StubMav):
        calls: int = 0

        def start_tlog(self, path: Path) -> bool:
            self.calls += 1
            raise PermissionError(13, "Permission denied")

    mav = Failing()
    f = flow(ground, mav)
    m = mission(MissionState.ACTIVE)
    f._follow_tlog(m)  # pyright: ignore[reportPrivateUsage]
    f._follow_tlog(m)  # pyright: ignore[reportPrivateUsage]
    assert mav.calls == 1
    assert not f.stop_tlog(str(m.id))


def test_closing_files_ends_the_recording_for_good(ground: Ground) -> None:
    """The flow's tick may still see the mission open while its files close."""
    mav = StubMav()
    f = flow(ground, mav)
    m = mission(MissionState.LANDED)
    f._follow_tlog(m)  # pyright: ignore[reportPrivateUsage]
    assert f.stop_tlog(str(m.id))
    f._follow_tlog(m)  # pyright: ignore[reportPrivateUsage]
    assert len(mav.started) == 1
