"""The demo's stack as processes and the pilot that flies it, for the end-to-end tests.

``demo_stack`` starts the whole stack as ``just demo`` does (``Stack(services=True)``:
Postgres with a fresh ``searchdrone_i<N>``, ``sd-ground``, SITL with builtin physics,
linksim, ``sd-onboard``, ``sd-brain`` with the scripted commander and SC01's script, and
on instance 0 the test picture through ``video_relay`` into MediaMTX). ``pilot`` is the
operator: a CLI pilot token on the panel API over real HTTP, and the live WebSocket with
pings (which keeps the operator present). Nothing runs in the test's own process.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
import struct
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import asyncpg
import httpx
import pytest
import yaml
from websockets.asyncio.client import connect

from sd_safety.targets import Plane, distance_m
from sd_schemas.base import new_id
from sd_sim import config, paths
from sd_sim.invariants import Sample
from sd_sim.launch import Stack, read_env_value
from sd_sim.pytest_plugin import worker_instance
from sd_sim.services import DB_HOST, DB_PORT, VEHICLE_ID, database_name
from sd_sim.util import new_run_dir

ROOT = paths.REPO
SCRIPT = ROOT / "simulation/scenarios/scripts/SC01.yaml"
FIXTURE: dict[str, Any] = yaml.safe_load(
    (ROOT / "simulation/scenarios/demo_fixture.yaml").read_text(encoding="utf-8")
)
H = {"X-SD-Client": "panel", "Content-Type": "application/json"}
ALT_M = 40.0
SPACING_M = 30.0
MIN_COVERAGE = 0.95
DEMO_LIMIT_S = 600.0  # D5-D9 (take-off accepted to disarmed)
CLOSE_LIMIT_S = 15.0  # CLOSED after disarming (closes after 10 s)
TERMINAL = frozenset(
    {
        "rejected_by_gate",
        "rejected_by_operator",
        "approval_expired",
        "rejected_onboard",
        "done",
        "failed",
        "superseded",
        "expired",
        "cancelled",
    }
)


def require_video() -> bool:
    """CI's end-to-end job runs MediaMTX and sets ``SD_SIM__REQUIRE_VIDEO=1``."""
    return os.environ.get("SD_SIM__REQUIRE_VIDEO") == "1"


@pytest.fixture
def demo_stack() -> Iterator[Stack]:
    """``just demo``'s stack with builtin physics, the scripted commander and a test rig
    pilot (the only differences from the owner's demo: physics, AI and the CLI pilot)."""
    instance = worker_instance()
    if not paths.arducopter_bin().is_file():
        pytest.skip(f"{paths.arducopter_bin()} saknas (just sim-setup)")
    values = {
        "physics": "builtin",
        "gui": "false",
        "video": "test",
        "brain": "scripted",
        "script": str(SCRIPT),
        "instance": str(instance),
        "open.browser": "false",
    }
    cfg = config.load(values, demo=True)
    cfg.open_browser = False
    stack = Stack(cfg, new_run_dir(instance, "sc01-demo"), services=True, test_rig=True)
    try:
        stack.up()
        yield stack
    finally:
        stack.down()


# --- the pilot ------------------------------------------------------------------------


@dataclass
class Demo:
    stack: Stack
    panel: httpx.AsyncClient
    plane: Plane
    mid: str = ""
    track: list[Sample] = field(default_factory=list[Sample])

    _live: dict[str, Any] = field(default_factory=dict[str, Any])

    @property
    def tel(self) -> dict[str, Any]:
        return self._live.get("telemetry") or {}

    def from_home_m(self) -> float:
        t = self.tel
        return distance_m(self.plane.lat0, self.plane.lon0, t["lat"], t["lon"])

    async def until(self, cond: Callable[[], Any], limit_s: float, what: str) -> float:
        start = time.monotonic()
        while True:
            value = cond()
            if asyncio.iscoroutine(value):
                value = await value
            if value:
                return time.monotonic() - start
            if time.monotonic() - start > limit_s:
                t = self.tel
                raise AssertionError(
                    f"{what}: inte inom {limit_s} s (läge {t.get('flight_mode')}, "
                    f"{t.get('alt_rel_home_m')} m, armerad {t.get('armed')})"
                )
            await asyncio.sleep(0.1)

    async def post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        return await self.panel.post(path, json=body, headers=H)

    async def get(self, path: str, **params: str) -> Any:
        res = await self.panel.get(path, params=params or None)
        assert res.status_code == 200, f"{path}: {res.status_code} {res.text}"
        return res.json()

    async def status(self) -> str:
        return str((await self.get(f"/api/v1/missions/{self.mid}"))["mission"]["status"])

    async def say(self, text: str) -> None:
        res = await self.post(
            f"/api/v1/missions/{self.mid}/chat", {"client_msg_id": str(new_id()), "text": text}
        )
        assert res.status_code == 201, res.text

    async def commands(self) -> list[dict[str, Any]]:
        return list(await self.get(f"/api/v1/missions/{self.mid}/commands"))

    async def wait_command(
        self, skill: str, status: str, limit_s: float, *, or_later: tuple[str, ...] = ()
    ) -> dict[str, Any]:
        """The AI's newest ``skill`` order reaches ``status`` (or one of ``or_later``, the
        statuses after it, when a poll can miss a short one); another end fails."""
        found: dict[str, Any] = {}
        wanted = (status, *or_later)

        async def reached() -> bool:
            cmds = [
                c
                for c in await self.commands()
                if c["skill"] == skill and c["issuer_kind"] == "brain"
            ]
            if not cmds:
                return False
            c = max(cmds, key=lambda c: c["created_at"])
            if c["status"] in TERMINAL and c["status"] not in wanted:
                raise AssertionError(f"{skill}: {c['status']} {c['status_reason']}")
            found.update(c)
            return c["status"] in wanted

        await self.until(reached, limit_s, f"{skill} {status}")
        return found

    async def gate(self, command_id: str) -> list[dict[str, Any]]:
        return list(
            await self.get(f"/api/v1/missions/{self.mid}/gate_decisions", command_id=command_id)
        )


async def _watch(demo: Demo) -> None:
    """The drone five times a second (the track): the ground station's view, and while the
    ground station does not answer (a restart) the test rig's own channel to the flight
    controller (serial0)."""
    while True:
        sample: Sample | None = None
        try:
            res = await demo.panel.get(f"/api/v1/vehicles/{VEHICLE_ID}/state")
            if res.status_code == 200:
                demo._live = res.json()
                t = demo._live.get("telemetry")
                if t:
                    sample = Sample(
                        t=time.monotonic(),
                        lat=t["lat"],
                        lon=t["lon"],
                        alt_m=t["alt_rel_home_m"],
                        groundspeed_ms=t["groundspeed_ms"],
                        mode=t["flight_mode"],
                        armed=t["armed"],
                    )
        except (httpx.HTTPError, ValueError):
            h: dict[str, Any] = demo.stack.harness_state()
            if h.get("heartbeat"):
                sample = Sample(
                    t=time.monotonic(),
                    lat=float(h["lat"]),
                    lon=float(h["lon"]),
                    alt_m=float(h["rel_alt_m"]),
                    groundspeed_ms=float(h["ground_speed_ms"]),
                    mode=str(h["mode"]),
                    armed=bool(h["armed"]),
                )
        if sample is not None:
            demo.track.append(sample)
        await asyncio.sleep(0.2)


async def _present(port: int, token: str) -> None:
    """The pilot's live connection with pings: the operator is present (no SC19 hold)."""
    url = f"ws://127.0.0.1:{port}/ws/v1/live"
    headers = {"Authorization": f"Bearer {token}", "X-SD-Client": "panel"}
    while True:
        with contextlib.suppress(OSError, Exception):
            async with connect(url, additional_headers=headers, proxy=None) as ws:
                await ws.recv()  # hello
                await ws.send(
                    json.dumps({"op": "subscribe", "topics": ["system"], "vehicle_id": VEHICLE_ID})
                )

                async def drain() -> None:
                    async for _ in ws:
                        pass

                reader = asyncio.ensure_future(drain())
                try:
                    while not reader.done():
                        await ws.send(json.dumps({"op": "ping"}))
                        await asyncio.sleep(5.0)
                finally:
                    reader.cancel()
        await asyncio.sleep(1.0)


@contextlib.asynccontextmanager
async def pilot(stack: Stack) -> AsyncIterator[Demo]:
    token = (stack.secrets_dir / "pilot_token").read_text(encoding="utf-8").strip()
    port = stack.ports["ground_panel"]
    home = FIXTURE["home"]
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}",
        headers={"Authorization": f"Bearer {token}"},
        trust_env=False,
        timeout=10,
    ) as panel:
        demo = Demo(stack, panel, Plane(float(home["lat"]), float(home["lon"])))
        tasks = [asyncio.ensure_future(_watch(demo)), asyncio.ensure_future(_present(port, token))]
        try:
            await demo.until(lambda: bool(demo.tel), 10, "telemetri från markstationen")
            yield demo
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


# --- checks on the database and the mission folder --------------------------------------


async def _db(stack: Stack) -> asyncpg.Connection:
    password = read_env_value(stack.repo / ".env", "SD_GROUND__DB_PASSWORD")
    return await asyncpg.connect(
        host=DB_HOST,
        port=DB_PORT,
        user="sd_owner",
        password=password,
        database=database_name(stack.cfg.instance),
    )


TABLES = (
    "missions",
    "charters",
    "chat_messages",
    "commands",
    "gate_decisions",
    "events",
    "telemetry",
)


async def check_database(stack: Stack, mid: str) -> dict[str, Any]:
    """F-08 and D10: one mission and vehicle in every table, telemetry >= 2 Hz with no gap
    over 3 s from take-off to landing, a ground gate decision for every order."""
    conn = await _db(stack)
    try:
        for table in TABLES:
            key = "id" if table == "missions" else "mission_id"
            rows = await conn.fetch(
                f"SELECT DISTINCT {key}::text AS m, vehicle_id FROM {table} WHERE {key} = $1::uuid",  # noqa: S608 - fixed table names
                mid,
            )
            assert rows, f"{table}: inga rader för uppdraget"
            assert {(r["m"], r["vehicle_id"]) for r in rows} == {(mid, VEHICLE_ID)}, table
            stray = await conn.fetchval(
                f"SELECT count(*) FROM {table} WHERE vehicle_id <> $1",  # noqa: S608
                VEHICLE_ID,
            )
            assert stray == 0, f"{table}: rader med annat vehicle_id"
        status = await conn.fetchval("SELECT status FROM missions WHERE id = $1::uuid", mid)
        assert status == "CLOSED"
        ungated = await conn.fetch(
            "SELECT c.skill FROM commands c WHERE c.mission_id = $1::uuid AND NOT EXISTS "
            "(SELECT 1 FROM gate_decisions g WHERE g.command_id = c.id)",
            mid,
        )
        assert not ungated, f"order utan grindbeslut: {[r['skill'] for r in ungated]}"
        times = [
            r["t"]
            for r in await conn.fetch(
                "SELECT t FROM telemetry WHERE mission_id = $1::uuid AND armed ORDER BY t", mid
            )
        ]
        assert len(times) > 10, "ingen telemetri under flygningen"
        span = (times[-1] - times[0]).total_seconds()
        gaps = [(b - a).total_seconds() for a, b in itertools.pairwise(times)]
        assert max(gaps) <= 3.0, f"lucka i telemetrin: {max(gaps):.1f} s"
        assert len(times) / span >= 2.0, f"telemetri {len(times) / span:.2f} Hz"
        # D8: no telemetry sample from arming to disarming outside the charter's area.
        outside = await conn.fetchval(
            "SELECT count(*) FROM telemetry t JOIN charters c ON c.mission_id = t.mission_id "
            "AND c.status = 'approved' WHERE t.mission_id = $1::uuid AND t.armed "
            "AND NOT ST_Within(ST_Force2D(t.pos), c.area)",
            mid,
        )
        assert outside == 0, f"{outside} telemetripunkter utanför ramens område"
        charter = await conn.fetchrow(
            "SELECT approved_at, approved_by, autonomy_mode, fence_verified_at FROM charters "
            "WHERE mission_id = $1::uuid AND status = 'approved'",
            mid,
        )
        assert charter is not None
        return dict(charter)
    finally:
        await conn.close()


def _boxes(data: bytes, start: int = 0, end: int | None = None) -> Iterator[tuple[str, int, int]]:
    """ISO BMFF boxes: (type, payload start, box end)."""
    end = len(data) if end is None else end
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack(">I4s", data[pos : pos + 8])
        head = 8
        if size == 1:
            size = struct.unpack(">Q", data[pos + 8 : pos + 16])[0]
            head = 16
        elif size == 0:
            size = end - pos
        if size < head:
            return
        yield kind.decode("latin-1"), pos + head, pos + size
        pos += size


def _child(data: bytes, start: int, end: int, path: str) -> tuple[int, int] | None:
    for name in path.split("/"):
        hit = next(((s, e) for k, s, e in _boxes(data, start, end) if k == name), None)
        if hit is None:
            return None
        start, end = hit
    return start, end


def fmp4_h264_seconds(path: Path) -> float:
    """The video length of a fragmented MP4 that MediaMTX recorded, or 0 if it has no
    H.264 track (no ffprobe in the SITL image: the boxes are read directly)."""
    data = path.read_bytes()
    trak = _child(data, 0, len(data), "moov/trak")
    if trak is None:
        return 0.0
    stsd = _child(data, *trak, "mdia/minf/stbl/stsd")
    mdhd = _child(data, *trak, "mdia/mdhd")
    if stsd is None or mdhd is None:
        return 0.0
    entry = data[stsd[0] + 8 + 4 : stsd[0] + 8 + 8]  # version/flags, count, size, type
    if entry not in (b"avc1", b"avc3"):
        return 0.0
    version = data[mdhd[0]]
    timescale = struct.unpack(">I", data[mdhd[0] + (20 if version == 1 else 12) :][:4])[0]
    total = 0
    for kind, s, e in _boxes(data):
        if kind != "moof":
            continue
        for k2, s2, e2 in _boxes(data, s, e):
            if k2 != "traf":
                continue
            default = 0
            for k3, s3, _e3 in _boxes(data, s2, e2):
                flags = int.from_bytes(data[s3 + 1 : s3 + 4], "big")
                if k3 == "tfhd":
                    off = s3 + 8 + (8 if flags & 0x1 else 0) + (4 if flags & 0x2 else 0)
                    if flags & 0x8:
                        default = struct.unpack(">I", data[off : off + 4])[0]
                elif k3 == "trun":
                    count = struct.unpack(">I", data[s3 + 4 : s3 + 8])[0]
                    off = s3 + 8 + (4 if flags & 0x1 else 0) + (4 if flags & 0x4 else 0)
                    per = 4 * sum(bool(flags & b) for b in (0x100, 0x200, 0x400, 0x800))
                    if flags & 0x100:
                        total += sum(
                            struct.unpack(">I", data[off + i * per : off + i * per + 4])[0]
                            for i in range(count)
                        )
                    else:
                        total += default * count
    return total / timescale if timescale else 0.0


def check_folder(stack: Stack, mid: str, flight_s: float) -> list[str]:
    """F-09: the mission folder's files. Returns the criteria not checked (no video)."""
    folder = stack.stack_paths().data_dir / "missions" / mid
    names = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    for want in ("flight.tlog", "events.jsonl", "mission.json"):
        assert want in names, f"{want} saknas i uppdragsmappen: {sorted(names)}"
    assert any(n.endswith(".BIN") for n in names), f"*.BIN saknas: {sorted(names)}"
    if not stack.video_on or stack.ground_process_env()["SD_GROUND__VIDEO__ENABLED"] != "true":
        assert not require_video(), "video krävs (SD_SIM__REQUIRE_VIDEO=1) men MediaMTX kör inte"
        return ["F-09 video (MediaMTX kör inte; just video-up)"]
    videos = sorted(folder.glob("video/**/*.mp4"))
    assert videos, f"ingen video i uppdragsmappen: {sorted(names)}"
    seconds = sum(fmp4_h264_seconds(v) for v in videos)
    assert seconds >= flight_s - 5.0, f"video {seconds:.0f} s, uppdraget {flight_s:.0f} s"
    return []


DEMO_REQUEST = (
    "Flyg till P1 och sök av en ruta på 100 gånger 100 meter där, på 40 meters höjd. "
    "Landa hemma sen."
)


async def airborne(d: Demo) -> None:
    """The demo's opening without its checks (D2-D5): the request, "Godkänn ram" in mode
    B, the onboard checks, "Fritt runt drönaren", and the AI's take-off to 40 m."""
    res = await d.post(f"/api/v1/vehicles/{VEHICLE_ID}/missions/open", {})
    assert res.status_code == 200, res.text
    d.mid = str(res.json()["id"])
    await d.say(DEMO_REQUEST)
    cards: list[dict[str, Any]] = []

    async def card() -> bool:
        cards[:] = [
            a
            for a in await d.get(f"/api/v1/missions/{d.mid}/approvals", status="pending")
            if a["kind"] == "charter"
        ]
        return bool(cards)

    await d.until(card, 10, "AI:ns ramkort")
    res = await d.post(
        f"/api/v1/approvals/{cards[0]['id']}/decision",
        {
            "decision": "approve",
            "subject_version": cards[0]["subject_version"],
            "autonomy_mode": "WITHIN_CHARTER",
        },
    )
    assert res.status_code == 200, res.text
    await d.until(is_status(d, "READY"), 90, "uppdraget READY")
    await d.until(
        lambda: (d._live.get("snapshot") or {}).get("preflight_ok") is True,
        90,
        "ombordsdatorns förflygningskontroll",
    )
    res = await d.post(f"/api/v1/missions/{d.mid}/clearance", {})
    assert res.status_code == 200, res.text
    await d.wait_command("takeoff", "done", 90)


def is_status(d: Demo, status: str) -> Callable[[], Any]:
    async def check() -> bool:
        return await d.status() == status

    return check


def still_for(samples: list[Sample], seconds: float) -> bool:
    """Consecutive samples (0.2 s apart) covering ``seconds``."""
    run_start: float | None = None
    last = None
    for s in samples:
        if last is None or s.t - last > 0.6:
            run_start = s.t
        last = s.t
        if run_start is not None and s.t - run_start >= seconds:
            return True
    return False


def box_enu() -> list[tuple[float, float]]:
    box = FIXTURE["survey_box"]
    p = next(p for p in FIXTURE["pins"] if p["id"] == box["center_pin"])
    w, h = (float(x) / 2 for x in box["size_m"])
    cx, cy = float(p["east_m"]), float(p["north_m"])
    return [(cx - w, cy - h), (cx + w, cy - h), (cx + w, cy + h), (cx - w, cy + h)]
