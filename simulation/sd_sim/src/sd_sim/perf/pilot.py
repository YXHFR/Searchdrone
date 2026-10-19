"""The operator of ``just perf``: flies the demo on a running stack as the pilot would.

A port of the pilot in ``simulation/tests/scenarios/demo_rig.py`` (test-only) to plain
threads, so the synchronous perf driver can use it: a CLI pilot token on the panel API
over real HTTP (loopback, never a proxy), a watcher that reads the vehicle's state five
times a second, and the live WebSocket with pings, which keeps the operator present (no
SC19 hold). The demo's opening is the same as SC01's, plus the pin P1 on the map that a
real AI (unlike the scripted one) needs: the request in the chat, "Godkänn ram" in mode B,
the onboard checks, "Fritt runt drönaren" and the AI's take-off.

``chat_latencies_s`` reads the instance's database afterwards: for every operator chat
message the AI answered, the time from the message to its first reply, an order or a chat
line (P8). The demo's only message is the request, which the AI answers with a charter
proposal and a chat line, not an order: the take-off comes later on the clearance and
replies to no message, so counting orders alone would leave P8 empty.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from sd_sim import fixture
from sd_sim.errors import InfraError, VehicleError
from sd_sim.launch import read_env_value
from sd_sim.services import DB_HOST, DB_PORT, VEHICLE_ID, database_name
from sd_sim.util import LOOPBACK_OPENER

if TYPE_CHECKING:
    from sd_sim.launch import Stack

__all__ = [
    "CHAT_LATENCY_SQL",
    "DEMO_REQUEST",
    "HttpPanel",
    "Panel",
    "Pilot",
    "chat_latencies_s",
    "flight_done",
    "is_brake",
    "latencies_s",
    "new_uuid7",
    "not_brake",
    "remaining_s",
]

# SC01's request (D3), word for word as in demo_rig.
DEMO_REQUEST = (
    "Flyg till P1 och sök av en ruta på 100 gånger 100 meter där, på 40 meters höjd. "
    "Landa hemma sen."
)
DEMO_PIN = "P1"  # the pin the request names; the AI only knows the pins on the map
EARTH_RADIUS_M = 6_378_137.0
HEADERS = {"X-SD-Client": "panel", "Content-Type": "application/json"}
POLL_S = 0.2  # the watcher: five times a second, as demo_rig
PING_S = 5.0  # the live WebSocket's pings (demo_rig's _present)
TELEMETRY_S = 10.0  # the first telemetry from the ground station after start()
HTTP_TIMEOUT_S = 10.0
# A command status after which it never changes (demo_rig.TERMINAL).
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
DONE_STATUSES = frozenset({"LANDED", "CLOSED"})  # the mission after the flight

# P8: every operator message the AI answered, and its first reply of either kind: an order
# (``commands.in_reply_to``) or a chat line (``chat_messages.in_reply_to``, role brain).
CHAT_LATENCY_SQL = """
SELECT m.created_at AS said_at, min(r.created_at) AS replied_at
FROM chat_messages m
JOIN (
    SELECT c.in_reply_to, c.created_at FROM commands c WHERE c.issuer_kind = 'brain'
    UNION ALL
    SELECT b.in_reply_to, b.created_at FROM chat_messages b WHERE b.role = 'brain'
) r ON r.in_reply_to = m.id
WHERE m.role = 'operator' AND ($1::uuid IS NULL OR m.mission_id = $1::uuid)
GROUP BY m.id, m.created_at
ORDER BY m.created_at
"""


# --- pure parts ---------------------------------------------------------------------------


def new_uuid7() -> uuid.UUID:
    """A UUIDv7 (the API's ids must be v7; RFC 9562): 48 bits of Unix milliseconds, the
    version, 74 random bits and the variant."""
    ms = time.time_ns() // 1_000_000
    value = (ms & ((1 << 48) - 1)) << 80 | int.from_bytes(os.urandom(10), "big")
    value = (value & ~(0xF << 76)) | (0x7 << 76)
    value = (value & ~(0x3 << 62)) | (0x2 << 62)
    return uuid.UUID(int=value)


def is_brake(mode: str | None) -> bool:
    return mode == "BRAKE"


def not_brake(mode: str | None) -> bool:
    """A known flight mode other than BRAKE (no telemetry is not "not braking")."""
    return mode is not None and mode != "BRAKE"


def flight_done(live: dict[str, Any], status: str | None, *, seen_airborne: bool) -> bool:
    """The flight is over: the mission is LANDED or CLOSED, or the drone that was in the
    air stands disarmed on the ground."""
    if status in DONE_STATUSES:
        return True
    tel = live.get("telemetry") or {}
    return (
        seen_airborne
        and tel.get("armed") is False
        and tel.get("landed_state") == "on_ground"
        and not live.get("airborne")
    )


def latencies_s(rows: Iterable[tuple[datetime | None, datetime | None]]) -> list[float]:
    """(message created_at, first reply created_at) rows as seconds, in row order."""
    return [
        (ordered - said).total_seconds()
        for said, ordered in rows
        if said is not None and ordered is not None
    ]


def remaining_s(deadline: float, now: float | None = None) -> float:
    """The seconds left until a ``time.monotonic()`` deadline, never below zero."""
    return max(0.0, deadline - (time.monotonic() if now is None else now))


def pin_latlon(fx: fixture.Fixture, name: str) -> tuple[float, float]:
    """A fixture pin as (lat, lon): metres east/north of home on a flat earth, as the
    world is (a few hundred metres, so the error is millimetres)."""
    pin, home = fx.pins[name], fx.home
    lat = home.lat + math.degrees(pin.north_m / EARTH_RADIUS_M)
    lon = home.lon + math.degrees(pin.east_m / (EARTH_RADIUS_M * math.cos(math.radians(home.lat))))
    return lat, lon


def newest_brain_command(commands: Sequence[dict[str, Any]], skill: str) -> dict[str, Any] | None:
    mine = [c for c in commands if c.get("skill") == skill and c.get("issuer_kind") == "brain"]
    return max(mine, key=lambda c: str(c.get("created_at"))) if mine else None


# --- the panel API ------------------------------------------------------------------------


class Panel(Protocol):
    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> tuple[int, Any]:
        """(HTTP status, decoded JSON body or None)."""
        ...


class HttpPanel:
    """The panel API with a bearer token, on loopback only."""

    def __init__(self, base_url: str, token: str, timeout_s: float = HTTP_TIMEOUT_S) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.timeout_s = timeout_s

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> tuple[int, Any]:
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(  # noqa: S310 - base_url is http://127.0.0.1
            url,
            data=data,
            method=method,
            headers={"Authorization": f"Bearer {self._token}", **HEADERS},
        )
        try:
            with LOOPBACK_OPENER.open(req, timeout=self.timeout_s) as resp:
                status, raw = int(resp.status), resp.read()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read()
        except (urllib.error.URLError, OSError) as exc:
            raise InfraError(f"sd-ground svarar inte ({method} {path}): {exc}") from None
        try:
            return status, json.loads(raw) if raw else None
        except ValueError:
            return status, raw.decode("utf-8", "replace")


# --- the pilot ----------------------------------------------------------------------------


class Pilot:
    """The operator on a running ``Stack`` (``test_rig=True`` for the pilot token).

    Use as a context manager (or ``start()``/``close()``): the watcher and the present
    operator's WebSocket run in daemon threads until ``close()``."""

    def __init__(self, stack: Stack, *, panel: Panel | None = None) -> None:
        self.stack = stack
        self.port = int(stack.ports["ground_panel"])
        self._token = ""
        if panel is None:
            path = stack.secrets_dir / "pilot_token"
            try:
                self._token = path.read_text(encoding="utf-8").strip()
            except OSError:
                raise InfraError(
                    "Pilotens token saknas (stacken måste startas med test_rig=True)."
                ) from None
            panel = HttpPanel(f"http://127.0.0.1:{self.port}", self._token)
        self.panel = panel
        self.mid = ""
        self._live: dict[str, Any] = {}
        self._seen_airborne = False
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # --- life cycle -----------------------------------------------------------------------

    def __enter__(self) -> Pilot:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def start(self, telemetry_s: float = TELEMETRY_S) -> Pilot:
        """Start the watcher and the live WebSocket; wait for the first telemetry."""
        if self._threads:
            return self
        self._stop.clear()
        self._threads = [threading.Thread(target=self._watch, name="perf-watch", daemon=True)]
        if self._token:
            self._threads.append(
                threading.Thread(target=self._present, name="perf-present", daemon=True)
            )
        try:
            for thread in self._threads:
                thread.start()
            deadline = time.monotonic() + telemetry_s
            while not self.tel:
                if remaining_s(deadline) <= 0:
                    raise InfraError(
                        f"Ingen telemetri från markstationen inom {telemetry_s:.0f} s."
                    )
                time.sleep(0.1)
        except BaseException:
            # __exit__ never runs when __enter__ raises: stop what was started, so the
            # operator is not left present and a later start() waits for telemetry again.
            self.close()
            raise
        return self

    def close(self) -> None:
        """Stop the threads (idempotent, never raises)."""
        self._stop.set()
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=PING_S + 2.0)
        self._threads = []

    def _watch(self) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(InfraError):
                self.refresh()
            self._stop.wait(POLL_S)

    def _present(self) -> None:
        """The pilot's live connection with pings: the operator is present."""
        from websockets.sync.client import connect

        url = f"ws://127.0.0.1:{self.port}/ws/v1/live"
        headers = {"Authorization": f"Bearer {self._token}", "X-SD-Client": "panel"}
        subscribe = {"op": "subscribe", "topics": ["system"], "vehicle_id": VEHICLE_ID}
        while not self._stop.is_set():
            with (
                contextlib.suppress(Exception),
                connect(url, additional_headers=headers, proxy=None, open_timeout=5) as ws,
            ):
                ws.recv(timeout=5)  # hello
                ws.send(json.dumps(subscribe))
                next_ping = 0.0
                while not self._stop.is_set():
                    if time.monotonic() >= next_ping:
                        ws.send(json.dumps({"op": "ping"}))
                        next_ping = time.monotonic() + PING_S
                    with contextlib.suppress(TimeoutError):
                        ws.recv(timeout=0.5)  # drain the events
            self._stop.wait(1.0)

    # --- what the drone does --------------------------------------------------------------

    def refresh(self) -> dict[str, Any]:
        """Read the vehicle's state once (the watcher does this five times a second)."""
        status, body = self.panel.request("GET", f"/api/v1/vehicles/{VEHICLE_ID}/state")
        if status == 200 and isinstance(body, dict):
            self._live = dict(body)  # pyright: ignore[reportUnknownArgumentType]
            if self._live.get("airborne"):
                self._seen_airborne = True
        return self._live

    def _current(self) -> dict[str, Any]:
        return self._live if self._threads else self.refresh()

    @property
    def tel(self) -> dict[str, Any]:
        return self._live.get("telemetry") or {}

    def mode(self) -> str | None:
        mode = (self._current().get("telemetry") or {}).get("flight_mode")
        return None if mode is None else str(mode)

    def airborne(self) -> bool:
        """The ground station's strict definition (``VehicleLive.airborne``)."""
        return bool(self._current().get("airborne"))

    def _until(self, cond: Callable[[], object], timeout_s: float, what: str) -> float:
        start = time.monotonic()
        while True:
            if cond():
                return time.monotonic() - start
            if time.monotonic() - start > timeout_s:
                t = self.tel
                raise VehicleError(
                    f"{what}: inte inom {timeout_s:.0f} s (läge {t.get('flight_mode')}, "
                    f"{t.get('alt_rel_home_m')} m, armerad {t.get('armed')})"
                )
            time.sleep(0.1)

    def wait_mode(self, pred: Callable[[str | None], bool], timeout_s: float) -> str | None:
        """Wait until ``pred(mode())`` holds; returns that mode."""
        self._until(lambda: pred(self.mode()), timeout_s, "flygläget")
        return self.mode()

    # --- the panel's buttons --------------------------------------------------------------

    def _post(self, path: str, body: dict[str, Any], want: int, what: str) -> Any:
        status, reply = self.panel.request("POST", path, body)
        if status != want:
            raise VehicleError(f"{what}: HTTP {status} {reply}")
        return reply

    def _get(self, path: str, **params: str) -> Any:
        status, reply = self.panel.request("GET", path, params=params or None)
        if status != 200:
            raise VehicleError(f"{path}: HTTP {status} {reply}")
        return reply

    def _action(self, action: str) -> int:
        body = {"client_action_id": str(new_uuid7())}
        path = f"/api/v1/vehicles/{VEHICLE_ID}/actions/{action}"
        t0 = time.monotonic_ns()
        self._post(path, body, 202, action.upper())
        return t0

    def press_stop(self) -> int:
        """STOP; returns ``time.monotonic_ns()`` taken right before the POST."""
        return self._action("stop")

    def press_resume(self) -> int:
        """The pilot's Resume; returns ``time.monotonic_ns()`` taken right before the POST."""
        return self._action("resume")

    def status(self) -> str:
        return str(self._get(f"/api/v1/missions/{self.mid}")["mission"]["status"])

    def place_pin(self, name: str) -> None:
        """The pilot's pin on the map, where the demo fixture has it."""
        lat, lon = pin_latlon(fixture.load(), name)
        body = {
            "kind": "pin",
            "label": name,
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
        }
        self._post(f"/api/v1/missions/{self.mid}/annotations", body, 201, f"nålen {name}")

    def charter_failure(self, request_id: str | None = None) -> list[str] | None:
        """The error codes when the AI gave up its charter proposal (its line with
        ``refs.charter_failed``), or ``turn_failed:<why>`` when its answer to the line
        ``request_id`` says no card follows (``refs.turn_failed``: no answer in time, a
        reply it could not read); else None. Reads the newest 200 lines. The pilot never
        answers a question in the chat: only a real card is approved."""
        page = self._get(f"/api/v1/missions/{self.mid}/chat", limit="200")
        for m in page.get("items", []):
            refs = m.get("refs") or {}
            if m.get("role") != "brain":
                continue
            if refs.get("charter_failed"):
                return [str(c) for c in refs.get("codes") or []]
            replies = request_id is not None and str(m.get("in_reply_to")) == request_id
            if replies and refs.get("turn_failed"):
                return [f"turn_failed:{refs['turn_failed']}"]
        return None

    def say(self, text: str) -> str:
        """Writes ``text`` in the chat as the pilot; returns the line's id."""
        body = {"client_msg_id": str(new_uuid7()), "text": text}
        reply = self._post(f"/api/v1/missions/{self.mid}/chat", body, 201, "chatten")
        return str((reply or {}).get("id"))

    def wait_command(self, skill: str, status: str, timeout_s: float) -> dict[str, Any]:
        """The AI's newest ``skill`` order reaches ``status``; another end fails."""
        found: dict[str, Any] = {}

        def reached() -> bool:
            c = newest_brain_command(
                list(self._get(f"/api/v1/missions/{self.mid}/commands")), skill
            )
            if c is None:
                return False
            if c["status"] in TERMINAL and c["status"] != status:
                raise VehicleError(f"{skill}: {c['status']} {c.get('status_reason')}")
            found.update(c)
            return c["status"] == status

        self._until(reached, timeout_s, f"AI:ns {skill} {status}")
        return found

    def fly_demo_until_airborne(self, timeout_s: float = 300.0) -> str:
        """The demo's opening (D2-D5): open a mission, the pin P1 on the map (the AI
        places its search box by the pin's label), the request in the chat, "Godkänn
        ram" on the AI's charter card in mode B, the onboard checks, "Fritt runt
        drönaren" and the AI's take-off to 40 m. Every step may use what is left of
        ``timeout_s``. When the AI gives up its charter (``refs.charter_failed``), or its
        answer to the request says no card follows (``refs.turn_failed``), the run fails
        at once with the error codes. Returns the mission id."""
        self.start()
        deadline = time.monotonic() + timeout_s
        opened = self._post(f"/api/v1/vehicles/{VEHICLE_ID}/missions/open", {}, 200, "uppdraget")
        self.mid = str(opened["id"])
        self.place_pin(DEMO_PIN)
        request_id = self.say(DEMO_REQUEST)
        cards: list[dict[str, Any]] = []

        def card() -> bool:
            pending = self._get(f"/api/v1/missions/{self.mid}/approvals", status="pending")
            cards[:] = [a for a in pending if a.get("kind") == "charter"]
            if not cards and (codes := self.charter_failure(request_id)) is not None:
                raise VehicleError(
                    "AI:n gav upp ramförslaget: " + (", ".join(codes) or "okänt fel")
                )
            return bool(cards)

        self._until(card, remaining_s(deadline), "AI:ns ramkort")
        decision = {
            "decision": "approve",
            "subject_version": cards[0]["subject_version"],
            "autonomy_mode": "WITHIN_CHARTER",
        }
        self._post(f"/api/v1/approvals/{cards[0]['id']}/decision", decision, 200, "Godkänn ram")
        self._until(lambda: self.status() == "READY", remaining_s(deadline), "uppdraget READY")
        self._until(
            lambda: (self._live.get("snapshot") or {}).get("preflight_ok") is True,
            remaining_s(deadline),
            "ombordsdatorns förflygningskontroll",
        )
        self._post(f"/api/v1/missions/{self.mid}/clearance", {}, 200, "Fritt runt drönaren")
        self.wait_command("takeoff", "done", remaining_s(deadline))
        self._until(self.airborne, remaining_s(deadline), "i luften")
        return self.mid

    def wait_flight_done(self, timeout_s: float = 900.0) -> None:
        """Until the mission is LANDED/CLOSED or the drone stands disarmed on the ground."""
        self.start()
        asked_at = 0.0
        status: str | None = None

        def done() -> bool:
            nonlocal asked_at, status
            if self.mid and time.monotonic() - asked_at >= 1.0:  # the mission once a second
                asked_at, status = time.monotonic(), self.status()
            return flight_done(self._live, status, seen_airborne=self._seen_airborne)

        self._until(done, timeout_s, "landad och avarmerad")

    def chat_latencies_s(self, stack: Stack | None = None) -> list[float]:
        """P8 for this pilot's mission (every mission when none was opened)."""
        return chat_latencies_s(stack or self.stack, self.mid or None)


# --- the database -------------------------------------------------------------------------


async def _chat_rows(stack: Stack, mission_id: str | None) -> list[tuple[datetime, datetime]]:
    import asyncpg

    password = read_env_value(stack.repo / ".env", "SD_GROUND__DB_PASSWORD")
    conn = await asyncpg.connect(
        host=DB_HOST,
        port=DB_PORT,
        user="sd_owner",
        password=password,
        database=database_name(stack.cfg.instance),
    )
    try:
        rows = await conn.fetch(CHAT_LATENCY_SQL, mission_id)
    finally:
        await conn.close()
    return [(r["said_at"], r["replied_at"]) for r in rows]


def chat_latencies_s(stack: Stack, mission_id: str | None = None) -> list[float]:
    """Operator chat message to the AI's first reply (order or chat line), in seconds (P8),
    from the instance's database (reply ``created_at`` - ``chat_messages.created_at``).
    An empty list means no message was answered; the report shows that as no data."""
    try:
        return latencies_s(asyncio.run(_chat_rows(stack, mission_id)))
    except OSError as exc:
        raise InfraError(f"Databasen svarar inte på {DB_HOST}:{DB_PORT}: {exc}") from None
