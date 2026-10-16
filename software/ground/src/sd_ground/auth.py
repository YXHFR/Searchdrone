"""Who is calling: login codes, PIN pairing, sessions, CLI tokens and the commander token.

Everything here is thread-safe: the ingress thread (its own event loop) checks sessions for
STOP / RETURN HOME while the main loop handles everything else. Sessions are checked against
memory, so a logged-in panel keeps working when the database is down.

Only SHA-256 hashes of codes, PINs and tokens are kept. Plain values are returned once to
the caller and never logged.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import secrets
import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Literal

from sd_obs.clock import Clock

__all__ = [
    "ATTEMPT_LIMIT",
    "ATTEMPT_WINDOW_S",
    "SESSION_HEADROOM",
    "AuthMode",
    "BrainToken",
    "LoginError",
    "PendingLogins",
    "Principal",
    "Role",
    "SessionRecord",
    "SessionStore",
    "hash_token",
    "new_token",
]

Role = Literal["pilot", "observer", "commander", "system"]
AuthMode = Literal["local", "pin", "cli", "cloudflare_access"]
Via = Literal["local", "lan", "remote"]

SESSION_IDLE = dt.timedelta(hours=12)
SESSION_CAP = dt.timedelta(days=7)
SESSION_HEADROOM = dt.timedelta(hours=2)  # takeoff and charter approval need this much left
PAIR_TTL_S = 300.0
ATTEMPT_LIMIT = 5  # failed POST /session per client address ...
ATTEMPT_WINDOW_S = 300.0  # ... within this window -> 429
PIN_GLOBAL_LIMIT = 5  # failed PIN attempts from all addresses -> the PIN is invalidated
TOKEN_DEFAULT = dt.timedelta(hours=24)
TOKEN_MAX = dt.timedelta(days=30)


def new_token() -> str:
    """32 random bytes, base64url."""
    return secrets.token_urlsafe(32)


def hash_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Principal:
    """The caller as the server sees it. Never taken from the request body."""

    role: Role
    issuer_ref: str
    auth_mode: AuthMode
    via: Via = "local"
    session_id: str | None = None
    display_name: str = ""

    @property
    def interactive(self) -> bool:
        """A person in the panel (login code or PIN), not a script with a CLI token."""
        return self.auth_mode in ("local", "pin")


class LoginError(Exception):
    """A failed login. ``code`` is the stable API code, ``status`` the HTTP status."""

    def __init__(self, status: int, code: str, detail_sv: str) -> None:
        super().__init__(code)
        self.status, self.code, self.detail_sv = status, code, detail_sv


@dataclass
class _Pending:
    code_hash: str
    expires_mono: float
    auth_mode: AuthMode
    pin_hash: str | None = None
    used: bool = False
    pin_failures: int = 0
    invalidated: bool = False


class PendingLogins:
    """Login codes and PINs. Kept only in memory; a restart forgets them."""

    def __init__(
        self,
        clock: Clock,
        *,
        code_ttl_s: float = 600.0,
        on_pin_locked: Callable[[], None] | None = None,
    ) -> None:
        self._clock = clock
        self._ttl = code_ttl_s
        self._lock = threading.Lock()
        self._codes: list[_Pending] = []
        self._pair: _Pending | None = None
        self._attempts: dict[str, deque[float]] = {}
        self._on_pin_locked = on_pin_locked

    def create_login_code(self) -> str:
        code = new_token()
        with self._lock:
            self._prune()
            self._codes.append(
                _Pending(hash_token(code), self._clock.monotonic() + self._ttl, "local")
            )
        return code

    def create_pair(self) -> tuple[str, str]:
        """A new LAN code and its 6-digit PIN. Any earlier PIN stops working."""
        code = new_token()
        pin = f"{secrets.randbelow(1_000_000):06d}"
        with self._lock:
            self._pair = _Pending(
                hash_token(code), self._clock.monotonic() + PAIR_TTL_S, "pin", hash_token(pin)
            )
        return code, pin

    def _prune(self) -> None:
        now = self._clock.monotonic()
        self._codes = [p for p in self._codes if not p.used and p.expires_mono > now]

    def _limited(self, client: str) -> bool:
        now = self._clock.monotonic()
        window = self._attempts.setdefault(client, deque())
        while window and window[0] <= now - ATTEMPT_WINDOW_S:
            window.popleft()
        return len(window) >= ATTEMPT_LIMIT

    def _fail(self, client: str) -> None:
        self._attempts.setdefault(client, deque()).append(self._clock.monotonic())

    def redeem(self, client: str, *, login_code: str | None, pin: str | None) -> AuthMode:
        """Use a code or a PIN once. Raises ``LoginError`` (401, 429)."""
        locked = False
        try:
            with self._lock:
                if self._limited(client):
                    raise LoginError(
                        429, "TOO_MANY_ATTEMPTS", "För många försök. Vänta några minuter."
                    )
                now = self._clock.monotonic()
                if login_code is not None:
                    mode = self._redeem_code(hash_token(login_code), now)
                else:
                    mode, locked = self._redeem_pin(hash_token(pin or ""), now)
                if mode is None:
                    self._fail(client)
                    if self._pair is not None and self._pair.invalidated and pin is not None:
                        raise LoginError(
                            401, "PIN_INVALIDATED", "PIN-koden är spärrad. Kör: sd-ground pair"
                        )
                    raise LoginError(401, "LOGIN_FAILED", "Koden gäller inte.")
                return mode
        finally:
            if locked and self._on_pin_locked is not None:
                self._on_pin_locked()

    def _redeem_code(self, code_hash: str, now: float) -> AuthMode | None:
        for pending in self._codes:
            if (
                hmac.compare_digest(pending.code_hash, code_hash)
                and not pending.used
                and pending.expires_mono > now
            ):
                pending.used = True
                return pending.auth_mode
        pair = self._pair
        if (
            pair is not None
            and hmac.compare_digest(pair.code_hash, code_hash)
            and not pair.used
            and not pair.invalidated
            and pair.expires_mono > now
        ):
            pair.used = True
            return "pin"
        return None

    def _redeem_pin(self, pin_hash: str, now: float) -> tuple[AuthMode | None, bool]:
        pair = self._pair
        if pair is None or pair.invalidated or pair.used or pair.expires_mono <= now:
            return None, False
        if pair.pin_hash is not None and hmac.compare_digest(pair.pin_hash, pin_hash):
            pair.used = True
            return "pin", False
        pair.pin_failures += 1
        if pair.pin_failures >= PIN_GLOBAL_LIMIT:
            pair.invalidated = True
            return None, True
        return None, False


@dataclass(frozen=True)
class SessionRecord:
    id: str
    kind: Literal["session", "token"]
    role: Literal["pilot", "observer"]
    issuer_ref: str
    auth_mode: AuthMode
    token_sha256: str = field(repr=False)
    created_at: dt.datetime
    last_used_at: dt.datetime
    expires_at: dt.datetime | None  # tokens: fixed; sessions: None (12 h idle, 7 days cap)
    name: str | None = None
    revoked: bool = False
    pending_persist: bool = False
    extended_logged: bool = False

    def natural_expiry(self) -> dt.datetime:
        """When the session ends if no vehicle is in the air."""
        if self.kind == "token":
            return self.expires_at or self.created_at
        return min(self.last_used_at + SESSION_IDLE, self.created_at + SESSION_CAP)

    def cap(self) -> dt.datetime:
        """The hard cap that keeping the session in use cannot move."""
        if self.kind == "token":
            return self.expires_at or self.created_at
        return self.created_at + SESSION_CAP


class SessionStore:
    """All valid sessions and CLI tokens, in memory. The database is only their backup.

    ``airborne_since`` tells when the current flight began (``None`` on the ground). A pilot
    session that was still valid when the flight began never expires before landing.
    """

    def __init__(
        self,
        clock: Clock,
        *,
        airborne_since: Callable[[], dt.datetime | None] = lambda: None,
        on_extended: Callable[[SessionRecord], None] | None = None,
    ) -> None:
        self._clock = clock
        self._airborne_since = airborne_since
        self._on_extended = on_extended
        self._lock = threading.Lock()
        self._by_hash: dict[str, SessionRecord] = {}
        self._revoked: set[str] = set()  # logged out; never brought back by load()
        self._revoked_saved: set[str] = set()

    def create_session(
        self, *, role: Literal["pilot", "observer"], issuer_ref: str, auth_mode: AuthMode
    ) -> tuple[str, SessionRecord]:
        token = new_token()
        now = self._clock.utc_now()
        record = SessionRecord(
            id=secrets.token_hex(16),
            kind="session",
            role=role,
            issuer_ref=issuer_ref,
            auth_mode=auth_mode,
            token_sha256=hash_token(token),
            created_at=now,
            last_used_at=now,
            expires_at=None,
            pending_persist=True,
        )
        with self._lock:
            self._by_hash[record.token_sha256] = record
        return token, record

    def load(self, records: Iterable[SessionRecord]) -> None:
        """Replace what the database knows; keep sessions not yet persisted."""
        with self._lock:
            fresh = {r.token_sha256: r for r in records if r.token_sha256 not in self._revoked}
            for key, old in self._by_hash.items():
                if key in fresh:
                    keep = fresh[key]
                    if old.last_used_at > keep.last_used_at:
                        keep = replace(keep, last_used_at=old.last_used_at)
                    fresh[key] = replace(keep, extended_logged=old.extended_logged)
                elif old.pending_persist:
                    fresh[key] = old
            self._by_hash = fresh

    def pending(self) -> list[SessionRecord]:
        with self._lock:
            return [r for r in self._by_hash.values() if r.pending_persist]

    def mark_persisted(self, token_sha256: str) -> None:
        with self._lock:
            rec = self._by_hash.get(token_sha256)
            if rec is not None:
                self._by_hash[token_sha256] = replace(rec, pending_persist=False)

    def revoke(self, token_sha256: str) -> None:
        """Ends a session now. Stays ended even if the database still lists it."""
        with self._lock:
            self._by_hash.pop(token_sha256, None)
            self._revoked.add(token_sha256)

    def unsaved_revocations(self) -> list[str]:
        with self._lock:
            return list(self._revoked - self._revoked_saved)

    def revocation_saved(self, token_sha256: str) -> None:
        with self._lock:
            self._revoked_saved.add(token_sha256)

    def _valid(self, rec: SessionRecord, now: dt.datetime) -> tuple[bool, bool]:
        """(valid, extended_by_flight)."""
        if rec.revoked:
            return False, False
        natural = rec.natural_expiry()
        if natural > now:
            return True, False
        since = self._airborne_since()
        if rec.kind == "session" and rec.role == "pilot" and since is not None and natural > since:
            return True, True
        return False, False

    def lookup(self, token: str, *, touch: bool = True) -> SessionRecord | None:
        key = hash_token(token)
        extended: SessionRecord | None = None
        with self._lock:
            rec = self._by_hash.get(key)
            if rec is None:
                return None
            now = self._clock.utc_now()
            valid, by_flight = self._valid(rec, now)
            if not valid:
                return None
            if by_flight and not rec.extended_logged:
                rec = replace(rec, extended_logged=True)
                extended = rec
            if touch and rec.kind == "session" and not by_flight:
                rec = replace(rec, last_used_at=now)
            self._by_hash[key] = rec
        if extended is not None and self._on_extended is not None:
            self._on_extended(extended)
        return rec

    def expires_at(self, rec: SessionRecord) -> dt.datetime | None:
        """What the panel is told: ``None`` while a flight keeps the session alive."""
        now = self._clock.utc_now()
        _, by_flight = self._valid(rec, now)
        if by_flight or (rec.role == "pilot" and self._airborne_since() is not None):
            return None
        return rec.natural_expiry()

    def headroom(self, rec: SessionRecord) -> dt.timedelta:
        """Time left until the session's cap."""
        return rec.cap() - self._clock.utc_now()

    def all(self) -> list[SessionRecord]:
        with self._lock:
            return list(self._by_hash.values())


class BrainToken:
    """The commander token: only its hash is kept; compared in constant time."""

    def __init__(self, token: str | None) -> None:
        self._hash = hash_token(token) if token else None

    @property
    def configured(self) -> bool:
        return self._hash is not None

    def matches(self, token: str) -> bool:
        return self._hash is not None and hmac.compare_digest(self._hash, hash_token(token))


def token_expiry(now: dt.datetime, expires: dt.timedelta | None) -> dt.datetime:
    """The expiry of a new CLI token: 24 h by default, at most 30 days, never none."""
    span = TOKEN_DEFAULT if expires is None else expires
    if span <= dt.timedelta(0) or span > TOKEN_MAX:
        raise ValueError("--expires måste vara mellan 1 minut och 30 dygn")
    return now + span


def parse_duration(text: str) -> dt.timedelta:
    """``30m``, ``24h``, ``7d``."""
    units = {"m": "minutes", "h": "hours", "d": "days"}
    if len(text) < 2 or text[-1] not in units or not text[:-1].isdigit():
        raise ValueError(f"ogiltig tid {text!r} (t.ex. 30m, 24h, 7d)")
    return dt.timedelta(**{units[text[-1]]: int(text[:-1])})


def bearer_of(authorization: str) -> str | None:
    """The token of an ``Authorization: Bearer <token>`` header."""
    scheme, _, value = authorization.partition(" ")
    return value.strip() if scheme.lower() == "bearer" and value.strip() else None


def session_for(
    store: SessionStore, *, authorization: str, cookie: str | None
) -> SessionRecord | None:
    """A Bearer header counts only for CLI tokens, the cookie only for panel sessions."""
    token = bearer_of(authorization)
    if token is not None:
        rec = store.lookup(token)
        return rec if rec is not None and rec.kind == "token" else None
    if cookie:
        rec = store.lookup(cookie)
        return rec if rec is not None and rec.kind == "session" else None
    return None


def principal_for(rec: SessionRecord, listener: object) -> Principal:
    """The caller of a request on ``listener`` (``"lan"`` or the local listener)."""
    return Principal(
        role=rec.role,
        issuer_ref=rec.issuer_ref,
        auth_mode=rec.auth_mode,
        via="lan" if listener == "lan" else "local",
        session_id=rec.id,
        display_name=rec.issuer_ref.split(":", 1)[-1],
    )
