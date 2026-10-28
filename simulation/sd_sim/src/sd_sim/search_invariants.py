"""The search scenarios' invariants (E1, E2, E8), checked after each SK run.

* ``no_confirm_without_human`` (E1): every confirmed or rejected candidate has a person's
  decision (``operator:`` reference, a session) at or before the revision that put it
  there, and every confirmation was preceded by a large view of one of the candidate's
  pictures in the same session.
* ``no_candidate_deleted`` (E2): a candidate id seen once is there at every later look,
  and the revisions of a candidate only grow.
* ``keepout_never_entered`` (E8): after a keep-out cylinder is in force, the flown track
  never comes inside it. A drone already inside when the cylinder is made (it was made
  around a person under the drone) may leave it, but never come back in. A cylinder can
  also come with its edge closer than the drone can stop from at its speed (a person seen
  just ahead): no reaction keeps it out, so the drone may brake into it by what the stop
  (``stop_distance_m``) leaves, while braking, and then leave it as above; any deeper is
  an entry.
  ``keepout_min_distance_m`` is the smallest horizontal distance from the track to an
  active cylinder's edge after the drone was outside it (negative: inside).

The rows are the mission's own, read from the ground's database by the scenario; the
cylinders are what the onboard executor held (``keepouts_in_force``), sampled during the
flight. Nothing here touches the flight stack.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from sd_safety.targets import distance_m
from sd_sim.invariants import Sample

__all__ = [
    "KEEPOUT_TOLERANCE_M",
    "CandidateWatch",
    "SeenKeepout",
    "keepout_distances",
    "keepout_never_entered",
    "no_confirm_without_human",
    "stop_distance_m",
]

# The flight controller's position wobbles and the track is sampled; a metre inside the
# edge is noise, not an entry.
KEEPOUT_TOLERANCE_M: Final = 1.0
CLEAR_M: Final = 5.0
# How far a drone goes on from the moment a cylinder is in force on board until LOITER has
# stopped it: the reflex's tick (0.1 s), the mode change and LOITER's braking ramp, then
# ArduCopter's LOITER braking (LOIT_BRK_ACCEL, 2.5 m/s²). The time it takes, with
# ``STOP_SETTLE_S`` for the jerk-limited ramp and the position controller settling.
STOP_REACT_S: Final = 0.5
STOP_DECEL_MS2: Final = 2.5
STOP_SETTLE_S: Final = 1.0
# FLYG HEM goes past every cylinder (``sd_safety.policy.KEEPOUT_EXEMPT``): ArduPilot flies
# straight home and lands, so E8 skips those samples. LAND counts as the way home only
# after SMART_RTL or RTL: it is also the end of STOP, which is not exempt.
_FLY_HOME_MODES: Final = frozenset({"SMART_RTL", "RTL"})


def _ts(value: Any) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def no_confirm_without_human(
    revisions: Iterable[Mapping[str, Any]],
    decisions: Iterable[Mapping[str, Any]],
    views: Iterable[Mapping[str, Any]],
    observed: Mapping[str, Iterable[str]],
) -> list[str]:
    """``revisions``: candidate rows (candidate_id, rev, state, created_at);
    ``decisions``: candidate_decisions rows (candidate_id, decision, operator_ref,
    session_id, created_at); ``views``: candidate_views rows (detection_id, session_id,
    created_at); ``observed``: candidate id -> the detection ids it observed."""
    broken: list[str] = []
    by_cand: dict[str, list[Mapping[str, Any]]] = {}
    for d in decisions:
        by_cand.setdefault(str(d["candidate_id"]), []).append(d)
        ref = str(d["operator_ref"])
        if not ref.startswith("operator:") or not d.get("session_id"):
            broken.append(f"E1: beslut på {d['candidate_id']} utan en person ({ref})")
    seen: dict[tuple[str, str], dt.datetime] = {}
    for v in views:
        key = (str(v["session_id"]), str(v["detection_id"]))
        at = _ts(v["created_at"])
        seen[key] = min(at, seen.get(key, at))
    want = {"confirmed": "confirm", "rejected": "reject"}
    for r in revisions:
        state = str(r["state"])
        if state not in want:
            continue
        cid, at = str(r["candidate_id"]), _ts(r["created_at"])
        made = [
            d
            for d in by_cand.get(cid, [])
            if d["decision"] == want[state] and _ts(d["created_at"]) <= at
        ]
        if not made:
            broken.append(f"E1: {cid} rev {r['rev']} blev {state} utan en persons beslut")
    dets = {c: {str(x) for x in ds} for c, ds in observed.items()}
    for cid, ds in by_cand.items():
        for d in ds:
            if d["decision"] != "confirm":
                continue
            at, session = _ts(d["created_at"]), str(d["session_id"])
            if not any(seen.get((session, det), at + dt.timedelta(1)) <= at
                       for det in dets.get(cid, ())):  # fmt: skip
                broken.append(f"E1: {cid} bekräftad utan stor vy i samma session")
    return broken


@dataclass
class CandidateWatch:
    """E2 over repeated looks at a mission's candidates (id -> highest rev)."""

    seen: dict[str, int] = field(default_factory=dict[str, int])
    broken: list[str] = field(default_factory=list[str])
    looks: int = 0

    def look(self, revs: Mapping[str, int]) -> None:
        self.looks += 1
        for cid, rev in self.seen.items():
            now = revs.get(cid)
            if now is None:
                self.broken.append(f"E2: kandidaten {cid} försvann (titt {self.looks})")
            elif now < rev:
                self.broken.append(f"E2: kandidaten {cid} gick från rev {rev} till {now}")
        for cid, rev in revs.items():
            self.seen[cid] = max(rev, self.seen.get(cid, 0))


@dataclass(frozen=True)
class SeenKeepout:
    """A cylinder the onboard executor held, from the first time the test saw it."""

    keepout_id: str
    lat: float
    lon: float
    radius_m: float
    since: float  # monotonic seconds, the clock of the track's samples
    until: float = math.inf  # when it was no longer in force


def _searching(track: Sequence[Sample]) -> list[Sample]:
    """The track without the samples on the way home: SMART_RTL, RTL and the LAND
    that follows them."""
    out: list[Sample] = []
    homing = False
    for s in track:
        if s.mode in _FLY_HOME_MODES:
            homing = True
        elif s.mode != "LAND" or not homing:
            homing = False
            out.append(s)
    return out


def stop_distance_m(speed_ms: float) -> float:
    """How far a drone at ``speed_ms`` goes on before it stands still, once a cylinder is
    in force on board."""
    v = max(0.0, speed_ms)
    return v * STOP_REACT_S + v * v / (2.0 * STOP_DECEL_MS2)


def _edge_distances(k: SeenKeepout, track: Sequence[Sample]) -> list[tuple[float, float, float]]:
    """(t, distance outside the edge, ground speed) for the armed samples while ``k`` was
    in force, except those on the way home."""
    return [
        (s.t, distance_m(k.lat, k.lon, s.lat, s.lon) - k.radius_m, s.groundspeed_ms)
        for s in _searching(track)
        if s.armed and k.since <= s.t <= k.until
    ]


def keepout_distances(keepouts: Iterable[SeenKeepout], track: Sequence[Sample]) -> float | None:
    """``keepout_min_distance_m``: the smallest distance to an active cylinder's edge
    once the drone was clearly outside it, ``CLEAR_M`` past the edge (a cylinder is
    born around what the drone just saw, so the way out starts at the edge; ``None``
    with no cylinder or no such sample)."""
    best: float | None = None
    for k in keepouts:
        out = False
        for _, d, _ in _edge_distances(k, track):
            out = out or d >= CLEAR_M
            if out:
                best = d if best is None else min(best, d)
    return best


def keepout_never_entered(
    keepouts: Iterable[SeenKeepout],
    track: Sequence[Sample],
    tolerance_m: float = KEEPOUT_TOLERANCE_M,
) -> list[str]:
    broken: list[str] = []
    for k in keepouts:
        rows = _edge_distances(k, track)
        if not rows:
            continue
        # Came with its edge within the stop: braking may take the drone this far in.
        t0, d0, v0 = rows[0]
        stop = stop_distance_m(v0)
        brake_m = stop - d0 if 0.0 <= d0 < stop else 0.0
        braking_until = t0 + STOP_REACT_S + v0 / STOP_DECEL_MS2 + STOP_SETTLE_S
        out = False
        for t, d, _ in rows:
            if brake_m > 0.0 and t <= braking_until:
                if d < -(brake_m + tolerance_m):
                    broken.append(
                        f"E8: banan gick {-d:.1f} m in i cylindern {k.keepout_id} "
                        f"({t - k.since:.0f} s efter att den kom, {d0:.1f} m bort i "
                        f"{v0:.1f} m/s: inbromsningen tar den {brake_m:.1f} m in)"
                    )
                    break
                continue
            if d >= 0:
                out = True
            elif out and d < -tolerance_m:
                broken.append(
                    f"E8: banan gick {-d:.1f} m in i cylindern {k.keepout_id} "
                    f"({t - k.since:.0f} s efter att den kom)"
                )
                break
    return broken
