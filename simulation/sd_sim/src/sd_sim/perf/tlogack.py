"""Find the vehicle's STOP acknowledgements in a radio tlog.

A STOP is a DO_SET_MODE command (176). The vehicle accepted it when it sends
COMMAND_ACK with ``command == 176`` and ``result == 0`` (MAV_RESULT_ACCEPTED). The
tlog (``sd_sim.tlog``) holds one record per frame, an 8-byte big-endian timestamp and
the raw frame; line ``i`` of the ``.idx.jsonl`` sidecar describes record ``i`` with
the direction and the host's monotonic time. The two files are read in step. Records
the link simulator dropped never reached the ground and are skipped, and so is a last
record that is only partly written.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from pymavlink.dialects.v20 import ardupilotmega as mavlink2

from sd_sim.tlog import split_frames
from sd_sim.udprelay import FROM_VEHICLE

MAV_CMD_DO_SET_MODE = 176
MAV_RESULT_ACCEPTED = 0
MSG_COMMAND_ACK = 77
_STAMP_LEN = 8
# A MAVLink v2 frame is at most 10 header + 255 payload + 2 checksum + 13 signature bytes.
_MAX_FRAME = 280


def _records(data: bytes) -> Iterator[bytes]:
    """Raw frames of a tlog, in order. Stops at a record that is cut off or broken."""
    i = 0
    while i + _STAMP_LEN < len(data):
        start = i + _STAMP_LEN
        frames = split_frames(data[start : start + _MAX_FRAME])
        if not frames or not data.startswith(frames[0].raw, start):
            return
        yield frames[0].raw
        i = start + len(frames[0].raw)


def _index(idx: Path) -> Iterator[dict[str, Any]]:
    with idx.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                return  # a half-written last line
            if isinstance(entry, dict):
                yield entry
            else:
                return


def stop_acks(tlog: Path, idx: Path) -> list[int]:
    """Host monotonic ns of every accepted DO_SET_MODE COMMAND_ACK from the vehicle."""
    try:
        data = tlog.read_bytes()
        entries = list(_index(idx))
    except OSError:
        return []
    parser = mavlink2.MAVLink(None)
    out: list[int] = []
    for raw, entry in zip(_records(data), entries, strict=False):
        if entry.get("dir") != FROM_VEHICLE or entry.get("dropped"):
            continue
        if entry.get("msgid") != MSG_COMMAND_ACK:
            continue
        try:
            msg = parser.decode(bytearray(raw))
        except Exception:  # noqa: S112 - a frame pymavlink cannot decode is not an ACK
            continue
        if msg.get_msgId() != MSG_COMMAND_ACK:
            continue
        command = getattr(msg, "command", None)
        result = getattr(msg, "result", None)
        mono = entry.get("mono_ns")
        if (
            command == MAV_CMD_DO_SET_MODE
            and result == MAV_RESULT_ACCEPTED
            and isinstance(mono, int)
        ):
            out.append(mono)
    return out


def first_after(acks: Sequence[int], t0_ns: int, within_s: float = 3.0) -> int | None:
    """The first ack at or after ``t0_ns`` and at most ``within_s`` seconds later."""
    limit = t0_ns + int(within_s * 1e9)
    hits = [t for t in acks if t0_ns <= t <= limit]
    return min(hits) if hits else None
