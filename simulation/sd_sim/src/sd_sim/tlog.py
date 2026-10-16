"""Record MAVLink datagrams as a pymavlink tlog plus a JSON-lines index.

The tlog holds every MAVLink frame with an 8-byte big-endian wall-clock timestamp in
microseconds (the format ``mavlogdump.py`` reads). The sidecar ``<name>.idx.jsonl`` has
one line per frame with the direction, the monotonic host time and the ids, so a test
can order frames from several taps on one timeline.
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

MAGIC_V1 = 0xFE
MAGIC_V2 = 0xFD
_V2_SIGNED = 0x01


@dataclass(frozen=True)
class Frame:
    raw: bytes
    sysid: int
    compid: int
    msgid: int


def split_frames(data: bytes) -> list[Frame]:
    """Split one datagram into MAVLink v1/v2 frames. Bytes that are not MAVLink are skipped."""
    frames: list[Frame] = []
    i = 0
    n = len(data)
    while i < n:
        magic = data[i]
        if magic == MAGIC_V2 and i + 10 <= n:
            length = data[i + 1]
            signed = data[i + 2] & _V2_SIGNED
            total = 10 + length + 2 + (13 if signed else 0)
            if i + total > n:
                break
            msgid = data[i + 7] | (data[i + 8] << 8) | (data[i + 9] << 16)
            frames.append(Frame(data[i : i + total], data[i + 5], data[i + 6], msgid))
            i += total
        elif magic == MAGIC_V1 and i + 6 <= n:
            length = data[i + 1]
            total = 6 + length + 2
            if i + total > n:
                break
            frames.append(Frame(data[i : i + total], data[i + 3], data[i + 4], data[i + 5]))
            i += total
        else:
            i += 1
    return frames


class TlogWriter:
    """Append-only writer for ``<path>`` and ``<path stem>.idx.jsonl``."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._tlog: IO[bytes] = path.open("ab")
        self._idx: IO[str] = path.with_suffix(".idx.jsonl").open("a", encoding="utf-8")
        self._count = 0

    def write(self, data: bytes, direction: str, *, dropped: bool = False) -> None:
        mono = time.monotonic_ns()
        wall_us = time.time_ns() // 1000
        for frame in split_frames(data):
            self._tlog.write(struct.pack(">Q", wall_us) + frame.raw)
            entry = {
                "i": self._count,
                "dir": direction,
                "mono_ns": mono,
                "sysid": frame.sysid,
                "compid": frame.compid,
                "msgid": frame.msgid,
            }
            if dropped:
                entry["dropped"] = True
            self._idx.write(json.dumps(entry) + "\n")
            self._count += 1

    def flush(self) -> None:
        self._tlog.flush()
        self._idx.flush()

    def close(self) -> None:
        self._tlog.close()
        self._idx.close()
