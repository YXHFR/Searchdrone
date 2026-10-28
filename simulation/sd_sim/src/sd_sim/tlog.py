"""Record MAVLink datagrams as a pymavlink tlog plus a JSON-lines index.

The tlog holds every MAVLink frame with an 8-byte big-endian wall-clock timestamp in
microseconds (the format ``mavlogdump.py`` reads). The sidecar ``<name>.idx.jsonl`` has
one line per frame with the direction, the monotonic host time and the ids, so a test
can order frames from several taps on one timeline.

The files are written on a thread of their own. The taps sit in the path of every
datagram (``mavtap``, ``linksim``), and a write that waits for a busy disk would hold up
the link it records: on a loaded CI host that stalled both relays for seconds, long
enough for the ground's companion watchdog to see a silent onboard computer and STOP.
"""

from __future__ import annotations

import json
import queue
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Final

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


_FLUSH: Final = b""  # in the queue: flush both files
_CLOSE: Final = None  # in the queue: the last item


class TlogWriter:
    """Append-only writer for ``<path>`` and ``<path stem>.idx.jsonl``. ``write`` and
    ``flush`` only queue; the writer thread does the file work in order, so a stalled
    disk never holds up the caller. ``close`` waits until everything is on file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._tlog: IO[bytes] = path.open("ab")
        self._idx: IO[str] = path.with_suffix(".idx.jsonl").open("a", encoding="utf-8")
        self._count = 0
        self._queue: queue.SimpleQueue[tuple[bytes, str] | bytes | None] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._drain, name=f"tlog-{path.name}", daemon=True)
        self._thread.start()

    def write(self, data: bytes, direction: str, *, dropped: bool = False) -> None:
        mono = time.monotonic_ns()
        wall_us = time.time_ns() // 1000
        for frame in split_frames(data):
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
            self._queue.put((struct.pack(">Q", wall_us) + frame.raw, json.dumps(entry) + "\n"))
            self._count += 1

    def flush(self) -> None:
        self._queue.put(_FLUSH)

    def close(self) -> None:
        if self._thread.is_alive():
            self._queue.put(_CLOSE)
            self._thread.join()
        self._tlog.close()
        self._idx.close()

    def _drain(self) -> None:
        while (item := self._queue.get()) is not _CLOSE:
            if item == _FLUSH:
                self._tlog.flush()
                self._idx.flush()
            elif isinstance(item, tuple):
                record, line = item
                self._tlog.write(record)
                self._idx.write(line)
