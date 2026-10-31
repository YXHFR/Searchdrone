"""Record MAVLink datagrams as a pymavlink tlog plus a JSON-lines index.

The tlog holds every MAVLink frame with an 8-byte big-endian wall-clock timestamp in
microseconds (the format ``mavlogdump.py`` reads). The sidecar ``<name>.idx.jsonl`` has
one line per frame with the direction, the monotonic host time and the ids, so a test
can order frames from several taps on one timeline.

The files are written on a thread of their own. The taps sit in the path of every
datagram (``mavtap``, ``linksim``), and a write that waits for a busy disk would hold up
the link it records: on a loaded CI host that stalled both relays for seconds, long
enough for the ground's companion watchdog to see a silent onboard computer and STOP.

The queue to that thread has a cap (``QUEUE_MAX`` frames), so a disk that stays slow can
never eat the memory. When it is full a new frame is dropped, not written, and counted:
the ``i`` numbers in the index skip it (records and index lines stay in step), and
``close`` writes the count to ``<name>.lost.json`` when any was lost. The link itself is
never held up.
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
# At most this many frames wait for the disk (a frame and its index line are under 0.5 kB,
# so some 25 MB). A link at a few hundred frames per second fills it only when the disk has
# stalled for minutes.
QUEUE_MAX: Final = 50_000


class TlogWriter:
    """Append-only writer for ``<path>`` and ``<path stem>.idx.jsonl``. ``write`` and
    ``flush`` only queue; the writer thread does the file work in order, so a stalled
    disk never holds up the caller. With ``queue_max`` frames waiting, a new frame is
    dropped and counted in ``lost``. ``close`` waits until everything is on file. A write
    that fails stops the thread; the next call (or ``close``) raises that error."""

    def __init__(self, path: Path, *, queue_max: int = QUEUE_MAX) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._tlog: IO[bytes] = path.open("ab")
        self._idx: IO[str] = path.with_suffix(".idx.jsonl").open("a", encoding="utf-8")
        self._count = 0
        self.lost = 0  # frames dropped because the queue was full
        self._error: Exception | None = None
        # +2: room for the flush and close markers, which are never dropped.
        self._queue: queue.Queue[tuple[bytes, str] | bytes | None] = queue.Queue(queue_max + 2)
        self._frames_max = queue_max
        self._flush_queued = threading.Event()
        self._thread = threading.Thread(target=self._drain, name=f"tlog-{path.name}", daemon=True)
        self._thread.start()

    def write(self, data: bytes, direction: str, *, dropped: bool = False) -> None:
        self._raise_failed()
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
            self._count += 1
            if self._queue.qsize() >= self._frames_max:
                self.lost += 1  # the disk is behind: drop this frame, never wait
                continue
            self._queue.put_nowait(
                (struct.pack(">Q", wall_us) + frame.raw, json.dumps(entry) + "\n")
            )

    def flush(self) -> None:
        self._raise_failed()
        if self._flush_queued.is_set():
            return  # one is already waiting: a second one would only take a frame's place
        self._flush_queued.set()
        try:
            self._queue.put_nowait(_FLUSH)
        except queue.Full:  # cannot happen (the markers have room), but never wait
            self._flush_queued.clear()

    def close(self) -> None:
        if self._thread.is_alive():
            self._queue.put(_CLOSE)  # waits only for the thread, which drains the queue
            self._thread.join()
        self._tlog.close()
        self._idx.close()
        if self.lost:
            lost = self.path.with_suffix(".lost.json")
            lost.write_text(json.dumps({"lost_frames": self.lost}) + "\n", encoding="utf-8")
        self._raise_failed()

    def _raise_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def _drain(self) -> None:
        try:
            while (item := self._queue.get()) is not _CLOSE:
                if item == _FLUSH:
                    self._flush_queued.clear()
                    self._tlog.flush()
                    self._idx.flush()
                elif isinstance(item, tuple):
                    record, line = item
                    self._tlog.write(record)
                    self._idx.write(line)
        except Exception as exc:  # the caller sees it on its next write, flush or close
            self._error = exc
