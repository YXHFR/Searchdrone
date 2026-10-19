import json
import struct
from pathlib import Path

from pymavlink.dialects.v20 import ardupilotmega as mavlink2

from sd_sim.perf.tlogack import first_after, stop_acks
from sd_sim.tlog import TlogWriter, split_frames
from sd_sim.udprelay import FROM_VEHICLE, TO_VEHICLE

MAV = mavlink2.MAVLink(None, srcSystem=1, srcComponent=1)
GCS = mavlink2.MAVLink(None, srcSystem=255, srcComponent=190)


def ack(command: int, result: int) -> bytes:
    return MAV.command_ack_encode(command, result, 0, 0, 255, 190).pack(MAV)


def heartbeat() -> bytes:
    return MAV.heartbeat_encode(2, 3, 0, 0, 4).pack(MAV)


def set_mode_cmd() -> bytes:
    return GCS.command_long_encode(1, 1, 176, 0, 1, 17, 0, 0, 0, 0, 0).pack(GCS)


def write_log(path: Path, records: list[tuple[bytes, str, int, bool]]) -> tuple[Path, Path]:
    """Same layout as ``TlogWriter``, with chosen monotonic times."""
    idx = path.with_suffix(".idx.jsonl")
    with path.open("wb") as tlog, idx.open("w", encoding="utf-8") as lines:
        for i, (raw, direction, mono, dropped) in enumerate(records):
            frame = split_frames(raw)[0]
            tlog.write(struct.pack(">Q", 1_700_000_000_000_000 + i) + raw)
            entry = {
                "i": i,
                "dir": direction,
                "mono_ns": mono,
                "sysid": frame.sysid,
                "compid": frame.compid,
                "msgid": frame.msgid,
            }
            if dropped:
                entry["dropped"] = True
            lines.write(json.dumps(entry) + "\n")
    return path, idx


def test_stop_acks_filters(tmp_path: Path) -> None:
    tlog, idx = write_log(
        tmp_path / "radio.tlog",
        [
            (heartbeat(), FROM_VEHICLE, 100, False),
            (set_mode_cmd(), TO_VEHICLE, 200, False),
            (ack(176, 0), FROM_VEHICLE, 300, False),  # the one we want
            (ack(176, 4), FROM_VEHICLE, 400, False),  # FAILED
            (ack(400, 0), FROM_VEHICLE, 500, False),  # ARM, not a mode change
            (ack(176, 0), FROM_VEHICLE, 600, True),  # dropped by the link simulator
            (ack(176, 0), TO_VEHICLE, 700, False),  # wrong direction
            (heartbeat(), FROM_VEHICLE, 800, False),
            (ack(176, 0), FROM_VEHICLE, 900, False),
        ],
    )
    assert stop_acks(tlog, idx) == [300, 900]


def test_stop_acks_partial_trailing_record(tmp_path: Path) -> None:
    tlog, idx = write_log(
        tmp_path / "radio.tlog",
        [(ack(176, 0), FROM_VEHICLE, 10, False), (ack(176, 0), FROM_VEHICLE, 20, False)],
    )
    data = tlog.read_bytes()
    tlog.write_bytes(data[:-5])  # the writer was in the middle of the second record
    assert stop_acks(tlog, idx) == [10]
    tlog.write_bytes(data[: -len(ack(176, 0)) - 3])  # cut inside the timestamp
    assert stop_acks(tlog, idx) == [10]


def test_stop_acks_index_behind_and_half_line(tmp_path: Path) -> None:
    tlog, idx = write_log(
        tmp_path / "radio.tlog",
        [(ack(176, 0), FROM_VEHICLE, 10, False), (ack(176, 0), FROM_VEHICLE, 20, False)],
    )
    lines = idx.read_text(encoding="utf-8").splitlines()
    idx.write_text(lines[0] + "\n" + lines[1][:12], encoding="utf-8")
    assert stop_acks(tlog, idx) == [10]


def test_stop_acks_missing_files(tmp_path: Path) -> None:
    assert stop_acks(tmp_path / "none.tlog", tmp_path / "none.idx.jsonl") == []


def test_stop_acks_reads_tlog_writer_output(tmp_path: Path) -> None:
    path = tmp_path / "radio.tlog"
    writer = TlogWriter(path)
    writer.write(heartbeat() + ack(176, 0), FROM_VEHICLE)  # two frames in one datagram
    writer.write(set_mode_cmd(), TO_VEHICLE)
    writer.write(ack(176, 0), FROM_VEHICLE, dropped=True)
    writer.close()
    idx = path.with_suffix(".idx.jsonl")
    entries = [json.loads(line) for line in idx.read_text(encoding="utf-8").splitlines()]
    assert stop_acks(path, idx) == [entries[1]["mono_ns"]]


def test_first_after() -> None:
    acks = [1_000, 5_000_000_000, 6_000_000_000, 9_000_000_000]
    assert first_after(acks, 4_000_000_000) == 5_000_000_000
    assert first_after(acks, 5_000_000_000) == 5_000_000_000
    assert first_after(acks, 6_100_000_000, within_s=2.5) is None  # 9 s is 2.9 s later
    assert first_after(acks, 6_100_000_000) == 9_000_000_000
    assert first_after([], 0) is None
    assert first_after([9_000_000_000, 5_000_000_000], 0, within_s=10) == 5_000_000_000
