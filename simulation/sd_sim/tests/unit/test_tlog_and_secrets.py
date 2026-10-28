import json
import struct
import threading
import time
from pathlib import Path

from pymavlink.dialects.v20 import ardupilotmega as mavlink2

from sd_sim import launch
from sd_sim.errors import InfraError
from sd_sim.tlog import TlogWriter, split_frames


def heartbeat(sysid: int, compid: int) -> bytes:
    mav = mavlink2.MAVLink(None, srcSystem=sysid, srcComponent=compid)
    return mav.heartbeat_encode(6, 8, 0, 0, 4).pack(mav)


def test_split_frames_v2_and_v1() -> None:
    data = heartbeat(250, 190) + b"\x00junk" + heartbeat(1, 1)
    frames = split_frames(data)
    assert [(f.sysid, f.compid, f.msgid) for f in frames] == [(250, 190, 0), (1, 1, 0)]
    v1 = bytes([0xFE, 1, 7, 253, 190, 42, 0xAA, 0x11, 0x22])
    assert [(f.sysid, f.msgid) for f in split_frames(v1)] == [(253, 42)]
    assert split_frames(heartbeat(1, 1)[:-3]) == []


def test_tlog_writer(tmp_path: Path) -> None:
    path = tmp_path / "radio.tlog"
    writer = TlogWriter(path)
    frame = heartbeat(250, 190)
    writer.write(frame, "to_vehicle")
    writer.write(frame, "from_vehicle", dropped=True)
    writer.close()
    raw = path.read_bytes()
    assert len(raw) == 2 * (8 + len(frame))
    assert struct.unpack(">Q", raw[:8])[0] > 0 and raw[8 : 8 + len(frame)] == frame
    idx = [json.loads(line) for line in path.with_suffix(".idx.jsonl").read_text().splitlines()]
    assert idx[0]["dir"] == "to_vehicle" and idx[0]["sysid"] == 250 and "dropped" not in idx[0]
    assert idx[1]["dropped"] is True


def test_tlog_writer_does_not_wait_for_the_disk(tmp_path: Path) -> None:
    # The relays write every datagram to their tap before they forward it: a disk that
    # stalls must not stall the link.
    path = tmp_path / "radio.tlog"
    writer = TlogWriter(path)
    real, disk_free = writer._tlog, threading.Event()

    class StalledDisk:
        def write(self, data: bytes) -> int:
            disk_free.wait()
            return real.write(data)

        def flush(self) -> None:
            real.flush()

        def close(self) -> None:
            real.close()

    writer._tlog = StalledDisk()  # type: ignore[assignment]
    frame = heartbeat(250, 190)
    threading.Timer(2.0, disk_free.set).start()  # the stall ends, even when it would block
    start = time.monotonic()
    for _ in range(50):
        writer.write(frame, "to_vehicle")
    writer.flush()
    assert time.monotonic() - start < 0.5
    disk_free.set()
    writer.close()
    assert len(path.read_bytes()) == 50 * (8 + len(frame))
    idx = path.with_suffix(".idx.jsonl").read_text().splitlines()
    assert [json.loads(line)["i"] for line in idx] == list(range(50))


def test_write_secret_modes(tmp_path: Path) -> None:
    folder = tmp_path / "secrets"
    path = launch.write_secret(folder, "linksim_token", "abc")
    assert path.stat().st_mode & 0o777 == 0o600
    assert folder.stat().st_mode & 0o777 == 0o700
    launch.check_secret_modes(folder)
    path.chmod(0o644)
    try:
        launch.check_secret_modes(folder)
    except InfraError:
        pass
    else:
        raise AssertionError("expected InfraError for mode 0644")
