"""Stills from Gazebo's cameras, written to a folder. Runs with the system Python.

Gazebo's Python bindings (``python3-gz-transport13``, ``python3-gz-msgs10``) are
installed for the system's Python 3.12 only, so this file is a standalone script with
nothing but the standard library and those bindings, imported only where Gazebo is
needed (so the rest can be tested anywhere); ``sd_sim.render_stills`` starts it as a
separate process (``/usr/bin/python3.12 gz_stills.py shots ...``) and reads the folder.

    shots --world W --model M --poses FILE --image-topic T --boxes-topic B --out DIR
        Moves the static camera model M to every pose in FILE (a JSON list of
        ``{"name", "x", "y", "z", "roll", "pitch", "yaw"}``, metres and radians in the
        world frame) and saves one picture per pose, named after the pose: ``DIR/<name>.ppm``
        plus ``DIR/<name>.json`` (simulator time, size, the pose and the bounding-box
        camera's labelled boxes of the same moment). The JSON file is written last, by
        rename, so a reader that sees it also finds the whole picture.

Gazebo Transport stays on loopback: the caller sets ``GZ_IP=127.0.0.1`` and the
instance's ``GZ_PARTITION``.
"""
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

RGB_INT8 = 3  # gz.msgs.PixelFormatType
BOX_WAIT_S = 3.0
SHOT_WAIT_S = 30.0
SET_POSE_TRIES = 5
SETTLE_FRAMES = 4
MAX_SETTLE_TRIES = 20  # pictures per pose to wait for boxes that hold still


def _stamp_ns(msg: Any) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nsec)


def _boxes(msg: Any) -> list[dict[str, float]]:
    return [
        {
            "label": int(b.label),
            "x0": float(b.box.min_corner.x),
            "y0": float(b.box.min_corner.y),
            "x1": float(b.box.max_corner.x),
            "y1": float(b.box.max_corner.y),
        }
        for b in msg.annotated_box
    ]


class Recorder:
    """Keeps the newest pictures and box sets by simulator time."""

    def __init__(self) -> None:
        self.lock = threading.Condition()
        self.images: dict[int, tuple[int, int, bytes]] = {}
        self.boxes: dict[int, list[dict[str, float]]] = {}
        self.newest = -1

    def on_image(self, msg: Any) -> None:
        if msg.pixel_format_type != RGB_INT8:
            return
        stamp = _stamp_ns(msg)
        with self.lock:
            self.images[stamp] = (int(msg.width), int(msg.height), bytes(msg.data))
            self.newest = max(self.newest, stamp)
            for old in [s for s in self.images if s < stamp - 10_000_000_000]:
                del self.images[old]
            self.lock.notify_all()

    def on_boxes(self, msg: Any) -> None:
        stamp = _stamp_ns(msg)
        with self.lock:
            self.boxes[stamp] = _boxes(msg)
            for old in [s for s in self.boxes if s < stamp - 10_000_000_000]:
                del self.boxes[old]
            self.lock.notify_all()

    def wait_image_after(self, stamp: int, timeout: float) -> int | None:
        end = time.monotonic() + timeout
        with self.lock:
            while self.newest <= stamp:
                left = end - time.monotonic()
                if left <= 0:
                    return None
                self.lock.wait(left)
            return self.newest

    def boxes_at(self, stamp: int, timeout: float) -> list[dict[str, float]] | None:
        end = time.monotonic() + timeout
        with self.lock:
            while stamp not in self.boxes:
                left = end - time.monotonic()
                if left <= 0:
                    return None
                self.lock.wait(left)
            return self.boxes[stamp]


def write_still(
    out: Path,
    name: str,
    stamp: int,
    image: tuple[int, int, bytes],
    boxes: list[dict[str, float]] | None,
    extra: dict[str, Any] | None = None,
) -> None:
    width, height, data = image
    ppm = out / f"{name}.ppm"
    tmp = out / f".{name}.ppm.tmp"
    with tmp.open("wb") as f:
        f.write(b"P6 %d %d 255\n" % (width, height))
        f.write(data)
    tmp.replace(ppm)
    meta = {"name": name, "sim_time_ns": stamp, "width": width, "height": height, "boxes": boxes}
    meta.update(extra or {})
    tmp = out / f".{name}.json.tmp"
    tmp.write_text(json.dumps(meta) + "\n", encoding="utf-8")
    tmp.replace(out / f"{name}.json")


def quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def shots(args: argparse.Namespace) -> int:
    from gz.msgs10.annotated_axis_aligned_2d_box_v_pb2 import AnnotatedAxisAligned2DBox_V
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.image_pb2 import Image
    from gz.msgs10.pose_pb2 import Pose
    from gz.transport13 import Node

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    poses = json.loads(Path(args.poses).read_text(encoding="utf-8"))
    rec = Recorder()
    node = Node()
    node.subscribe(Image, args.image_topic, rec.on_image)
    node.subscribe(AnnotatedAxisAligned2DBox_V, args.boxes_topic, rec.on_boxes)
    if rec.wait_image_after(-1, SHOT_WAIT_S) is None:
        sys.stderr.write("ingen bild från kameran\n")
        return 1
    for pose in poses:
        req = Pose()
        req.name = args.model
        req.position.x, req.position.y, req.position.z = pose["x"], pose["y"], pose["z"]
        w, x, y, z = quat(pose["roll"], pose["pitch"], pose["yaw"])
        req.orientation.w, req.orientation.x = w, x
        req.orientation.y, req.orientation.z = y, z
        for _ in range(SET_POSE_TRIES):
            ok, rep = node.request(f"/world/{args.world}/set_pose", req, Pose, Boolean, 5000)
            if ok and rep.data:
                break
            time.sleep(1.0)
        else:
            sys.stderr.write(f"set_pose misslyckades för {pose['name']}\n")
            return 1
        # The bounding-box camera can lag the picture by a frame or two after a move, so
        # wait several pictures and keep one whose boxes match the next picture's.
        stamp = rec.newest
        for _ in range(SETTLE_FRAMES):
            newer = rec.wait_image_after(stamp, SHOT_WAIT_S)
            if newer is None:
                sys.stderr.write(f"ingen ny bild efter flytten till {pose['name']}\n")
                return 1
            stamp = newer
        for _ in range(MAX_SETTLE_TRIES):
            boxes = rec.boxes_at(stamp, BOX_WAIT_S)
            nxt = rec.wait_image_after(stamp, SHOT_WAIT_S)
            if nxt is None:
                sys.stderr.write(f"inga fler bilder vid {pose['name']}\n")
                return 1
            if boxes is not None and rec.boxes_at(nxt, BOX_WAIT_S) == boxes:
                break
            stamp = nxt
        else:
            sys.stderr.write(f"facitrutorna blev aldrig stilla vid {pose['name']}\n")
            return 1
        with rec.lock:
            image = rec.images[stamp]
        write_still(out, pose["name"], stamp, image, boxes, {"pose": pose})
    return 0


def main(argv: list[str]) -> int:
    if os.environ.get("GZ_IP") != "127.0.0.1":
        sys.stderr.write("GZ_IP ska vara 127.0.0.1 (Gazebo Transport bara på loopback)\n")
        return 2
    parser = argparse.ArgumentParser(prog="gz_stills")
    sub = parser.add_subparsers(dest="mode", required=True)
    sh = sub.add_parser("shots")
    for key in ("--world", "--model", "--poses", "--image-topic", "--boxes-topic", "--out"):
        sh.add_argument(key, required=True)
    args = parser.parse_args(argv)
    return shots(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
