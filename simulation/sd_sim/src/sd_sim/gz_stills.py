"""Stills from Gazebo's cameras, written to a folder. Runs with the system Python.

Gazebo's Python bindings (``python3-gz-transport13``, ``python3-gz-msgs10``) are
installed for the system's Python 3.12 only, so this file is a standalone script with
nothing but the standard library and those bindings, imported only where Gazebo is
needed (so the rest can be tested anywhere); ``sd_sim`` starts it as a separate process
(``/usr/bin/python3.12 gz_stills.py ...``) and reads the folder.

    capture --image-topic T --boxes-topic B --out DIR [--keep N]
            [--poses-topic P --model M [--link L]] [--archive DIR2]
        Every picture on T becomes ``DIR/<n>.ppm`` plus ``DIR/<n>.json`` (simulator time,
        size and the labelled boxes of the same moment); only the newest N are kept (the
        reader deletes what it takes). Runs until it is stopped. Used by the simulator
        stack for a search scenario: the onboard perception reads the folder.
        With ``--poses-topic`` (the world's ``dynamic_pose/info``) every picture also
        gets the camera's pose in the world, ``"camera": {"position_m", "orientation_wxyz"}``
        (x east, y north, z up), from model M's pose and its camera link L's pose of the
        same moment and the camera's fixed mount on L. ``--archive`` keeps a copy of
        every picture that has a pose in DIR2, never pruned: the simulator farm's raw
        stills.

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
# The dynamic poses come at about 60 Hz; a pose this close to the picture is its pose.
POSE_MATCH_NS = 20_000_000
# The stills camera on its link: pitched +90 degrees about y (``<pose>0 0 0 0 90 0``).
MOUNT_WXYZ = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)

Vec = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # w, x, y, z


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


def quat_mul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def rotate(q: Quat, v: Vec) -> Vec:
    w, x, y, z = q
    vx, vy, vz = v
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def camera_pose(model: tuple[Vec, Quat], link: tuple[Vec, Quat]) -> dict[str, list[float]]:
    """The camera's pose in the world from the model's pose (in the world) and its camera
    link's pose (in the model), with the camera's mount on the link."""
    (pm, qm), (pl, ql) = model, link
    off = rotate(qm, pl)
    q = quat_mul(quat_mul(qm, ql), MOUNT_WXYZ)
    norm = math.sqrt(sum(c * c for c in q))
    return {
        "position_m": [round(pm[i] + off[i], 4) for i in range(3)],
        "orientation_wxyz": [round(c / norm, 6) for c in q],
    }


def _pose_of(p: Any) -> tuple[Vec, Quat]:
    pos, rot = p.position, p.orientation
    return (
        (float(pos.x), float(pos.y), float(pos.z)),
        (float(rot.w), float(rot.x), float(rot.y), float(rot.z)),
    )


class Recorder:
    """Keeps the newest pictures, box sets and camera poses by simulator time."""

    def __init__(self, model: str = "", link: str = "camera_link") -> None:
        self.lock = threading.Condition()
        self.images: dict[int, tuple[int, int, bytes]] = {}
        self.boxes: dict[int, list[dict[str, float]]] = {}
        self.cameras: dict[int, dict[str, list[float]]] = {}
        self.newest = -1
        self.newest_pose = -1
        self.model, self.link = model, link

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

    def on_poses(self, msg: Any) -> None:
        """A ``Pose_V`` of the world's dynamic poses: models in the world, links in their
        model. Keeps the camera's pose when both the model and its link are there."""
        found = {p.name: _pose_of(p) for p in msg.pose if p.name in (self.model, self.link)}
        if self.model not in found or self.link not in found:
            return
        stamp = _stamp_ns(msg)
        with self.lock:
            self.cameras[stamp] = camera_pose(found[self.model], found[self.link])
            self.newest_pose = max(self.newest_pose, stamp)
            for old in [s for s in self.cameras if s < stamp - 10_000_000_000]:
                del self.cameras[old]
            self.lock.notify_all()

    def camera_at(self, stamp: int, timeout: float) -> dict[str, list[float]] | None:
        """The camera's pose nearest ``stamp`` (within ``POSE_MATCH_NS``), waiting until a
        pose after it has come; ``None`` without one."""
        end = time.monotonic() + timeout
        with self.lock:
            while self.newest_pose < stamp:
                left = end - time.monotonic()
                if left <= 0:
                    break
                self.lock.wait(left)
            near = [s for s in self.cameras if abs(s - stamp) <= POSE_MATCH_NS]
            if not near:
                return None
            return self.cameras[min(near, key=lambda s: abs(s - stamp))]

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


def prune(out: Path, keep: int) -> int:
    """Delete all but the newest ``keep`` stills in ``out``; returns how many went."""
    metas = sorted(out.glob("[!.]*.json"))
    gone = 0
    for meta in metas[: max(0, len(metas) - keep)]:
        meta.with_suffix(".ppm").unlink(missing_ok=True)
        meta.unlink(missing_ok=True)
        gone += 1
    return gone


def capture(args: argparse.Namespace) -> int:
    from gz.msgs10.annotated_axis_aligned_2d_box_v_pb2 import AnnotatedAxisAligned2DBox_V
    from gz.msgs10.image_pb2 import Image
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.transport13 import Node

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = Path(args.archive) if args.archive else None
    if archive is not None:
        archive.mkdir(parents=True, exist_ok=True)
    rec = Recorder(args.model or "", args.link)
    node = Node()
    if not node.subscribe(Image, args.image_topic, rec.on_image) or not node.subscribe(
        AnnotatedAxisAligned2DBox_V, args.boxes_topic, rec.on_boxes
    ):
        sys.stderr.write("kan inte prenumerera på kamerans ämnen\n")
        return 1
    if args.poses_topic and not node.subscribe(Pose_V, args.poses_topic, rec.on_poses):
        sys.stderr.write("kan inte prenumerera på världens lägen\n")
        return 1
    last, count = -1, 0
    while True:
        stamp = rec.wait_image_after(last, 3600.0)
        if stamp is None:
            continue
        last = stamp
        boxes = rec.boxes_at(stamp, BOX_WAIT_S)
        with rec.lock:
            image = rec.images.get(stamp)
        if image is None:
            continue
        camera = rec.camera_at(stamp, BOX_WAIT_S) if args.poses_topic else None
        extra = {"camera": camera} if camera is not None else None
        name = f"{count:08d}"
        write_still(out, name, stamp, image, boxes, extra)
        if archive is not None and camera is not None:
            write_still(archive, name, stamp, image, boxes, extra)
        count += 1
        prune(out, args.keep)


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
    cap = sub.add_parser("capture")
    for key in ("--image-topic", "--boxes-topic", "--out"):
        cap.add_argument(key, required=True)
    cap.add_argument("--keep", type=int, default=20)
    cap.add_argument("--poses-topic", default="")
    cap.add_argument("--model", default="")
    cap.add_argument("--link", default="camera_link")
    cap.add_argument("--archive", default="")
    sh = sub.add_parser("shots")
    for key in ("--world", "--model", "--poses", "--image-topic", "--boxes-topic", "--out"):
        sh.add_argument(key, required=True)
    args = parser.parse_args(argv)
    if args.mode == "capture" and bool(args.poses_topic) != bool(args.model):
        parser.error("--poses-topic och --model hör ihop")
    return capture(args) if args.mode == "capture" else shots(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
