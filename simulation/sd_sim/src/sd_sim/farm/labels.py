"""Facit for the farm's pictures: boxes from the ground truth, checked against Gazebo.

For every picture the people in the world's facit (``truth.json``) are projected through
the camera's pose (``sd_sim.farm.camera``): the box around each person's eight box
corners is the label. The simulator's bounding-box camera, which sees the same picture
and reports what is actually visible of every labelled model, is the check: each of its
person boxes must lie inside a projected box (``containment``), and an unhidden person's
projected box must overlap it well (``iou``). A person whose projected box is in view but
whom the bounding-box camera does not see (under the trees) gets no label; the metadata
lists them as hidden.

Formats: YOLO (one ``<image>.txt`` per picture, class 0 = person, centre and size in
fractions of the picture) and COCO (one ``coco.json`` per round, category 1 = person).
The metadata JSON per picture (``simfarm.image/v1``) has the world, time of day,
weather, height, the camera's pose and intrinsics, the git commit, the people with their
boxes and the person models' credits.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sd_sim import png
from sd_sim.farm.camera import Box, CameraPose, Intrinsics, box
from sd_sim.farm.people import PERSON_LABEL, Person

IMAGE_SCHEMA = "simfarm.image/v1"
YOLO_CLASS = 0
COCO_CATEGORY = {"id": 1, "name": "person", "supercategory": "person"}
MIN_BOX_PX = 2.0  # a label smaller than this (either side) is too small to keep
MIN_IN_VIEW = 0.25  # at least this share of a person's box inside the picture
CONTAINED = 0.8  # a camera box lies inside a projected box when this share of it does
UNHIDDEN = 0.8  # visible share of the projected box above which the person counts as open
BLACK_MEAN = 8.0  # mean grey level (0-255) under which a picture is black
FLAT_STD = 2.0  # spread of grey levels under which a picture is one flat colour


@dataclass(frozen=True)
class PersonLabel:
    name: str
    box: Box  # projected, clipped to the picture
    in_view: float  # share of the unclipped projected box inside the picture
    seen: Box | None  # the bounding-box camera's box of this person, if any
    model: str
    pose: str

    @property
    def iou(self) -> float | None:
        return None if self.seen is None else self.box.iou(self.seen)

    @property
    def containment(self) -> float | None:
        if self.seen is None or self.seen.area <= 0:
            return None
        return self.seen.intersection(self.box) / self.seen.area

    @property
    def visible(self) -> float:
        """Share of the projected box the camera sees (0: hidden)."""
        if self.seen is None or self.box.area <= 0:
            return 0.0
        return min(1.0, self.seen.area / self.box.area)

    @property
    def labelled(self) -> bool:
        return (
            self.seen is not None
            and self.in_view >= MIN_IN_VIEW
            and self.box.w >= MIN_BOX_PX
            and self.box.h >= MIN_BOX_PX
        )

    def as_dict(self) -> dict[str, Any]:
        def b(v: Box | None) -> list[float] | None:
            return (
                None
                if v is None
                else [round(v.x0, 2), round(v.y0, 2), round(v.x1, 2), round(v.y1, 2)]
            )

        return {
            "name": self.name,
            "model": self.model,
            "pose": self.pose,
            "box_xyxy": b(self.box),
            "seen_xyxy": b(self.seen),
            "in_view": round(self.in_view, 3),
            "visible": round(self.visible, 3),
            "iou": None if self.iou is None else round(self.iou, 3),
            "labelled": self.labelled,
        }


@dataclass
class ImageFacit:
    people: list[PersonLabel] = field(default_factory=list)
    unmatched_seen: list[Box] = field(default_factory=list)  # camera boxes with no person

    @property
    def labels(self) -> list[PersonLabel]:
        return [p for p in self.people if p.labelled]

    @property
    def hidden(self) -> list[PersonLabel]:
        return [p for p in self.people if p.seen is None]


def people_in(truth: Mapping[str, Any]) -> list[tuple[Person, float]]:
    """Every person in the world's facit with the ground height under it."""
    out = []
    for row in truth.get("objects", []):
        if row.get("truth", {}).get("cls") == "person":
            out.append((Person.from_truth(row), float(row.get("ground_m", 0.0))))
    return out


def camera_boxes(boxes: Iterable[Mapping[str, Any]] | None, label: int = PERSON_LABEL) -> list[Box]:
    return [
        Box(float(b["x0"]), float(b["y0"]), float(b["x1"]), float(b["y1"]))
        for b in boxes or []
        if int(b["label"]) == label
    ]


def facit(
    truth: Mapping[str, Any],
    cam: Intrinsics,
    pose: CameraPose,
    seen_boxes: Sequence[Box],
) -> ImageFacit:
    """Project every person and pair each with at most one bounding-box camera box (the
    one most contained in it, greedily by containment then IoU). A person mostly outside
    the picture (``in_view`` below ``MIN_IN_VIEW``) is still matched, so the camera's box
    of its visible part is not taken for a stray, but it gets no label."""
    projected: list[tuple[Person, Box, float]] = []
    for person, ground in people_in(truth):
        got = box(cam, pose, person.corners(ground))
        if got is None:
            continue
        projected.append((person, got[0], got[1]))
    pairs: list[tuple[float, float, int, int]] = []
    for i, (_, pbox, _) in enumerate(projected):
        for j, sbox in enumerate(seen_boxes):
            if sbox.area <= 0:
                continue
            contain = sbox.intersection(pbox) / sbox.area
            if contain > 0:
                pairs.append((contain, pbox.iou(sbox), i, j))
    pairs.sort(reverse=True)
    match: dict[int, int] = {}
    used: set[int] = set()
    for _, _, i, j in pairs:
        if i in match or j in used:
            continue
        match[i] = j
        used.add(j)
    out = ImageFacit()
    for i, (person, pbox, in_view) in enumerate(projected):
        seen = seen_boxes[match[i]] if i in match else None
        out.people.append(PersonLabel(person.name, pbox, in_view, seen, person.model, person.pose))
    out.unmatched_seen = [b for j, b in enumerate(seen_boxes) if j not in used]
    return out


def yolo_lines(labels: Sequence[PersonLabel], cam: Intrinsics) -> str:
    rows = []
    for lab in labels:
        b = lab.box
        cx = (b.x0 + b.x1) / 2 / cam.width
        cy = (b.y0 + b.y1) / 2 / cam.height
        rows.append(f"{YOLO_CLASS} {cx:.6f} {cy:.6f} {b.w / cam.width:.6f} {b.h / cam.height:.6f}")
    return "".join(r + "\n" for r in rows)


class Coco:
    """One round's COCO file, built picture by picture."""

    def __init__(self, info: Mapping[str, Any]) -> None:
        self.data: dict[str, Any] = {
            "info": dict(info),
            "licenses": [],
            "categories": [COCO_CATEGORY],
            "images": [],
            "annotations": [],
        }

    def add(self, file_name: str, cam: Intrinsics, labels: Sequence[PersonLabel]) -> int:
        image_id = len(self.data["images"]) + 1
        self.data["images"].append(
            {"id": image_id, "file_name": file_name, "width": cam.width, "height": cam.height}
        )
        for lab in labels:
            b = lab.box
            self.data["annotations"].append(
                {
                    "id": len(self.data["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": COCO_CATEGORY["id"],
                    "bbox": [round(b.x0, 2), round(b.y0, 2), round(b.w, 2), round(b.h, 2)],
                    "area": round(b.area, 2),
                    "iscrowd": 0,
                    "attributes": {"pose": lab.pose, "visible": round(lab.visible, 3)},
                }
            )
        return image_id

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.data, indent=1) + "\n", encoding="utf-8")


def read_ppm(path: Path) -> NDArray[np.uint8]:
    """A binary PPM (P6, 8 bit) as written by ``gz_stills``."""
    data = path.read_bytes()
    tokens: list[bytes] = []
    i = 0
    while len(tokens) < 4 and i < len(data):
        c = data[i : i + 1]
        if c == b"#":  # a comment runs to the end of the line
            end = data.find(b"\n", i)
            i = len(data) if end < 0 else end + 1
        elif c.isspace():
            i += 1
        else:
            start = i
            while i < len(data) and not data[i : i + 1].isspace():
                i += 1
            tokens.append(data[start:i])
    # Exactly one whitespace byte ends the header; the pixels follow (any byte value).
    if len(tokens) < 4 or tokens[0] != b"P6" or tokens[3] != b"255" or i >= len(data):
        raise ValueError(f"{path}: ingen 8-bitars P6-bild")
    try:
        width, height = int(tokens[1]), int(tokens[2])
    except ValueError as exc:
        raise ValueError(f"{path}: ingen 8-bitars P6-bild") from exc
    body = data[i + 1 :]
    if len(body) < width * height * 3:
        raise ValueError(f"{path}: bilden är kortare än {width} x {height}")
    return np.frombuffer(body[: width * height * 3], dtype=np.uint8).reshape(height, width, 3)


def brightness(image: NDArray[np.uint8]) -> tuple[float, float]:
    """Mean and spread of the grey level, 0-255."""
    grey = image.astype(np.float32).mean(axis=2)
    return float(grey.mean()), float(grey.std())


def is_black(image: NDArray[np.uint8]) -> bool:
    mean, std = brightness(image)
    return mean < BLACK_MEAN or std < FLAT_STD


def write_png(image: NDArray[np.uint8], path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(png.encode(image))
    tmp.replace(path)
