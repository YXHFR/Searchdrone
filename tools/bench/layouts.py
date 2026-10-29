"""How each dataset's archive is laid out: which members are pictures and labels, the
image id and sequence of each picture, and its person boxes.

The **sequence** groups pictures that may look nearly alike: the same flight, or the
same place on the same day. The split (``tools/bench/splits.py``) puts a whole sequence
in one part, so no near-copy of a test picture is ever trained on.

* ``heridal_voc``: ``JPEGImages/<stem>.jpg`` with ``Annotations/<stem>.xml`` (Pascal
  VOC). The id is the stem (``test_TRS_0001``), the sequence the place code (``TRS``):
  HERIDAL's own train and test pictures share places, so the place is the safe group.
* ``wisard_vis``: the visible-light folders (``<date>_<place>_..._VIS...``), one
  ``<stem>.jpg`` with a YOLO ``<stem>.txt`` each; folders without labels are left out.
  The id is ``<folder>/<stem>``, the sequence ``<date>_<place>``.
* ``kios_yolo``: KIOS's two archives side by side, ``Images/<part>/<stem>.jpg`` with
  ``Annotations/Yolo/<part>/<stem>.txt`` (YOLO, class 0 = people; the VOC files are not
  read: some name another picture in their ``<filename>``). The archive's own
  Train/Valid/Test folders are ignored: they hold frames of the same videos. The stem is
  ``<n>_<recording>_<frame>[_crop]`` and is the id; the sequence is the **site** of the
  recording (``KIOS_SITES``, read off the pictures). Recordings that show the same site
  are one sequence, and so are recordings that move between sites: ``people_0042``
  starts at the civil-defence exercise and goes on over the campus, so the campus and
  the exercise are one sequence. Every KIOS picture shows people, so a picture without
  its YOLO file, or of a recording with no site, refuses the whole set.
* ``simfarm``: the sim farm's rounds (``<round>/images/<stem>.png`` with
  ``<round>/labels/<stem>.txt``, YOLO class 0). The id is ``<round>/<stem>``, the
  sequence the round. Only for pre-training, never measured on.

Boxes are ``(x0, y0, x1, y1)`` in picture pixels; there is one class, person.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath

from bench.common import BenchError

Box = tuple[float, float, float, float]

MAX_LABEL_BYTES = 1 << 20
MAX_BOXES = 500


@dataclass(frozen=True)
class Member:
    """A picture in an archive and its label file (``None``: no label, no persons)."""

    image_id: str
    seq: str
    image: str
    label: str | None


@dataclass
class Layout:
    name: str
    members: Callable[[list[str]], list[Member]]
    boxes: Callable[[bytes | None, int, int], list[Box]]


def _clip(box: Box, width: int, height: int) -> Box | None:
    x0, y0, x1, y1 = box
    x0, x1 = max(0.0, min(x0, x1)), min(float(width), max(x0, x1))
    y0, y1 = max(0.0, min(y0, y1)), min(float(height), max(y0, y1))
    return (x0, y0, x1, y1) if x1 - x0 >= 1 and y1 - y0 >= 1 else None


def yolo_boxes(data: bytes | None, width: int, height: int, classes: frozenset[int]) -> list[Box]:
    """YOLO lines ``class cx cy w h`` in fractions of the picture."""
    if not data:
        return []
    if len(data) > MAX_LABEL_BYTES:
        raise ValueError("label file too large")
    out: list[Box] = []
    for line in data.decode("utf-8", errors="strict").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"bad YOLO line: {line[:80]!r}")
        cls, cx, cy, w, h = int(parts[0]), *(float(p) for p in parts[1:])
        if cls not in classes:
            continue
        box = _clip(
            (
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            ),
            width,
            height,
        )
        if box:
            out.append(box)
    if len(out) > MAX_BOXES:
        raise ValueError("too many boxes in one label file")
    return out


def voc_boxes(data: bytes | None, width: int, height: int) -> list[Box]:
    """Pascal VOC objects named ``person``."""
    if not data:
        return []
    if len(data) > MAX_LABEL_BYTES or b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        raise ValueError("label file too large or with a DTD")
    root = ET.fromstring(data)  # noqa: S314 - checked archive, no DTD (refused above)
    out: list[Box] = []
    for obj in root.iter("object"):
        if (obj.findtext("name") or "").strip().lower() != "person":
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        vals = [float(bb.findtext(k) or "nan") for k in ("xmin", "ymin", "xmax", "ymax")]
        box = _clip((vals[0], vals[1], vals[2], vals[3]), width, height)
        if box:
            out.append(box)
    if len(out) > MAX_BOXES:
        raise ValueError("too many boxes in one label file")
    return out


def _stem(name: str) -> str:
    return PurePosixPath(name).stem


def _heridal(names: list[str]) -> list[Member]:
    present = set(names)
    out: list[Member] = []
    for name in names:
        p = PurePosixPath(name)
        if p.parts[0] == "__MACOSX" or len(p.parts) < 2 or p.parent.name != "JPEGImages":
            continue
        if p.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        stem = p.stem
        if not re.fullmatch(r"(train|test)_[A-Z]+_\d+", stem):
            continue
        label = str(p.parent.parent / "Annotations" / f"{stem}.xml")
        seq = sequence_of("heridal_voc", stem)
        out.append(Member(stem, seq, name, label if label in present else None))
    return out


def _wisard(names: list[str]) -> list[Member]:
    present = set(names)
    out: list[Member] = []
    for name in names:
        p = PurePosixPath(name)
        if (
            len(p.parts) != 2
            or "_VIS" not in p.parts[0]
            or p.suffix.lower() not in (".jpg", ".jpeg")
        ):
            continue
        label = str(p.with_suffix(".txt"))
        if label not in present:
            continue  # flights without labels are left out
        image_id = f"{p.parts[0]}/{p.stem}"
        out.append(Member(image_id, sequence_of("wisard_vis", image_id), name, label))
    return out


# KIOS recording -> site. Read off the pictures; a recording not listed here is refused
# (``sequence_of``), so a changed archive can never slip a new site into a part unseen.
KIOS_SITES: dict[str, str] = {
    "people_0007": "campus_exercise",
    "people_0008": "campus_exercise",
    "people_0042": "campus_exercise",  # the exercise first, then the campus
    "people_0043": "campus_exercise",
    "people_0049": "campus_exercise",
    "people_0050": "campus_exercise",
    "people_0085": "campus_exercise",
    "peoplepegia": "pegia_coast",
    "ppl": "shore",
    "ppl_2": "shore",
    "ppl_3": "yard",
    "ppl_4": "meadow",
}
_KIOS_STEM = re.compile(r"\d+_(?P<rec>[a-z]+(?:_\d+)?)_\d+(?:_crop)?")
_KIOS_PARTS = ("Train", "Valid", "Test")


def _kios(names: list[str]) -> list[Member]:
    present = set(names)
    out: list[Member] = []
    for name in names:
        p = PurePosixPath(name)
        if len(p.parts) != 3 or p.parts[0] != "Images" or p.parts[1] not in _KIOS_PARTS:
            continue
        if p.suffix.lower() != ".jpg" or not _KIOS_STEM.fullmatch(p.stem):
            continue
        label = f"Annotations/Yolo/{p.parts[1]}/{p.stem}.txt"
        try:
            seq = sequence_of("kios_yolo", p.stem)
        except ValueError as exc:
            raise BenchError(f"kios: {name} hör till en inspelning utan plats ({exc})") from exc
        # every KIOS picture has people and a YOLO file; one without would count as empty
        if label not in present:
            raise BenchError(f"kios: {name} saknar etikettfil {label}")
        out.append(Member(p.stem, seq, name, label))
    return out


def _simfarm(names: list[str]) -> list[Member]:
    present = set(names)
    out: list[Member] = []
    for name in names:
        p = PurePosixPath(name)
        if len(p.parts) != 3 or p.parts[1] != "images" or p.suffix.lower() != ".png":
            continue
        image_id = f"{p.parts[0]}/{p.stem}"
        label = f"{p.parts[0]}/labels/{p.stem}.txt"
        seq = sequence_of("simfarm", image_id)
        out.append(Member(image_id, seq, name, label if label in present else None))
    return out


PERSON = frozenset({0})

LAYOUTS: dict[str, Layout] = {
    "heridal_voc": Layout("heridal_voc", _heridal, voc_boxes),
    "wisard_vis": Layout("wisard_vis", _wisard, lambda d, w, h: yolo_boxes(d, w, h, PERSON)),
    "kios_yolo": Layout("kios_yolo", _kios, lambda d, w, h: yolo_boxes(d, w, h, PERSON)),
    "simfarm": Layout("simfarm", _simfarm, lambda d, w, h: yolo_boxes(d, w, h, PERSON)),
}


def sequence_of(layout: str, image_id: str) -> str:
    """The sequence an image id belongs to: the one rule the layouts and the split use."""
    if layout == "heridal_voc":
        return image_id.split("_")[1]
    if layout == "wisard_vis":
        return "_".join(image_id.split("/")[0].split("_")[:2])
    if layout == "kios_yolo":
        m = _KIOS_STEM.fullmatch(image_id)
        if m is None or m["rec"] not in KIOS_SITES:
            raise ValueError(f"KIOS picture of an unknown recording: {image_id!r}")
        return KIOS_SITES[m["rec"]]
    if layout == "simfarm":
        return image_id.split("/")[0]
    raise KeyError(layout)
