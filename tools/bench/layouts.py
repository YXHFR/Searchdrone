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
  its YOLO file, of a recording with no site, or with a name of another form refuses
  the whole set.
* ``kios_carperson``: KIOS's parking-lot flights (Zenodo 21216709). Three picture
  archives hold one zip per video (``<flight>/DJI_<time>_<NNNN>_D.zip``, opened as
  nested zips: ``<flight>/DJI_<time>_<NNNN>_D/DJI_<time>_<NNNN>_D/frame_<n>.jpg``); the
  labels are ``YOLO/<NN>D/frame_<n>.txt`` (class 0 = Human, 1 = Car; only Human is read).
  Neighbouring frames of a video are near-copies, so only every tenth frame is taken
  (``KIOS_CARPERSON_STEP``). The id is ``<video>/frame_<n>``, the sequence the video:
  a whole video is in one part. A frame without its label file refuses the set.
* ``seadronessee_coco``: SeaDronesSee Object Detection v2 (``images/<part>/<n>.jpg``
  with ``annotations/instances_<part>.json``, COCO). The labels are one file for all
  pictures, so this layout reads them (``Layout.find`` with a reader). Only the class
  ``swimmer`` is a person; boats, jet skis, buoys and life-saving appliances are left
  out, and a picture with none of them is an empty picture (water, no person). The
  authors' own train and val parts hold frames of the same videos, so they are pooled:
  the id is ``<video>/<n>`` and the sequence the video (``source.folder_name``).
* ``simfarm``: the sim farm's rounds (``<round>/images/<stem>.png`` with
  ``<round>/labels/<stem>.txt``, YOLO class 0). The id is ``<round>/<stem>``, the
  sequence the round. Only for pre-training, never measured on.

Boxes are ``(x0, y0, x1, y1)`` in picture pixels; there is one class, person.
"""

from __future__ import annotations

import json
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
    """A picture in an archive and its label file (``None``: no label, no persons).
    ``boxes``: the person boxes when the layout has read them already (one label file
    for all pictures); they are clipped to the picture when it is indexed."""

    image_id: str
    seq: str
    image: str
    label: str | None
    boxes: tuple[Box, ...] | None = None


Reader = Callable[[str], bytes]


@dataclass
class Layout:
    name: str
    members: Callable[[list[str]], list[Member]]
    boxes: Callable[[bytes | None, int, int], list[Box]]
    # pictures in zips inside the archive: each ``<a>.zip`` member is opened and its
    # files are named ``<a>/<name>``
    nested: bool = False
    # a set whose labels are one file for all pictures: its members from the names and
    # a reader of those label files
    catalogue: Callable[[list[str], Reader], list[Member]] | None = None

    def find(self, names: list[str], read: Reader | None = None) -> list[Member]:
        """The members among ``names`` (``read``: reads a label file, for a catalogue)."""
        if self.catalogue is None:
            return self.members(names)
        if read is None:
            raise BenchError(f"{self.name}: etiketterna måste läsas för att hitta bilderna")
        return self.catalogue(names, read)


def clip(box: Box, width: int, height: int) -> Box | None:
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
        box = clip(
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
        box = clip((vals[0], vals[1], vals[2], vals[3]), width, height)
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
        if p.suffix.lower() != ".jpg":
            continue
        if not _KIOS_STEM.fullmatch(p.stem):
            # a picture named in a way not seen before could be a new recording
            raise BenchError(f"kios: {name} har ett okänt namn")
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


# KIOS car/person: only every tenth frame (a video's neighbouring frames are near-copies)
KIOS_CARPERSON_STEP = 10
_KCP_FLIGHTS = ("40m-random-walking", "60m-random-walking", "60m-crowd-moving-forward")
_KCP_VIDEO = re.compile(r"DJI_\d{14}_(?P<n>\d{4})_D")
_KCP_FRAME = re.compile(r"frame_(?P<n>\d{6})")


def _kios_carperson(names: list[str]) -> list[Member]:
    present = set(names)
    out: list[Member] = []
    for name in names:
        p = PurePosixPath(name)
        if len(p.parts) != 4 or p.parts[0] not in _KCP_FLIGHTS or p.suffix.lower() != ".jpg":
            continue
        video = _KCP_VIDEO.fullmatch(p.parts[1])
        frame = _KCP_FRAME.fullmatch(p.stem)
        if video is None or p.parts[2] != p.parts[1] or frame is None or p.suffix != ".jpg":
            # a picture named in a way not seen before could be a video of another kind
            raise BenchError(f"kios_carperson: {name} har ett okänt namn")
        if int(frame["n"]) % KIOS_CARPERSON_STEP:
            continue
        label = f"YOLO/{video['n'][-2:]}D/{p.stem}.txt"
        # every frame shows people; one without its labels would count as empty
        if label not in present:
            raise BenchError(f"kios_carperson: {name} saknar etikettfil {label}")
        image_id = f"{p.parts[1]}/{p.stem}"
        out.append(Member(image_id, sequence_of("kios_carperson", image_id), name, label))
    return out


# SeaDronesSee: the COCO class that is a person, and a video folder name safe in an id
SDS_PERSON = "swimmer"
_SDS_FOLDER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}")
_SDS_FILE = re.compile(r"\d{1,9}\.jpg")
_SDS_LABELS = re.compile(r"annotations/instances_(?P<part>train|val)\.json")
MAX_COCO_BYTES = 64 << 20


def _labels_first(names: list[str]) -> list[Member]:
    raise BenchError("seadronessee: etiketterna måste läsas för att hitta bilderna")


def _coco(data: bytes, where: str) -> tuple[list[tuple[int, str, str]], dict[int, list[Box]]]:
    """A SeaDronesSee COCO file: its pictures ``(id, file name, video folder)`` and the
    person boxes ``(x0, y0, x1, y1)`` by picture id."""
    if len(data) > MAX_COCO_BYTES:
        raise BenchError(f"seadronessee: {where} är orimligt stor")
    try:
        body = json.loads(data)
        person = {int(c["id"]) for c in body["categories"] if c["name"] == SDS_PERSON}
        if len(person) != 1:
            raise ValueError(f"no single class {SDS_PERSON!r}")
        boxes: dict[int, list[Box]] = {}
        for a in body["annotations"]:
            if int(a["category_id"]) in person:
                x, y, w, h = (float(v) for v in a["bbox"])
                boxes.setdefault(int(a["image_id"]), []).append((x, y, x + w, y + h))
        images = [
            (int(im["id"]), str(im["file_name"]), str(im["source"]["folder_name"]))
            for im in body["images"]
        ]
    except (ValueError, KeyError, TypeError) as exc:
        raise BenchError(f"seadronessee: {where} går inte att läsa ({exc})") from exc
    return images, boxes


def _seadronessee(names: list[str], read: Reader) -> list[Member]:
    present = set(names)
    out: list[Member] = []
    for labels in sorted(n for n in names if _SDS_LABELS.fullmatch(n)):
        part = labels.removeprefix("annotations/instances_").removesuffix(".json")
        images, boxes = _coco(read(labels), labels)
        for key, file_name, folder in images:
            image = f"images/{part}/{file_name}"
            if not _SDS_FOLDER.fullmatch(folder) or not _SDS_FILE.fullmatch(file_name):
                raise BenchError(f"seadronessee: {image} har ett okänt namn ({folder})")
            if image not in present:
                raise BenchError(f"seadronessee: {image} saknas")
            found = boxes.get(key, [])
            if len(found) > MAX_BOXES:
                raise BenchError(f"seadronessee: för många personer i {image}")
            image_id = f"{folder}/{PurePosixPath(file_name).stem}"
            seq = sequence_of("seadronessee_coco", image_id)
            out.append(Member(image_id, seq, image, labels, tuple(found)))
    if not out:
        raise BenchError("seadronessee: inga etiketter (annotations/instances_*.json) hittades")
    if len({m.image_id for m in out}) != len(out):
        raise BenchError("seadronessee: samma bild står två gånger i etiketterna")
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
    # class 0 is Human, 1 Car
    "kios_carperson": Layout(
        "kios_carperson",
        _kios_carperson,
        lambda d, w, h: yolo_boxes(d, w, h, PERSON),
        nested=True,
    ),
    # the boxes come with the members (``Member.boxes``), read from the COCO files
    "seadronessee_coco": Layout(
        "seadronessee_coco", _labels_first, lambda d, w, h: [], catalogue=_seadronessee
    ),
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
    if layout in ("simfarm", "kios_carperson", "seadronessee_coco"):
        return image_id.split("/")[0]
    raise KeyError(layout)
