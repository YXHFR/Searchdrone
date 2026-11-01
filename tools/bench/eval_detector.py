"""``just bench-detector``: measure a person detector on the image bank.

    just bench-detector model=null mini=1          # self-check: recall 0
    just bench-detector model=facit mini=1         # self-check: recall 1
    just bench-detector model=selfcheck mini=1     # both, fails unless 0 and 1
    just bench-detector model=<name> part=test     # the locked test part (see README)

Keys: ``model`` (``null``, ``facit``, ``selfcheck``, or a trained model by name such as
``person_yolox_s`` or ``coco_yolox_s``, read from ``<data>/models/detector/<name>.json``),
``part`` (``test``, default, or ``val``), ``sets`` (datasets, comma separated; default
every measured set that is fetched), ``mini=1`` (the small CI bank instead), ``limit``
(first N pictures, for a quick look), ``data``,
``out`` (result file; default ``build/bench/det_<model>_<part>.json``), ``compare``
(an earlier result: fails when recall over 32 px drops more than 2 points, for the
weakest dataset, as ``just pick-detector`` ranks, or for any one dataset, or when a
dataset of the earlier result is missing; see ``compare``).

The detector sees one picture at a time as the drone's camera would give it; the
result file holds only numbers (no picture, no file name, no box).
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from bench.common import (
    RESULTS,
    ROOT,
    BenchError,
    data_dir,
    flag,
    inside,
    parse_args,
    safe_relpath,
    say,
    warn,
)
from bench.datasets import Image, image_root, index_path, load_register, read_index
from bench.metrics import (
    FP_PER_12MP,
    PictureResult,
    Summary,
    match_picture,
    recall_by_set,
    summarise,
    weakest_recall,
)
from bench.splits import CURRENT, MINI_VERSION, load_mini, load_split

from sd_detect.api import BBox, Detection, Detector, Frame
from sd_detect.fake import NullDetector

KEYS = ("model", "part", "sets", "mini", "limit", "data", "out", "compare")
RESULT_SCHEMA = "bench.detector/v1"
COMPARE_DROP = 0.02
PERSON = "person"


class FacitDetector:
    """Returns the picture's own person boxes with score 1: recall must come out 1.
    Only for the self-check; it is told the truth before each picture."""

    name: ClassVar[str] = "facit"
    license: ClassVar[str] = "Apache-2.0"

    def __init__(self) -> None:
        self.truth: tuple[tuple[float, float, float, float], ...] = ()

    def detect(self, image: Frame) -> list[Detection]:
        del image
        return [
            Detection(
                label=PERSON,
                score=1.0,
                bbox=BBox(x=b[0], y=b[1], w=b[2] - b[0], h=b[3] - b[1]),
                source="model",
            )
            for b in self.truth
        ]


_MODEL_NAME = re.compile(r"^[a-z0-9_]{1,40}$")

# model name -> factory; trained models are found by name in <data>/models/detector
DETECTORS: dict[str, Callable[[], Detector]] = {
    "null": NullDetector,
    "facit": FacitDetector,
}


@dataclass(frozen=True)
class Picture:
    image: Image
    path: Path


def pictures(args: dict[str, str], data: Path) -> tuple[str, list[Picture]]:
    """The pictures to measure on and the name of what they are (for the result)."""
    if flag(args, "mini"):
        split_version, rows = load_mini(MINI_VERSION)
        wanted = {r.image_id for r in rows}
        images = [i for i in read_index(data, "heridal", mini=True) if i.image_id in wanted]
        if len(images) != len(wanted):
            raise BenchError("Minibanken är ofullständig. Kör: just bench-fetch yes=1 mini=1")
        root = image_root(data, "heridal", mini=True)
        source = f"{MINI_VERSION} (splits/{split_version})"
    else:
        part = args.get("part", "test")
        if part not in ("test", "val"):
            raise BenchError("part ska vara test eller val", 2)
        split = load_split(CURRENT)
        register = load_register()
        measured = [n for n, ds in register.items() if "measure" in ds.use]
        if args.get("sets"):
            names = [s.strip() for s in args["sets"].split(",") if s.strip()]
            bad = [n for n in names if n not in measured]
            if bad:
                raise BenchError(f"Mäts inte: {', '.join(bad)}. Mätbara: {', '.join(measured)}", 2)
        else:
            names = [n for n in measured if index_path(data, n).is_file()]
            if not names:
                raise BenchError("Ingen datamängd är hämtad. Kör: just bench-fetch yes=1", 2)
        images, roots = [], {}
        for name in names:
            roots[name] = image_root(data, name)
            index = read_index(data, name)
            split.check(name, [(i.local_id, i.seq) for i in index])
            images += [i for i in index if split.part_of(name, i.seq) == part]
        source = f"splits/{split.version} {part} ({', '.join(names)})"
        if args.get("limit"):
            source += f" limit={int(args['limit'])}"  # never compared with a full run
        out = [Picture(i, inside(roots[i.dataset], safe_relpath(i.file))) for i in images]
        return source, _limit(out, args)
    return source, _limit([Picture(i, inside(root, safe_relpath(i.file))) for i in images], args)


def _limit(items: list[Picture], args: dict[str, str]) -> list[Picture]:
    items.sort(key=lambda p: p.image.image_id)
    if args.get("limit"):
        return items[: int(args["limit"])]
    return items


def load_frame(path: Path) -> Frame:
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return Frame(width=rgb.shape[1], height=rgb.shape[0], rgb=rgb.tobytes(), ts_mono_ms=0)


def run(
    detector: Detector, items: Sequence[Picture], clock: Callable[[], float] = time.perf_counter
) -> Iterator[PictureResult]:
    for n, pic in enumerate(items, 1):
        frame = load_frame(pic.path)
        if (frame.width, frame.height) != (pic.image.width, pic.image.height):
            raise BenchError(f"{pic.image.image_id}: bilden har annan storlek än i indexet")
        if isinstance(detector, FacitDetector):
            detector.truth = pic.image.boxes
        start = clock()
        found = detector.detect(frame)
        ms = (clock() - start) * 1000
        dets = [
            (d.score, (d.bbox.x, d.bbox.y, d.bbox.x + d.bbox.w, d.bbox.y + d.bbox.h))
            for d in found
            if d.label == PERSON
        ]
        yield match_picture(pic.image.dataset, frame.width, frame.height, pic.image.boxes, dets, ms)
        if n % 25 == 0:
            say(f"  {n}/{len(items)} bilder")


def model_detector(model: str, data: Path) -> Detector:
    """A built-in detector, or a trained one: ``<data>/models/detector/<model>.json``
    (written by ``just export-detector`` or ``just detector-fetch``), run in tiles as on
    the drone."""
    if model in DETECTORS:
        return DETECTORS[model]()
    if not _MODEL_NAME.match(model):
        raise BenchError("model= ska vara små bokstäver, siffror och _", 2)
    manifest = data / "models" / "detector" / f"{model}.json"
    if not manifest.is_file():
        raise BenchError(
            f"Okänd modell {model!r}: varken inbyggd ({', '.join(sorted(DETECTORS))}) eller "
            f"tränad ({manifest} finns inte; kör just export-detector eller just detector-fetch).",
            2,
        )
    from sd_detect.onnx import ModelError, load_model

    try:
        return load_model(manifest).tiled
    except ModelError as exc:
        raise BenchError(str(exc), 1) from exc


def measure(model: str, items: Sequence[Picture], data: Path | None = None) -> Summary:
    if model not in DETECTORS and data is None:
        raise BenchError(f"Okänd modell {model!r}. Kända: {', '.join(sorted(DETECTORS))}.", 2)
    detector = model_detector(model, data) if data is not None else DETECTORS[model]()
    return summarise(list(run(detector, items)))


def result_body(model: str, source: str, summary: Summary) -> dict[str, object]:
    return {
        "schema": RESULT_SCHEMA,
        "model": model,
        "source": source,
        "fp_per_12mp_target": FP_PER_12MP,
        "measured": datetime.now(UTC).strftime("%Y-%m-%d"),
        **summary.as_dict(),
    }


def _pooled_gt32(body: Mapping[str, Any]) -> float:
    return float(body["recall"]["gt32"]["recall"])


def compare(body: dict[str, object], earlier: Path) -> str | None:
    """Why ``body`` is worse than ``earlier`` (``None``: it is not), on recall over 32 px.
    It fails when the weakest dataset's recall (what ``pick-detector`` ranks on) drops
    more than 2 points, when any dataset in both drops more than 2 points, or when a
    dataset of the earlier result is missing from the new one. An earlier result without
    per-dataset rows is compared pooled against pooled, and the message says so."""
    old = json.loads(earlier.read_text(encoding="utf-8"))
    was, now = recall_by_set(old, "gt32"), recall_by_set(body, "gt32")
    if not was:
        drop = _pooled_gt32(old) - _pooled_gt32(body)
        if drop > COMPARE_DROP:
            return (
                f"recall över 32 px (sammanslagen; {earlier.name} har ingen per datamängd) "
                f"sjönk {drop * 100:.1f} procentenheter"
            )
        return None
    why = []
    missing = sorted(set(was) - set(now))
    if missing:
        why.append(f"{', '.join(missing)} saknas i den nya mätningen")
    drop = weakest_recall(old, "gt32") - weakest_recall(body, "gt32")
    if drop > COMPARE_DROP:
        why.append(f"svagaste datamängden sjönk {drop * 100:.1f} procentenheter")
    for name in sorted(set(was) & set(now)):
        drop = was[name] - now[name]
        if drop > COMPARE_DROP:
            why.append(f"{name} sjönk {drop * 100:.1f} procentenheter")
    if why:
        return f"recall över 32 px mot {earlier.name}: {'; '.join(why)}"
    return None


def selfcheck(items: Sequence[Picture], source: str = "") -> tuple[list[str], dict[str, object]]:
    """Null must find nothing and the facit everything; the errors and both results."""
    errors = []
    bodies: dict[str, object] = {}
    for model, want in (("null", 0.0), ("facit", 1.0)):
        s = measure(model, items)
        for cls, row in s.recall.items():
            if row["n"] and row["recall"] != want:
                errors.append(f"{model}: recall {cls} = {row['recall']}, väntat {want}")
        bodies[model] = result_body(model, source, s)
        say(f"{model}: recall {s.recall['all']['recall']} över {s.recall['all']['n']} personer")
    return errors, bodies


def private_out(out: Path, sets: Sequence[str], register: Mapping[str, Any]) -> str | None:
    """A warning when numbers that may not leave this computer land where git could take them.

    Outside the repository, and under the ignored build/, is the owner's own business."""
    where = out.resolve()
    if not where.is_relative_to(ROOT) or where.is_relative_to(ROOT / "build"):
        return None
    private = [s for s in sets if s in register and not register[s].publish_results]
    if not private:
        return None
    return (
        f"{out} ligger i förrådet utanför build/ och innehåller resultat på "
        f"{', '.join(private)}, som inte får publiceras; checka inte in den."
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv, KEYS, "bench-detector")
        model = args.get("model", "")
        if not model:
            raise BenchError("Ange model=. Kända: selfcheck, " + ", ".join(sorted(DETECTORS)), 2)
        data = data_dir(args)
        source, items = pictures(args, data)
        if not items:
            raise BenchError("Inga bilder att mäta på.", 2)
        say(f"Mäter {model} på {len(items)} bilder ({source})")
        part = "mini" if flag(args, "mini") else args.get("part", "test")
        out = Path(args.get("out") or RESULTS / f"det_{model}_{part}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        if model == "selfcheck":
            errors, bodies = selfcheck(items, source)
            out.write_text(
                json.dumps({"errors": errors, **bodies}, indent=1) + "\n", encoding="utf-8"
            )
            for e in errors:
                warn(e)
            return 1 if errors else 0
        body = result_body(model, source, measure(model, items, data))
        out.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
        by_set = body["recall_by_dataset"]
        assert isinstance(by_set, dict)  # noqa: S101 - built just above
        private = private_out(out, list(by_set), load_register())
        if private:
            warn(private)
        rec = body["recall"]
        assert isinstance(rec, dict)  # noqa: S101 - built just above
        say(
            f"recall < 16 px {rec['lt16']['recall']}, 16-32 px {rec['16to32']['recall']}, "
            f"> 32 px {rec['gt32']['recall']}; AP50 {body['ap50']}; skrivet till {out}"
        )
        for name, row in by_set.items():
            cls = row["by_class"]
            say(
                f"  {name}: recall {row['recall']} över {row['n']} personer "
                f"(16-32 px {cls['16to32']['recall']}, > 32 px {cls['gt32']['recall']}; "
                f"egen tröskel {row['threshold']}, {row['fp_per_12mp']} falsklarm per 12 MP)"
            )
        if args.get("compare"):
            why = compare(body, Path(args["compare"]))
            if why:
                warn(why)
                return 1
        return 0
    except BenchError as exc:
        warn(str(exc))
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
