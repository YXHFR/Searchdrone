"""One farm round: the spec it runs from, the run, and the dataset it leaves behind.

The loop (``sd_sim.farm.loop``) and the round are separate processes, and the round runs
from the newest ``main`` in the farm's worktree, so this file is the contract between
them: the loop writes ``spec.json`` (``SPEC_SCHEMA``) into an empty folder and runs
``python -m sd_sim.farm round spec=<folder>/spec.json``; the round fills the folder and
writes ``round.json`` (``ROUND_SCHEMA``) last. Exit codes: 0 the round passed its quality
check, 1 it failed it, 2 the round runner is not built, 3 the round could not run.

A round folder::

    spec.json          what the loop asked for (seed, commander, commit)
    round.json         the outcome: scenario, quality check, counts, credits
    truth.json         the world's facit the pictures were labelled from, plus
                       ``targets`` (lat/lon of the people to find, for the grade)
    images/<id>.png    the pictures from the air (no black ones)
    labels/<id>.txt    YOLO, class 0 = person
    meta/<id>.json     per picture: world, time, weather, height, camera, people
    coco.json          COCO for the whole round
    runs/<run>/        what the grade reads of each flight: ``flight.json`` and, when
                       sd-brain decided, ``brain.json``
    grade.json         (optional) the AI commander's grade, when sd-brain can grade

``finalize()`` turns a runner's raw stills (``<name>.ppm`` plus ``<name>.json`` with the
simulator time, the bounding-box camera's boxes and the camera's pose, as ``gz_stills``
writes them) into that layout; it is independent of how the round was flown.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from sd_schemas.simfarm import ROUND_SCHEMA
from sd_sim import _args
from sd_sim.farm import labels, qc
from sd_sim.farm.camera import CameraPose, Intrinsics
from sd_sim.farm.people import BY_ID

SPEC_SCHEMA = "simfarm.spec/v1"
COMMANDERS = ("script", "sd-brain")
EXIT_OK, EXIT_QC, EXIT_ERROR = 0, 1, 3
EXIT_NOT_BUILT = _args.NOT_BUILT_EXIT
GRADE_MODULE = "sd_brain.simeval.farm"
GRADE_TIMEOUT_S = 600
DEFAULT_WORLDS = ("sd_lake_forest",)
# Asked in a child process, so sd_brain is never imported into the farm's own process.
_HAS_MODULE = (
    "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)"
)


class RoundError(RuntimeError):
    """The round could not run (the simulator, a missing model, a broken still)."""


class SkippedError(RoundError):
    """The round is not flown (its height cannot clear the trees under the charter's
    ceiling). Exit 3 like an error, with status ``skipped`` and the reason."""


class NotBuiltError(RoundError):
    """The round runner needs a part of the simulator that is not built yet. The round
    contract's exit 2: the loop stops instead of counting a failure. No runner raises it
    today; it is kept for a world or vehicle a runner cannot fly yet."""


@dataclass(frozen=True)
class RoundSpec:
    round_id: str
    seed: int
    commander: str
    commit: str
    created_utc: str
    worlds: tuple[str, ...] = DEFAULT_WORLDS  # the worlds the scenario may draw from
    schema: str = SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.commander not in COMMANDERS:
            raise ValueError(f"commander ska vara {' eller '.join(COMMANDERS)}")
        object.__setattr__(self, "worlds", tuple(self.worlds))
        if not self.worlds:
            raise ValueError("worlds får inte vara tom")

    @classmethod
    def new(
        cls,
        seed: int,
        commander: str,
        commit: str,
        now: dt.datetime | None = None,
        worlds: Sequence[str] = DEFAULT_WORLDS,
    ) -> RoundSpec:
        now = now or dt.datetime.now(dt.UTC)
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        return cls(
            f"{stamp}-s{seed}-{commit[:8]}", seed, commander, commit, now.isoformat(), tuple(worlds)
        )

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> RoundSpec:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != SPEC_SCHEMA:
            raise RoundError(f"{path}: okänt schema {data.get('schema')!r}")
        return cls(**data)


@dataclass
class RawRound:
    """What a runner hands over: the raw stills, the facit and what it knows of the flight."""

    stills_dir: Path
    truth: Mapping[str, Any]
    scenario: Mapping[str, Any]
    camera: Intrinsics
    max_height_m: float | None = None  # from the flight log, if the runner has it
    extra: dict[str, Any] = field(default_factory=dict)  # run folder, mission id, ...
    runs_dir: Path | None = None  # <run>/flight.json and brain.json, for the grade


class Runner(Protocol):
    def run(self, spec: RoundSpec, work: Path) -> RawRound: ...


def _height(meta: Mapping[str, Any], pose: CameraPose, truth: Mapping[str, Any]) -> float:
    """The camera's height above the ground: the runner's own value, else above the mean
    ground height of the facit's objects."""
    if meta.get("agl_m") is not None:
        return float(meta["agl_m"])
    grounds = [float(o.get("ground_m", 0.0)) for o in truth.get("objects", [])]
    ground = sum(grounds) / len(grounds) if grounds else 0.0
    return pose.position[2] - ground


def _pose(meta: Mapping[str, Any]) -> CameraPose:
    cam = meta.get("camera")
    if not isinstance(cam, Mapping):
        raise RoundError(f"stillbild {meta.get('name')!r} saknar kamerans läge")
    p = [float(v) for v in cam["position_m"]]
    q = [float(v) for v in cam["orientation_wxyz"]]
    norm = math.sqrt(sum(v * v for v in q))
    if len(p) != 3 or len(q) != 4 or not 0.99 < norm < 1.01:
        raise RoundError(f"stillbild {meta.get('name')!r} har ett ogiltigt kameraläge")
    return CameraPose((p[0], p[1], p[2]), (q[0], q[1], q[2], q[3]))


RUN_FILES = ("flight.json", "brain.json")


def copy_runs(runs_dir: Path | None, out: Path) -> list[str]:
    """The flights' records into ``out/runs/<run>/``: only ``RUN_FILES``, only regular
    files (never through a link). Returns the runs copied."""
    if runs_dir is None or not runs_dir.is_dir():
        return []
    copied = []
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir() and not p.is_symlink()):
        files = [run / f for f in RUN_FILES if (run / f).is_file() and not (run / f).is_symlink()]
        if not files:
            continue
        dest = out / "runs" / run.name
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            (dest / f.name).write_bytes(f.read_bytes())
        copied.append(run.name)
    return copied


def credits(truth: Mapping[str, Any]) -> list[dict[str, str]]:
    used = sorted(
        {
            o["truth"]["model"]
            for o in truth.get("objects", [])
            if o.get("truth", {}).get("cls") == "person"
        }
    )
    return [BY_ID[m].attribution() for m in used if m in BY_ID]


def finalize(spec: RoundSpec, raw: RawRound, out: Path) -> tuple[qc.QcResult, int]:
    """Write the round's dataset into ``out`` and check it; returns the check and the
    number of pictures kept."""
    for sub in ("images", "labels", "meta"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    truth = {**raw.truth, "targets": targets(raw.truth)}
    (out / "truth.json").write_text(json.dumps(truth, indent=1) + "\n", encoding="utf-8")
    info = {
        "description": "Searchdrone sim farm, synthetic aerial pictures with person boxes",
        "round_id": spec.round_id,
        "commit": spec.commit,
        "date_created": spec.created_utc,
        "credits": credits(raw.truth),
    }
    coco = labels.Coco(info)
    tally = qc.RoundTally(max_height_m=raw.max_height_m or 0.0)
    kept = 0
    for meta_path in sorted(raw.stills_dir.glob("[!.]*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ppm = meta_path.with_suffix(".ppm")
        if not ppm.is_file():
            continue
        pose = _pose(meta)
        height = _height(meta, pose, raw.truth)
        tally.max_height_m = max(tally.max_height_m, height)
        if height < qc.MIN_FLIGHT_M:
            tally.on_ground += 1
            continue
        tally.pictures += 1
        image = labels.read_ppm(ppm)
        if labels.is_black(image):
            tally.black += 1
            continue
        cam = raw.camera
        if image.shape[1] != cam.width or image.shape[0] != cam.height:
            raise RoundError(f"{ppm.name}: storleken stämmer inte med kameran")
        facit = labels.facit(raw.truth, cam, pose, labels.camera_boxes(meta.get("boxes")))
        tally.add(facit)
        name = f"{spec.round_id}_{kept:04d}"
        labels.write_png(image, out / "images" / f"{name}.png")
        (out / "labels" / f"{name}.txt").write_text(
            labels.yolo_lines(facit.labels, cam), encoding="utf-8"
        )
        coco.add(f"images/{name}.png", cam, facit.labels)
        mean, std = labels.brightness(image)
        image_meta = {
            "schema": labels.IMAGE_SCHEMA,
            "round_id": spec.round_id,
            "image": f"images/{name}.png",
            "sim_time_ns": meta.get("sim_time_ns"),
            "commit": spec.commit,
            "commander": spec.commander,
            "world": raw.scenario.get("world"),
            "day_of_year": raw.scenario.get("day_of_year"),
            "solar_hour": raw.scenario.get("solar_hour"),
            "sun": raw.scenario.get("sun"),
            "weather": raw.scenario.get("weather"),
            "light": raw.scenario.get("light"),
            "height_m": round(height, 2),
            "camera": {
                **pose.as_dict(),
                "width": cam.width,
                "height": cam.height,
                "hfov_rad": cam.hfov_rad,
            },
            "brightness": {"mean": round(mean, 1), "std": round(std, 1)},
            "people": [p.as_dict() for p in facit.people],
            "label_source": "projection of the facit, checked against the bounding-box camera",
            "credits": info["credits"],
        }
        (out / "meta" / f"{name}.json").write_text(
            json.dumps(image_meta, indent=1) + "\n", encoding="utf-8"
        )
        kept += 1
    coco.write(out / "coco.json")
    copy_runs(raw.runs_dir, out)
    return qc.check(tally), kept


def _has_grader(module: str = GRADE_MODULE) -> bool:
    try:
        probe = subprocess.run(  # noqa: S603 - fixed argv
            [sys.executable, "-c", _HAS_MODULE, module],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def grade(
    out: Path,
    available: Callable[[], bool] | None = None,
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any] | None:
    """Let the AI commander grade the round, when this checkout's sd-brain can
    (``sd_brain.simeval.farm`` exists): runs ``python -m sd_brain eval grade --round
    <out>``, which writes ``grade.json``. ``None`` while there is no grader."""
    if not (available or _has_grader)():
        return None
    cmd = [sys.executable, "-m", "sd_brain", "eval", "grade", "--round", str(out)]
    try:
        run(cmd, check=True, timeout=GRADE_TIMEOUT_S)
        data = json.loads((out / "grade.json").read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("grade.json är inget objekt")
        return data
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"error": f"betyget gick inte att räkna: {exc}"}


def targets(truth: Mapping[str, Any]) -> list[dict[str, float]]:
    """Where the people to find are (WGS84), as ``sd-brain eval grade`` reads them."""
    return [
        {"lat": float(row["lat"]), "lon": float(row["lon"])}
        for row in truth.get("objects", [])
        if row.get("truth", {}).get("role") == "target" and "lat" in row and "lon" in row
    ]


def write_result(out: Path, spec: RoundSpec, result: Mapping[str, Any]) -> None:
    """``round.json``, written last and by rename: a reader that sees it sees a whole round.
    ``"grade": None`` means the grade is pending (the grader runs after this write)."""
    data = {
        **result,  # the contract's own keys below always win
        "schema": ROUND_SCHEMA,
        "round_id": spec.round_id,
        "seed": spec.seed,
        "commit": spec.commit,
        "commander_requested": spec.commander,
        "created_utc": spec.created_utc,
        "finished_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    if (out / "truth.json").is_file():
        data["truth"] = "truth.json"
    tmp = out / ".round.json.tmp"
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    tmp.replace(out / "round.json")


def read_result(out: Path) -> dict[str, Any] | None:
    path = out / "round.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("schema") == ROUND_SCHEMA else None


def execute(spec_path: Path, runner: Runner) -> int:
    """The round process: run, finalize, write ``round.json``, grade; returns the exit code.
    Anything unexpected is exit 3 (error), never 1 (a failed check)."""
    try:
        return _execute(spec_path, runner)
    except Exception as exc:
        print(f"sd_sim.farm round: {type(exc).__name__}: {exc}", file=sys.stderr)  # noqa: T201
        return EXIT_ERROR


def _execute(spec_path: Path, runner: Runner) -> int:
    out = spec_path.parent
    spec = RoundSpec.read(spec_path)
    work = out / "work"
    try:
        raw = runner.run(spec, work)
        check, kept = finalize(spec, raw, out)
    except NotBuiltError as exc:
        write_result(out, spec, {"status": "not_built", "error": str(exc)})
        return EXIT_NOT_BUILT
    except SkippedError as exc:
        write_result(out, spec, {"status": "skipped", "error": str(exc)})
        print(f"sd_sim.farm round: {exc}", file=sys.stderr)  # noqa: T201
        return EXIT_ERROR
    except (RoundError, OSError, ValueError, KeyError) as exc:
        write_result(out, spec, {"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return EXIT_ERROR
    result: dict[str, Any] = {
        **raw.extra,  # the runner's own keys (``runs``, ...) never replace the ones below
        "status": "ok" if check.ok else "failed",
        "error": None if check.ok else "; ".join(check.reasons),
        "scenario": dict(raw.scenario),
        "qc": check.as_dict(),
        "images": kept,
        "credits": credits(raw.truth),
    }
    if not check.ok:
        # Only a round that passed is graded: a failed one says something about the
        # simulator or the recording, not about the decision maker.
        write_result(out, spec, {**result, "grade": {"skipped": "qc_failed"}})
        return EXIT_QC
    # The grader reads the whole round, ``round.json`` included; its grade is added after.
    write_result(out, spec, {**result, "grade": None})
    write_result(out, spec, {**result, "grade": grade(out)})
    return EXIT_OK
