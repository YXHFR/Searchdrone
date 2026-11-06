"""Train the person detector on a Modal GPU: ``just train-detector-cloud gpu=A100-80GB ...``.

    just train-detector-cloud model=yolox_s                    # A100 80 GB, asks first
    just train-detector-cloud gpu=H100 model=damoyolo_s yes=1  # no question
    just train-detector-cloud model=yolox_s sets=heridal,kios,uit_drone4 scales=native,wide,tele

The training is the one ``just train-detector`` runs (``train_detector.py`` and
``export_onnx.py``, unchanged); only the computer differs. Every key of
``train-detector`` works the same, except ``device`` (always the GPU) and ``simfarm``
(the simulator's pictures stay on this computer). Keys of its own: ``gpu`` (A100-80GB,
the default, H100 or L40S), ``yes`` (1: start without asking), ``hours`` (your own guess
of the training time on that GPU, for the cost shown; default a rough guess),
``max_hours`` (default 24: the most GPU hours this command may use; when they run out
it stops, and the same command with a larger ``max_hours`` carries on) and ``name`` (the
exported model's name, default ``person_<model>``, as ``just export-detector``).

What happens:

1. The pictures the training reads (the ``train`` and ``val`` parts of the chosen sets,
   never the locked ``test`` part), each set's ``index.json``, the COCO weights and an
   ``init=`` file are uploaded to a private Modal Volume (``VOLUME``), laid out as the
   data folder is. Only new or changed files are sent (``plan_upload``). Every file is
   checked before it is sent (``guard``): only a set that may be trained on, with a
   confirmed open licence and not kept on this computer, ever leaves it. A set that may
   only be measured on (GPL, no licence, non-commercial, WiSARD) is never uploaded.
   Nothing that looks like an orthophoto is sent either (``check_no_orthophotos``).
2. The cost is printed (the GPU's price per hour from ``PRICES`` plus the container's
   CPU and memory, ``EXTRA_PER_HOUR``, times the hours) and you are asked before
   anything is started.
3. ``train_detector.py`` and then ``export_onnx.py`` run on the GPU (``modal_app.py``),
   with their checkpoints on the volume (``gpu_call.py``). The training saves
   ``last.pt`` every ``save_minutes`` (default 10) and after every epoch, not when it is
   stopped. A Modal call lasts at most 24 hours, so each call is given a time budget a
   little shorter (and within what is left of ``max_hours``), measured in the container
   on its own clock; then the script is stopped and the next call starts from the last
   save, until the run is done. Up to about 15 minutes of training (since the last
   save, and the volume is committed every 5 minutes) may be done again. ``fresh=1``
   is done here, before anything else: the old run's files are removed from the volume
   (``FRESH_FILES``), so every call after that carries on from the new run's own
   ``last.pt``, or starts clean when there is none (``start_fresh``). No retries are
   asked of Modal, though it may move a call to another machine after a preemption;
   this side stops waiting, and cancels the call, when its time is up. Each call's
   worst case is counted against ``max_hours`` before it starts, for this run folder in
   all (``LEDGER``), and corrected only when the container reports its own time, so the
   ceiling holds. A stopped command (Ctrl+C, a lost line) carries on the same way when
   run again.
4. ``best.pt`` and ``settings.json`` come back to ``<data>/train/<run>`` (never over a
   local training's folder: ``check_local_run``) and the ONNX file and its manifest to
   ``<data>/models/detector`` (never over a model exported here: ``check_local_model``),
   where the local training puts them, so
   ``just bench-detector``, ``just pick-detector`` and the simulator use them unchanged.

The Modal token is read by Modal itself from ``~/.modal.toml`` or the environment
(``MODAL_TOKEN_ID``, ``MODAL_TOKEN_SECRET``); it is never read, printed or stored here.
This file never imports ``modal`` at the top: the parts that decide what is uploaded and
what it costs are tested without it, the network or an account.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from bench.common import BenchError, flag, inside, parse_args, safe_relpath, say, warn
from bench.datasets import (
    NC_PREFIXES,
    OPEN_LICENCES,
    Dataset,
    index_path,
    load_register,
    read_source,
)
from gpu_call import STOP_GRACE_S

HERE = Path(__file__).resolve().parent

# The Modal Volume that holds the uploaded pictures and the runs (private to the account).
VOLUME = "searchdrone-train"
# Where the volume is mounted in the container: the data folder there.
REMOTE_DATA = PurePosixPath("/data")

# USD per GPU hour, from modal.com/pricing, read 2026-11-04. The container's CPU and
# memory are billed on top (EXTRA_PER_HOUR); the volume is not counted.
PRICES_SOURCE = "modal.com/pricing, läst 2026-11-04"
PRICES: dict[str, float] = {"A100-80GB": 2.50, "H100": 3.95, "L40S": 1.95}
# The container beside the GPU (modal_app.py asks for exactly these, as both request
# and limit, so it is never billed for more), and their price from the same page: USD
# per CPU core hour and per GiB hour.
CPU_CORES = 8.0
MEMORY_MIB = 32768
CPU_CORE_HOUR = 0.0472
GIB_HOUR = 0.0080
EXTRA_PER_HOUR = CPU_CORES * CPU_CORE_HOUR + MEMORY_MIB / 1024 * GIB_HOUR  # about 0.63
DEFAULT_GPU = "A100-80GB"
# How much faster than the RTX 3070 the README's times are for: a rough guess from the
# cards' published mixed-precision throughput, not measured on this training.
SPEEDUP: dict[str, float] = {"A100-80GB": 3.0, "H100": 4.5, "L40S": 2.5}
# The README's upper times for 20 epochs on HERIDAL and KIOS on an RTX 3070, in hours
# (yolox_m: about 2.8 times yolox_s's work per tile).
BASE_HOURS: dict[str, float] = {"yolox_s": 3.0, "damoyolo_s": 4.0, "yolox_m": 8.5}
# Modal stops a call after 24 hours; the run is stopped this long before, so its last
# save is on the volume.
CALL_LIMIT_S = 24 * 3600
STOP_MARGIN_S = 30 * 60
DEFAULT_MAX_HOURS = 24.0
# Keys of the cloud command itself; all other keys are train-detector's.
OWN_KEYS = ("gpu", "yes", "hours", "max_hours", "name")
# train-detector keys that make no sense here: the GPU is the device, the data folder is
# the volume, and the simulator's pictures stay on this computer.
LOCAL_ONLY_KEYS = ("device", "simfarm", "simfarm_epochs")
# The record of what was uploaded, per volume, in the data folder (size, mtime_ns).
RECORD = "train/.modal-{volume}.json"
# The GPU seconds used so far, per run on the volume, in the data folder: max_hours= is
# a ceiling for all calls of a run together, also over several commands.
LEDGER = "train/.modal-hours-{run}.json"
# Each call ends this long after its deadline at most: the script's own stop
# (STOP_GRACE_S) and the last commit of the volume. Modal's timeout is set to it.
FINISH_SLACK_S = 5 * 60
# How long the container may take to start before its own clock (the budget, Modal's
# timeout) begins; this side waits at most that much longer, then cancels the call.
START_SLACK_S = 15 * 60
# A call shorter than this is not worth starting (the container and data take minutes).
MIN_CALL_S = 15 * 60
# The local run folder of a run made here, with what came back from which run on the
# volume; a folder without it (or with last.pt) is a local training's and is never
# written to.
MARKER = "modal-run.json"
# What fresh=1 removes from the run's folder on the volume: what train_detector.py
# resumes from (last.pt, checked against settings.json) and the old run's best.pt.
FRESH_FILES = ("last.pt", "settings.json", "best.pt")
# An init= file: a best.pt, of at most this size (yolox_m's is about 200 MB).
INIT_SUFFIXES = (".pt", ".pth")
MAX_INIT_BYTES = 1_000_000_000
# Where the repository's files lie in the container (bench.common.ROOT there).
IN_CONTAINER = PurePosixPath("/root/sd")
# Exactly the files of tools/ the container needs to run train_detector.py and
# export_onnx.py (and this module and modal_app.py, which Modal imports there), as
# globs relative to tools/. Nothing else is put in the image: no .env, no local
# settings, no weights or data (``image_files`` refuses them even if listed).
IMAGE_FILES: tuple[str, ...] = (
    "bench/common.py",
    "bench/datasets.py",
    "bench/datasets.toml",
    "bench/layouts.py",
    "bench/splits.py",
    "bench/splits/*.json",
    "train/cloud.py",
    "train/evaluate.py",
    "train/export_onnx.py",
    "train/families.py",
    "train/gpu_call.py",
    "train/modal_app.py",
    "train/tiles.py",
    "train/train_detector.py",
    "train/vendor/NOTICE.md",
    "train/vendor/damoyolo/*.py",
    "train/vendor/damoyolo/LICENSE",
    "train/vendor/damoyolo/NOTICE",
    "train/vendor/damoyolo/tinynas_L25_k1kx.txt",
    "train/vendor/yolox/*.py",
    "train/vendor/yolox/LICENSE",
)
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".onnx", ".npy", ".npz", ".jpg", ".png", ".tif")
# Files sent per batch; the record is written after each, so a stopped upload carries on.
CHUNK_FILES = 500
CHUNK_BYTES = 2_000_000_000


def _literal(path: Path, name: str) -> Any:
    """A constant assigned at the top of ``path`` (``KEYS``, ``PRETRAINED``), read
    without importing it: those modules need PyTorch, which this side does not."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"{name} saknas i {path.name}")


def train_keys() -> tuple[str, ...]:
    return tuple(_literal(HERE / "train_detector.py", "KEYS"))


def pretrained_files() -> dict[str, str]:
    return dict(_literal(HERE / "families.py", "PRETRAINED"))


def default_sets() -> tuple[str, ...]:
    """``train_detector.Settings.sets``, the sets ``train-detector`` trains on when
    ``sets=`` is not given, read the same way."""
    tree = ast.parse((HERE / "train_detector.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Settings":
            for item in node.body:
                if (
                    isinstance(item, ast.AnnAssign)
                    and isinstance(item.target, ast.Name)
                    and item.target.id == "sets"
                    and item.value is not None
                ):
                    return tuple(ast.literal_eval(item.value))
    raise RuntimeError("Settings.sets saknas i train_detector.py")


def image_files(tools: Path) -> list[tuple[Path, PurePosixPath]]:
    """(file here, path in the container) of every file ``IMAGE_FILES`` names under
    ``tools``. Raises if a pattern finds nothing or a file is one that must never be
    sent (secrets, local settings, weights, pictures)."""
    out: list[tuple[Path, PurePosixPath]] = []
    for pattern in IMAGE_FILES:
        found = sorted(p for p in tools.glob(pattern) if p.is_file())
        if not found:
            raise BenchError(f"{pattern} saknas under {tools}.", 1)
        for path in found:
            name = path.name
            if (
                name.startswith(".")
                or ".local." in name
                or name.endswith(FORBIDDEN_SUFFIXES)
                or "__pycache__" in path.parts
            ):
                raise BenchError(f"{path} får inte skickas till Modal.", 1)
            rel = path.relative_to(tools)
            out.append((path, IN_CONTAINER / "tools" / PurePosixPath(*rel.parts)))
    return out


def check_init(path: Path) -> int:
    """An ``init=`` file must be a ``best.pt`` of a sane size; checked here, before
    anything is sent, and again by ``guard``. Returns its size."""
    if not path.is_file():
        raise BenchError(f"init={path} finns inte.", 2)
    if path.suffix not in INIT_SUFFIXES:
        raise BenchError(f"init= ska vara en .pt-fil (best.pt), inte {path.name}.", 2)
    size = path.stat().st_size
    if size > MAX_INIT_BYTES:
        raise BenchError(
            f"init={path.name} är {size / 1e9:.1f} GB; en best.pt är högst "
            f"{MAX_INIT_BYTES / 1e9:.0f} GB. Fel fil?",
            2,
        )
    return size


# --- what may leave this computer ---------------------------------------------------


def refusal(ds: Dataset) -> str | None:
    """Why ``ds`` may never be uploaded (``None``: it may). Only a set that is trained on,
    under a confirmed open licence, and not kept on this computer, may."""
    if ds.no_grant:
        return "licensen (GPL eller ingen) ger ingen rätt att bygga vidare; den får bara mätas på"
    if ds.licence.startswith(NC_PREFIXES):
        return "licensen är icke-kommersiell; den får bara mätas på, på den här datorn"
    if "train" not in ds.use:
        return "den får bara mätas på" if "measure" in ds.use else "den tränas inte på"
    if not ds.licence_confirmed:
        return "datalicensen är inte bekräftad"
    if ds.licence not in OPEN_LICENCES:
        return f"licensen {ds.licence} är inte öppen"
    if ds.local_only:
        return "den stannar på den här datorn (local_only)"
    return None


def cloud_sets(requested: Sequence[str], register: Mapping[str, Dataset], data: Path) -> list[str]:
    """The sets of ``requested`` to upload and train on. A set that may never leave
    this computer is refused (exit 2), except one that is only measured on: it is never
    trained on locally either (``tiles.pictures``), so it is left out, with a word."""
    unknown = [s for s in requested if s not in register]
    if unknown:
        raise BenchError(f"Okända set i sets=: {', '.join(unknown)}.", 2)
    out: list[str] = []
    for name in requested:
        ds = register[name]
        why = refusal(ds)
        if why is None:
            if index_path(data, name).is_file():
                out.append(name)
            else:
                warn(
                    f"{name} är inte hämtad och tränas inte på (just bench-fetch yes=1 sets={name})"
                )
        elif (
            set(ds.use) == {"measure"}
            and not ds.no_grant
            and not ds.licence.startswith(NC_PREFIXES)
        ):
            say(f"{name} laddas inte upp: {why}.")
        else:
            raise BenchError(
                f"{name} laddas aldrig upp till Modal: {why}. Ta bort det ur sets=.", 2
            )
    if not out:
        raise BenchError("Inga bilder att träna på. Kör: just bench-fetch yes=1", 2)
    return out


@dataclass(frozen=True)
class Upload:
    local: Path
    remote: PurePosixPath  # relative to the volume's root (the data folder there)


def guard(remote: PurePosixPath, register: Mapping[str, Dataset], size: int) -> None:
    """The hard check every upload passes, file by file, just before it is sent: the
    path must be one the training reads, a dataset path must name a set ``refusal``
    lets go, and an ``init/`` file must be a ``.pt``/``.pth`` of at most
    ``MAX_INIT_BYTES`` (``size``). Raises ``BenchError`` (exit 1) otherwise, and nothing
    is sent."""
    parts: tuple[str, ...] = remote.parts
    if remote.is_absolute() or ".." in parts or len(parts) < 2:
        raise BenchError(f"Uppladdningen stoppad: {remote} är ingen tillåten sökväg.", 1)
    if parts[0] == "datasets" and len(parts) >= 3:
        ds = register.get(parts[1])
        why = "okänt set" if ds is None else refusal(ds)
        if why is not None:
            raise BenchError(f"Uppladdningen stoppad: {remote} hör till {parts[1]} ({why}).", 1)
        if parts[2:] == ("index.json",) or (parts[2] == "raw" and len(parts) >= 4):
            return
    elif parts[0] == "models" and len(parts) == 3 and parts[1] == "pretrained":
        if parts[2] in pretrained_files().values():
            return
    elif parts[0] == "init" and len(parts) == 3:
        if remote.suffix in INIT_SUFFIXES and 0 <= size <= MAX_INIT_BYTES:
            return
    raise BenchError(f"Uppladdningen stoppad: {remote} är ingen tillåten sökväg.", 1)


def orthophoto_findings(path: Path) -> list[str]:
    """What makes ``path`` look like an orthophoto (``tools/check_no_orthophotos.py``)."""
    import check_no_orthophotos as ortho

    return ortho.check_paths([path])


def plan(
    data: Path,
    sets: Sequence[str],
    model: str,
    limit: int = 0,
    init: Path | None = None,
    init_sha256: str = "",
) -> list[Upload]:
    """Every file the training reads: each set's ``index.json``, the pictures of its
    ``train`` and ``val`` parts (never ``test``), the COCO weights and an ``init=`` file.
    The pictures are the ones ``tiles.pictures`` gives the training, ``limit`` included."""
    from tiles import pictures

    out: list[Upload] = []
    for name in sets:
        if (read_source(data, name) or "").startswith("local:"):
            raise BenchError(f"{name} pekar på bilder i repot; de laddas inte upp.", 2)
        out.append(Upload(index_path(data, name), PurePosixPath("datasets", name, "index.json")))
    for part in ("train", "val"):
        for im, root in pictures(data, part, sets, limit):
            rel = safe_relpath(im.file)
            local = inside(root, rel)
            out.append(Upload(local, PurePosixPath("datasets", im.dataset, "raw", *rel.parts)))
    weights = pretrained_files()[model]
    out.append(
        Upload(
            data / "models" / "pretrained" / weights, PurePosixPath("models", "pretrained", weights)
        )
    )
    if init is not None:
        out.append(Upload(init, init_remote(init, init_sha256)))
    seen: dict[PurePosixPath, Upload] = {}
    for u in out:
        seen.setdefault(u.remote, u)
    return list(seen.values())


def init_remote(init: Path, sha256: str) -> PurePosixPath:
    """Where an ``init=`` file goes: by its content, so two runs' ``best.pt`` never meet."""
    return PurePosixPath("init", sha256[:16], init.name)


def stat_of(path: Path) -> tuple[int, int]:
    st = path.stat()
    return st.st_size, st.st_mtime_ns


def plan_upload(
    uploads: Iterable[Upload],
    local: Mapping[PurePosixPath, tuple[int, int]],
    remote: Mapping[str, int],
    record: Mapping[str, Sequence[int]],
) -> list[Upload]:
    """The uploads to send: a file the volume lacks, one whose size differs there, or
    one changed here since it was sent (size or modification time against ``record``)."""
    todo = []
    for u in uploads:
        key = u.remote.as_posix()
        size, mtime = local[u.remote]
        if remote.get(key) != size or list(record.get(key, ())) != [size, mtime]:
            todo.append(u)
    return todo


# --- cost -----------------------------------------------------------------------------


def guess_hours(model: str, gpu: str, epochs: int, scales: int = 1, upscale: int = 1) -> float:
    """A rough training time on ``gpu``: the README's RTX 3070 time for 20 epochs, scaled
    by the epochs, the extra scales (about one more native's worth of tiles each) and
    ``upscale``² tiles, and divided by the GPU's guessed speed-up."""
    work = BASE_HOURS[model] * epochs / 20 * max(1, scales) * upscale**2
    return work / SPEEDUP[gpu]


@dataclass(frozen=True)
class Cost:
    gpu: str
    gpu_price: float  # USD per hour, the GPU alone
    extra: float  # USD per hour, the container's CPU and memory
    hours: float  # the guess
    max_hours: float  # the ceiling

    @property
    def price(self) -> float:
        """USD per hour, all billed."""
        return self.gpu_price + self.extra

    @property
    def usd(self) -> float:
        return self.price * min(self.hours, self.max_hours)

    @property
    def max_usd(self) -> float:
        return self.price * self.max_hours

    def lines(self, guessed: bool) -> list[str]:
        how = "grov gissning" if guessed else "din uppskattning (hours=)"
        return [
            f"Per timme: {self.gpu} {self.gpu_price:.2f} $ + processor och minne "
            f"(exakt {CPU_CORES:g} kärnor, {MEMORY_MIB // 1024} GiB) {self.extra:.2f} $ "
            f"= {self.price:.2f} $/h ({PRICES_SOURCE}).",
            f"Ungefärlig kostnad: {self.price:.2f} $/h x {self.hours:.1f} h "
            f"= {self.usd:.0f} $ ({how}).",
            f"Högst: {self.max_hours:g} h x {self.price:.2f} $/h = {self.max_usd:.0f} $ "
            "(max_hours=; sedan stannar den, och samma kommando fortsätter).",
        ]


def cost(gpu: str, hours: float, max_hours: float) -> Cost:
    if gpu not in PRICES:
        raise BenchError(f"gpu= ska vara en av {', '.join(PRICES)}, inte {gpu!r}.", 2)
    if not (hours > 0 and math.isfinite(hours)) or not (max_hours > 0 and math.isfinite(max_hours)):
        raise BenchError("hours= och max_hours= ska vara positiva tal.", 2)
    return Cost(gpu, PRICES[gpu], EXTRA_PER_HOUR, hours, max_hours)


# --- where things go ----------------------------------------------------------------


@dataclass(frozen=True)
class Places:
    local_run: Path  # <data>/train/<run>: best.pt comes back here
    local_models: Path  # <data>/models/detector: the ONNX file and manifest
    remote_run: PurePosixPath  # absolute, in the container
    remote_models: PurePosixPath
    name: str  # the exported model's name

    def results(self) -> list[tuple[PurePosixPath, Path]]:
        """(path on the volume, relative to its root; local file) of what comes back."""
        run, models = (
            self.remote_run.relative_to(REMOTE_DATA),
            self.remote_models.relative_to(REMOTE_DATA),
        )
        return [
            (run / "best.pt", self.local_run / "best.pt"),
            (run / "settings.json", self.local_run / "settings.json"),
            (models / f"{self.name}.onnx", self.local_models / f"{self.name}.onnx"),
            (models / f"{self.name}.json", self.local_models / f"{self.name}.json"),
        ]


def places(data: Path, model: str, out: str = "", name: str = "") -> Places:
    """The local run folder is ``out=`` or ``<data>/train/<model>``, as for
    ``train-detector``. On the volume the run is ``train/<folder name>-<hash>``, the
    hash of the whole local path, so two ``out=`` folders with the same name never share
    a run there."""
    local_run = Path(out).expanduser().resolve() if out else data / "train" / model
    if not local_run.name or local_run.name.startswith("."):
        raise BenchError(f"out= ska sluta på ett mappnamn, inte {out!r}.", 2)
    run = f"{local_run.name}-{hashlib.sha256(str(local_run).encode()).hexdigest()[:8]}"
    name = name or f"person_{model}"
    if "/" in name or name.startswith("."):
        raise BenchError(f"name= ska vara ett filnamn, inte {name!r}.", 2)
    return Places(
        local_run=local_run,
        local_models=data / "models" / "detector",
        remote_run=REMOTE_DATA / "train" / run,
        remote_models=REMOTE_DATA / "models" / "detector",
        name=name,
    )


def train_argv(args: Mapping[str, str], sets: Sequence[str], where: Places, init: str) -> list[str]:
    """``train_detector.py``'s arguments in the container: the keys as given, with the
    data folder, the run folder, the sets and ``init=`` moved to the volume. ``fresh``
    is left out: it is done before the calls, on the volume (``remove_old_run``)."""
    moved = {"data", "out", "sets", "init", "fresh", *OWN_KEYS}
    argv = [f"{k}={v}" for k, v in args.items() if k not in moved]
    argv += [
        f"data={REMOTE_DATA}",
        f"out={where.remote_run}",
        f"sets={','.join(sets)}",
        "device=cuda",
    ]
    if init:
        argv.append(f"init={init}")
    return argv


def export_argv(model: str, where: Places) -> list[str]:
    return [
        f"model={model}",
        f"data={REMOTE_DATA}",
        f"train={where.remote_run}",
        f"name={where.name}",
        f"out={where.remote_models}",
    ]


# --- the Modal side, behind a seam the tests fake ---------------------------------------


class Backend(Protocol):
    def remote_sizes(self) -> dict[str, int]:
        """Every file on the volume: path relative to its root -> size."""
        ...

    def upload(self, uploads: Sequence[Upload]) -> None: ...

    def remove(self, paths: Sequence[PurePosixPath]) -> None:
        """Remove these files (relative to the volume's root); one that is not there is
        no error."""
        ...

    def run(
        self, train: Sequence[str], export: Sequence[str], budget_s: float, timeout_s: int
    ) -> dict[str, Any]:
        """One call on the GPU (``gpu_call.call``): the scripts are stopped ``budget_s``
        after the container starts, by its own clock; Modal stops the call after
        ``timeout_s``, and this side stops waiting and cancels it ``START_SLACK_S``
        later. ``{"status": "done" | "stopped" | "failed" | "no_gpu", "code": int | None,
        "seconds": float}``. ``stopped``: the time was up; the next call carries on from
        the last save on the volume. ``no_gpu``: the call never got a container before
        this side stopped waiting. ``seconds`` (how long the container ran) is there only
        when the container itself reported it."""
        ...

    def fetch(self, remote: PurePosixPath, local: Path) -> None: ...


def modal_backend(gpu: str) -> Backend:  # pragma: no cover - needs Modal and an account
    import modal_app

    return modal_app.ModalBackend(gpu)


def _read_record(path: Path) -> dict[str, list[int]]:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return body if isinstance(body, dict) else {}


def upload(
    backend: Backend, uploads: Sequence[Upload], register: Mapping[str, Dataset], record_file: Path
) -> int:
    """Send what is new or changed; every file passes ``guard`` and the orthophoto check
    first, and nothing is sent if one fails. Returns how many files were sent."""
    for u in uploads:
        if not u.local.is_file():
            raise BenchError(f"{u.local} saknas. Kör just bench-fetch igen.", 2)
    local = {u.remote: stat_of(u.local) for u in uploads}
    for u in uploads:
        guard(u.remote, register, local[u.remote][0])
    record = _read_record(record_file)
    todo = plan_upload(uploads, local, backend.remote_sizes(), record)
    findings = [f for u in todo for f in orthophoto_findings(u.local)]
    if findings:
        raise BenchError("Uppladdningen stoppad, ser ut som ortofoto: " + "; ".join(findings), 1)
    total = sum(local[u.remote][0] for u in todo)
    say(f"Laddar upp {len(todo)} av {len(uploads)} filer ({total / 1e9:.2f} GB) till {VOLUME}.")
    record_file.parent.mkdir(parents=True, exist_ok=True)
    for chunk in chunks(todo, {u.remote: local[u.remote][0] for u in todo}):
        backend.upload(chunk)
        for u in chunk:
            record[u.remote.as_posix()] = list(local[u.remote])
        tmp = record_file.with_name(f".{record_file.name}.tmp")
        tmp.write_text(json.dumps(record, indent=0, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(record_file)
    return len(todo)


def chunks(uploads: Sequence[Upload], sizes: Mapping[PurePosixPath, int]) -> list[list[Upload]]:
    """``uploads`` in batches of at most ``CHUNK_FILES`` files and ``CHUNK_BYTES`` (a
    larger file alone)."""
    out: list[list[Upload]] = []
    size = 0
    for u in uploads:
        if not out or len(out[-1]) >= CHUNK_FILES or size + sizes[u.remote] > CHUNK_BYTES:
            out.append([])
            size = 0
        out[-1].append(u)
        size += sizes[u.remote]
    return out


def confirm(yes: bool, ask: Callable[[str], str] = input) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty() and ask is input:
        raise BenchError("Bekräfta med yes=1 när ingen kan svara på frågan.", 2)
    return ask("Starta? Skriv ja för att fortsätta: ").strip().lower() in ("ja", "j", "yes", "y")


def _read_ledger(ledger: Path) -> tuple[float, bool]:
    """(seconds used, a ``fresh=1`` whose removal is not yet confirmed) from ``LEDGER``."""
    try:
        body = json.loads(ledger.read_text(encoding="utf-8"))
        return float(body["seconds"]), bool(body.get("fresh_pending", False))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return 0.0, False


def _write_ledger(ledger: Path, seconds: float, fresh_pending: bool) -> None:
    """Write ``LEDGER`` so that a power loss leaves the old or the new one, whole."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    tmp = ledger.with_name(f".{ledger.name}.tmp")
    body = {"seconds": round(seconds, 1), "fresh_pending": fresh_pending}
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(body) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(ledger)
    with contextlib.suppress(OSError):  # not every file system can sync a folder
        folder = os.open(ledger.parent, os.O_RDONLY)
        try:
            os.fsync(folder)
        finally:
            os.close(folder)


def hours_used(ledger: Path) -> float:
    """The GPU hours the run has used so far (``LEDGER``)."""
    return _read_ledger(ledger)[0] / 3600


def fresh_pending(ledger: Path) -> bool:
    """``fresh=1`` was asked for, but the old run's files are not yet known to be gone."""
    return _read_ledger(ledger)[1]


def start_fresh(ledger: Path) -> None:
    """``fresh=1`` confirmed: the run's hours from 0, and its old files to be removed
    (``remove_old_run``) by this command or, if it stops first, the next one."""
    _write_ledger(ledger, 0.0, True)


def remove_old_run(backend: Backend, where: Places, ledger: Path) -> None:
    """Remove the old run's ``FRESH_FILES`` from the volume, then mark it done in
    ``LEDGER``. Safe to do again: a file already gone is no error."""
    run = where.remote_run.relative_to(REMOTE_DATA)
    backend.remove([run / name for name in FRESH_FILES])
    _write_ledger(ledger, _read_ledger(ledger)[0], False)


def again_text(fresh_given: bool, ledger: Path) -> str:
    """What to run to carry on: without ``fresh=1`` once the old run is removed (or it
    would be removed again, with what the new run has saved), as given before that."""
    if fresh_given and not fresh_pending(ledger):
        return "samma kommando utan fresh=1"
    return "samma kommando"


def call_budget(left_s: float) -> float:
    """How long the next call may train when ``left_s`` of ``max_hours`` is left: under
    Modal's 24 hours with a margin, and so that the call, with its start, its stop and
    last commit (``START_SLACK_S``, ``STOP_GRACE_S``, ``FINISH_SLACK_S``), ends within
    ``left_s`` even by this side's clock. 0: not worth starting."""
    room = min(CALL_LIMIT_S - STOP_MARGIN_S, left_s)
    budget = room - STOP_GRACE_S - FINISH_SLACK_S - START_SLACK_S
    return budget if budget >= MIN_CALL_S else 0.0


def run_until_done(
    backend: Backend,
    train: Sequence[str],
    export: Sequence[str],
    max_hours: float,
    ledger: Path,
    again: str = "samma kommando",
    clock: Callable[[], float] = time.monotonic,
) -> float:
    """Call the GPU until the run is done, together at most ``max_hours`` for the run
    (all calls, also earlier commands', from ``ledger``).

    Each call's worst case (Modal's timeout and the wait for the container) is written
    to ``ledger`` before the call. It is corrected only when the container reported the
    seconds it ran: then to those or the client's wall time, whichever is longer. Without
    them (the client stopped waiting, Ctrl+C, a lost line, a hard stop of this computer)
    the container may still have run that long, so the worst case stays.
    ``again`` is what the messages tell you to run to carry on. Returns the hours used
    by this command."""
    used_here = 0.0
    while True:
        spent = _read_ledger(ledger)[0]
        budget = call_budget(max_hours * 3600 - spent)
        if not budget:
            raise BenchError(
                f"max_hours={max_hours:g} är slut ({spent / 3600:.1f} h använda av den här "
                f"körningen). Kör {again} med större max_hours= för att fortsätta från "
                "senaste sparningen.",
                1,
            )
        timeout_s = math.ceil(budget + STOP_GRACE_S + FINISH_SLACK_S)
        worst = float(timeout_s + START_SLACK_S)
        _write_ledger(ledger, spent + worst, False)
        started = clock()
        res: Mapping[str, Any] = {}
        try:
            res = backend.run(train, export, budget, timeout_s)
        finally:
            seconds = worst
            if res.get("seconds") is not None:
                seconds = max(clock() - started, float(res["seconds"]))
                _write_ledger(ledger, spent + seconds, False)
            used_here += seconds
        status = res.get("status")
        if status == "done":
            return used_here / 3600
        if status == "no_gpu":
            raise BenchError(
                f"Modal gav inget grafikkort på {(clock() - started) / 60:.0f} minuter, så "
                f"inget tränades. Försök igen senare ({again}), eller med en annan gpu=.",
                1,
            )
        if status != "stopped":
            raise BenchError(
                f"Träningen på Modal slutade med kod {res.get('code')}; se utskriften ovan. "
                f"Kör {again} för att fortsätta från senaste sparningen.",
                1,
            )
        say(f"Anropet stoppades efter {seconds / 3600:.1f} h; fortsätter från senaste sparningen.")


def check_local_run(where: Places) -> None:
    """Refuse to write into a local training's folder: one with ``last.pt``, or with
    ``best.pt`` or ``settings.json`` but no ``MARKER`` naming this run on the volume.
    Checked before anything is sent and again before anything comes back."""
    folder = where.local_run
    if (folder / "last.pt").exists():
        raise BenchError(
            f"{folder} har en lokal träning (last.pt). Välj en annan mapp med out=.", 2
        )
    ours = bool(_read_marker(where))
    if not ours and any((folder / f).exists() for f in ("best.pt", "settings.json")):
        raise BenchError(
            f"{folder} har redan en träning som inte kommer från den här körningen i Modal. "
            "Välj en annan mapp med out=.",
            2,
        )


def _read_marker(where: Places) -> dict[str, Any]:
    """The run folder's ``MARKER`` when it names this run on the volume, else ``{}``."""
    try:
        body = json.loads((where.local_run / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if (
        not isinstance(body, dict)
        or body.get("volume") != VOLUME
        or body.get("remote_run") != where.remote_run.name
    ):
        return {}
    return body


def _write_marker(where: Places, onnx_sha256: str = "") -> None:
    """Mark the run folder as this run's; ``onnx_sha256`` adds an ONNX file fetched for
    it (written before that file is moved in place, so a stop between never locks the
    next fetch out)."""
    old = _read_marker(where)
    shas = [s for s in old.get("onnx_sha256", []) if isinstance(s, str)]
    if onnx_sha256 and onnx_sha256 not in shas:
        shas.append(onnx_sha256)
    body = {
        "volume": VOLUME,
        "remote_run": where.remote_run.name,
        "name": where.name,
        "onnx_sha256": shas,
    }
    where.local_run.mkdir(parents=True, exist_ok=True)
    tmp = where.local_run / f".{MARKER}.tmp"
    tmp.write_text(json.dumps(body) + "\n", encoding="utf-8")
    tmp.replace(where.local_run / MARKER)


def check_local_model(where: Places) -> None:
    """Refuse to write over ``<name>.onnx`` or ``<name>.json`` in the models folder
    unless they are what an earlier fetch of this run brought (the ONNX file's SHA-256
    is in ``MARKER``). Checked before anything is sent and again before the fetch."""
    onnx = where.local_models / f"{where.name}.onnx"
    manifest = where.local_models / f"{where.name}.json"
    if not onnx.exists() and not manifest.exists():
        return
    body = _read_marker(where)
    ours = body.get("name") == where.name and (
        not onnx.exists()
        or hashlib.sha256(onnx.read_bytes()).hexdigest() in body.get("onnx_sha256", [])
    )
    if not ours:
        raise BenchError(
            f"{onnx.name} finns redan i {where.local_models} och kommer inte från den här "
            "körningen i Modal. Välj ett annat namn med name=.",
            2,
        )


def remote_error(exc: BaseException, again: str = "samma kommando") -> BenchError:
    """A short Swedish message for an error from Modal or the network, instead of a
    traceback: only the kind of error, never its text (which may hold anything)."""
    if isinstance(exc, ImportError):
        return BenchError(
            "Modal är inte installerat här. Kör via: just train-detector-cloud ...", 2
        )
    return BenchError(
        f"Modal eller nätet svarade inte som väntat ({type(exc).__name__}). Kontrollera "
        "nätet och inloggningen (cd tools/train && uv run --no-sync modal token info). "
        f"Kör sedan {again} för att fortsätta där det slutade.",
        1,
    )


def fetch_results(backend: Backend, where: Places) -> None:
    """Fetch ``best.pt``, ``settings.json``, the ONNX file and its manifest. Each file
    is written beside and then moved in place; the run folder is marked as this run's
    first (``MARKER``), after the checks again (``check_local_run``,
    ``check_local_model``). The manifest must name ``<name>.onnx`` itself, and its
    SHA-256 must match the file."""
    check_local_run(where)
    check_local_model(where)
    _write_marker(where)
    onnx = where.local_models / f"{where.name}.onnx"
    for remote, local in where.results():
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_name(f".{local.name}.part")
        backend.fetch(remote, tmp)
        if local == onnx:
            _write_marker(where, hashlib.sha256(tmp.read_bytes()).hexdigest())
        tmp.replace(local)
    try:
        manifest = json.loads((where.local_models / f"{where.name}.json").read_text("utf-8"))
        named, sha = manifest["onnx"], manifest["sha256"]
    except (OSError, ValueError, KeyError, TypeError):
        raise BenchError(f"{where.name}.json är inget giltigt manifest; kör igen.", 1) from None
    if named != onnx.name:
        raise BenchError(f"Manifestet pekar på {named!r}, inte {onnx.name}; kör igen.", 1)
    if hashlib.sha256(onnx.read_bytes()).hexdigest() != sha:
        raise BenchError(f"{onnx.name} stämmer inte med manifestets SHA-256; kör igen.", 1)


def main(
    argv: Sequence[str] | None = None, backend: Callable[[str], Backend] = modal_backend
) -> int:
    try:
        keys = (*train_keys(), *OWN_KEYS)
        args = parse_args(sys.argv[1:] if argv is None else argv, keys, "train-detector-cloud")
        for key in LOCAL_ONLY_KEYS:
            if key in args:
                raise BenchError(
                    f"{key}= finns inte i molnet: det är alltid grafikkortet, och simulatorns "
                    "bilder stannar på den här datorn. Ta bort det.",
                    2,
                )
        model = args.get("model", "")
        if model not in BASE_HOURS:
            raise BenchError(f"model= ska vara en av {', '.join(BASE_HOURS)}", 2)
        if not args.get("data"):
            raise BenchError("data= saknas (just train-detector-cloud fyller i den).", 2)
        data = Path(args["data"]).expanduser().resolve()
        gpu = args.get("gpu", DEFAULT_GPU)
        if gpu not in PRICES:
            raise BenchError(f"gpu= ska vara en av {', '.join(PRICES)}, inte {gpu!r}.", 2)
        register = load_register()
        requested = [s for s in args.get("sets", ",".join(default_sets())).split(",") if s]
        sets = cloud_sets(requested, register, data)
        where = places(data, model, args.get("out", ""), args.get("name", ""))
        init_local = Path(args["init"]).expanduser().resolve() if args.get("init") else None
        init_sha, init_remote_path = "", ""
        if init_local is not None:
            check_init(init_local)
            init_sha = hashlib.sha256(init_local.read_bytes()).hexdigest()
            init_remote_path = str(REMOTE_DATA / init_remote(init_local, init_sha))
        try:
            epochs = int(args.get("epochs", "20"))
            limit = int(args.get("limit", "0"))
            upscale = int(args.get("upscale", "1"))
            guessed = "hours" not in args
            hours = float(args["hours"]) if not guessed else 0.0
            max_hours = float(args.get("max_hours", str(DEFAULT_MAX_HOURS)))
        except ValueError:
            raise BenchError(
                "epochs=, limit=, upscale=, hours= och max_hours= ska vara tal.", 2
            ) from None
        if guessed:
            n_scales = len([s for s in args.get("scales", "native").split(",") if s])
            hours = guess_hours(model, gpu, epochs, n_scales, upscale)
        price = cost(gpu, hours, max_hours)
        weights = data / "models" / "pretrained" / pretrained_files()[model]
        if not weights.is_file():
            raise BenchError(
                f"COCO-vikterna saknas ({weights.name}). Kör: just detector-fetch yes=1", 2
            )
        check_local_run(where)
        check_local_model(where)
        fresh = flag(args, "fresh")
        ledger = data / LEDGER.format(run=where.remote_run.name)
        before = 0.0 if fresh else hours_used(ledger)
        uploads = plan(data, sets, model, limit, init_local, init_sha)
        say(f"Tränar {model} på {', '.join(sets)} i Modal ({len(uploads)} filer behövs).")
        for line in price.lines(guessed):
            say(line)
        if before:
            say(
                f"Körningen har redan använt {before:.1f} h av max_hours={max_hours:g} "
                f"(ungefär {before * price.price:.0f} $)."
            )
        if fresh:
            say(
                "fresh=1: den gamla träningen i Modal (last.pt, best.pt, settings.json) tas "
                "bort först, och körningen börjar om från noll; dess timmar räknas från 0. "
                "Fortsätt sedan med samma kommando utan fresh=1."
            )
        elif fresh_pending(ledger):
            say(
                "Ett tidigare fresh=1 hann inte ta bort den gamla träningen i Modal; det "
                "görs nu, och körningen börjar om från noll."
            )
        say(f"Resultatet hamnar i {where.local_run} och {where.local_models}.")
        if not confirm(flag(args, "yes")):
            say("Avbrutet; inget startades.")
            return 1
        if fresh:
            start_fresh(ledger)
        started = time.monotonic()
        try:
            client = backend(gpu)
            if fresh_pending(ledger):
                remove_old_run(client, where, ledger)
            upload(client, uploads, register, data / RECORD.format(volume=VOLUME))
            used = run_until_done(
                client,
                train_argv(args, sets, where, init_remote_path),
                export_argv(model, where),
                max_hours,
                ledger,
                again_text(fresh, ledger),
            )
            fetch_results(client, where)
        except KeyboardInterrupt:
            warn(
                f"Avbruten. Kör {again_text(fresh, ledger)} för att fortsätta där det "
                "sparades senast."
            )
            return 130
        except BenchError:
            raise
        except Exception as exc:  # Modal's and the network's errors, in plain words
            raise remote_error(exc, again_text(fresh, ledger)) from exc
        say(
            f"Klart på {used:.1f} timmar i Modal (ungefär {used * price.price:.0f} $), "
            f"{(time.monotonic() - started) / 3600:.1f} h i allt."
        )
        say(f"Modellen: {where.local_models / (where.name + '.json')}")
        say(
            f"Nästa steg: just bench-detector model={where.name} part=val, sedan just pick-detector"
        )
        return 0
    except BenchError as exc:
        warn(str(exc))
        return exc.code
    except KeyboardInterrupt:
        warn("Avbruten. Kör samma kommando igen för att fortsätta där det sparades senast.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
