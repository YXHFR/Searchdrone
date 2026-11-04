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
2. The cost is printed (GPU price per hour from ``PRICES``, times the hours) and you
   are asked before anything is started.
3. ``train_detector.py`` and then ``export_onnx.py`` run on the GPU (``modal_app.py``),
   with their checkpoints on the volume. Modal stops a call after at most 24 hours; the
   run is stopped a little before, saved, and started again from ``last.pt`` until it is
   done or ``max_hours`` is used up. A stopped command (Ctrl+C, a lost line) carries on
   from the last save when run again.
4. ``best.pt`` and ``settings.json`` come back to ``<data>/train/<run>`` and the ONNX file
   and its manifest to ``<data>/models/detector``, where the local training puts them, so
   ``just bench-detector``, ``just pick-detector`` and the simulator use them unchanged.

The Modal token is read by Modal itself from ``~/.modal.toml`` or the environment
(``MODAL_TOKEN_ID``, ``MODAL_TOKEN_SECRET``); it is never read, printed or stored here.
This file never imports ``modal`` at the top: the parts that decide what is uploaded and
what it costs are tested without it, the network or an account.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
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

HERE = Path(__file__).resolve().parent

# The Modal Volume that holds the uploaded pictures and the runs (private to the account).
VOLUME = "searchdrone-train"
# Where the volume is mounted in the container: the data folder there.
REMOTE_DATA = PurePosixPath("/data")

# USD per GPU hour, from modal.com/pricing, read 2026-11-04. The container's CPU and
# memory are billed on top (a few percent of the GPU's price); the volume is not counted.
PRICES_SOURCE = "modal.com/pricing, läst 2026-11-04"
PRICES: dict[str, float] = {"A100-80GB": 2.50, "H100": 3.95, "L40S": 1.95}
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


def guard(remote: PurePosixPath, register: Mapping[str, Dataset]) -> None:
    """The hard check every upload passes, file by file, just before it is sent: the
    path must be one the training reads, and a dataset path must name a set
    ``refusal`` lets go. Raises ``BenchError`` (exit 1) otherwise, and nothing is sent."""
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
    price: float  # USD per hour
    hours: float  # the guess
    max_hours: float  # the ceiling

    @property
    def usd(self) -> float:
        return self.price * min(self.hours, self.max_hours)

    @property
    def max_usd(self) -> float:
        return self.price * self.max_hours

    def lines(self, guessed: bool) -> list[str]:
        how = "grov gissning" if guessed else "din uppskattning (hours=)"
        return [
            f"GPU: {self.gpu}, {self.price:.2f} $/h ({PRICES_SOURCE}).",
            f"Ungefärlig kostnad: {self.gpu} {self.price:.2f} $/h x {self.hours:.1f} h "
            f"= {self.usd:.0f} $ ({how}). CPU och minne tillkommer, några procent.",
            f"Högst: {self.max_hours:g} h x {self.price:.2f} $/h = {self.max_usd:.0f} $ "
            "(max_hours=; sedan stannar den, och samma kommando fortsätter).",
        ]


def cost(gpu: str, hours: float, max_hours: float) -> Cost:
    if gpu not in PRICES:
        raise BenchError(f"gpu= ska vara en av {', '.join(PRICES)}, inte {gpu!r}.", 2)
    if not (hours > 0 and math.isfinite(hours)) or not (max_hours > 0 and math.isfinite(max_hours)):
        raise BenchError("hours= och max_hours= ska vara positiva tal.", 2)
    return Cost(gpu, PRICES[gpu], hours, max_hours)


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
    ``train-detector``; on the volume the run has the same folder name under ``train/``."""
    local_run = Path(out).expanduser().resolve() if out else data / "train" / model
    run = local_run.name
    if not run or run.startswith("."):
        raise BenchError(f"out= ska sluta på ett mappnamn, inte {out!r}.", 2)
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
    data folder, the run folder, the sets and ``init=`` moved to the volume."""
    moved = {"data", "out", "sets", "init", *OWN_KEYS}
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

    def run(self, train: Sequence[str], export: Sequence[str], budget_s: float) -> dict[str, Any]:
        """One call on the GPU: ``{"status": "done" | "stopped" | "failed", "code": int,
        "seconds": float}``. ``stopped``: the budget ran out and the last save is on the
        volume."""
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
        guard(u.remote, register)
        if not u.local.is_file():
            raise BenchError(f"{u.local} saknas. Kör just bench-fetch igen.", 2)
    local = {u.remote: stat_of(u.local) for u in uploads}
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


def run_until_done(
    backend: Backend, train: Sequence[str], export: Sequence[str], max_hours: float
) -> float:
    """Call the GPU until the run is done, each call at most ``CALL_LIMIT_S -
    STOP_MARGIN_S`` long, together at most ``max_hours``. Returns the hours used."""
    spent = 0.0
    while True:
        left = max_hours * 3600 - spent
        if left < STOP_MARGIN_S:
            raise BenchError(
                f"max_hours={max_hours:g} är slut ({spent / 3600:.1f} h). Kör samma kommando med "
                "större max_hours= för att fortsätta från senaste sparningen.",
                1,
            )
        budget = min(CALL_LIMIT_S - STOP_MARGIN_S, left)
        res = backend.run(train, export, budget)
        spent += float(res.get("seconds", budget))
        status = res.get("status")
        if status == "done":
            return spent / 3600
        if status != "stopped":
            raise BenchError(
                f"Träningen på Modal slutade med kod {res.get('code')}; se utskriften ovan. "
                "Samma kommando fortsätter från senaste sparningen.",
                1,
            )
        say(
            f"Anropet tog slut efter {res.get('seconds', 0) / 3600:.1f} h; fortsätter från last.pt."
        )


def fetch_results(backend: Backend, where: Places) -> None:
    for remote, local in where.results():
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_name(f".{local.name}.part")
        backend.fetch(remote, tmp)
        tmp.replace(local)
    manifest = json.loads((where.local_models / f"{where.name}.json").read_text(encoding="utf-8"))
    onnx = where.local_models / manifest["onnx"]
    if hashlib.sha256(onnx.read_bytes()).hexdigest() != manifest["sha256"]:
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
        requested = [s for s in args.get("sets", "heridal,kios,wisard").split(",") if s]
        sets = cloud_sets(requested, register, data)
        where = places(data, model, args.get("out", ""), args.get("name", ""))
        init_local = Path(args["init"]).expanduser().resolve() if args.get("init") else None
        init_sha, init_remote_path = "", ""
        if init_local is not None:
            if not init_local.is_file():
                raise BenchError(f"init={init_local} finns inte.", 2)
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
        uploads = plan(data, sets, model, limit, init_local, init_sha)
        say(f"Tränar {model} på {', '.join(sets)} i Modal ({len(uploads)} filer behövs).")
        for line in price.lines(guessed):
            say(line)
        say(f"Resultatet hamnar i {where.local_run} och {where.local_models}.")
        if not confirm(flag(args, "yes")):
            say("Avbrutet; inget startades.")
            return 1
        started = time.monotonic()
        client = backend(gpu)
        upload(client, uploads, register, data / RECORD.format(volume=VOLUME))
        used = run_until_done(
            client,
            train_argv(args, sets, where, init_remote_path),
            export_argv(model, where),
            max_hours,
        )
        fetch_results(client, where)
        say(
            f"Klart på {used:.1f} GPU-timmar (ungefär {used * price.price:.0f} $), "
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
