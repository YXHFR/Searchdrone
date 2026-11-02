"""The dataset register (``tools/bench/datasets.toml``), its licence rule and the
per-dataset index that ``just bench-fetch`` writes.

The index (``<data_dir>/datasets/<name>/index.json``, schema ``bench.index/v1``) lists
every picture once: its id, sequence, file (relative to ``raw/``), size and person
boxes. The training and the measurement both read only the index, never the archive.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bench.common import BENCH, ROOT, BenchError
from bench.layouts import LAYOUTS, Box

REGISTER = BENCH / "datasets.toml"
INDEX_SCHEMA = "bench.index/v1"
Use = Literal["train", "measure", "pretrain"]
USES: tuple[Use, ...] = ("train", "measure", "pretrain")

# Open licences: free to train and measure on, results may be published with credit.
OPEN_LICENCES = frozenset({"CC0-1.0", "CC-BY-3.0", "CC-BY-4.0", "MIT", "Apache-2.0"})
# Non-commercial licences: only for measuring on the owner's computer, never trained on.
NC_PREFIXES = ("CC-BY-NC",)
# No grant to build on (GPL, or no data licence at all: "NONE"): only measured on, on the
# owner's computer, never trained on, and no weights or results published (the owner's
# decision of 2026-11-01).
NO_GRANT_PREFIXES = ("GPL-",)
NO_GRANT_LICENCES = frozenset({"NONE"})


@dataclass(frozen=True)
class Archive:
    """One more picture archive of a set published as several (``more_archives``)."""

    url: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class Dataset:
    name: str
    title: str
    licence: str
    attribution: str
    layout: str
    use: tuple[Use, ...]
    publish_weights: bool
    local_only: bool
    # whether numbers measured on it may leave this computer (a profile, a README figure)
    publish_results: bool = True
    url: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    ranges: bool = False
    local_path: str | None = None
    licence_note: str | None = None
    source_page: str | None = None
    checked: str | None = None
    # True only when the licence is known to cover the pictures; a set is never trained
    # on without it (a row that leaves it out is not confirmed)
    licence_confirmed: bool = False
    # a second archive with the labels, for a set published as pictures and labels apart
    labels_url: str | None = None
    labels_size_bytes: int | None = None
    labels_sha256: str | None = None
    # more picture archives, for a set published as several (fetched and unpacked with
    # the first, each checked against its own size and SHA-256)
    more_archives: tuple[Archive, ...] = ()
    # a set published as loose files on a WebDAV share, not as an archive: ``url`` is the
    # share's folder, ``share_folders`` the folders under it that are fetched,
    # ``share_files`` how many files they hold, ``size_bytes`` their total size and
    # ``sha256`` the SHA-256 of their ``sha256sum`` lines sorted by path (fetch.py)
    share_folders: tuple[str, ...] = ()
    share_files: int | None = None

    @property
    def is_share(self) -> bool:
        return bool(self.share_folders)

    @property
    def no_grant(self) -> bool:
        """GPL or no data licence: measured on locally, never trained on or published."""
        return self.licence.startswith(NO_GRANT_PREFIXES) or self.licence in NO_GRANT_LICENCES

    @property
    def archive_name(self) -> str:
        return f"{self.name}.zip"

    @property
    def labels_archive_name(self) -> str:
        return f"{self.name}-labels.zip"


def licence_problems(ds: Dataset) -> list[str]:
    """What breaks the licence rule (empty: the row is fine)."""
    problems: list[str] = []
    is_open = ds.licence in OPEN_LICENCES
    is_nc = ds.licence.startswith(NC_PREFIXES)
    if not (is_open or is_nc or ds.no_grant):
        problems.append(f"{ds.name}: licence {ds.licence!r} is not allowed")
    if is_nc and (set(ds.use) != {"measure"} or not ds.local_only or ds.publish_weights):
        problems.append(f"{ds.name}: a non-commercial set may only be measured on, locally")
    if ds.no_grant and (
        set(ds.use) != {"measure"} or not ds.local_only or ds.publish_weights or ds.publish_results
    ):
        problems.append(
            f"{ds.name}: a GPL or unlicensed set may only be measured on, locally, "
            "with publish_weights and publish_results false"
        )
    if "train" in ds.use and not ds.licence_confirmed:
        problems.append(f"{ds.name}: never trained on until the data licence is confirmed")
    if "pretrain" in ds.use and len(ds.use) > 1:
        problems.append(f"{ds.name}: synthetic pre-training data is never trained or measured on")
    if not ds.attribution.strip():
        problems.append(f"{ds.name}: attribution missing")
    if ds.url is None and ds.local_path is None:
        problems.append(f"{ds.name}: neither url nor local_path")
    if ds.url is not None and not ds.url.startswith("https://"):
        problems.append(f"{ds.name}: url must be https")
    if ds.labels_url is not None and (
        ds.url is None or not ds.labels_url.startswith("https://") or not ds.labels_size_bytes
    ):
        problems.append(f"{ds.name}: labels_url must be https, with url and labels_size_bytes")
    for extra in ds.more_archives:
        if ds.url is None or not extra.url.startswith("https://") or extra.size_bytes <= 0:
            problems.append(f"{ds.name}: more_archives must be https, with url and a size")
    if ds.is_share and (
        ds.url is None
        or not ds.url.endswith("/")
        or ds.labels_url is not None
        or ds.more_archives
        or not ds.share_files
        or not ds.size_bytes
    ):
        problems.append(
            f"{ds.name}: a share needs url (a folder, ending in /), share_files and "
            "size_bytes, and no labels_url or more_archives"
        )
    for folder in ds.share_folders:
        if not re.fullmatch(r"[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)*", folder):
            problems.append(f"{ds.name}: share folder {folder!r} is not a plain relative path")
    return problems


def _flag(row: dict[str, Any], key: str, default: bool | None = None) -> bool:
    """A yes/no field: only TOML's true or false (``bool("false")`` would be True)."""
    value = row[key] if default is None else row.get(key, default)
    if not isinstance(value, bool):
        raise BenchError(
            f"datasets.toml: {key} i {row.get('name')} ska vara true eller false, inte {value!r}",
            2,
        )
    return value


def _dataset(row: dict[str, Any]) -> Dataset:
    try:
        use = tuple(row["use"])
        ds = Dataset(
            name=row["name"],
            title=row["title"],
            licence=row["licence"],
            attribution=row["attribution"],
            layout=row["layout"],
            use=use,  # type: ignore[arg-type]
            publish_weights=_flag(row, "publish_weights"),
            local_only=_flag(row, "local_only"),
            publish_results=_flag(row, "publish_results", True),
            url=row.get("url"),
            size_bytes=row.get("size_bytes"),
            sha256=row.get("sha256"),
            ranges=_flag(row, "ranges", False),
            local_path=row.get("local_path"),
            licence_note=row.get("licence_note"),
            source_page=row.get("source_page"),
            checked=row.get("checked"),
            licence_confirmed=_flag(row, "licence_confirmed", False),
            labels_url=row.get("labels_url"),
            labels_size_bytes=row.get("labels_size_bytes"),
            labels_sha256=row.get("labels_sha256"),
            more_archives=tuple(
                Archive(url=str(a["url"]), size_bytes=int(a["size_bytes"]), sha256=str(a["sha256"]))
                for a in row.get("more_archives", [])
            ),
            share_folders=tuple(str(f) for f in row.get("share_folders", [])),
            share_files=row.get("share_files"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchError(f"datasets.toml: felaktig rad ({exc})", 2) from exc
    if ds.layout not in LAYOUTS or any(u not in USES for u in use):
        raise BenchError(f"datasets.toml: okänd layout eller användning i {ds.name}", 2)
    return ds


def load_register(path: Path = REGISTER) -> dict[str, Dataset]:
    rows = tomllib.loads(path.read_text(encoding="utf-8")).get("dataset", [])
    out: dict[str, Dataset] = {}
    for row in rows:
        ds = _dataset(row)
        if ds.name in out:
            raise BenchError(f"datasets.toml: {ds.name} finns två gånger", 2)
        problems = licence_problems(ds)
        if problems:
            raise BenchError("datasets.toml bryter licensregeln: " + "; ".join(problems), 2)
        out[ds.name] = ds
    return out


# --- the index ------------------------------------------------------------------------


@dataclass(frozen=True)
class Image:
    image_id: str  # "<dataset>:<id>"
    dataset: str
    seq: str
    file: str  # relative to the dataset's raw/ folder
    width: int
    height: int
    boxes: tuple[Box, ...]

    @property
    def local_id(self) -> str:
        return self.image_id.split(":", 1)[1]


def dataset_dir(data: Path, name: str, mini: bool = False) -> Path:
    """``<data>/datasets/<name>``; the mini bank lives apart in ``datasets-mini``."""
    return data / ("datasets-mini" if mini else "datasets") / name


def index_path(data: Path, name: str, mini: bool = False) -> Path:
    return dataset_dir(data, name, mini) / "index.json"


def write_index(
    data: Path, ds: Dataset, images: list[Image], source: str, mini: bool = False
) -> Path:
    path = index_path(data, ds.name, mini)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema": INDEX_SCHEMA,
        "dataset": ds.name,
        "licence": ds.licence,
        "source": source,
        "images": [
            {
                "id": im.image_id,
                "seq": im.seq,
                "file": im.file,
                "width": im.width,
                "height": im.height,
                "boxes": [[round(v, 2) for v in b] for b in im.boxes],
            }
            for im in images
        ],
    }
    tmp = path.with_name(".index.json.tmp")
    tmp.write_text(json.dumps(body) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def read_index(data: Path, name: str, mini: bool = False) -> list[Image]:
    path = index_path(data, name, mini)
    if not path.is_file():
        how = "mini=1" if mini else f"sets={name}"
        raise BenchError(f"{name} är inte hämtad. Kör: just bench-fetch yes=1 {how}", 2)
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("schema") != INDEX_SCHEMA or body.get("dataset") != name:
        raise BenchError(f"{path} har fel format; kör just bench-fetch igen.", 2)
    return [
        Image(
            image_id=row["id"],
            dataset=name,
            seq=row["seq"],
            file=row["file"],
            width=int(row["width"]),
            height=int(row["height"]),
            boxes=tuple((float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in row["boxes"]),
        )
        for row in body["images"]
    ]


def read_source(data: Path, name: str, mini: bool = False) -> str | None:
    """The ``source`` an existing index was built from (``None``: no index)."""
    path = index_path(data, name, mini)
    if not path.is_file():
        return None
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("source"))
    except ValueError:
        return None


def raw_dir(data: Path, name: str, mini: bool = False) -> Path:
    return dataset_dir(data, name, mini) / "raw"


def image_root(data: Path, name: str, mini: bool = False) -> Path:
    """The folder an index's ``file`` paths are relative to: ``raw/``, or for a local
    set (the sim farm) the folder in the repository it was indexed from."""
    source = read_source(data, name, mini) or ""
    if source.startswith("local:"):
        root = (ROOT / source.removeprefix("local:")).resolve()
        if not root.is_relative_to(ROOT.resolve()):
            raise BenchError(f"{name}: indexet pekar utanför repot ({root})", 2)
        return root
    return raw_dir(data, name, mini)
