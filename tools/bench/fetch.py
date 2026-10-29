"""``just bench-fetch``: download the image bank to ``<data_dir>/datasets/`` and index it.

    just bench-fetch yes=1                 # every set whose SHA-256 is pinned
    just bench-fetch yes=1 sets=heridal    # one set
    just bench-fetch yes=1 sets=simfarm    # index the sim farm's local pictures
    just bench-fetch yes=1 mini=1          # the small fixed test bank (about 300 MB)

Without ``yes=1`` it only says what would be downloaded, how large it is and under
which licence. Each archive is checked against the size and SHA-256 in
``datasets.toml``; a wrong file is never unpacked. Only the members the layout reads
(pictures and labels) are unpacked, each under a checked name inside ``raw/``. A stopped
run picks up where it was: the download resumes, pictures already unpacked are kept.
The archive is removed once its index is written (``keep=1`` keeps it).

The mini bank (``splits/mini_v1.json``) is read straight out of the archive on the
server with byte ranges, picture by picture, each checked against its own SHA-256; it
lands in ``<data_dir>/datasets-mini/``. The nightly CI run uses it and keeps nothing.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import re
import shutil
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path

from bench.common import (
    ROOT,
    BenchError,
    RangeFile,
    data_dir,
    download,
    flag,
    inside,
    parse_args,
    safe_relpath,
    say,
    warn,
)
from bench.datasets import (
    Dataset,
    Image,
    dataset_dir,
    load_register,
    raw_dir,
    read_source,
    write_index,
)
from bench.layouts import LAYOUTS, Member
from bench.splits import load_mini, load_split

KEYS = ("yes", "sets", "mini", "data", "keep")
MAX_MEMBER_BYTES = 200 << 20  # no picture or label is this large; refuse zip bombs


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        return im.size


def _index_members(ds: Dataset, members: list[Member], raw: Path) -> list[Image]:
    layout = LAYOUTS[ds.layout]
    images: list[Image] = []
    for m in members:
        file = inside(raw, safe_relpath(m.image))
        if not file.is_file():
            continue
        width, height = _image_size(file)
        label = None
        if m.label is not None:
            lpath = inside(raw, safe_relpath(m.label))
            label = lpath.read_bytes() if lpath.is_file() else None
        try:
            boxes = layout.boxes(label, width, height)
        except ValueError as exc:
            raise BenchError(f"{ds.name}: felaktig etikett för {m.image_id} ({exc})") from exc
        images.append(
            Image(
                image_id=f"{ds.name}:{m.image_id}",
                dataset=ds.name,
                seq=m.seq,
                file=m.image,
                width=width,
                height=height,
                boxes=tuple(boxes),
            )
        )
    return images


def _wanted(ds: Dataset, names: list[str]) -> tuple[list[Member], list[str]]:
    members = LAYOUTS[ds.layout].members(names)
    files = sorted({m.image for m in members} | {m.label for m in members if m.label})
    return members, files


def _extract(zf: zipfile.ZipFile, files: list[str], raw: Path) -> None:
    infos = {i.filename: i for i in zf.infolist()}
    for n, name in enumerate(files, 1):
        info = infos[name]
        if info.file_size > MAX_MEMBER_BYTES:
            raise BenchError(f"{name} i arkivet är orimligt stor ({info.file_size} byte)")
        target = inside(raw, safe_relpath(name))
        if target.is_file() and target.stat().st_size == info.file_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        with zf.open(info) as src, tmp.open("wb") as dst:  # zipfile checks each CRC-32
            shutil.copyfileobj(src, dst, 1 << 20)
        tmp.replace(target)
        if n % 500 == 0:
            say(f"  {n}/{len(files)} filer uppackade")


def _progress(name: str):
    last = [-1]

    def show(have: int, size: int) -> None:
        pct = int(100 * have / size)
        if pct != last[0] and pct % 5 == 0:
            last[0] = pct
            say(f"  {name}: {pct} % ({_gb(have)} av {_gb(size)})")

    return show


def _is_sha256(value: str | None) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", value or "") is not None


def pinned(ds: Dataset) -> bool:
    """Whether the archive's SHA-256 is known, and the labels archive's if it has one
    (only then is it ever downloaded)."""
    return _is_sha256(ds.sha256) and (ds.labels_url is None or _is_sha256(ds.labels_sha256))


def _archives(ds: Dataset, folder: Path) -> list[tuple[str, int, str, Path]]:
    """``(url, size, sha256, local file)`` of the set's archives: the pictures, then the
    labels when they are published apart."""
    assert ds.url and ds.size_bytes and ds.sha256  # noqa: S101 - checked by the caller
    out = [(ds.url, ds.size_bytes, ds.sha256, folder / ds.archive_name)]
    if ds.labels_url:
        assert ds.labels_size_bytes and ds.labels_sha256  # noqa: S101 - checked by pinned()
        out.append(
            (ds.labels_url, ds.labels_size_bytes, ds.labels_sha256, folder / ds.labels_archive_name)
        )
    return out


def fetch_archive(ds: Dataset, data: Path, keep: bool) -> int:
    assert ds.url and ds.size_bytes  # noqa: S101 - checked by the caller
    if ds.sha256 is None or not pinned(ds):
        raise BenchError(
            f"{ds.name}: arkivets SHA-256 är inte fastställd än i datasets.toml, så det "
            "hämtas inte. Se tools/bench/README.md.",
            2,
        )
    source = f"{ds.url}#sha256={ds.sha256}"
    if ds.labels_url:
        source += f"+{ds.labels_url}#sha256={ds.labels_sha256}"
    if read_source(data, ds.name) == source:
        say(f"{ds.name}: redan hämtad och indexerad.")
        return 0
    folder = dataset_dir(data, ds.name)
    archives = _archives(ds, folder)
    for url, size, sha, archive in archives:
        say(f"{ds.name}: hämtar {_gb(size)} ({ds.licence}) till {archive}")
        download(url, archive, size, sha, progress=_progress(archive.name))
    raw = raw_dir(data, ds.name)
    with contextlib.ExitStack() as stack:
        zips = [stack.enter_context(zipfile.ZipFile(a[3])) for a in archives]
        owner: dict[str, zipfile.ZipFile] = {}
        for zf in zips:
            for name in zf.namelist():
                if name in owner and not name.endswith("/"):
                    raise BenchError(f"{ds.name}: {name} finns i två arkiv")
                owner[name] = zf
        members, files = _wanted(ds, list(owner))
        say(f"{ds.name}: packar upp {len(files)} filer till {raw}")
        for zf in zips:
            _extract(zf, [f for f in files if owner[f] is zf], raw)
    images = _index_members(ds, members, raw)
    if "measure" in ds.use:  # the archive must hold exactly the pictures the split was drawn from
        load_split().check(ds.name, [(i.local_id, i.seq) for i in images])
    path = write_index(data, ds, images, source)
    say(f"{ds.name}: {len(images)} bilder, {sum(len(i.boxes) for i in images)} personer -> {path}")
    if not keep:
        for *_, archive in archives:
            archive.unlink()
    return 0


def index_local(ds: Dataset, data: Path) -> int:
    """The sim farm: its pictures already lie in the repository's ``build/``; nothing is
    copied, the index points there (``image_root``)."""
    assert ds.local_path  # noqa: S101 - checked by the register
    base = (ROOT / ds.local_path).resolve()
    if not base.is_dir():
        raise BenchError(f"{ds.name}: {base} finns inte. Kör först: just sim-farm", 2)
    names = sorted(p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file())
    members, _ = _wanted(ds, names)
    images = _index_members(ds, members, base)
    path = write_index(data, ds, images, f"local:{ds.local_path}")
    say(f"{ds.name}: {len(images)} bilder (bara förträning) -> {path}")
    return 0


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_mini(register: dict[str, Dataset], data: Path) -> int:
    split, rows = load_mini()
    ds = register["heridal"]
    assert ds.url  # noqa: S101 - the register row has one
    source = f"{ds.url}#mini={split}"
    raw = raw_dir(data, ds.name, mini=True)
    todo = []
    for row in rows:
        for member, sha, size in (
            (row.image_member, row.image_sha256, row.image_bytes),
            (row.label_member, row.label_sha256, row.label_bytes),
        ):
            if member is None or sha is None or size is None:
                continue
            target = inside(raw, safe_relpath(member))
            if target.is_file() and target.stat().st_size == size:
                continue  # checked when it was written
            todo.append((member, sha, size, target))
    if todo:
        total = sum(t[2] for t in todo)
        say(f"Minibanken: hämtar {len(todo)} filer ({total / 1e6:.0f} MB) ur {ds.name}-arkivet")
        with zipfile.ZipFile(io.BufferedReader(RangeFile(ds.url), 1 << 20)) as zf:
            infos = {i.filename: i for i in zf.infolist()}
            for n, (member, sha, size, target) in enumerate(todo, 1):
                info = infos.get(member)
                if (
                    info is None
                    or info.file_size != size
                    or size > MAX_MEMBER_BYTES
                    or info.compress_size > MAX_MEMBER_BYTES
                ):
                    raise BenchError(f"{member} saknas i arkivet eller har fel storlek")
                body = zf.read(info)
                if _sha256(body) != sha:
                    raise BenchError(f"Kontrollsumman stämmer inte för {member}; inget sparas.")
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + ".part")
                tmp.write_bytes(body)
                tmp.replace(target)
                if n % 10 == 0:
                    say(f"  {n}/{len(todo)}")
    wanted = {r.image_id.split(":", 1)[1] for r in rows}
    names = [r.image_member for r in rows] + [r.label_member for r in rows if r.label_member]
    members = [m for m in LAYOUTS[ds.layout].members(names) if m.image_id in wanted]
    images = _index_members(ds, members, raw)
    path = write_index(data, ds, images, source, mini=True)
    say(f"Minibanken: {len(images)} bilder, {sum(len(i.boxes) for i in images)} personer -> {path}")
    return 0


def plan(register: dict[str, Dataset], names: list[str], mini: bool) -> str:
    lines = ["Det här skulle hämtas (kör igen med yes=1 för att hämta):"]
    if mini:
        _, rows = load_mini()
        size = sum(r.image_bytes + (r.label_bytes or 0) for r in rows)
        lines.append(
            f"  minibanken: {len(rows)} bilder ur HERIDAL, {size / 1e6:.0f} MB (CC-BY-3.0)"
        )
        return "\n".join(lines)
    for name in names:
        ds = register[name]
        if ds.url:
            what = _gb((ds.size_bytes or 0) + (ds.labels_size_bytes or 0))
        else:
            what = f"lokala bilder i {ds.local_path}, inget hämtas"
        lines.append(f"  {name}: {what}, licens {ds.licence}")
        if ds.licence_note:
            lines.append(f"    ({ds.licence_note})")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv, KEYS, "bench-fetch")
        register = load_register()
        mini = flag(args, "mini")
        if args.get("sets"):
            names = [s.strip() for s in args["sets"].split(",") if s.strip()]
            unknown = [n for n in names if n not in register]
            if unknown:
                known = ", ".join(sorted(register))
                raise BenchError(f"Okänd datamängd: {', '.join(unknown)}. Kända: {known}.", 2)
        else:
            names = [n for n, ds in register.items() if ds.url and pinned(ds)]
            for n, ds in register.items():
                if ds.url and not pinned(ds) and not mini:
                    say(f"{n} hoppas över: arkivets SHA-256 är inte fastställd än.")
        if not flag(args, "yes"):
            say(plan(register, names, mini))
            return 2
        data = data_dir(args)
        if mini:
            return fetch_mini(register, data)
        for name in names:
            ds = register[name]
            if ds.url:
                fetch_archive(ds, data, keep=flag(args, "keep"))
            else:
                index_local(ds, data)
        return 0
    except BenchError as exc:
        warn(str(exc))
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
