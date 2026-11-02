"""Draw the locked split and the mini bank (a maintainer's tool; the owner never runs it).

    python -m bench.make_splits split version=v1 seed=v1 heridal=<listing> wisard=<listing>
    python -m bench.make_splits mini version=mini_v1 split=v1 n=64
    python -m bench.make_splits extend version=v2 base=v1 kios=<listing> kios_labels=<listing>

``split`` reads each archive's member list (``<name>\\t<size>\\t<crc>`` per line, as
``list`` writes it) and writes ``splits/<version>.json``. ``list`` writes such a listing
straight from the server with byte ranges (``python -m bench.make_splits list
heridal > heridal.txt``), without downloading the archive.

``extend`` writes a new version that keeps every dataset of ``base`` exactly as it was
and draws only the datasets given (for a set published as two archives, the second
listing is ``<name>_labels``). ``<name>=index`` takes a fetched set's index instead of
a listing (in ``data=``): SeaDronesSee's sequences are in its labels, not its member
names. A set that is only trained on gets no test part (``RATIOS_TRAIN_ONLY``):

    python -m bench.make_splits extend version=v3 base=v2 kios_carperson=index seadronessee=index

``mini`` picks ``n`` pictures of the HERIDAL test part, spread evenly over its
sequences, reads each picture and label out of the archive on the server and writes
their SHA-256 to ``splits/<version>.json``. A test split that is already published is
never redrawn: a change is a new version.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from bench.common import BenchError, RangeFile, data_dir, parse_args, say, warn
from bench.datasets import load_register, read_index
from bench.layouts import LAYOUTS
from bench.splits import (
    MINI_SCHEMA,
    RATIOS_TRAIN_ONLY,
    SPLITS,
    build_split_file,
    extend_split_file,
    load_split,
)


def _listing(path: Path) -> list[str]:
    return [
        line.split("\t", 1)[0] for line in path.read_text(encoding="utf-8").splitlines() if line
    ]


def _write(path: Path, body: object) -> None:
    if path.exists():
        raise BenchError(f"{path.name} finns redan; en ändring blir en ny version.", 2)
    path.write_text(json.dumps(body, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    say(f"skrev {path}")


def cmd_list(name: str, labels: bool = False) -> int:
    ds = load_register()[name]
    url = ds.labels_url if labels else ds.url
    if not url or not ds.ranges:
        raise BenchError(f"{name} kan inte listas med byte-intervall", 2)
    with zipfile.ZipFile(io.BufferedReader(RangeFile(url), 1 << 20)) as zf:
        for info in zf.infolist():
            say(f"{info.filename}\t{info.file_size}\t{info.CRC}")
    return 0


def _sequences(args: dict[str, str]) -> dict[str, dict[str, list[str]]]:
    """Dataset -> sequence -> image ids, for every measured or trained-on dataset given
    a listing, or ``index`` (its index from ``just bench-fetch``, in ``data=``)."""
    listing: dict[str, dict[str, list[str]]] = {}
    for name, ds in load_register().items():
        if name not in args or not {"measure", "train"} & set(ds.use):
            continue
        seqs: dict[str, list[str]] = defaultdict(list)
        if args[name] == "index":
            for im in read_index(data_dir(args), name):
                seqs[im.seq].append(im.local_id)
        else:
            names = _listing(Path(args[name]))
            if args.get(f"{name}_labels"):
                names += _listing(Path(args[f"{name}_labels"]))
            for m in LAYOUTS[ds.layout].members(names):
                seqs[m.seq].append(m.image_id)
        listing[name] = dict(seqs)
        say(f"{name}: {sum(len(v) for v in seqs.values())} bilder i {len(seqs)} sekvenser")
    return listing


def cmd_split(args: dict[str, str]) -> int:
    listing = _sequences(args)
    version = args.get("version", "v1")
    body = build_split_file(version, args.get("seed", version), listing)
    _write(SPLITS / f"{version}.json", body)
    return 0


def cmd_extend(args: dict[str, str]) -> int:
    base = args.get("base", "v1")
    load_split(base)  # a published, well-formed file
    body = json.loads((SPLITS / f"{base}.json").read_text(encoding="utf-8"))
    version = args.get("version", "v2")
    listing = _sequences(args)
    register = load_register()
    # a set that is never measured on gets no test part (splits.RATIOS_TRAIN_ONLY)
    ratios = {n: RATIOS_TRAIN_ONLY for n in listing if "measure" not in register[n].use}
    _write(
        SPLITS / f"{version}.json",
        extend_split_file(version, args.get("seed", version), body, listing, ratios),
    )
    return 0


def pick_mini(test_ids: dict[str, list[str]], n: int) -> list[str]:
    """``n`` ids spread over the sequences: each gets its share of ``n`` by size
    (largest remainder), taken evenly spaced through its sorted ids."""
    queues = {s: sorted(v) for s, v in sorted(test_ids.items()) if v}
    total = sum(len(v) for v in queues.values())
    if n >= total:
        return sorted(i for v in queues.values() for i in v)
    exact = {s: n * len(v) / total for s, v in queues.items()}
    share = {s: int(x) for s, x in exact.items()}
    for s in sorted(exact, key=lambda s: (share[s] - exact[s], s))[: n - sum(share.values())]:
        share[s] += 1
    picked: list[str] = []
    for s, ids in queues.items():
        step = len(ids) / share[s] if share[s] else 0
        picked.extend(ids[int(k * step)] for k in range(share[s]))
    return sorted(picked)


def cmd_mini(args: dict[str, str]) -> int:
    register = load_register()
    ds = register["heridal"]
    assert ds.url  # noqa: S101 - the register row has one
    split_version = args.get("split", "v1")
    split = load_split(split_version)
    with zipfile.ZipFile(io.BufferedReader(RangeFile(ds.url), 1 << 20)) as zf:
        members = LAYOUTS[ds.layout].members(zf.namelist())
        by_id = {m.image_id: m for m in members}
        test: dict[str, list[str]] = defaultdict(list)
        for m in members:
            if split.part_of("heridal", m.seq) == "test":
                test[m.seq].append(m.image_id)
        chosen = pick_mini(dict(test), int(args.get("n", "64")))
        rows = []
        for local in chosen:
            m = by_id[local]
            row: dict[str, object] = {"id": f"heridal:{local}"}
            for role, member in (("image", m.image), ("label", m.label)):
                if member is None:
                    row[role] = None
                    continue
                body = zf.read(member)
                row[role] = {
                    "member": member,
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "bytes": len(body),
                }
            rows.append(row)
            say(f"  {local}")
    version = args.get("version", "mini_v1")
    _write(
        SPLITS / f"{version}.json",
        {
            "schema": MINI_SCHEMA,
            "version": version,
            "split": split_version,
            "dataset": "heridal",
            "rule": "test-part pictures spread evenly over the sequences",
            "images": rows,
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if not argv:
            raise BenchError("ange split, extend, mini eller list", 2)
        cmd, rest = argv[0], argv[1:]
        if cmd == "list" and len(rest) in (1, 2):
            if rest[1:] not in ([], ["labels=1"]):
                raise BenchError("list <set> [labels=1]", 2)
            return cmd_list(rest[0], labels=rest[1:] == ["labels=1"])
        names = tuple(load_register())
        sets = (*names, *(f"{n}_labels" for n in names))
        if cmd == "split":
            return cmd_split(parse_args(rest, ("version", "seed", *sets), "make_splits split"))
        if cmd == "extend":
            keys = ("version", "seed", "base", "data", *sets)
            return cmd_extend(parse_args(rest, keys, "make_splits extend"))
        if cmd == "mini":
            return cmd_mini(parse_args(rest, ("version", "split", "n"), "make_splits mini"))
        raise BenchError(f"okänt kommando {cmd!r}", 2)
    except BenchError as exc:
        warn(str(exc))
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
