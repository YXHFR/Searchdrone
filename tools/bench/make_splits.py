"""Draw the locked split and the mini bank (a maintainer's tool; the owner never runs it).

    python -m bench.make_splits split version=v1 seed=v1 heridal=<listing> wisard=<listing>
    python -m bench.make_splits mini version=mini_v1 split=v1 n=64

``split`` reads each archive's member list (``<name>\\t<size>\\t<crc>`` per line, as
``list`` writes it) and writes ``splits/<version>.json``. ``list`` writes such a listing
straight from the server with byte ranges (``python -m bench.make_splits list
heridal > heridal.txt``), without downloading the archive.

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

from bench.common import BenchError, RangeFile, parse_args, say, warn
from bench.datasets import load_register
from bench.layouts import LAYOUTS
from bench.splits import MINI_SCHEMA, SPLITS, build_split_file, load_split


def _listing(path: Path) -> list[str]:
    return [
        line.split("\t", 1)[0] for line in path.read_text(encoding="utf-8").splitlines() if line
    ]


def _write(path: Path, body: object) -> None:
    if path.exists():
        raise BenchError(f"{path.name} finns redan; en ändring blir en ny version.", 2)
    path.write_text(json.dumps(body, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    say(f"skrev {path}")


def cmd_list(name: str) -> int:
    ds = load_register()[name]
    if not ds.url or not ds.ranges:
        raise BenchError(f"{name} kan inte listas med byte-intervall", 2)
    with zipfile.ZipFile(io.BufferedReader(RangeFile(ds.url), 1 << 20)) as zf:
        for info in zf.infolist():
            say(f"{info.filename}\t{info.file_size}\t{info.CRC}")
    return 0


def cmd_split(args: dict[str, str]) -> int:
    register = load_register()
    listing: dict[str, dict[str, list[str]]] = {}
    for name, ds in register.items():
        if name not in args or "measure" not in ds.use:
            continue
        seqs: dict[str, list[str]] = defaultdict(list)
        for m in LAYOUTS[ds.layout].members(_listing(Path(args[name]))):
            seqs[m.seq].append(m.image_id)
        listing[name] = dict(seqs)
        say(f"{name}: {sum(len(v) for v in seqs.values())} bilder i {len(seqs)} sekvenser")
    version = args.get("version", "v1")
    body = build_split_file(version, args.get("seed", version), listing)
    _write(SPLITS / f"{version}.json", body)
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
            raise BenchError("ange split, mini eller list", 2)
        cmd, rest = argv[0], argv[1:]
        if cmd == "list" and len(rest) == 1:
            return cmd_list(rest[0])
        if cmd == "split":
            names = tuple(load_register())
            return cmd_split(parse_args(rest, ("version", "seed", *names), "make_splits split"))
        if cmd == "mini":
            return cmd_mini(parse_args(rest, ("version", "split", "n"), "make_splits mini"))
        raise BenchError(f"okänt kommando {cmd!r}", 2)
    except BenchError as exc:
        warn(str(exc))
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
