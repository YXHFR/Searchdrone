"""The locked split of the image bank into train, val and test, by sequence.

``splits/v1.json`` (schema ``bench.split/v1``) lists, per dataset, every sequence with
its part, its number of pictures and a SHA-256 of their sorted ids (so an archive that
changed under us is noticed, ``Split.check``). A picture's part is its sequence's part;
its sequence follows from its id (``tools/bench/layouts.py``). A sequence (one flight,
or one place on one day) is never cut: all its pictures are in the same part, so no
near-copy of a test picture is trained on. The test part is locked: a change is a new
file (``v2``) and every number is measured again.

``make_split`` is how v1 was drawn (and how a v2 would be): sequences, largest first,
in an order fixed by a SHA-256 of the seed and the name, each to the part furthest
below its target share of pictures (70/10/20). ``splits/mini_v1.json`` (schema
``bench.mini/v1``) is the small, fixed test set for the nightly CPU run: picture and
label members with their SHA-256, read straight out of the archive with byte ranges.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bench.common import BENCH, BenchError

SPLITS = BENCH / "splits"
SPLIT_SCHEMA = "bench.split/v1"
MINI_SCHEMA = "bench.mini/v1"
Part = Literal["train", "val", "test"]
PARTS: tuple[Part, ...] = ("train", "val", "test")
RATIOS: dict[Part, float] = {"train": 0.7, "val": 0.1, "test": 0.2}


@dataclass(frozen=True)
class Split:
    version: str
    sequences: dict[tuple[str, str], Part]  # (dataset, sequence) -> part
    counts: dict[tuple[str, str], int]
    digests: dict[tuple[str, str], str]

    def part_of(self, dataset: str, seq: str) -> Part | None:
        return self.sequences.get((dataset, seq))

    def check(self, dataset: str, images: Iterable[tuple[str, str]]) -> None:
        """``images``: ``(local id, sequence)`` of a whole indexed dataset. Every
        sequence must be in the split with the same pictures."""
        found: dict[str, list[str]] = defaultdict(list)
        for local, seq in images:
            found[seq].append(local)
        for seq, ids in found.items():
            key = (dataset, seq)
            if key not in self.sequences:
                raise BenchError(
                    f"{dataset}: sekvensen {seq} finns inte i splits/{self.version}", 1
                )
            if self.counts[key] != len(ids) or self.digests[key] != ids_digest(ids):
                raise BenchError(
                    f"{dataset}: sekvensen {seq} har andra bilder än i splits/{self.version}", 1
                )


def ids_digest(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def make_split(
    sequences: Mapping[str, list[str]], seed: str, ratios: Mapping[Part, float] = RATIOS
) -> dict[str, Part]:
    """Sequence -> part (module doc). ``sequences``: sequence -> its image ids."""
    total = sum(len(v) for v in sequences.values())
    if total == 0:
        return {}

    def order(name: str) -> tuple[int, str]:
        tie = hashlib.sha256(f"{seed}/{name}".encode()).hexdigest()
        return (-len(sequences[name]), tie)

    have: dict[Part, int] = dict.fromkeys(PARTS, 0)
    out: dict[str, Part] = {}
    for name in sorted(sequences, key=order):
        part = max(PARTS, key=lambda p: (ratios[p] * total - have[p]) / max(ratios[p] * total, 1))
        out[name] = part
        have[part] += len(sequences[name])
    return out


def build_split_file(
    version: str, seed: str, listing: Mapping[str, Mapping[str, list[str]]]
) -> dict[str, object]:
    """The body of ``splits/<version>.json`` from dataset -> sequence -> image ids."""
    datasets: dict[str, object] = {}
    for name in sorted(listing):
        seqs = listing[name]
        parts = make_split(seqs, f"{seed}/{name}")
        datasets[name] = {
            "sequences": {
                s: {"part": parts[s], "images": len(seqs[s]), "ids_sha256": ids_digest(seqs[s])}
                for s in sorted(seqs)
            },
        }
    return {
        "schema": SPLIT_SCHEMA,
        "version": version,
        "seed": seed,
        "ratios": dict(RATIOS),
        "rule": "whole sequences per part, largest first, to the part furthest below its share",
        "datasets": datasets,
    }


def load_split(version: str = "v1", path: Path | None = None) -> Split:
    path = path or SPLITS / f"{version}.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("schema") != SPLIT_SCHEMA or body.get("version") != version:
        raise BenchError(f"{path.name}: fel format eller version", 2)
    seqs: dict[tuple[str, str], Part] = {}
    counts: dict[tuple[str, str], int] = {}
    digests: dict[tuple[str, str], str] = {}
    for ds, entry in body["datasets"].items():
        for seq, row in entry["sequences"].items():
            if row["part"] not in PARTS:
                raise BenchError(f"{path.name}: okänd del {row['part']!r}", 2)
            seqs[(ds, seq)] = row["part"]
            counts[(ds, seq)] = int(row["images"])
            digests[(ds, seq)] = str(row["ids_sha256"])
    return Split(version, seqs, counts, digests)


@dataclass(frozen=True)
class MiniImage:
    image_id: str
    image_member: str
    image_sha256: str
    image_bytes: int
    label_member: str | None
    label_sha256: str | None
    label_bytes: int | None


MINI_VERSION = "mini_v1"


def load_mini(version: str = MINI_VERSION, path: Path | None = None) -> tuple[str, list[MiniImage]]:
    """The split version the mini bank was drawn from, and its pictures."""
    path = path or SPLITS / f"{version}.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("schema") != MINI_SCHEMA or body.get("version", version) != version:
        raise BenchError(f"{path.name}: fel format eller version", 2)
    images = [
        MiniImage(
            image_id=row["id"],
            image_member=row["image"]["member"],
            image_sha256=row["image"]["sha256"],
            image_bytes=int(row["image"]["bytes"]),
            label_member=row["label"]["member"] if row.get("label") else None,
            label_sha256=row["label"]["sha256"] if row.get("label") else None,
            label_bytes=int(row["label"]["bytes"]) if row.get("label") else None,
        )
        for row in body["images"]
    ]
    return str(body["split"]), images
