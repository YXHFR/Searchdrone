"""The locked split (tools/bench/splits/v1.json) and the mini bank."""

from __future__ import annotations

import json
from typing import Any

import pytest
from bench.common import BenchError
from bench.layouts import LAYOUTS, sequence_of
from bench.splits import (
    PARTS,
    RATIOS,
    SPLITS,
    Split,
    ids_digest,
    load_mini,
    load_split,
    make_split,
)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    assert len(keys) == len(set(keys)), f"duplicate keys: {keys}"
    return dict(pairs)


def test_split_no_leakage() -> None:
    """A picture's part is its sequence's part, and each sequence is listed once with one
    part: so no sequence has pictures in both train and test."""
    body = json.loads(
        (SPLITS / "v1.json").read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys
    )
    split = load_split("v1")
    for ds, entry in body["datasets"].items():
        train = {s for s, r in entry["sequences"].items() if r["part"] == "train"}
        test = {s for s, r in entry["sequences"].items() if r["part"] == "test"}
        val = {s for s, r in entry["sequences"].items() if r["part"] == "val"}
        assert train and test and val, ds
        assert not train & test and not train & val and not val & test
        for seq in entry["sequences"]:
            assert split.part_of(ds, seq) in PARTS


@pytest.mark.parametrize(
    ("layout", "names"),
    [
        # HERIDAL's own train and test pictures share places: the place is the group
        ("heridal_voc", ["h/JPEGImages/train_TRS_0001.jpg", "h/JPEGImages/test_TRS_0002.jpg"]),
        # two flights over one place on one day are one sequence
        (
            "wisard_vis",
            [
                "210924_FHL_Enterprise_VIS_0126/DJI_0001.jpg",
                "210924_FHL_Enterprise_VIS_0126/DJI_0001.txt",
                "210924_FHL_Enterprise_VIS_0401/DJI_0002.jpg",
                "210924_FHL_Enterprise_VIS_0401/DJI_0002.txt",
            ],
        ),
    ],
)
def test_near_copies_share_a_sequence(layout: str, names: list[str]) -> None:
    members = LAYOUTS[layout].members(names)
    assert len(members) == 2
    assert members[0].seq == members[1].seq


def test_split_check_refuses_other_pictures() -> None:
    split = Split("t", {("d", "s"): "test"}, {("d", "s"): 2}, {("d", "s"): ids_digest(["a", "b"])})
    split.check("d", [("a", "s"), ("b", "s")])
    with pytest.raises(BenchError, match="finns inte"):
        split.check("d", [("a", "s"), ("b", "s"), ("c", "other")])
    with pytest.raises(BenchError, match="andra bilder"):
        split.check("d", [("a", "s"), ("c", "s")])


def test_split_covers_every_part_near_its_share() -> None:
    split = load_split("v1")
    for ds in ("heridal", "wisard"):
        n = dict.fromkeys(PARTS, 0)
        for (d, _), part in split.sequences.items():
            if d == ds:
                n[part] += split.counts[(d, _)]
        total = sum(n.values())
        for part in PARTS:
            # whole sequences: the share is only near the target
            assert abs(n[part] / total - RATIOS[part]) < 0.15, (ds, part, n)


def test_split_is_reproducible_from_its_sizes() -> None:
    body = json.loads((SPLITS / "v1.json").read_text(encoding="utf-8"))
    for ds, entry in body["datasets"].items():
        sizes = {s: [f"{s}/{i}" for i in range(r["images"])] for s, r in entry["sequences"].items()}
        again = make_split(sizes, f"{body['seed']}/{ds}")
        assert again == {s: r["part"] for s, r in entry["sequences"].items()}


def test_make_split_keeps_sequences_whole() -> None:
    seqs = {f"s{i}": [f"s{i}_{j}" for j in range(i + 1)] for i in range(20)}
    parts = make_split(seqs, "t")
    assert set(parts) == set(seqs)
    assert set(parts.values()) == set(PARTS)
    assert make_split(seqs, "t") == parts


def test_mini_bank_is_drawn_from_the_test_part() -> None:
    version, rows = load_mini()
    split = load_split(version)
    assert 32 <= len(rows) <= 300
    assert len({r.image_id for r in rows}) == len(rows)
    for r in rows:
        ds, local = r.image_id.split(":", 1)
        assert split.part_of(ds, sequence_of("heridal_voc", local)) == "test", r.image_id
        assert len(r.image_sha256) == 64
        assert r.image_member.endswith(".jpg")
    size = sum(r.image_bytes + (r.label_bytes or 0) for r in rows)
    assert size < 400e6


def test_pick_mini_spreads_by_sequence_size() -> None:
    from bench.make_splits import pick_mini

    ids = {"A": [f"A{i:02d}" for i in range(60)], "B": [f"B{i}" for i in range(30)], "C": ["C0"]}
    picked = pick_mini(ids, 9)
    assert len(picked) == len(set(picked)) == 9
    counts = {s: sum(p.startswith(s) for p in picked) for s in ids}
    assert counts == {"A": 6, "B": 3, "C": 0}  # 5.93, 2.97, 0.10 by largest remainder
    assert pick_mini(ids, 9) == picked  # the same every time
    assert pick_mini(ids, 500) == sorted(i for v in ids.values() for i in v)
