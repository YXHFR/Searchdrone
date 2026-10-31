"""The locked splits (tools/bench/splits/v1.json, v2.json) and the mini bank."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from bench.common import BenchError
from bench.layouts import LAYOUTS, sequence_of
from bench.splits import (
    CURRENT,
    PARTS,
    RATIOS,
    SPLITS,
    Split,
    extend_split_file,
    ids_digest,
    load_mini,
    load_split,
    make_split,
)

VERSIONS = ("v1", "v2")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    assert len(keys) == len(set(keys)), f"duplicate keys: {keys}"
    return dict(pairs)


@pytest.mark.parametrize("version", VERSIONS)
def test_split_no_leakage(version: str) -> None:
    """A picture's part is its sequence's part, and each sequence is listed once with one
    part: so no sequence has pictures in both train and test."""
    body = json.loads(
        (SPLITS / f"{version}.json").read_text(encoding="utf-8"),
        object_pairs_hook=_no_duplicate_keys,
    )
    split = load_split(version)
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


@pytest.mark.parametrize("version", VERSIONS)
def test_split_covers_every_part_near_its_share(version: str) -> None:
    split = load_split(version)
    for ds in {d for d, _ in split.sequences}:
        n = dict.fromkeys(PARTS, 0)
        for (d, _), part in split.sequences.items():
            if d == ds:
                n[part] += split.counts[(d, _)]
        total = sum(n.values())
        for part in PARTS:
            # whole sequences: the share is only near the target
            assert abs(n[part] / total - RATIOS[part]) < 0.15, (ds, part, n)


@pytest.mark.parametrize("version", VERSIONS)
def test_split_is_reproducible_from_its_sizes(version: str) -> None:
    body = json.loads((SPLITS / f"{version}.json").read_text(encoding="utf-8"))
    for ds, entry in body["datasets"].items():
        sizes = {s: [f"{s}/{i}" for i in range(r["images"])] for s, r in entry["sequences"].items()}
        again = make_split(sizes, entry.get("seed", f"{body['seed']}/{ds}"))
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


# --- v2: v1 as it was, plus KIOS by site -------------------------------------------------


def test_v2_keeps_every_v1_dataset_as_it_was() -> None:
    """HERIDAL's test part (and WiSARD's) is locked: numbers measured on v1 stay
    comparable."""
    v1 = json.loads((SPLITS / "v1.json").read_text(encoding="utf-8"))
    v2 = json.loads((SPLITS / "v2.json").read_text(encoding="utf-8"))
    assert v2["base"] == "v1"
    for ds, entry in v1["datasets"].items():
        kept = {k: v for k, v in v2["datasets"][ds].items() if k != "seed"}
        assert kept == entry, ds
        assert v2["datasets"][ds]["seed"] == f"{v1['seed']}/{ds}"
    old, new = load_split("v1"), load_split("v2")
    for key, part in old.sequences.items():
        assert new.sequences[key] == part


def test_the_current_split_is_v2_with_kios_by_site() -> None:
    from bench.layouts import KIOS_SITES

    assert CURRENT == "v2"
    split = load_split()
    sites = {seq: part for (ds, seq), part in split.sequences.items() if ds == "kios"}
    assert set(sites) == set(KIOS_SITES.values())  # a site is a sequence: never cut
    assert set(sites.values()) == set(PARTS)
    assert sum(split.counts[("kios", s)] for s in sites) == 3136  # the record's 3,136 pictures


def test_extend_never_redraws_a_locked_dataset() -> None:
    base = json.loads((SPLITS / "v1.json").read_text(encoding="utf-8"))
    with pytest.raises(BenchError, match="låst"):
        extend_split_file("v9", "v9", base, {"heridal": {"TRS": ["test_TRS_0001"]}})


def test_extend_draws_only_the_new_dataset() -> None:
    base = {
        "schema": "bench.split/v1",
        "version": "v1",
        "seed": "v1",
        "datasets": {"a": {"sequences": {"s": {"part": "test", "images": 1, "ids_sha256": "x"}}}},
    }
    sites = {f"site{i}": [f"site{i}/{j}" for j in range(10 * (i + 1))] for i in range(6)}
    body = extend_split_file("v2", "v2", base, {"b": sites})
    assert body["datasets"]["a"] == {"seed": "v1/a", **base["datasets"]["a"]}  # type: ignore[index]
    drawn = body["datasets"]["b"]  # type: ignore[index]
    assert drawn["seed"] == "v2/b"
    assert {s: r["part"] for s, r in drawn["sequences"].items()} == make_split(sites, "v2/b")
    for s, r in drawn["sequences"].items():
        assert r["images"] == len(sites[s]) and r["ids_sha256"] == ids_digest(sites[s])


@pytest.mark.parametrize(
    ("argv", "called"),
    [
        (["list", "kios"], ("kios", False)),
        (["list", "kios", "labels=1"], ("kios", True)),
        (["list", "kios", "labels=0"], None),
        (["list", "kios", "yes"], None),
        (["list"], None),
        (["list", "kios", "labels=1", "extra"], None),
    ],
)
def test_list_takes_a_set_and_an_optional_labels_flag(
    argv: list[str], called: tuple[str, bool] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``make_splits list <set> [labels=1]``: anything else is refused before any
    archive is opened."""
    from bench import make_splits

    calls: list[tuple[str, bool]] = []

    def fake_list(name: str, labels: bool = False) -> int:
        calls.append((name, labels))
        return 0

    monkeypatch.setattr(make_splits, "cmd_list", fake_list)
    code = make_splits.main(argv)
    assert calls == ([called] if called else [])
    assert (code == 0) == (called is not None)


def test_extend_from_the_command_line_needs_the_labels_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``make_splits extend`` reads the pictures' and the labels' archive listings; with
    the pictures alone every KIOS picture lacks its labels and nothing is written."""
    from bench import make_splits

    pictures = ["Images/Train/1_people_0007_0.jpg", "Images/Test/2_ppl_4_00001.jpg"]
    labels = [
        f"Annotations/Yolo/{p.split('/')[1]}/{p.split('/')[2].removesuffix('.jpg')}.txt"
        for p in pictures
    ]
    (tmp_path / "kios.txt").write_text("\n".join(pictures) + "\n", encoding="utf-8")
    (tmp_path / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    out = tmp_path / "splits"
    out.mkdir()
    (out / "v1.json").write_text((SPLITS / "v1.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(make_splits, "SPLITS", out)
    base = ["extend", "version=v9", "base=v1", f"kios={tmp_path / 'kios.txt'}"]
    assert make_splits.main(base) != 0
    assert not (out / "v9.json").exists()
    assert make_splits.main([*base, f"kios_labels={tmp_path / 'labels.txt'}"]) == 0
    body = json.loads((out / "v9.json").read_text(encoding="utf-8"))
    assert body["base"] == "v1" and body["datasets"]["heridal"] == json.loads(
        (SPLITS / "v1.json").read_text(encoding="utf-8")
    )["datasets"]["heridal"] | {"seed": "v1/heridal"}
    assert set(body["datasets"]["kios"]["sequences"]) == {"campus_exercise", "meadow"}
