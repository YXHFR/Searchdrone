"""The detector measures (tools/bench/metrics.py, eval_detector.py, profiles.py)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from bench import eval_detector as ev
from bench import pick_detector
from bench.common import BenchError
from bench.datasets import Image
from bench.metrics import (
    MP12,
    PictureResult,
    ap50,
    match_picture,
    size_class,
    summarise,
    threshold_for,
    weakest_recall,
    wilson,
)
from bench.profiles import START, build_profile
from PIL import Image as PILImage


def _pictures(tmp_path: Path) -> list[ev.Picture]:
    """Three synthetic pictures with persons of each size class (no real picture)."""
    out = []
    boxes = [
        ((10.0, 10.0, 20.0, 20.0), (100.0, 100.0, 125.0, 120.0)),
        ((50.0, 50.0, 90.0, 110.0),),
        (),
    ]
    for n, bx in enumerate(boxes):
        path = tmp_path / f"p{n}.png"
        PILImage.new("RGB", (320, 240), (30, 80, 30)).save(path)
        out.append(ev.Picture(Image(f"t:p{n}", "t", "s", path.name, 320, 240, bx), path))
    return out


def test_null_detector_gives_recall_zero(tmp_path: Path) -> None:
    s = ev.measure("null", _pictures(tmp_path))
    assert s.recall["all"]["n"] == 3
    assert all(row["recall"] == 0.0 for row in s.recall.values())
    assert s.ap50 == 0.0


def test_facit_detector_gives_recall_one(tmp_path: Path) -> None:
    s = ev.measure("facit", _pictures(tmp_path))
    for cls in ("lt16", "16to32", "gt32", "all"):
        assert s.recall[cls]["n"] == 1 if cls != "all" else 3
        assert s.recall[cls]["recall"] == 1.0
    assert s.ap50 == pytest.approx(1.0)
    assert s.fp_per_12mp == 0.0
    assert s.precision_at_score_min == 1.0


def test_selfcheck_passes_on_the_synthetic_bank(tmp_path: Path) -> None:
    errors, bodies = ev.selfcheck(_pictures(tmp_path))
    assert errors == []
    assert set(bodies) == {"null", "facit"}


def test_result_holds_only_numbers(tmp_path: Path) -> None:
    pics = _pictures(tmp_path)
    body = ev.result_body("facit", "test", ev.measure("facit", pics))
    text = json.dumps(body)
    assert body["schema"] == "bench.detector/v1"
    assert "p0" not in text and ".png" not in text and str(tmp_path) not in text


def test_compare_flags_a_drop_over_two_points(tmp_path: Path) -> None:
    def body(r: float) -> dict[str, object]:
        return {"recall": {"gt32": {"recall": r}}}

    old = tmp_path / "old.json"
    old.write_text(json.dumps(body(0.90)), encoding="utf-8")
    assert ev.compare(body(0.885), old) is None
    assert ev.compare(body(0.87), old) is not None


def _per_set_body(pooled: float, **sets: float) -> dict[str, object]:
    def row(r: float) -> dict[str, object]:
        return {"by_class": {"gt32": {"recall": r, "n": 50}}}

    return {
        "recall": {"gt32": {"recall": pooled}},
        "recall_by_dataset": {name: row(r) for name, r in sets.items()},
    }


def test_compare_fails_when_the_weakest_dataset_drops(tmp_path: Path) -> None:
    """A pooled gain carried by KIOS never hides a drop on HERIDAL, the set
    ``pick-detector`` ranks on."""
    old = tmp_path / "old.json"
    old.write_text(json.dumps(_per_set_body(0.80, heridal=0.78, kios=0.80)), encoding="utf-8")
    worse = _per_set_body(0.88, heridal=0.70, kios=0.90)  # pooled up 8 points
    assert weakest_recall(worse, "gt32") == 0.70
    why = ev.compare(worse, old)
    assert why is not None and "svagaste" in why and "heridal sjönk" in why
    assert ev.compare(_per_set_body(0.70, heridal=0.77, kios=0.79), old) is None


def test_compare_fails_when_any_one_dataset_drops(tmp_path: Path) -> None:
    """KIOS losing 20 points fails although HERIDAL stays the weakest and holds."""
    old = tmp_path / "old.json"
    old.write_text(json.dumps(_per_set_body(0.90, heridal=0.70, kios=0.95)), encoding="utf-8")
    why = ev.compare(_per_set_body(0.75, heridal=0.70, kios=0.75), old)
    assert why is not None and "kios sjönk 20.0" in why and "svagaste" not in why


def test_compare_fails_when_a_dataset_was_not_measured(tmp_path: Path) -> None:
    old = tmp_path / "old.json"
    old.write_text(json.dumps(_per_set_body(0.80, heridal=0.78, kios=0.80)), encoding="utf-8")
    why = ev.compare(_per_set_body(0.95, kios=0.95), old)
    assert why is not None and "heridal saknas" in why


def test_compare_against_an_older_pooled_result_is_pooled_and_says_so(tmp_path: Path) -> None:
    """An earlier result without per-dataset rows: pooled against pooled, never the rule
    change mistaken for a drop."""
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"recall": {"gt32": {"recall": 0.80}}}), encoding="utf-8")
    assert ev.compare(_per_set_body(0.80, heridal=0.60, kios=0.82), old) is None
    why = ev.compare(_per_set_body(0.75, heridal=0.75, kios=0.75), old)
    assert why is not None and "sammanslagen" in why


@pytest.mark.parametrize(
    ("box", "cls"),
    [
        ((0, 0, 15.9, 10), "lt16"),
        ((0, 0, 16, 4), "16to32"),
        ((0, 0, 10, 32), "16to32"),
        ((0, 0, 33, 5), "gt32"),
    ],
)
def test_size_class_by_longest_side(box: tuple[float, ...], cls: str) -> None:
    assert size_class(box) == cls


def test_matching_is_greedy_by_score_and_one_to_one() -> None:
    truth = [(0, 0, 10, 10)]
    dets = [(0.4, (0, 0, 10, 10)), (0.9, (0, 0, 10, 11)), (0.8, (50, 50, 60, 60))]
    r = match_picture("d", 100, 100, truth, dets)
    assert r.truth_hit_score == [0.9]
    assert sorted(r.fp_scores) == [0.4, 0.8]


def test_low_overlap_is_a_miss() -> None:
    r = match_picture("d", 100, 100, [(0, 0, 10, 10)], [(0.9, (6, 6, 16, 16))])
    assert r.truth_hit_score == [None]
    assert r.fp_scores == [0.9]


def test_threshold_keeps_false_alarms_within_half_per_12mp() -> None:
    # 4 pictures of 12 MP: 2 false alarms allowed
    pics = [match_picture("d", 4000, 3000, [], [(s, (0, 0, 9, 9))]) for s in (0.9, 0.8, 0.7, 0.6)]
    assert sum(p.pixels for p in pics) == 4 * MP12
    t = threshold_for(pics)
    assert 0.7 < t <= 0.8 + 1e-9
    s = summarise(pics)
    assert s.fp_per_12mp == pytest.approx(0.5)


def test_threshold_counts_other_picture_sizes_by_area() -> None:
    # 2 pictures of 3 MP = 0.5 x 12 MP: 0.25 false alarms allowed, so none
    pics = [match_picture("d", 2000, 1500, [], [(0.9, (0, 0, 9, 9))]) for _ in range(2)]
    assert threshold_for(pics) > 0.9


def test_threshold_ties_are_kept_out_together() -> None:
    pics = [match_picture("d", 4000, 3000, [], [(0.5, (0, 0, 9, 9))]) for _ in range(4)]
    assert threshold_for(pics) > 0.5


def test_threshold_is_zero_without_false_alarms() -> None:
    assert threshold_for([match_picture("d", 10, 10, [(0, 0, 5, 5)], [(0.3, (0, 0, 5, 5))])]) == 0.0


def test_ap50_of_a_ranked_list() -> None:
    # hit, miss, hit over 2 persons: AP = 0.5 * 1 + 0.5 * 2/3
    r = match_picture(
        "d",
        100,
        100,
        [(0, 0, 10, 10), (20, 20, 30, 30)],
        [(0.9, (0, 0, 10, 10)), (0.8, (60, 60, 70, 70)), (0.7, (20, 20, 30, 30))],
    )
    assert ap50([r]) == pytest.approx(0.5 + 0.5 * 2 / 3)


def test_wilson_interval() -> None:
    lo, hi = wilson(9, 10)
    assert lo < 0.9 < hi
    assert wilson(0, 0) == (0.0, 1.0)
    assert math.isclose(wilson(10, 10)[1], 1.0)


def test_profile_has_the_oracle_format() -> None:
    result = {
        "schema": "bench.detector/v1",
        "source": "splits/v1 test (heridal)",
        "measured": "2026-11-02",
        "fp_per_12mp_target": 0.5,
        "recall": {c: {"recall": 0.5, "n": 10} for c in ("lt16", "16to32", "gt32", "all")},
        "latency_ms": {"p50": 800.0, "p95": 1200.0},
        "score_hit": {"a": 5.0, "b": 2.0},
        "score_fp": None,
    }
    start = json.loads(START.read_text(encoding="utf-8"))
    p = build_profile("yolox_s", 1, result, start)
    assert p["profile_id"] == "det_yolox_s_v1"
    for key in (
        "recall",
        "occlusion_factor",
        "fp_per_12mp",
        "score_hit",
        "score_fp",
        "latency_ms",
        "source",
    ):
        assert key in p and key in start
    assert set(p["recall"]) == {"lt16", "16to32", "gt32"}  # type: ignore[arg-type]
    assert p["score_hit"] == {"a": 5.0, "b": 2.0}
    assert p["not_measured"] == ["occlusion_factor", "score_fp"]


def test_profile_refuses_a_result_pooled_over_datasets() -> None:
    """The simulator's profile must describe one kind of terrain, not mostly KIOS."""
    row = {"recall": 0.5, "n": 10}
    result = {
        "schema": "bench.detector/v1",
        "recall": dict.fromkeys(("lt16", "16to32", "gt32", "all"), row),
        "recall_by_dataset": {"heridal": row, "kios": row},
        "latency_ms": {"p50": 800.0, "p95": 1200.0},
    }
    start = json.loads(START.read_text(encoding="utf-8"))
    with pytest.raises(BenchError, match="flera datamängder"):
        build_profile("yolox_s", 1, result, start)


# --- per dataset ----------------------------------------------------------------------


def _person(dataset: str, side: float, score: float | None, fps: list[float]) -> PictureResult:
    """A 12 MP picture with one person of ``side`` px, hit at ``score`` (None: missed)."""
    return PictureResult(dataset, MP12, [size_class((0, 0, side, side))], [score], fps)


def _two_sets() -> list[PictureResult]:
    # "a": two big people (one hit at 0.9, one at 0.4), one small one at 0.5, and a false
    # alarm at 0.8 in each of its 2 pictures. "b": four pictures with no false alarm.
    a = [
        _person("a", 40, 0.9, [0.8]),
        _person("a", 40, 0.4, [0.8]),
        _person("a", 20, 0.5, []),
    ]
    b = [_person("b", 40, 0.3, []), _person("b", 40, 0.2, []), _person("b", 20, None, [])]
    b.append(PictureResult("b", MP12, [], [], []))
    return a + b


def test_each_dataset_is_measured_at_its_own_threshold() -> None:
    """Pooled, the 7 pictures allow 3 false alarms and the threshold falls to 0; "a"
    alone allows 1 of its 2, so it must keep only what scores above 0.8."""
    s = summarise(_two_sets())
    assert s.threshold == 0.0
    a, b = s.recall_by_dataset["a"], s.recall_by_dataset["b"]
    assert a["threshold"] == pytest.approx(0.8)
    assert a["fp_per_12mp"] == 0.0
    assert a["by_class"] == {
        "lt16": {"n": 0, "hits": 0, "recall": 0.0, "lo": 0.0, "hi": 1.0},
        "16to32": pytest.approx({"n": 1, "hits": 0, "recall": 0.0, "lo": 0.0, "hi": 0.7935}),
        "gt32": pytest.approx({"n": 2, "hits": 1, "recall": 0.5, "lo": 0.0945, "hi": 0.9055}),
    }
    assert (a["n"], a["hits"]) == (3, 1)
    assert b["threshold"] == 0.0
    b_class = b["by_class"]
    assert isinstance(b_class, dict)
    assert (b_class["gt32"]["hits"], b_class["16to32"]["hits"]) == (2, 0)
    # each row is what a run on that dataset alone gives
    for name in ("a", "b"):
        alone = summarise([r for r in _two_sets() if r.dataset == name])
        assert s.recall_by_dataset[name]["by_class"] == {
            c: alone.recall[c] for c in ("lt16", "16to32", "gt32")
        }
        assert s.recall_by_dataset[name]["threshold"] == alone.as_dict()["threshold"]


def test_pick_ranks_on_what_summarise_writes() -> None:
    """The picker reads the rows summarise writes: the weakest dataset, and a size class
    with nobody in it is left out."""
    body = summarise(_two_sets()).as_dict()
    assert weakest_recall(body, "gt32") == 0.5  # "a" at its own threshold
    assert weakest_recall(body, "16to32") == 0.0
    assert pick_detector._per_set(body) == " (a 0.500/0.000, b 1.000/0.000)"
    body["recall_by_dataset"]["b"]["by_class"]["16to32"]["n"] = 0  # type: ignore[index]
    assert "b 1.000/-" in pick_detector._per_set(body)


# --- the locked test part ------------------------------------------------------------


def _indexed_bank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fetched 'heridal' with a test place (TST), a val place (VAL) and a train place
    (TRN), and the split that says so."""
    from bench import datasets
    from bench.splits import Split, ids_digest

    data = tmp_path / "data"
    raw = data / "datasets" / "heridal" / "raw"
    raw.mkdir(parents=True)
    images = []
    for seq in ("TST", "VAL", "TRN"):
        name = f"test_{seq}_0001.png"
        PILImage.new("RGB", (64, 48)).save(raw / name)
        images.append(Image(f"heridal:test_{seq}_0001", "heridal", seq, name, 64, 48, ()))
    datasets.write_index(data, datasets.load_register()["heridal"], images, "test")
    parts = {"TST": "test", "VAL": "val", "TRN": "train"}
    split = Split(
        "v1",
        {("heridal", s): p for s, p in parts.items()},  # type: ignore[misc]
        {("heridal", s): 1 for s in parts},
        {("heridal", s): ids_digest([f"test_{s}_0001"]) for s in parts},
    )
    monkeypatch.setattr(ev, "load_split", lambda *a: split)
    return data


@pytest.mark.parametrize(("part", "seq"), [("test", "TST"), ("val", "VAL")])
def test_only_the_asked_part_is_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, part: str, seq: str
) -> None:
    data = _indexed_bank(tmp_path, monkeypatch)
    source, items = ev.pictures({"part": part, "sets": "heridal"}, data)
    assert [p.image.seq for p in items] == [seq]
    assert source == f"splits/v1 {part} (heridal)"


def test_train_part_and_unmeasured_sets_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _indexed_bank(tmp_path, monkeypatch)
    with pytest.raises(BenchError, match="part"):
        ev.pictures({"part": "train"}, data)
    with pytest.raises(BenchError, match="Mäts inte"):
        ev.pictures({"sets": "simfarm"}, data)


def test_an_index_that_does_not_match_the_split_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bench import datasets

    data = _indexed_bank(tmp_path, monkeypatch)
    extra = Image("heridal:test_TST_0002", "heridal", "TST", "x.png", 64, 48, ())
    images = [*datasets.read_index(data, "heridal"), extra]
    datasets.write_index(data, datasets.load_register()["heridal"], images, "test")
    with pytest.raises(BenchError, match="andra bilder"):
        ev.pictures({"part": "test"}, data)


def test_the_mini_bank_measures_only_its_listed_pictures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bench import datasets
    from bench.splits import MiniImage

    data = tmp_path / "data"
    raw = data / "datasets-mini" / "heridal" / "raw"
    raw.mkdir(parents=True)
    images = []
    for seq in ("AAA", "BBB"):
        name = f"test_{seq}_0001.png"
        PILImage.new("RGB", (64, 48)).save(raw / name)
        images.append(Image(f"heridal:test_{seq}_0001", "heridal", seq, name, 64, 48, ()))
    register = datasets.load_register()
    datasets.write_index(data, register["heridal"], images, "mini", mini=True)

    def mini(*ids: str) -> None:
        rows = [MiniImage(i, "m.jpg", "0" * 64, 1, None, None, None) for i in ids]
        monkeypatch.setattr(ev, "load_mini", lambda *a: ("v1", rows))

    mini("heridal:test_AAA_0001")
    source, items = ev.pictures({"mini": "1"}, data)
    assert [p.image.seq for p in items] == ["AAA"]
    assert source == "mini_v1 (splits/v1)"
    mini("heridal:test_AAA_0001", "heridal:test_CCC_0001")
    with pytest.raises(BenchError, match="Minibanken är ofullständig"):
        ev.pictures({"mini": "1"}, data)


def test_profiles_main_writes_a_profile(tmp_path: Path) -> None:
    from bench import profiles

    pics = _pictures(tmp_path)
    body = ev.result_body("facit", "splits/v1 test (t)", ev.measure("facit", pics))
    result = tmp_path / "r.json"
    result.write_text(json.dumps(body), encoding="utf-8")
    out = tmp_path / "p.json"
    assert profiles.main(["model=facit_x", "version=2", f"result={result}", f"out={out}"]) == 0
    p = json.loads(out.read_text(encoding="utf-8"))
    assert p["profile_id"] == "det_facit_x_v2" and p["recall"]["gt32"] == 1.0
    assert profiles.main(["model=Bad-Name", f"result={result}", f"out={out}"]) == 2
