"""The whole training chain on CPU with a few made-up pictures and one epoch: tiles,
COCO start, training, stop and resume, ONNX export. No real data, no download."""

from __future__ import annotations

import dataclasses
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import families
import numpy as np
import pytest
import tiles
import torch
import train_detector
from bench.common import BenchError
from bench.datasets import Image, load_register, write_index
from bench.splits import Split, ids_digest
from evaluate import ap50
from export_onnx import export, val_ap50_rule
from PIL import Image as PILImage

TILE = 128
SEQS = {"AAA": "train", "BBB": "train", "CCC": "val"}


def _picture(path: Path, boxes: list[tuple[int, int, int, int]], seed: int) -> None:
    rng = np.random.default_rng(seed)
    rgb = rng.integers(60, 120, size=(200, 480, 3), dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        rgb[y0:y1, x0:x1] = (220, 40, 40)
    PILImage.fromarray(rgb).save(path)


@pytest.fixture
def bank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fetched 'heridal' of six 480 x 200 pictures in three places, and COCO-shaped
    weights for both families (random, 80 classes)."""
    data = tmp_path / "data"
    raw = data / "datasets" / "heridal" / "raw"
    raw.mkdir(parents=True)
    images: list[Image] = []
    for i, seq in enumerate([s for s in SEQS for _ in range(2)]):
        boxes = [
            (30 + 10 * i + 110 * k, 40 + 50 * (k % 2), 50 + 10 * i + 110 * k, 70 + 50 * (k % 2))
            for k in range(4)
        ]
        name = f"{seq}_{i}.jpg"
        _picture(raw / name, boxes, i)
        images.append(Image(f"heridal:{seq}_{i}", "heridal", seq, name, 480, 200, tuple(boxes)))
    write_index(data, load_register()["heridal"], images, "test")
    by_seq: dict[str, list[str]] = {}
    for im in images:
        by_seq.setdefault(im.seq, []).append(im.local_id)
    split = Split(
        "v1",
        {("heridal", s): p for s, p in SEQS.items()},  # type: ignore[misc]
        {("heridal", s): len(v) for s, v in by_seq.items()},
        {("heridal", s): ids_digest(v) for s, v in by_seq.items()},
    )
    monkeypatch.setattr(tiles, "load_split", lambda *a, **k: split)
    pre = data / "models" / "pretrained"
    pre.mkdir(parents=True)
    for model, file in families.PRETRAINED.items():
        torch.manual_seed(0)
        net = families.build(model, num_classes=80)
        state = net.state_dict()
        if model == "damoyolo_s":  # the released weights: one class channel more, "module."
            net = __import__("vendor.damoyolo", fromlist=["DamoYolo"]).DamoYolo(80, legacy=True)
            state = {f"module.{k}": v for k, v in net.state_dict().items()}
        torch.save({"model": state}, pre / file)
    return data


def test_tiles_keep_people_and_some_ground() -> None:
    cut = tiles.cut(300, 200, [(10.0, 10.0, 30.0, 40.0), (120.0, 120.0, 140.0, 199.0)], 128)
    assert len(cut) == 3 * 2
    with_people = [t for t, kept in cut if kept]
    assert with_people
    # a person cut in half by a tile edge is kept only where at least half of them is
    for (x0, y0, x1, y1), kept in cut:
        for b in kept:
            assert 0 <= b[0] < b[2] <= x1 - x0 and 0 <= b[1] < b[3] <= y1 - y0


@pytest.mark.parametrize("file", ["../outside.jpg", "/etc/outside.jpg"])
def test_a_picture_outside_its_folder_is_refused(tmp_path: Path, file: str) -> None:
    im = Image("t:x", "t", "s", file, 64, 64, ())
    with pytest.raises(BenchError):
        tiles.build_cache([(im, tmp_path / "pictures")], tmp_path / "tiles", progress=False)


def test_test_part_and_measure_only_sets_are_never_trained_on(bank: Path) -> None:
    train = tiles.pictures(bank, "train", ["heridal"])
    val = tiles.pictures(bank, "val", ["heridal"])
    assert {im.seq for im, _ in train} == {"AAA", "BBB"}
    assert {im.seq for im, _ in val} == {"CCC"}
    nc = [d.name for d in load_register().values() if "train" not in d.use]
    for name in nc:
        assert tiles.pictures(bank, "train", [name]) == []


@pytest.mark.parametrize("model", list(families.FAMILIES))
def test_coco_person_channel_starts_the_person_head(bank: Path, model: str) -> None:
    net = families.build(model)
    person = families.load_coco(
        net, model, bank / "models" / "pretrained" / families.PRETRAINED[model]
    )
    assert person
    coco: Any = torch.load(
        bank / "models" / "pretrained" / families.PRETRAINED[model], weights_only=True
    )
    coco = {k.removeprefix("module."): v for k, v in coco["model"].items()}
    for name in person:
        assert torch.equal(net.state_dict()[name], coco[name][:1])


@pytest.mark.parametrize("model", list(families.FAMILIES))
def test_train_one_epoch_resume_and_export(
    bank: Path, model: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "run"
    argv = [f"model={model}", f"data={bank}", f"out={out}", "epochs=1", "batch=2",
            f"tile={TILE}", "device=cpu", "workers=0", "save_minutes=0"]  # fmt: skip

    calls = {"n": 0}
    real_loss = families.loss

    def stop_on_third(*a: Any) -> torch.Tensor:
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real_loss(*a)

    monkeypatch.setattr(families, "loss", stop_on_third)
    assert train_detector.main(argv) == 130  # stopped by hand
    saved = torch.load(out / "last.pt", weights_only=True)
    assert saved["step"] == 2 and saved["epoch"] == 0
    monkeypatch.setattr(families, "loss", real_loss)
    assert train_detector.main(argv) == 0  # carries on from step 2
    best = torch.load(out / "best.pt", weights_only=True)
    assert best["epoch"] == 1 and best["val_ap50"] is not None
    # another setting in the same folder is refused, not mixed
    assert train_detector.main([*argv[:-1], "batch=4"]) == 2
    assert json.loads((out / "settings.json").read_text())["balance"] == "sets"
    assert train_detector.main([*argv, "balance=tiles"]) == 2  # nor another balance
    assert best["balance"] == "sets" and best["val_ap50_pick"] == ["heridal"]

    onnx_path, manifest_path = export(out / "best.pt", tmp_path / "models", f"person_{model}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["family"] == families.FAMILIES[model].family
    assert manifest["input_px"] == TILE and manifest["classes"] == ["person"]
    assert manifest["onnx"] == onnx_path.name and len(manifest["sha256"]) == 64
    assert [t["dataset"] for t in manifest["trained_on"]] == ["heridal", "wisard"][:1]
    assert manifest["publish"] is True
    assert manifest["balance"] == "sets" and manifest["val_ap50_pick"] == ["heridal"]
    assert manifest["val_ap50_rule"] == "min_over_datasets"  # heridal is all there is


def _bce_as_cuda_autocast_allows(real: Any) -> Any:
    """``binary_cross_entropy`` as CUDA's autocast treats it: refused while autocast is on,
    and the inputs must have one dtype. CPU autocast quietly runs it in float32, so
    without this a CPU test would miss what crashes on the graphics card."""

    def bce(input: torch.Tensor, target: torch.Tensor, *a: Any, **k: Any) -> torch.Tensor:
        if any(torch.is_autocast_enabled(d) for d in ("cpu", "cuda")):
            raise RuntimeError("binary_cross_entropy is unsafe to autocast")
        if input.dtype != target.dtype:
            raise RuntimeError(f"Found dtype {target.dtype} but expected {input.dtype}")
        return real(input, target, *a, **k)

    return bce


@pytest.mark.parametrize("model", list(families.FAMILIES))
def test_train_step_under_mixed_precision(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torch.nn.functional,
        "binary_cross_entropy",
        _bce_as_cuda_autocast_allows(torch.nn.functional.binary_cross_entropy),
    )
    torch.manual_seed(0)
    net = families.build(model).train()
    images = torch.rand(2, 3, TILE, TILE) * 255
    boxes = [torch.tensor([[10.0, 10.0, 40.0, 50.0], [60.0, 60.0, 90.0, 100.0]]), torch.zeros(0, 4)]
    targets = families.train_targets(model, boxes, torch.device("cpu"))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = families.loss(net, images, targets)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize(
    ("error", "falls_back"), [(torch.OutOfMemoryError, True), (RuntimeError, False)]
)
def test_damo_assignment_moves_to_the_cpu_only_when_memory_runs_out(
    error: type[Exception], falls_back: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vendor.damoyolo.ota_assigner import AlignOTAAssigner

    assigner = AlignOTAAssigner(center_radius=2.5, cls_weight=1.0, iou_weight=3.0)
    real = assigner._assign
    calls: list[int] = []

    def fail_once(*a: Any, **k: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise error("first try")
        return real(*a, **k)

    monkeypatch.setattr(assigner, "_assign", fail_once)
    priors = torch.tensor([[16.0, 16.0, 8.0, 8.0], [48.0, 48.0, 8.0, 8.0]])
    args = (
        torch.full((2, 1), 0.5),
        priors,
        torch.tensor([[8.0, 8.0, 24.0, 24.0], [40.0, 40.0, 56.0, 56.0]]),
        torch.tensor([[6.0, 6.0, 26.0, 26.0]]),
        torch.zeros(1, dtype=torch.long),
    )
    if falls_back:
        with pytest.warns(UserWarning, match="OOM"):
            result = assigner.assign(*args)
        assert len(calls) == 2 and result.gt_inds.shape == (2,)
    else:
        with pytest.raises(RuntimeError, match="first try"):
            assigner.assign(*args)
        assert len(calls) == 1


def test_missing_weights_say_what_to_run(bank: Path, tmp_path: Path) -> None:
    (bank / "models" / "pretrained" / "yolox_s.pth").unlink()
    with pytest.raises(BenchError, match="detector-fetch"):
        train_detector.train(
            train_detector.Settings("yolox_s", epochs=1, batch=2, sets=("heridal",), tile=TILE),
            bank, tmp_path / "o", torch.device("cpu"), 0, 10.0, fresh=False,
        )  # fmt: skip


def test_ap50_is_one_for_a_perfect_tile_and_zero_for_a_miss() -> None:
    truth = np.array([[10, 10, 20, 20]], dtype=np.float32)
    assert ap50([(truth.copy(), np.array([0.9]), truth)]) == pytest.approx(1.0)
    assert ap50([(truth + 50, np.array([0.9]), truth)]) == 0.0


# --- going on from earlier weights, and val per dataset ------------------------------------


def _earlier_best(path: Path, model: str = "yolox_s") -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    net = families.build(model)
    state = net.state_dict()
    torch.save(
        {
            "model": model,
            "settings": {"sets": ["heridal"], "simfarm_epochs": 3},
            "ema": state,
            "epoch": 12,
        },
        path,
    )
    return state


def test_init_starts_from_earlier_weights_and_keeps_their_credit(
    bank: Path, tmp_path: Path
) -> None:
    init = tmp_path / "heridal_run" / "best.pt"
    init.parent.mkdir()
    earlier = _earlier_best(init)
    net = families.build("yolox_s")
    seen = train_detector.load_init(net, "yolox_s", init)
    for k, v in net.state_dict().items():
        assert torch.equal(v, earlier[k]), k
    assert (seen["trained_on"], seen["pretrained_on"]) == (["heridal"], ["simfarm"])
    assert seen["started_from"]["epoch"] == 12

    out = tmp_path / "mix"
    argv = [
        "model=yolox_s", f"data={bank}", f"out={out}", "epochs=1", "batch=2", f"tile={TILE}",
        "device=cpu", "workers=0", f"init={init}",
    ]  # fmt: skip
    assert train_detector.main(argv) == 0
    best = torch.load(out / "best.pt", weights_only=True)
    assert best["trained_on"] == ["heridal"] and best["pretrained_on"] == ["simfarm"]
    assert json.loads((out / "settings.json").read_text())["init"] == str(init)
    _, manifest_path = export(out / "best.pt", tmp_path / "models", "person_mix", check_images=1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [t["dataset"] for t in manifest["trained_on"]] == ["heridal"]
    assert manifest["pretrained_on"] == ["simfarm"]  # the earlier run's, credited still
    assert set(manifest["val_ap50_by_dataset"]) == {"heridal"}
    assert manifest["val_ap50_rule"] == "min_over_datasets"
    assert "own threshold" in manifest["recall_rule"]
    # where it started is named and pinned, but the owner's local path never leaves
    assert manifest["started_from"]["file"] == "best.pt"
    assert len(manifest["started_from"]["sha256"]) == 64
    assert manifest["base_weights"].startswith("best.pt (sha256 ")
    assert manifest["settings"]["init"] == "best.pt"
    assert str(tmp_path) not in manifest_path.read_text(encoding="utf-8")


def test_measure_only_sets_are_never_credited_as_trained_on() -> None:
    measure_only = [d.name for d in load_register().values() if "train" not in d.use]
    assert measure_only  # WiSARD, until its data licence is confirmed
    assert train_detector.trained_sets(["heridal", *measure_only]) == ["heridal"]
    # a set gone from the register stays, so the export refuses it instead of dropping it
    assert train_detector.trained_sets(["heridal", "gone"]) == ["heridal", "gone"]


@pytest.mark.parametrize("licence", ["GPL-3.0-only", "NONE"])
def test_gpl_or_unlicensed_set_is_refused_not_trained_on(
    bank: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    licence: str,
) -> None:
    """The owner's decision of 2026-11-01: such a set is measured on, never trained on."""
    import bench.datasets

    register = load_register()
    measured = dataclasses.replace(
        register["heridal"],
        name="gplset",
        licence=licence,
        use=("measure",),
        local_only=True,
        publish_weights=False,
        publish_results=False,
        licence_confirmed=False,
    )
    assert bench.datasets.licence_problems(measured) == []
    with_it = {**register, "gplset": measured}
    monkeypatch.setattr(bench.datasets, "load_register", lambda *a, **k: with_it)
    monkeypatch.setattr(tiles, "load_register", lambda *a, **k: with_it)
    out = tmp_path / "run"
    argv = ["model=yolox_s", f"data={bank}", f"out={out}", "epochs=1", "sets=heridal,gplset",
            "device=cpu", "workers=0"]  # fmt: skip
    assert train_detector.main(argv) == 2
    assert "gplset får bara mätas på" in capsys.readouterr().err
    assert not (out / "last.pt").exists()
    assert tiles.pictures(bank, "train", ["gplset"]) == []  # nor would the tiles take it


def test_old_checkpoints_get_their_credit_from_their_settings() -> None:
    """One rule for a ``best.pt`` from before ``trained_on``/``pretrained_on`` were saved,
    used by both ``init=`` and the export."""
    measure_only = [d.name for d in load_register().values() if "train" not in d.use]
    old = {"settings": {"sets": ["heridal", *measure_only], "simfarm_epochs": 2}}
    assert train_detector.seen_by(old) == (["heridal"], ["simfarm"])
    assert train_detector.seen_by({"settings": {"sets": ["heridal"]}}) == (["heridal"], [])
    new = {**old, "trained_on": ["kios"], "pretrained_on": []}
    assert train_detector.seen_by(new) == (["kios"], [])


def test_export_of_old_weights_credits_only_what_was_trained_on(bank: Path, tmp_path: Path) -> None:
    """Weights from before ``trained_on`` was saved fall back on their settings' sets,
    filtered the way the training filters them: a measure-only set is never credited."""
    measure_only = [d.name for d in load_register().values() if "train" not in d.use]
    best = tmp_path / "old" / "best.pt"
    best.parent.mkdir()
    _earlier_best(best)
    ck = torch.load(best, weights_only=True)
    ck.pop("trained_on", None)
    ck["settings"] = {
        **train_detector.Settings("yolox_s").key(),
        "sets": ["heridal", *measure_only],
    }
    torch.save(ck, best)
    _, manifest_path = export(best, tmp_path / "models", "person_x", check_images=1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [d["dataset"] for d in manifest["trained_on"]] == ["heridal"]


def test_export_refuses_weights_trained_on_an_unregistered_set(bank: Path, tmp_path: Path) -> None:
    best = tmp_path / "old" / "best.pt"
    best.parent.mkdir()
    _earlier_best(best)
    ck = torch.load(best, weights_only=True)
    torch.save({**ck, "trained_on": ["heridal", "gone"]}, best)
    with pytest.raises(BenchError, match="gone"):
        export(best, tmp_path / "models", "person_x", check_images=1)


def test_init_inside_out_or_of_another_model_is_refused(bank: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    _earlier_best(out / "best.pt")
    base = ["model=yolox_s", f"data={bank}", "epochs=1", "batch=2", "device=cpu"]
    # the run would write over the weights it starts from
    assert train_detector.main([*base, f"out={out}", f"init={out / 'best.pt'}"]) == 2
    assert train_detector.main([*base, f"out={tmp_path / 'x'}", f"init={tmp_path / 'no.pt'}"]) == 2
    with pytest.raises(BenchError, match="damoyolo_s"):
        train_detector.load_init(families.build("damoyolo_s"), "damoyolo_s", out / "best.pt")


def test_a_run_without_init_keeps_its_old_settings_key() -> None:
    """A run started before init= existed carries on: its saved settings have no init."""
    key = train_detector.Settings("yolox_s").key()
    assert "init" not in key
    assert train_detector.Settings("yolox_s", init="/x/best.pt").key()["init"] == "/x/best.pt"


def test_val_ap50_is_given_per_dataset_and_the_best_epoch_by_the_weakest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    batches = [(torch.zeros(2, 3, 8, 8), [truth, truth]), (torch.zeros(1, 3, 8, 8), [truth])]
    # tiles of set "a" are found, the one of set "b" is missed
    answers = iter([torch.stack([truth, truth]), torch.stack([truth + 50])])

    def predict(net: Any, model: str, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        boxes = next(answers)
        return boxes, torch.full(boxes.shape[:2], 0.9)

    monkeypatch.setattr(families, "predict", predict)
    per_set = train_detector.evaluate_val(
        torch.nn.Identity(),
        "yolox_s",
        batches,
        torch.device("cpu"),
        ["a", "a", "b"],  # type: ignore[arg-type]
    )
    assert per_set == {"a": pytest.approx(1.0), "b": 0.0}
    assert train_detector.val_score(per_set) == 0.0  # the weakest set, not the mean
    assert train_detector.val_score({}) == 0.0


# a big sea set, a middling land set and a small one, as SeaDronesSee drowned the others
SOURCES = ["seadronessee"] * 700 + ["heridal"] * 200 + ["kios"] * 100


def test_balance_sets_draws_as_many_tiles_from_every_set() -> None:
    order = train_detector.epoch_order(SOURCES, "sets", seed=0, epoch=0)
    assert len(order) == len(SOURCES)  # an epoch has as many steps as before
    counts = Counter(SOURCES[i] for i in order)
    assert sorted(counts.values()) == [333, 333, 334]
    # the small set is gone through whole, three times, before any tile a fourth time
    kios = Counter(i for i in order if SOURCES[i] == "kios")
    assert len(kios) == 100 and set(kios.values()) <= {3, 4}
    # the big one is drawn from, each tile at most once
    sea = Counter(i for i in order if SOURCES[i] == "seadronessee")
    assert len(sea) == 333 and set(sea.values()) == {1}


def test_balance_weights_share_out_the_epoch() -> None:
    weights = "seadronessee:1,heridal:2,kios:1"
    order = train_detector.epoch_order(SOURCES, weights, seed=0, epoch=0)
    counts = Counter(SOURCES[i] for i in order)
    assert counts == {"heridal": 500, "seadronessee": 250, "kios": 250}


@pytest.mark.parametrize("balance", ["sets", "tiles"])
def test_epoch_order_is_the_same_for_the_same_seed_and_epoch(balance: str) -> None:
    def order(seed: int, epoch: int) -> list[int]:
        return train_detector.epoch_order(SOURCES, balance, seed, epoch)

    assert order(3, 2) == order(3, 2)  # a resumed run gets the order it stopped in
    assert order(3, 2) != order(3, 3)
    assert order(3, 2) != order(4, 2)


def test_balance_tiles_keeps_the_old_order() -> None:
    for seed, epoch in [(0, 0), (0, 5), (7, 1)]:
        old = list(range(len(SOURCES)))
        random.Random(seed * 1000 + epoch).shuffle(old)  # noqa: S311 - as before balance=
        assert train_detector.epoch_order(SOURCES, "tiles", seed, epoch) == old


@pytest.mark.parametrize(
    "balance",
    ["even", "heridal:2", "heridal:2,kios:x", "heridal:0,kios:1", "heridal:1,kios:1,wisard:1"],
)
def test_bad_balance_is_refused(balance: str) -> None:
    with pytest.raises(BenchError):
        train_detector.parse_balance(balance, ["heridal", "kios"])


def test_bad_balance_or_pick_stops_before_training(bank: Path, tmp_path: Path) -> None:
    base = ["model=yolox_s", f"data={bank}", f"out={tmp_path / 'run'}", "sets=heridal,kios"]
    assert train_detector.main([*base, "balance=heridal:1"]) == 2
    assert train_detector.main([*base, "pick=seadronessee"]) == 2
    assert not (tmp_path / "run").exists()


def test_the_best_epoch_is_picked_on_the_measured_sets_by_default() -> None:
    four = ["heridal", "kios", "kios_carperson", "seadronessee"]
    assert train_detector.default_pick(four) == ("heridal", "kios")
    assert train_detector.default_pick(four[2:]) == tuple(four[2:])  # none measured: all
    # the last run: land fell while sea and parking rose
    per_set = {"heridal": 0.42, "kios": 0.31, "kios_carperson": 0.78, "seadronessee": 0.76}
    assert train_detector.val_score(per_set, ("heridal", "kios")) == 0.31
    assert train_detector.val_score({**per_set, "kios": 0.9}, ("heridal", "kios")) == 0.42
    assert train_detector.val_score(per_set, ("wisard",)) == 0.31  # none has val tiles: all


def test_balance_and_pick_are_part_of_the_settings_a_run_resumes_with() -> None:
    sets = ("heridal", "kios", "seadronessee")
    key = train_detector.Settings("yolox_s", sets=sets, pick=("heridal", "kios")).key()
    assert key["balance"] == "sets" and key["pick"] == ["heridal", "kios"]
    # a run from before balance= and pick= (every tile once, the weakest of all sets)
    old = train_detector.Settings("yolox_s", sets=sets, balance="tiles", pick=sets).key()
    assert "balance" not in old and "pick" not in old
    assert key != old


def test_the_manifest_says_which_sets_picked_the_epoch() -> None:
    by_dataset = {"heridal": 0.6, "seadronessee": 0.8}
    assert val_ap50_rule({}) == "pooled"
    assert val_ap50_rule({"val_ap50_by_dataset": by_dataset}) == "min_over_datasets"
    ck = {"val_ap50_by_dataset": by_dataset, "val_ap50_pick": ["heridal"]}
    assert val_ap50_rule(ck) == "min_over_pick"


# --- upscale: each tile covers tile / upscale picture pixels, enlarged to tile -------


def test_2x_tiles_cover_half_the_pixels_and_their_boxes_are_enlarged() -> None:
    boxes = [(10.0, 10.0, 30.0, 40.0), (100.0, 120.0, 112.0, 150.0)]
    plain = tiles.cut(300, 200, boxes, 128)
    big = tiles.cut(300, 200, boxes, 128, upscale=2)
    assert len(big) > len(plain)
    covered = np.zeros((200, 300), dtype=bool)
    for (x0, y0, x1, y1), kept in big:
        assert x1 - x0 == 64 and y1 - y0 == 64  # 128 / 2 picture pixels
        covered[y0:y1, x0:x1] = True
        for b in kept:
            assert 0 <= b[0] < b[2] <= 128 and 0 <= b[1] < b[3] <= 128
            # back in the picture, the box lies on a person
            back = (b[0] / 2 + x0, b[1] / 2 + y0, b[2] / 2 + x0, b[3] / 2 + y0)
            assert any(
                p[0] <= back[0] < back[2] <= p[2] and p[1] <= back[1] < back[3] <= p[3]
                for p in boxes
            )
    assert covered.all()  # every pixel up to the edges
    first = big[0]
    assert first[0] == (0, 0, 64, 64) and first[1] == [(20.0, 20.0, 60.0, 80.0)]


@pytest.mark.parametrize("upscale", [0, 3, 5])
def test_a_bad_upscale_cuts_nothing(upscale: int) -> None:
    with pytest.raises(ValueError, match="upscale"):
        tiles.cut(300, 200, [], 128, upscale=upscale)


def test_the_cache_key_is_unchanged_at_1x_and_differs_at_2x(bank: Path) -> None:
    items = tiles.pictures(bank, "train", ["heridal"])
    import hashlib

    old = hashlib.sha256(f"{tiles.CACHE_SCHEMA}|128|0.25".encode())  # before upscale=
    for im, _ in items:
        old.update(f"|{im.image_id}:{len(im.boxes)}".encode())
    assert tiles.cache_key(items, 128, 0.25) == old.hexdigest()[:16]
    assert tiles.cache_key(items, 128, 0.25, upscale=2) != old.hexdigest()[:16]


def test_the_2x_cache_holds_enlarged_tiles_with_the_people_under_their_boxes(
    bank: Path, tmp_path: Path
) -> None:
    items = tiles.pictures(bank, "train", ["heridal"], limit=1)
    cut = tiles.build_cache(
        items, tmp_path / "x2", TILE, empty_share=1.0, progress=False, upscale=2
    )
    plain = tiles.build_cache(items, tmp_path / "x1", TILE, empty_share=1.0, progress=False)
    assert len(cut) > len(plain)
    with_people = [t for t in cut if t.boxes]
    assert with_people
    for t in with_people:
        with PILImage.open(tmp_path / "x2" / t.file) as pic:
            rgb = np.asarray(pic.convert("RGB")).astype(int)
        assert rgb.shape[0] <= TILE and rgb.shape[1] <= TILE
        for x0, y0, x1, y1 in t.boxes:
            inner = rgb[int(y0) + 2 : int(y1) - 2, int(x0) + 2 : int(x1) - 2]
            assert inner.size and (inner[..., 0] - inner[..., 1]).mean() > 120  # red person
    # read back, the same tiles
    again = tiles.build_cache(
        items, tmp_path / "x2", TILE, empty_share=1.0, progress=False, upscale=2
    )
    assert again == cut


def test_upscale_is_part_of_the_settings_only_when_it_is_used() -> None:
    assert "upscale" not in train_detector.Settings("yolox_s").key()  # old runs carry on
    assert train_detector.Settings("yolox_s", upscale=2).key()["upscale"] == 2


@pytest.mark.parametrize("bad", ["upscale=3", "upscale=5", "upscale=0", "upscale=two"])
def test_a_bad_upscale_stops_before_training(bank: Path, tmp_path: Path, bad: str) -> None:
    argv = ["model=yolox_s", f"data={bank}", f"out={tmp_path / 'run'}", f"tile={TILE}", bad]
    assert train_detector.main(argv) == 2
    assert not (tmp_path / "run").exists()


def test_tile_over_upscale_may_not_be_under_32(bank: Path, tmp_path: Path) -> None:
    argv = ["model=yolox_s", f"data={bank}", f"out={tmp_path / 'run'}", "tile=64", "upscale=4"]
    assert train_detector.main(argv) == 2


@pytest.mark.parametrize("upscale", [1, 2])
def test_the_manifest_carries_the_runs_upscale(bank: Path, tmp_path: Path, upscale: int) -> None:
    best = tmp_path / "run" / "best.pt"
    best.parent.mkdir()
    _earlier_best(best)
    ck = torch.load(best, weights_only=True)
    ck["settings"] = train_detector.Settings("yolox_s", tile=TILE, upscale=upscale).key()
    torch.save(ck, best)
    _, manifest_path = export(best, tmp_path / "models", "person_x", check_images=1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["upscale"] == upscale and manifest["tile_px"] == TILE


def test_weights_from_before_upscale_export_as_1x(bank: Path, tmp_path: Path) -> None:
    best = tmp_path / "old" / "best.pt"
    best.parent.mkdir()
    _earlier_best(best)
    ck = torch.load(best, weights_only=True)
    ck["settings"] = {"sets": ["heridal"], "tile": TILE}  # no upscale key
    torch.save(ck, best)
    _, manifest_path = export(best, tmp_path / "models", "person_x", check_images=1)
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["upscale"] == 1


# --- yolox_m: the larger YOLOX --------------------------------------------------------


def test_yolox_m_is_megviis_yolox_m_and_reads_as_yolox_on_board() -> None:
    fam = families.FAMILIES["yolox_m"]
    assert (fam.family, fam.pad, fam.outputs) == ("yolox", 114, ("output",))
    assert families.PRETRAINED["yolox_m"] == "yolox_m.pth"
    coco = families.build("yolox_m", num_classes=80)
    # Megvii's README: YOLOX-m has 25.3 M parameters (YOLOX-s 9.0 M)
    assert sum(p.numel() for p in coco.parameters()) / 1e6 == pytest.approx(25.3, abs=0.1)
    small, big = families.build("yolox_s"), families.build("yolox_m")
    stem = "head.stems.0.conv.weight"  # 256 channels times the width: 0.50 and 0.75
    assert (small.state_dict()[stem].shape[0], big.state_dict()[stem].shape[0]) == (128, 192)
    assert len(big.state_dict()) > len(small.state_dict())  # deeper: depth 0.67 against 0.33


def test_yolox_m_exports_the_same_output_as_yolox_s(tmp_path: Path) -> None:
    """One tensor ``[1, n, 5 + 1]`` over the same cells: what sd_detect.onnx decodes."""
    import onnxruntime as ort

    shapes = {}
    for model in ("yolox_s", "yolox_m"):
        wrapper = families.ExportWrapper(families.build(model).eval(), model).eval()
        path = tmp_path / f"{model}.onnx"
        torch.onnx.export(
            wrapper, (torch.zeros(1, 3, TILE, TILE),), str(path), input_names=["images"],
            output_names=["output"], opset_version=17, dynamo=False,
        )  # fmt: skip
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (out,) = session.run(None, {"images": np.zeros((1, 3, TILE, TILE), np.float32)})
        shapes[model] = out.shape
    cells = sum((TILE // s) ** 2 for s in (8, 16, 32))
    assert shapes["yolox_m"] == shapes["yolox_s"] == (1, cells, 6)


def test_yolox_m_takes_batch_4_by_default(bank: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[train_detector.Settings] = []

    def fake_train(settings: train_detector.Settings, *a: Any) -> int:
        seen.append(settings)
        return 0

    monkeypatch.setattr(train_detector, "train", fake_train)
    base = [f"data={bank}", "device=cpu", f"out={bank / 'run'}"]
    assert train_detector.main(["model=yolox_m", *base]) == 0
    assert train_detector.main(["model=yolox_s", *base]) == 0
    assert train_detector.main(["model=yolox_m", "batch=2", *base]) == 0
    assert [s.batch for s in seen] == [4, 8, 2]
    assert train_detector.main(["model=yolox_m", "batch=many", *base]) == 2


def test_yolox_m_refuses_init_from_a_yolox_s_run(bank: Path, tmp_path: Path) -> None:
    init = tmp_path / "small" / "best.pt"
    init.parent.mkdir()
    _earlier_best(init, "yolox_s")
    with pytest.raises(BenchError, match=r"tränad som yolox_s, inte yolox_m.*COCO"):
        train_detector.load_init(families.build("yolox_m"), "yolox_m", init)
    argv = ["model=yolox_m", f"data={bank}", f"out={tmp_path / 'm'}", "epochs=1", "batch=2",
            f"tile={TILE}", "device=cpu", "workers=0", f"init={init}"]  # fmt: skip
    assert train_detector.main(argv) == 2
    assert not (tmp_path / "m" / "last.pt").exists()  # nothing was trained


def test_a_checkpoint_of_the_wrong_shape_is_refused_not_half_loaded(tmp_path: Path) -> None:
    init = tmp_path / "best.pt"
    _earlier_best(init, "yolox_s")
    ck = torch.load(init, weights_only=True)
    ck["model"] = "yolox_m"  # mislabelled: yolox_s weights under yolox_m's name
    torch.save(ck, init)
    with pytest.raises(BenchError, match="passar inte"):
        train_detector.load_init(families.build("yolox_m"), "yolox_m", init)


# --- scales: training pictures resampled to our cameras' person sizes -----------------


def test_scales_are_part_of_the_settings_only_when_they_are_used() -> None:
    key = train_detector.Settings("yolox_s").key()
    assert "scales" not in key and "max_up" not in key  # old runs carry on
    key = train_detector.Settings("yolox_s", scales=("native", "wide"), max_up=3.0).key()
    assert key["scales"] == ["native", "wide"] and key["max_up"] == 3.0


@pytest.mark.parametrize("bad", ["scales=zoom", "scales=0", "max_up=0.5", "max_up=nine"])
def test_a_bad_scale_stops_before_training(bank: Path, tmp_path: Path, bad: str) -> None:
    argv = ["model=yolox_s", f"data={bank}", f"out={tmp_path / 'run'}", f"tile={TILE}", bad]
    assert train_detector.main(argv) == 2
    assert not (tmp_path / "run").exists()


def test_a_scaled_run_resamples_the_train_part_only(bank: Path, tmp_path: Path) -> None:
    """The bank's 'heridal' is 50 px/m (the register): 120 px/m would enlarge it 2.4
    times, so it stops at max_up 2; the val part is cut as it is, like the test part."""
    out = tmp_path / "run"
    argv = ["model=yolox_s", f"data={bank}", f"out={out}", "epochs=1", "batch=2",
            f"tile={TILE}", "device=cpu", "workers=0", "scales=native,120"]  # fmt: skip
    assert train_detector.main(argv) == 0
    train_tiles = json.loads((out / "tiles" / "train" / "tiles.json").read_text())["tiles"]
    val_tiles = json.loads((out / "tiles" / "val" / "tiles.json").read_text())["tiles"]
    assert {t.get("scale", 1.0) for t in train_tiles} == {1.0, 2.0}
    assert all("scale" not in t for t in val_tiles)
    best = torch.load(out / "best.pt", weights_only=True)
    assert best["settings"]["scales"] == ["native", "120"]
    _, manifest_path = export(out / "best.pt", tmp_path / "models", "person_s", check_images=1)
    shown = json.loads(manifest_path.read_text(encoding="utf-8"))["settings"]
    assert shown["scales"] == ["native", "120"] and shown["max_up"] == 2.0
