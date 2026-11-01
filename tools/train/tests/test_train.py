"""The whole training chain on CPU with a few made-up pictures and one epoch: tiles,
COCO start, training, stop and resume, ONNX export. No real data, no download."""

from __future__ import annotations

import dataclasses
import json
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
from export_onnx import export
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

    onnx_path, manifest_path = export(out / "best.pt", tmp_path / "models", f"person_{model}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["family"] == families.FAMILIES[model].family
    assert manifest["input_px"] == TILE and manifest["classes"] == ["person"]
    assert manifest["onnx"] == onnx_path.name and len(manifest["sha256"]) == 64
    assert [t["dataset"] for t in manifest["trained_on"]] == ["heridal", "wisard"][:1]
    assert manifest["publish"] is True


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
