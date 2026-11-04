"""``just train-detector-cloud`` (``cloud.py``) without Modal, the network or an account:
what is uploaded (and what never is), the incremental upload, the cost, the paths, and
the whole command against a fake Modal."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import cloud
import pytest
import tiles
from bench.common import BenchError
from bench.datasets import Dataset, Image, load_register, write_index
from bench.splits import Split, ids_digest
from PIL import Image as PILImage

HERE = Path(__file__).resolve().parent
SEQS = {"AAA": "train", "BBB": "val", "CCC": "test"}


@pytest.fixture
def bank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fetched 'heridal' of two pictures in each part, a fetched 'wisard' (measure
    only) and the COCO weights' file."""
    data = tmp_path / "data"
    register = load_register()
    by_seq: dict[tuple[str, str], list[str]] = {}
    for name in ("heridal", "wisard"):
        raw = data / "datasets" / name / "raw"
        (raw / "sub").mkdir(parents=True)
        images = []
        for i, seq in enumerate([s for s in SEQS for _ in range(2)]):
            file = f"sub/{seq}_{i}.jpg"
            PILImage.new("RGB", (64, 48), (i * 20, 90, 90)).save(raw / file)
            images.append(Image(f"{name}:{seq}_{i}", name, seq, file, 64, 48, ((1, 1, 9, 9),)))
            by_seq.setdefault((name, seq), []).append(f"{seq}_{i}")
        write_index(data, register[name], images, "test")
    split = Split(
        "v1",
        {(d, s): SEQS[s] for d, s in by_seq},  # type: ignore[misc]
        {k: len(v) for k, v in by_seq.items()},
        {k: ids_digest(v) for k, v in by_seq.items()},
    )
    monkeypatch.setattr(tiles, "load_split", lambda *a, **k: split)
    pre = data / "models" / "pretrained"
    pre.mkdir(parents=True)
    (pre / cloud.pretrained_files()["yolox_s"]).write_bytes(b"coco weights")
    return data


class FakeModal:
    """``cloud.Backend`` with a dict for a volume. ``runs`` are the answers of the GPU
    calls in turn; a ``done`` one writes the results as the export would."""

    def __init__(self, runs: list[str] | None = None) -> None:
        self.volume: dict[str, bytes] = {}
        self.sent: list[list[str]] = []
        self.calls: list[tuple[list[str], list[str], float]] = []
        self.runs = runs or ["done"]

    def remote_sizes(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.volume.items()}

    def upload(self, uploads: Any) -> None:
        self.sent.append([u.remote.as_posix() for u in uploads])
        for u in uploads:
            self.volume[u.remote.as_posix()] = u.local.read_bytes()

    def run(self, train: Any, export: Any, budget_s: float) -> dict[str, Any]:
        self.calls.append((list(train), list(export), budget_s))
        status = self.runs[len(self.calls) - 1]
        if status == "done":
            args = dict(a.split("=", 1) for a in export)
            run = PurePosixPath(args["train"]).relative_to(cloud.REMOTE_DATA)
            models = PurePosixPath(args["out"]).relative_to(cloud.REMOTE_DATA)
            onnx = b"onnx bytes"
            self.volume[f"{run}/best.pt"] = b"best"
            self.volume[f"{run}/settings.json"] = b"{}"
            self.volume[f"{models}/{args['name']}.onnx"] = onnx
            manifest = {"onnx": f"{args['name']}.onnx", "sha256": hashlib.sha256(onnx).hexdigest()}
            self.volume[f"{models}/{args['name']}.json"] = json.dumps(manifest).encode()
        return {"status": status, "code": 0, "seconds": 3600.0}

    def fetch(self, remote: PurePosixPath, local: Path) -> None:
        local.write_bytes(self.volume[remote.as_posix()])


def _ds(**kw: Any) -> Dataset:
    base = load_register()["heridal"]
    return dataclasses.replace(base, **kw)


# --- what may leave this computer ---------------------------------------------------


def test_only_confirmed_open_train_sets_that_are_not_local_may_be_uploaded() -> None:
    register = load_register()
    allowed = {n for n, ds in register.items() if cloud.refusal(ds) is None}
    assert {"heridal", "kios"} <= allowed
    for name in ("wisard", "simfarm", "simstills"):  # measure only; kept on this computer
        assert name not in allowed
    for ds in allowed:
        d = register[ds]
        assert "train" in d.use and d.licence_confirmed and not d.local_only and not d.no_grant


@pytest.mark.parametrize(
    "change",
    [
        {"licence": "GPL-3.0-only", "use": ("measure",), "local_only": True},
        {"licence": "NONE", "use": ("measure",), "local_only": True},
        {"licence": "CC-BY-NC-4.0", "use": ("measure",), "local_only": True},
        {"use": ("measure",)},
        {"licence_confirmed": False},
        {"local_only": True},
        # a GPL set wrongly marked for training is still refused: the licence decides
        {"licence": "GPL-3.0-only"},
    ],
)
def test_measure_only_and_unlicensed_sets_are_refused(change: dict[str, Any]) -> None:
    assert cloud.refusal(_ds(**change)) is not None


def test_measure_only_sets_are_left_out_and_gpl_sets_stop_the_command(bank: Path) -> None:
    register = load_register()
    assert cloud.cloud_sets(["heridal", "kios", "wisard"], register, bank) == ["heridal"]
    gpl = {**register, "gplset": _ds(name="gplset", licence="GPL-3.0-only", use=("measure",))}
    with pytest.raises(BenchError, match="laddas aldrig upp"):
        cloud.cloud_sets(["heridal", "gplset"], gpl, bank)
    with pytest.raises(BenchError, match="laddas aldrig upp"):
        cloud.cloud_sets(["heridal", "simstills"], register, bank)


@pytest.mark.parametrize(
    "path",
    [
        "datasets/wisard/raw/a.jpg",
        "datasets/wisard/index.json",
        "datasets/simfarm/raw/a.png",
        "datasets/nosuchset/raw/a.jpg",
        "datasets/heridal/other.txt",
        "datasets/heridal/raw",
        "datasets/heridal/raw/../../wisard/raw/a.jpg",
        "/datasets/heridal/raw/a.jpg",
        "models/pretrained/other.pth",
        "build/ortho.npy",
        "secrets.toml",
    ],
)
def test_the_guard_stops_every_other_path(path: str) -> None:
    with pytest.raises(BenchError, match="Uppladdningen stoppad"):
        cloud.guard(PurePosixPath(path), load_register())


def test_the_guard_lets_the_training_files_through() -> None:
    register = load_register()
    for path in (
        "datasets/heridal/index.json",
        "datasets/heridal/raw/sub/a.jpg",
        "models/pretrained/yolox_s.pth",
        "init/0123456789abcdef/best.pt",
    ):
        cloud.guard(PurePosixPath(path), register)


def test_a_measure_only_upload_never_reaches_modal(bank: Path, tmp_path: Path) -> None:
    fake = FakeModal()
    sneaked = cloud.Upload(
        bank / "datasets" / "wisard" / "raw" / "sub" / "AAA_0.jpg",
        PurePosixPath("datasets/wisard/raw/sub/AAA_0.jpg"),
    )
    ok = cloud.Upload(
        bank / "datasets" / "heridal" / "index.json", PurePosixPath("datasets/heridal/index.json")
    )
    with pytest.raises(BenchError, match="wisard"):
        cloud.upload(fake, [ok, sneaked], load_register(), tmp_path / "record.json")
    assert fake.sent == [] and fake.volume == {}


def test_an_orthophoto_is_never_uploaded(bank: Path, tmp_path: Path) -> None:
    photo = bank / "datasets" / "heridal" / "raw" / "sub" / "ortho_0.jp2"
    photo.write_bytes(b"\0" * 16)
    fake = FakeModal()
    up = cloud.Upload(photo, PurePosixPath("datasets/heridal/raw/sub/ortho_0.jp2"))
    with pytest.raises(BenchError, match="ortofoto"):
        cloud.upload(fake, [up], load_register(), tmp_path / "record.json")
    assert fake.sent == []


# --- what is uploaded -----------------------------------------------------------------


def test_the_plan_holds_the_train_and_val_pictures_never_the_test_part(bank: Path) -> None:
    init = bank / "old" / "best.pt"
    init.parent.mkdir()
    init.write_bytes(b"old")
    sha = hashlib.sha256(b"old").hexdigest()
    plan = cloud.plan(bank, ["heridal"], "yolox_s", init=init, init_sha256=sha)
    remote = sorted(u.remote.as_posix() for u in plan)
    assert remote == sorted(
        [
            "datasets/heridal/index.json",
            "datasets/heridal/raw/sub/AAA_0.jpg",
            "datasets/heridal/raw/sub/AAA_1.jpg",
            "datasets/heridal/raw/sub/BBB_2.jpg",
            "datasets/heridal/raw/sub/BBB_3.jpg",
            "models/pretrained/yolox_s.pth",
            f"init/{sha[:16]}/best.pt",
        ]
    )
    assert not any("CCC" in r for r in remote)
    for u in plan:
        cloud.guard(u.remote, load_register())
    assert len(cloud.plan(bank, ["heridal"], "yolox_s", limit=1)) == 4  # index, 1+1, weights


def test_the_upload_sends_only_new_or_changed_files(bank: Path, tmp_path: Path) -> None:
    fake = FakeModal()
    record = tmp_path / "record.json"
    plan = cloud.plan(bank, ["heridal"], "yolox_s")
    assert cloud.upload(fake, plan, load_register(), record) == len(plan)
    assert cloud.upload(fake, plan, load_register(), record) == 0
    changed = bank / "datasets" / "heridal" / "raw" / "sub" / "AAA_0.jpg"
    os.utime(changed, ns=(1, 1))
    assert cloud.upload(fake, plan, load_register(), record) == 1
    assert fake.sent[-1] == ["datasets/heridal/raw/sub/AAA_0.jpg"]
    del fake.volume["datasets/heridal/index.json"]  # the volume lost it (or was emptied)
    assert cloud.upload(fake, plan, load_register(), record) == 1


def test_a_stopped_upload_keeps_what_was_sent(
    monkeypatch: pytest.MonkeyPatch, bank: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(cloud, "CHUNK_FILES", 2)
    record = tmp_path / "record.json"
    plan = cloud.plan(bank, ["heridal"], "yolox_s")

    class Breaks(FakeModal):
        def upload(self, uploads: Any) -> None:
            if self.sent:
                raise KeyboardInterrupt
            super().upload(uploads)

    broken = Breaks()
    with pytest.raises(KeyboardInterrupt):
        cloud.upload(broken, plan, load_register(), record)
    fake = FakeModal()
    fake.volume = dict(broken.volume)
    assert cloud.upload(fake, plan, load_register(), record) == len(plan) - 2


def test_chunks_split_by_count_and_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cloud, "CHUNK_FILES", 3)
    monkeypatch.setattr(cloud, "CHUNK_BYTES", 100)
    ups = [cloud.Upload(Path(f"/x/{i}"), PurePosixPath(f"r/{i}")) for i in range(5)]
    sizes = {u.remote: s for u, s in zip(ups, [10, 10, 10, 200, 10], strict=True)}
    assert [[u.remote.name for u in c] for c in cloud.chunks(ups, sizes)] == [
        ["0", "1", "2"],
        ["3"],
        ["4"],
    ]


# --- cost and paths -------------------------------------------------------------------


def test_the_cost_is_gpu_price_times_hours_with_its_source() -> None:
    c = cloud.cost("A100-80GB", 4.0, 24.0)
    assert c.usd == pytest.approx(10.0) and c.max_usd == pytest.approx(60.0)
    text = "\n".join(c.lines(guessed=True))
    assert "A100-80GB 2.50 $/h x 4.0 h = 10 $" in text
    assert "modal.com/pricing, läst 2026-11-04" in text
    assert cloud.cost("H100", 2.0, 1.0).usd == pytest.approx(3.95)  # capped at max_hours
    assert cloud.PRICES == {"A100-80GB": 2.50, "H100": 3.95, "L40S": 1.95}
    with pytest.raises(BenchError, match="gpu="):
        cloud.cost("T4", 1.0, 1.0)
    with pytest.raises(BenchError):
        cloud.cost("H100", -1.0, 1.0)


def test_the_guess_grows_with_the_work_and_shrinks_with_the_gpu() -> None:
    base = cloud.guess_hours("yolox_s", "A100-80GB", 20)
    assert base == pytest.approx(1.0)
    assert cloud.guess_hours("yolox_s", "A100-80GB", 40) == pytest.approx(2 * base)
    assert cloud.guess_hours("yolox_s", "A100-80GB", 20, scales=3) == pytest.approx(3 * base)
    assert cloud.guess_hours("yolox_s", "A100-80GB", 20, upscale=2) == pytest.approx(4 * base)
    assert (
        cloud.guess_hours("yolox_s", "H100", 20) < base < cloud.guess_hours("yolox_s", "L40S", 20)
    )


def test_results_come_back_where_the_local_training_puts_them(tmp_path: Path) -> None:
    where = cloud.places(tmp_path, "yolox_s")
    assert where.local_run == tmp_path / "train" / "yolox_s"
    assert where.remote_run == PurePosixPath("/data/train/yolox_s")
    assert [str(r) for r, _ in where.results()] == [
        "train/yolox_s/best.pt",
        "train/yolox_s/settings.json",
        "models/detector/person_yolox_s.onnx",
        "models/detector/person_yolox_s.json",
    ]
    assert where.results()[2][1] == tmp_path / "models" / "detector" / "person_yolox_s.onnx"
    mix = cloud.places(tmp_path, "yolox_s", str(tmp_path / "train" / "yolox_s_mix"), "person_mix")
    assert mix.remote_run == PurePosixPath("/data/train/yolox_s_mix")
    assert mix.local_run == tmp_path / "train" / "yolox_s_mix"
    with pytest.raises(BenchError):
        cloud.places(tmp_path, "yolox_s", name="../x")


def test_the_training_gets_its_own_keys_with_the_folders_moved(tmp_path: Path) -> None:
    where = cloud.places(tmp_path, "yolox_s")
    args = {"model": "yolox_s", "epochs": "5", "data": "/home/x", "gpu": "H100", "yes": "1"}
    argv = cloud.train_argv(args, ["heridal", "kios"], where, "/data/init/ab/best.pt")
    assert argv == [
        "model=yolox_s",
        "epochs=5",
        "data=/data",
        "out=/data/train/yolox_s",
        "sets=heridal,kios",
        "device=cuda",
        "init=/data/init/ab/best.pt",
    ]
    assert cloud.export_argv("yolox_s", where) == [
        "model=yolox_s",
        "data=/data",
        "train=/data/train/yolox_s",
        "name=person_yolox_s",
        "out=/data/models/detector",
    ]


def test_the_keys_and_weights_are_the_training_scripts_own() -> None:
    import families
    import train_detector

    assert cloud.train_keys() == train_detector.KEYS
    assert cloud.pretrained_files() == families.PRETRAINED
    assert set(cloud.BASE_HOURS) == set(families.FAMILIES)


def test_cloud_never_imports_modal_by_itself() -> None:
    env = {**os.environ, "PYTHONPATH": f"{HERE.parents[1]}{os.pathsep}{HERE.parent}"}
    code = "import sys, cloud; cloud.main(['model=x']); sys.exit('modal' in sys.modules)"
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, check=False)
    assert res.returncode == 0, res.stderr


# --- the whole command against a fake Modal --------------------------------------------


def test_the_command_uploads_trains_until_done_and_brings_the_model_home(
    bank: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeModal(runs=["stopped", "done"])
    code = cloud.main(
        [f"data={bank}", "model=yolox_s", "gpu=H100", "hours=2", "yes=1", "epochs=3"],
        backend=lambda gpu: fake,
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "H100 3.95 $/h x 2.0 h = 8 $" in out
    assert "wisard laddas inte upp" in out
    sent = [p for batch in fake.sent for p in batch]
    assert not any("wisard" in p or "CCC" in p for p in sent)
    assert len(fake.calls) == 2
    train, _export, budget = fake.calls[0]
    assert "sets=heridal" in train and "epochs=3" in train and f"data={bank}" not in train
    assert not any(a.startswith(("gpu=", "yes=", "hours=")) for a in train)
    assert budget == cloud.CALL_LIMIT_S - cloud.STOP_MARGIN_S
    assert (bank / "train" / "yolox_s" / "best.pt").read_bytes() == b"best"
    manifest = bank / "models" / "detector" / "person_yolox_s.json"
    assert json.loads(manifest.read_text())["onnx"] == "person_yolox_s.onnx"
    assert (bank / "models" / "detector" / "person_yolox_s.onnx").is_file()


def test_nothing_starts_without_a_yes(bank: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(cloud, "confirm", lambda yes: False)
    code = cloud.main([f"data={bank}", "model=yolox_s"], backend=lambda gpu: called.append(gpu))  # type: ignore[arg-type,func-returns-value]
    assert code == 1 and called == []


def test_without_a_terminal_yes_1_is_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", open(os.devnull))  # noqa: SIM115, PTH123
    with pytest.raises(BenchError, match="yes=1"):
        cloud.confirm(False)
    assert cloud.confirm(True)
    assert cloud.confirm(False, ask=lambda _: "ja")
    assert not cloud.confirm(False, ask=lambda _: "")


@pytest.mark.parametrize("arg", ["device=cpu", "simfarm=1", "gpu=T4", "model=yolo", "sets=gplnope"])
def test_bad_keys_stop_before_anything_is_sent(bank: Path, arg: str) -> None:
    fake = FakeModal()
    args = [f"data={bank}", "model=yolox_s", "yes=1", arg]
    assert cloud.main(args, backend=lambda gpu: fake) == 2
    assert fake.sent == [] and fake.calls == []


def test_max_hours_stops_the_calls(bank: Path) -> None:
    fake = FakeModal(runs=["stopped", "stopped", "stopped"])
    code = cloud.main(
        [f"data={bank}", "model=yolox_s", "yes=1", "max_hours=2"], backend=lambda gpu: fake
    )
    assert code == 1
    assert len(fake.calls) == 2  # 1 h each (FakeModal), then less than the margin is left
    assert fake.calls[0][2] == 2 * 3600


def test_a_failed_run_says_so_and_fetches_nothing(bank: Path) -> None:
    fake = FakeModal(runs=["failed"])
    assert cloud.main([f"data={bank}", "model=yolox_s", "yes=1"], backend=lambda gpu: fake) == 1
    assert not (bank / "models" / "detector").exists()
