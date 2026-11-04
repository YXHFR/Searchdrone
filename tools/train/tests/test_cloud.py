"""``just train-detector-cloud`` (``cloud.py``) without Modal, the network or an account:
what is uploaded (and what never is), the incremental upload, the cost, the paths, and
the whole command against a fake Modal."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import os
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
        self.calls: list[tuple[list[str], list[str], float, int]] = []
        self.resumed: list[bool] = []  # per call: carried on from last.pt
        self.runs = runs or ["done"]
        self.manifest_onnx = ""  # what the manifest names ("": <name>.onnx)

    def remote_sizes(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.volume.items()}

    def upload(self, uploads: Any) -> None:
        self.sent.append([u.remote.as_posix() for u in uploads])
        for u in uploads:
            self.volume[u.remote.as_posix()] = u.local.read_bytes()

    def run(self, train: Any, export: Any, budget_s: float, timeout_s: int) -> dict[str, Any]:
        self.calls.append((list(train), list(export), budget_s, timeout_s))
        status = self.runs[len(self.calls) - 1]
        if status == "lost":
            raise ConnectionError("secret-looking text\nmore lines")
        targs = dict(a.split("=", 1) for a in train)
        if "out" in targs:
            last = f"{PurePosixPath(targs['out']).relative_to(cloud.REMOTE_DATA)}/last.pt"
            # as train_detector.py: fresh=1 starts over, otherwise last.pt is carried on from
            self.resumed.append(last in self.volume and targs.get("fresh") != "1")
            self.volume[last] = b"state"
        if status == "done":
            args = dict(a.split("=", 1) for a in export)
            run = PurePosixPath(args["train"]).relative_to(cloud.REMOTE_DATA)
            models = PurePosixPath(args["out"]).relative_to(cloud.REMOTE_DATA)
            onnx = b"onnx bytes"
            self.volume[f"{run}/best.pt"] = b"best"
            self.volume[f"{run}/settings.json"] = b"{}"
            self.volume[f"{models}/{args['name']}.onnx"] = onnx
            named = self.manifest_onnx or f"{args['name']}.onnx"
            manifest = {"onnx": named, "sha256": hashlib.sha256(onnx).hexdigest()}
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
        "init/0123456789abcdef/best.onnx",
        "init/0123456789abcdef/.env",
        "init/0123456789abcdef/sub/best.pt",
    ],
)
def test_the_guard_stops_every_other_path(path: str) -> None:
    with pytest.raises(BenchError, match="Uppladdningen stoppad"):
        cloud.guard(PurePosixPath(path), load_register(), 10)


def test_the_guard_stops_an_init_file_that_is_too_large() -> None:
    path = PurePosixPath("init/0123456789abcdef/best.pt")
    cloud.guard(path, load_register(), cloud.MAX_INIT_BYTES)
    with pytest.raises(BenchError, match="Uppladdningen stoppad"):
        cloud.guard(path, load_register(), cloud.MAX_INIT_BYTES + 1)


def test_the_guard_lets_the_training_files_through() -> None:
    register = load_register()
    for path in (
        "datasets/heridal/index.json",
        "datasets/heridal/raw/sub/a.jpg",
        "models/pretrained/yolox_s.pth",
        "init/0123456789abcdef/best.pt",
        "init/0123456789abcdef/old.pth",
    ):
        cloud.guard(PurePosixPath(path), register, 10)


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
        cloud.guard(u.remote, load_register(), u.local.stat().st_size)
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
    assert pytest.approx(0.63, abs=0.01) == cloud.EXTRA_PER_HOUR  # 8 cores, 32 GiB
    assert c.price == pytest.approx(2.50 + cloud.EXTRA_PER_HOUR)
    assert c.usd == pytest.approx(4 * c.price) and c.max_usd == pytest.approx(24 * c.price)
    text = "\n".join(c.lines(guessed=True))
    assert "A100-80GB 2.50 $ + processor och minne (8 kärnor, 32 GiB) 0.63 $ = 3.13 $/h" in text
    assert "3.13 $/h x 4.0 h = 13 $" in text and "Högst: 24 h x 3.13 $/h = 75 $" in text
    assert "modal.com/pricing, läst 2026-11-04" in text and "procent" not in text
    h100 = cloud.cost("H100", 2.0, 1.0)
    assert h100.usd == pytest.approx(3.95 + cloud.EXTRA_PER_HOUR)  # capped at max_hours
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
    run = where.remote_run.name
    assert where.local_run == tmp_path / "train" / "yolox_s"
    assert run == "yolox_s-" + hashlib.sha256(str(where.local_run).encode()).hexdigest()[:8]
    assert where.remote_run == PurePosixPath("/data/train") / run
    assert [str(r) for r, _ in where.results()] == [
        f"train/{run}/best.pt",
        f"train/{run}/settings.json",
        "models/detector/person_yolox_s.onnx",
        "models/detector/person_yolox_s.json",
    ]
    assert where.results()[2][1] == tmp_path / "models" / "detector" / "person_yolox_s.onnx"
    mix = cloud.places(tmp_path, "yolox_s", str(tmp_path / "train" / "yolox_s_mix"), "person_mix")
    assert mix.remote_run.name.startswith("yolox_s_mix-")
    assert mix.local_run == tmp_path / "train" / "yolox_s_mix"
    # two out= folders of the same name never share a run on the volume
    other = cloud.places(tmp_path, "yolox_s", str(tmp_path / "elsewhere" / "yolox_s_mix"))
    assert other.remote_run != mix.remote_run
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
        f"out={where.remote_run}",
        "sets=heridal,kios",
        "device=cuda",
        "init=/data/init/ab/best.pt",
    ]
    assert cloud.export_argv("yolox_s", where) == [
        "model=yolox_s",
        "data=/data",
        f"train={where.remote_run}",
        "name=person_yolox_s",
        "out=/data/models/detector",
    ]


def test_the_keys_and_weights_are_the_training_scripts_own() -> None:
    import families
    import train_detector

    assert cloud.train_keys() == train_detector.KEYS
    assert cloud.pretrained_files() == families.PRETRAINED
    assert set(cloud.BASE_HOURS) == set(families.FAMILIES)
    assert cloud.default_sets() == train_detector.Settings(model="yolox_s").sets


def test_cloud_never_imports_modal_by_itself(
    bank: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With ``modal`` made unimportable, the whole command still runs against a fake."""
    monkeypatch.setitem(sys.modules, "modal", None)  # import modal -> ImportError
    monkeypatch.delitem(sys.modules, "modal_app", raising=False)
    importlib.reload(cloud)
    fake = FakeModal()
    code = cloud.main([f"data={bank}", "model=yolox_s", "yes=1"], backend=lambda gpu: fake)
    assert code == 0, capsys.readouterr()
    assert "modal_app" not in sys.modules


def test_the_image_holds_exactly_the_listed_files() -> None:
    files = cloud.image_files(HERE.parents[1])
    remote = {r.as_posix() for _, r in files}
    root = f"{cloud.IN_CONTAINER}/tools"
    for needed in (
        "train/train_detector.py",
        "train/export_onnx.py",
        "train/gpu_call.py",
        "train/modal_app.py",
        "bench/common.py",
        "bench/datasets.toml",
        "bench/splits/v1.json",
        "train/vendor/yolox/yolox.py",
        "train/vendor/damoyolo/tinynas_L25_k1kx.txt",
    ):
        assert f"{root}/{needed}" in remote
    for _, r in files:
        assert not r.name.startswith(".") and ".local." not in r.name
        assert r.suffix not in (".pt", ".pth", ".onnx", ".npy", ".pyc")
    assert not any("tests" in r.parts or "__pycache__" in r.parts for _, r in files)


def test_the_image_list_refuses_secrets_and_missing_files(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    for _, remote in cloud.image_files(HERE.parents[1]):
        rel = remote.relative_to(cloud.IN_CONTAINER / "tools")
        (tools / rel).parent.mkdir(parents=True, exist_ok=True)
        (tools / rel).write_text("x")
    assert cloud.image_files(tools)
    (tools / "bench" / "splits" / "keys.local.json").write_text("{}")
    with pytest.raises(BenchError, match="får inte skickas"):
        cloud.image_files(tools)
    (tools / "bench" / "splits" / "keys.local.json").unlink()
    (tools / "train" / "tiles.py").unlink()
    with pytest.raises(BenchError, match="saknas"):
        cloud.image_files(tools)


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
    assert "4.58 $/h x 2.0 h = 9 $" in out
    assert "wisard laddas inte upp" in out
    sent = [p for batch in fake.sent for p in batch]
    assert not any("wisard" in p or "CCC" in p for p in sent)
    assert len(fake.calls) == 2
    train, _export, budget, timeout_s = fake.calls[0]
    assert "sets=heridal" in train and "epochs=3" in train and f"data={bank}" not in train
    assert not any(a.startswith(("gpu=", "yes=", "hours=", "fresh=")) for a in train)
    # a duration, not a time of day: the container measures it on its own clock. The
    # call (with its start, stop and last commit) stays under Modal's 24 hours.
    assert budget == timeout_s - cloud.STOP_GRACE_S - cloud.FINISH_SLACK_S
    assert timeout_s + cloud.START_SLACK_S == cloud.CALL_LIMIT_S - cloud.STOP_MARGIN_S
    assert fake.resumed == [False, True]
    run = bank / "train" / "yolox_s"
    assert (run / "best.pt").read_bytes() == b"best"
    assert json.loads((run / cloud.MARKER).read_text())["volume"] == cloud.VOLUME
    remote = cloud.places(bank, "yolox_s").remote_run.name
    assert cloud.hours_used(bank / cloud.LEDGER.format(run=remote)) == pytest.approx(2.0)
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


def test_max_hours_is_a_ceiling_for_the_run_over_all_commands(bank: Path) -> None:
    fake = FakeModal(runs=["stopped"] * 5)
    args = [f"data={bank}", "model=yolox_s", "yes=1", "max_hours=2"]
    assert cloud.main(args, backend=lambda gpu: fake) == 1
    assert len(fake.calls) == 2  # 1 h each (FakeModal), then too little is left
    # this side waits at most timeout + START_SLACK_S: never past max_hours
    assert fake.calls[0][3] + cloud.START_SLACK_S == 2 * 3600
    assert fake.calls[1][3] + cloud.START_SLACK_S == 3600
    assert cloud.main(args, backend=lambda gpu: fake) == 1  # the same command: nothing left
    assert len(fake.calls) == 2
    assert cloud.main([*args[:-1], "max_hours=3"], backend=lambda gpu: fake) == 1
    assert len(fake.calls) == 3


def test_every_second_is_counted_even_when_a_call_breaks(tmp_path: Path) -> None:
    ticks = iter([0.0, 5000.0, 5000.0, 9000.0])
    ledger = tmp_path / "hours.json"

    class Slow(FakeModal):
        def run(self, *a: Any) -> dict[str, Any]:
            super().run(*a)
            return {"status": "stopped", "code": None, "seconds": 100.0}  # less than wall

    with pytest.raises(ConnectionError):
        cloud.run_until_done(FakeModal(["lost"]), [], [], 24, ledger, clock=lambda: next(ticks))
    assert cloud.hours_used(ledger) * 3600 == pytest.approx(5000.0)
    with pytest.raises(BenchError, match="max_hours"):
        cloud.run_until_done(Slow(["stopped"]), [], [], 2.5, ledger, clock=lambda: next(ticks))
    assert cloud.hours_used(ledger) * 3600 == pytest.approx(9000.0)  # the client's 4000 s


def test_a_call_budget_is_never_more_than_what_is_left() -> None:
    for left in (0, 600, 3600, 10 * 3600, 30 * 3600):
        budget = cloud.call_budget(left)
        if budget:
            slack = cloud.STOP_GRACE_S + cloud.FINISH_SLACK_S + cloud.START_SLACK_S
            assert budget + slack <= left
            assert budget <= cloud.CALL_LIMIT_S - cloud.STOP_MARGIN_S
    assert cloud.call_budget(600) == 0


def test_a_lost_line_gives_a_plain_message_not_a_traceback(
    bank: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeModal(runs=["lost"])
    assert cloud.main([f"data={bank}", "model=yolox_s", "yes=1"], backend=lambda gpu: fake) == 1
    err = capsys.readouterr().err
    assert "(ConnectionError)" in err and "secret-looking" not in err
    assert "modal token info" in err and "Traceback" not in err

    def no_modal(gpu: str) -> cloud.Backend:
        raise ModuleNotFoundError("No module named 'modal'")

    assert cloud.main([f"data={bank}", "model=yolox_s", "yes=1"], backend=no_modal) == 2
    assert "Modal är inte installerat" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["best.onnx", "big.pt"])
def test_a_bad_init_file_is_refused_before_anything_is_sent(
    bank: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setattr(cloud, "MAX_INIT_BYTES", 10)
    init = bank / "old" / name
    init.parent.mkdir()
    init.write_bytes(b"x" * 11)
    fake = FakeModal()
    args = [f"data={bank}", "model=yolox_s", "yes=1", f"init={init}"]
    assert cloud.main(args, backend=lambda gpu: fake) == 2
    assert fake.sent == [] and fake.calls == []


@pytest.mark.parametrize("found", ["last.pt", "best.pt", "settings.json"])
def test_a_local_training_is_never_written_over(bank: Path, found: str) -> None:
    run = bank / "train" / "yolox_s"
    run.mkdir(parents=True)
    (run / found).write_bytes(b"local")
    fake = FakeModal()
    assert cloud.main([f"data={bank}", "model=yolox_s", "yes=1"], backend=lambda gpu: fake) == 2
    assert fake.sent == [] and fake.calls == []
    assert (run / found).read_bytes() == b"local"


def test_the_same_cloud_run_may_come_back_again_but_not_over_a_new_local_one(
    bank: Path,
) -> None:
    args = [f"data={bank}", "model=yolox_s", "yes=1"]
    assert cloud.main(args, backend=lambda gpu: FakeModal()) == 0
    assert cloud.main(args, backend=lambda gpu: FakeModal()) == 0  # its own folder
    run = bank / "train" / "yolox_s"
    (run / "last.pt").write_bytes(b"local")  # a local training started there meanwhile
    where = cloud.places(bank, "yolox_s")
    with pytest.raises(BenchError, match=r"last\.pt"):
        cloud.fetch_results(FakeModal(), where)
    (run / "last.pt").unlink()
    (run / cloud.MARKER).write_text('{"volume": "searchdrone-train", "remote_run": "other"}')
    assert cloud.main(args, backend=lambda gpu: FakeModal()) == 2


def test_a_failed_run_says_so_and_fetches_nothing(bank: Path) -> None:
    fake = FakeModal(runs=["failed"])
    assert cloud.main([f"data={bank}", "model=yolox_s", "yes=1"], backend=lambda gpu: fake) == 1
    assert not (bank / "models" / "detector").exists()


def test_fresh_goes_to_the_first_call_only_and_resets_the_hours(bank: Path) -> None:
    remote = cloud.places(bank, "yolox_s").remote_run.name
    ledger = bank / cloud.LEDGER.format(run=remote)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text('{"seconds": 36000}')  # 10 h of an earlier run in this folder
    fake = FakeModal(runs=["stopped", "stopped", "done"])
    fake.volume[f"train/{remote}/last.pt"] = b"old run"
    args = [f"data={bank}", "model=yolox_s", "yes=1", "fresh=1", "max_hours=12"]
    assert cloud.main(args, backend=lambda gpu: fake) == 0
    assert ["fresh=1" in train for train, *_ in fake.calls] == [True, False, False]
    assert fake.resumed == [False, True, True]  # starts over once, then carries on
    assert cloud.hours_used(ledger) == pytest.approx(3.0)  # counted from 0 again


def test_the_message_after_a_stop_says_to_leave_fresh_out(
    bank: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeModal(runs=["lost"])
    args = [f"data={bank}", "model=yolox_s", "yes=1", "fresh=1"]
    assert cloud.main(args, backend=lambda gpu: fake) == 1
    assert "samma kommando utan fresh=1" in capsys.readouterr().err


@pytest.mark.parametrize("found", ["person_yolox_s.onnx", "person_yolox_s.json"])
def test_a_model_exported_here_is_never_written_over(bank: Path, found: str) -> None:
    models = bank / "models" / "detector"
    models.mkdir(parents=True)
    (models / found).write_bytes(b"local export")
    fake = FakeModal()
    assert cloud.main([f"data={bank}", "model=yolox_s", "yes=1"], backend=lambda gpu: fake) == 2
    assert fake.sent == [] and fake.calls == []
    assert (models / found).read_bytes() == b"local export"
    args = [f"data={bank}", "model=yolox_s", "yes=1", "name=person_cloud"]
    assert cloud.main(args, backend=lambda gpu: fake) == 0  # another name is fine


def test_the_same_cloud_model_may_come_back_but_not_over_a_new_local_export(bank: Path) -> None:
    args = [f"data={bank}", "model=yolox_s", "yes=1"]
    assert cloud.main(args, backend=lambda gpu: FakeModal()) == 0
    assert cloud.main(args, backend=lambda gpu: FakeModal()) == 0  # its own ONNX file
    onnx = bank / "models" / "detector" / "person_yolox_s.onnx"
    onnx.write_bytes(b"exported here meanwhile")
    assert cloud.main(args, backend=lambda gpu: FakeModal()) == 2
    assert onnx.read_bytes() == b"exported here meanwhile"


@pytest.mark.parametrize("named", ["../person_yolox_s.onnx", "sub/person_yolox_s.onnx", "x.onnx"])
def test_the_manifest_must_name_the_onnx_file_itself(bank: Path, named: str) -> None:
    fake = FakeModal()
    fake.manifest_onnx = named
    assert cloud.main([f"data={bank}", "model=yolox_s", "yes=1"], backend=lambda gpu: fake) == 1
    assert not (bank / "models" / named).exists()
