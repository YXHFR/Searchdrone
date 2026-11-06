"""``modal_app.py`` against a fake ``modal`` module: it imports in the container (where
Modal places it at ``/root/modal_app.py``) and here, the image holds only the listed
files, and a call is made with the GPU, the timeout and the deadline the client chose.
No network, no account and no real Modal."""

from __future__ import annotations

import contextlib
import enum
import importlib.machinery
import importlib.util
import json
import sys
import threading
import time
import types
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

import cloud
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "modal_app.py"


class FakeImage:
    def __init__(self) -> None:
        self.files: list[tuple[str, str]] = []
        self.uv: dict[str, Any] = {}

    def uv_sync(self, path: str, **kw: Any) -> FakeImage:
        self.uv = {"path": path, **kw}
        return self

    def env(self, _env: dict[str, str]) -> FakeImage:
        return self

    def add_local_file(self, local: Any, remote: str) -> FakeImage:
        self.files.append((str(local), remote))
        return self


class NotFoundError(Exception):
    pass


class ModalConnectionError(Exception):  # as modal.exception.ConnectionError
    pass


class InputCancellation(BaseException):  # as modal.exception
    pass


class InputStatus(enum.IntEnum):  # as modal.types
    PENDING = 0
    SUCCESS = 1
    TERMINATED = 4


class FakeVolume:
    """Files on the volume, such as the marks in ``modal_app.CALLS`` that a started
    container leaves. ``log`` is shared with the function, to see the order."""

    def __init__(self, log: list[Any]) -> None:
        self.files: dict[str, bytes] = {}
        self.log = log

    def listdir(self, path: str) -> list[types.SimpleNamespace]:
        self.log.append(("listdir", path))
        inside = [f for f in self.files if f.startswith(path.rstrip("/") + "/")]
        if not inside:  # as Modal: a folder that is not there
            raise NotFoundError(path)
        return [types.SimpleNamespace(path=f.lstrip("/")) for f in inside]

    def read_file_into_fileobj(self, path: str, fh: Any) -> None:
        if path not in self.files:
            raise FileNotFoundError(path)
        fh.write(self.files[path])

    def remove_file(self, path: str) -> None:
        if path not in self.files:
            raise FileNotFoundError(path)
        del self.files[path]

    def commit(self) -> None:
        self.log.append(("commit", None))


class FakeCall:
    object_id = "fc-1"

    def __init__(self, log: list[Any], late: bool) -> None:
        self.log, self.late = log, late

    def get(self, timeout: float) -> dict[str, Any]:
        self.log.append(("get", timeout))
        if self.late:
            raise TimeoutError
        return {"status": "done", "code": 0, "seconds": 1.0}

    def cancel(self, terminate_containers: bool = False) -> None:
        self.log.append(("cancel", terminate_containers))


class FakeFunction:
    late = False

    def __init__(self, fn: Any, kw: dict[str, Any], log: list[Any]) -> None:
        self.fn, self.kw, self.log = fn, kw, log

    def with_options(self, **kw: Any) -> FakeFunction:
        self.log.append(("options", kw))
        return self

    def spawn(self, *args: Any) -> FakeCall:
        self.log.append(("spawn", args))
        return FakeCall(self.log, self.late)


def fake_modal(local: bool) -> tuple[types.ModuleType, list[Any]]:
    log: list[Any] = []
    mod = types.ModuleType("modal")
    images: list[FakeImage] = []

    class Image:
        @staticmethod
        def debian_slim(**_kw: Any) -> FakeImage:
            images.append(FakeImage())
            return images[-1]

    class App:
        def __init__(self, name: str) -> None:
            self.name = name

        def function(self, **kw: Any) -> Any:
            log.append(("function", kw))
            return lambda fn: FakeFunction(fn, kw, log)

        @contextlib.contextmanager
        def run(self) -> Iterator[None]:
            yield

    volume = FakeVolume(log)

    class Volume:
        @staticmethod
        def from_name(name: str, create_if_missing: bool = False) -> FakeVolume:
            return volume

    @contextlib.contextmanager
    def enable_output() -> Iterator[None]:
        yield

    mod.Image = Image  # type: ignore[attr-defined]
    mod.App = App  # type: ignore[attr-defined]
    mod.Volume = Volume  # type: ignore[attr-defined]
    mod.enable_output = enable_output  # type: ignore[attr-defined]
    mod.is_local = lambda: local  # type: ignore[attr-defined]
    mod.exception = types.SimpleNamespace(  # type: ignore[attr-defined]
        NotFoundError=NotFoundError,
        ConnectionError=ModalConnectionError,
        InputCancellation=InputCancellation,
    )
    mod.images = images  # type: ignore[attr-defined]
    graphs: dict[str, list[Any]] = {}  # Modal's call graph per call id
    results: dict[str, Any] = {}  # what get() gives per call id: none yet, or an error
    cancelled: list[str] = []

    class FunctionCall:
        def __init__(self, call_id: str) -> None:
            self.call_id = call_id

        @staticmethod
        def from_id(call_id: str) -> FunctionCall:
            return FunctionCall(call_id)

        def get_call_graph(self) -> list[Any]:
            if self.call_id not in graphs:
                raise NotFoundError(self.call_id)
            return graphs[self.call_id]

        def get(self, timeout: float) -> Any:
            assert timeout == 0  # never waits
            if self.call_id not in graphs:
                raise NotFoundError(self.call_id)
            if self.call_id not in results:
                raise TimeoutError
            if isinstance(results[self.call_id], BaseException):
                raise results[self.call_id]
            return results[self.call_id]

        def cancel(self, terminate_containers: bool = False) -> None:
            assert terminate_containers
            if self.call_id not in graphs:
                raise NotFoundError(self.call_id)
            cancelled.append(self.call_id)

    mod.FunctionCall = FunctionCall  # type: ignore[attr-defined]
    mod.graphs = graphs  # type: ignore[attr-defined]
    mod.results = results  # type: ignore[attr-defined]
    mod.cancelled = cancelled  # type: ignore[attr-defined]
    return mod, log


def load(monkeypatch: pytest.MonkeyPatch, local: bool, at: str) -> tuple[Any, Any, list[Any]]:
    """Import modal_app's source as if it lay at ``at``, with a fake modal."""
    mod, log = fake_modal(local)
    monkeypatch.setitem(sys.modules, "modal", mod)
    monkeypatch.setitem(sys.modules, "modal.types", types.SimpleNamespace(InputStatus=InputStatus))
    loader = importlib.machinery.SourceFileLoader("modal_app", str(SOURCE))
    spec = importlib.util.spec_from_file_location("modal_app", at, loader=loader)
    assert spec is not None
    app = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "modal_app", app)
    loader.exec_module(app)
    return app, mod, log


def test_it_imports_in_the_container_at_root(monkeypatch: pytest.MonkeyPatch) -> None:
    app, mod, log = load(monkeypatch, local=False, at="/root/modal_app.py")
    assert Path("/root") == app.HERE
    assert app._uv_version(Path("/root")) is None
    assert mod.images[0].files == []  # in the container the files are already there
    function = next(kw for kind, kw in log if kind == "function")
    assert "retries" not in function
    assert function["timeout"] == cloud.CALL_LIMIT_S
    # request and limit alike, so the price per hour that cloud.cost prints holds
    assert function["cpu"] == (cloud.CPU_CORES, cloud.CPU_CORES)
    assert function["memory"] == (cloud.MEMORY_MIB, cloud.MEMORY_MIB)


def test_here_the_image_gets_only_the_listed_files(monkeypatch: pytest.MonkeyPatch) -> None:
    _app, mod, _log = load(monkeypatch, local=True, at=str(SOURCE))
    image = mod.images[0]
    assert image.uv["uv_version"]  # the repository's uv, from simulation/versions.lock
    tools = SOURCE.parents[1]
    assert sorted(image.files) == sorted(
        (str(local), str(remote)) for local, remote in cloud.image_files(tools)
    )
    remote = [r for _, r in image.files]
    assert f"{cloud.IN_CONTAINER}/tools/train/train_detector.py" in remote
    for _, r in image.files:
        name = r.rsplit("/", 1)[-1]
        assert not name.startswith(".env") and ".local." not in name
        assert not name.endswith((".pt", ".pth", ".onnx"))


def _mark(app: Any, token: str, seconds: float, ended: bool) -> None:
    body = json.dumps({"seconds": seconds, "ended": ended}).encode()
    app.volume.files[f"/{app.CALLS}/{token}.json"] = body


def _wall(monkeypatch: pytest.MonkeyPatch, app: Any, *now: float) -> None:
    """modal_app's wall clock reads ``now`` in turn, then stays at the last."""
    left = list(now)

    def wall() -> float:
        return left.pop(0) if len(left) > 1 else left[0]

    clock = types.SimpleNamespace(time=wall, monotonic=time.monotonic, sleep=time.sleep)
    monkeypatch.setattr(app, "time", clock)


def _run(app: Any, token: str = "t" * 32) -> tuple[dict[str, Any], list[str]]:
    ids: list[str] = []
    res = app.ModalBackend("H100").run(["a=1"], ["b=2"], 10.0, 60, token, ids.append)
    return res, ids


def test_a_call_uses_the_gpu_timeout_and_budget_given(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _mod, log = load(monkeypatch, local=True, at=str(SOURCE))
    ids: list[str] = []
    res = app.ModalBackend("H100").run(["a=1"], ["b=2"], 1234.5, 3600, "t" * 32, ids.append)
    assert res["status"] == "done" and ids == ["", "fc-1"]  # being sent, then its id
    assert ("options", {"gpu": "H100", "timeout": 3600}) in log
    assert ("spawn", (["a=1"], ["b=2"], 1234.5, "t" * 32)) in log
    assert ("get", app.WAIT_STEP_S) in log  # in short waits, against the wall clock
    assert not any(kind == "cancel" for kind, _ in log)


@pytest.mark.parametrize(("started", "status"), [(True, "stopped"), (False, "no_gpu")])
def test_a_call_that_runs_over_is_cancelled(
    monkeypatch: pytest.MonkeyPatch, started: bool, status: str
) -> None:
    app, _mod, log = load(monkeypatch, local=True, at=str(SOURCE))
    monkeypatch.setattr(app.train_on_gpu, "late", True)
    _wall(monkeypatch, app, 0.0, 30.0, 60.0, 90.0, 60 + cloud.START_SLACK_S)
    if started:  # the container left its mark when it ran
        _mark(app, "t" * 32, 30.0, ended=False)
    _mark(app, "o" * 32, 30.0, ended=False)  # another call's mark
    res, _ids = _run(app)
    assert res == {"status": status, "code": None}  # no "seconds": not returned
    kinds = [kind for kind, _ in log]
    assert kinds.index("listdir") < kinds.index("cancel")  # looked before cancelling
    assert ("listdir", f"/{app.CALLS}") in log  # the folder, not the file
    assert len(app.volume.files) == 1 + started  # the client removes marks later, not here
    assert [kind for kind in kinds if kind == "get"] == ["get"] * 3  # waited, in steps


def test_a_computer_that_slept_past_the_deadline_cancels_when_it_wakes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The monotonic clock stops while a laptop sleeps; the wall clock does not."""
    app, _mod, log = load(monkeypatch, local=True, at=str(SOURCE))
    monkeypatch.setattr(app.train_on_gpu, "late", True)
    _wall(monkeypatch, app, 0.0, 10.0, 8 * 3600.0)  # one wait, then a night's sleep
    _mark(app, "t" * 32, 30.0, ended=False)
    res, _ids = _run(app)
    assert res == {"status": "stopped", "code": None}
    kinds = [kind for kind, _ in log]
    assert kinds.count("get") == 1 and "cancel" in kinds


def test_an_old_call_is_cancelled_with_its_containers(monkeypatch: pytest.MonkeyPatch) -> None:
    app, mod, _log = load(monkeypatch, local=True, at=str(SOURCE))
    backend = app.ModalBackend("H100")
    mod.graphs["fc-1"] = [types.SimpleNamespace(status=InputStatus.PENDING)]
    backend.cancel("fc-1")
    backend.cancel("fc-gone")  # Modal no longer knows it: no error
    assert mod.cancelled == ["fc-1"]


def test_what_is_known_of_an_earlier_call(monkeypatch: pytest.MonkeyPatch) -> None:
    app, mod, _log = load(monkeypatch, local=True, at=str(SOURCE))
    backend = app.ModalBackend("H100")
    token = "t" * 32
    start_up, unseen = app.START_UP_S, app.UNSEEN_S
    # Modal cannot say (no id yet, or it no longer knows the call): may still run
    assert backend.call_state(token, "") == {"running": True, "seconds": start_up}
    assert backend.call_state(token, "fc-gone") == {"running": True, "seconds": start_up}
    mod.graphs["fc-1"] = [types.SimpleNamespace(status=InputStatus.PENDING)]
    assert backend.call_state(token, "fc-1")["running"]
    # ended, no function ran: only the container's start
    mod.graphs["fc-1"] = [types.SimpleNamespace(status=InputStatus.TERMINATED)]
    assert backend.call_state(token, "fc-1") == {"running": False, "seconds": start_up}
    # ended without its last word: its mark's seconds and what it may have run since
    _mark(app, token, 1000.0, ended=False)
    assert backend.call_state(token, "fc-1") == {"running": False, "seconds": 1000.0 + unseen}
    # its own last word, and its start-up
    _mark(app, token, 1234.5, ended=True)
    assert backend.call_state(token, "fc-1") == {"running": False, "seconds": 1234.5 + start_up}
    # an ended mark is not enough while Modal says the call runs (moved after a preemption)
    mod.graphs["fc-1"] = [types.SimpleNamespace(status=InputStatus.PENDING)]
    assert backend.call_state(token, "fc-1") == {"running": True, "seconds": 1234.5 + start_up}
    assert backend.call_state(token, "")["running"]  # nor without Modal
    backend.forget(token)
    backend.forget(token)  # already gone: no error
    assert app.volume.files == {}


@pytest.mark.parametrize(
    ("result", "ended"),
    [
        ({"status": "done"}, True),  # a result
        (InputCancellation(), True),  # cancelled
        (RuntimeError("failed"), True),  # the function failed, or its output expired
        (ModalConnectionError(), False),  # cannot say
        (NotFoundError(), False),
    ],
)
def test_a_call_whose_graph_still_says_pending_is_asked_for_its_result(
    monkeypatch: pytest.MonkeyPatch, result: Any, ended: bool
) -> None:
    """The call graph is best effort: a call just cancelled may still show as pending."""
    app, mod, _log = load(monkeypatch, local=True, at=str(SOURCE))
    backend = app.ModalBackend("H100")
    mod.graphs["fc-1"] = [types.SimpleNamespace(status=InputStatus.PENDING)]
    mod.results["fc-1"] = result
    assert backend.call_state("t" * 32, "fc-1")["running"] is not ended


def test_the_container_keeps_its_mark_with_its_seconds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app, _mod, _log = load(monkeypatch, local=False, at="/root/modal_app.py")
    monkeypatch.setattr(app, "REMOTE_DATA", PurePosixPath(tmp_path))
    monkeypatch.setattr(app, "COMMIT_EVERY_S", 0.05)
    mark = tmp_path / str(app.CALLS) / f"{'t' * 32}.json"
    seen: list[dict[str, Any]] = []

    def commit() -> None:  # what the volume holds at each commit
        seen.append(json.loads(mark.read_text()))

    monkeypatch.setattr(app.volume, "commit", commit)
    ran: list[Any] = []

    def call(train: Any, export: Any, budget_s: float, scripts: Any) -> dict[str, Any]:
        ran.append((train, export, budget_s, scripts))
        assert [m["ended"] for m in seen] == [False]  # written and committed first
        time.sleep(0.3)  # a few commits on the way, each with the mark brought up to date
        return {"status": "done", "code": 0}

    monkeypatch.setattr(app, "call", call)
    res = app.train_on_gpu.fn(["a=1"], ["b=2"], 99.0, "t" * 32)
    assert ran == [(["a=1"], ["b=2"], 99.0, cloud.IN_CONTAINER / "tools" / "train")]
    assert res["status"] == "done" and res["seconds"] > 0
    assert len(seen) >= 3
    assert all(not m["ended"] for m in seen[:-1]) and seen[-1]["ended"]
    assert [m["seconds"] for m in seen] == sorted(m["seconds"] for m in seen)
    assert seen[-1]["seconds"] >= 0.3
    assert seen[-1] == json.loads(mark.read_text())  # kept: only the client removes it
    assert not list(mark.parent.glob(".*.tmp"))


def test_a_container_moved_after_a_preemption_counts_the_earlier_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app, _mod, _log = load(monkeypatch, local=False, at="/root/modal_app.py")
    monkeypatch.setattr(app, "REMOTE_DATA", PurePosixPath(tmp_path))
    monkeypatch.setattr(app.volume, "commit", lambda: None)
    mark = tmp_path / str(app.CALLS) / f"{'t' * 32}.json"
    mark.parent.mkdir(parents=True)
    mark.write_text(json.dumps({"seconds": 1000.0, "ended": False}))  # the first attempt's
    monkeypatch.setattr(app, "call", lambda *a: {"status": "done", "code": 0})
    res = app.train_on_gpu.fn(["a=1"], ["b=2"], 99.0, "t" * 32)
    before = 1000.0 + app.UNSEEN_S  # its mark and what it may have run after it
    assert before <= res["seconds"] < before + 5
    assert json.loads(mark.read_text()) == {"seconds": round(res["seconds"], 1), "ended": True}


def test_two_runs_of_one_call_add_up_their_seconds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app, _mod, _log = load(monkeypatch, local=False, at="/root/modal_app.py")
    monkeypatch.setattr(app, "REMOTE_DATA", PurePosixPath(tmp_path))
    monkeypatch.setattr(app.volume, "commit", lambda: None)
    mark = tmp_path / str(app.CALLS) / f"{'t' * 32}.json"

    def call(*_a: Any) -> dict[str, Any]:
        time.sleep(0.1)
        return {"status": "done", "code": 0}

    monkeypatch.setattr(app, "call", call)
    first = app.train_on_gpu.fn(["a=1"], ["b=2"], 99.0, "t" * 32)["seconds"]
    second = app.train_on_gpu.fn(["a=1"], ["b=2"], 99.0, "t" * 32)["seconds"]
    # the first ended: its seconds and the second's start-up come first, never 0 again
    assert second >= round(first, 1) + app.START_UP_S + 0.1
    assert json.loads(mark.read_text()) == {"seconds": round(second, 1), "ended": True}


def _saving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, commit: Any, train_s: float
) -> tuple[Any, list[tuple[str, float]]]:
    """A container whose saves go through ``commit``, with short times, that trains for
    ``train_s``; returns the module and when (and why) it gave up."""
    app, _mod, _log = load(monkeypatch, local=False, at="/root/modal_app.py")
    monkeypatch.setattr(app, "REMOTE_DATA", PurePosixPath(tmp_path))
    monkeypatch.setattr(app, "COMMIT_EVERY_S", 0.1)
    monkeypatch.setattr(app, "SAVE_LIMIT_S", 0.25)  # one failed save is tried again
    monkeypatch.setattr(app, "WATCH_S", 0.01)
    monkeypatch.setattr(app.volume, "commit", commit)
    gave_up: list[tuple[str, float]] = []
    monkeypatch.setattr(app, "give_up", lambda why: gave_up.append((why, time.monotonic())))
    monkeypatch.setattr(app, "call", lambda *_a: time.sleep(train_s) or {"status": "done"})
    return app, gave_up


def test_one_save_that_fails_is_tried_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tries: list[float] = []

    def commit() -> None:
        tries.append(time.monotonic())
        if len(tries) == 2:  # the first save in the keeper fails, the next goes through
            raise OSError("volume")

    app, gave_up = _saving(monkeypatch, tmp_path, commit, train_s=0.45)
    res = app.train_on_gpu.fn(["a=1"], ["b=2"], 99.0, "t" * 32)
    assert gave_up == [] and res["status"] == "done"
    assert len(tries) >= 4  # the first, the failed one, the one after it, the last


@pytest.mark.parametrize("how", ["fails", "hangs"])
def test_a_container_that_cannot_save_stops_itself(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, how: str
) -> None:
    """Twice in a row, or a save that never returns: it stops once ``SAVE_LIMIT_S`` has
    passed since the last mark saved was written, within one ``WATCH_S``, which is what
    ``UNSEEN_S`` counts beyond that mark."""
    written: list[float] = []
    release = threading.Event()

    def commit() -> None:
        if written:
            if how == "hangs":
                release.wait(0.8)
            raise OSError("volume")
        written.append(time.monotonic())

    app, gave_up = _saving(monkeypatch, tmp_path, commit, train_s=0.5)
    with contextlib.suppress(OSError):  # the last save fails too
        app.train_on_gpu.fn(["a=1"], ["b=2"], 99.0, "t" * 32)
    release.set()
    assert len(gave_up) == 1
    why, at = gave_up[0]
    assert ("OSError" if how == "fails" else "inte klar") in why
    after = at - written[0]  # the mark was written just before this commit
    assert app.SAVE_LIMIT_S - 0.005 < after <= app.SAVE_LIMIT_S + app.WATCH_S + 0.03


def test_one_retry_fits_in_what_is_counted_after_a_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _mod, _log = load(monkeypatch, local=False, at="/root/modal_app.py")
    # the retry comes two commits after the last mark saved; a third would not be waited for
    assert 2 * app.COMMIT_EVERY_S < app.SAVE_LIMIT_S < 3 * app.COMMIT_EVERY_S
    assert app.UNSEEN_S == app.START_UP_S + app.SAVE_LIMIT_S + app.WATCH_S


def test_give_up_ends_the_container(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _mod, _log = load(monkeypatch, local=False, at="/root/modal_app.py")
    codes: list[int] = []
    monkeypatch.setattr(app.os, "_exit", codes.append)
    app.give_up("x")
    assert codes == [3]


def test_removing_old_files_is_safe_to_repeat(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _mod, _log = load(monkeypatch, local=True, at=str(SOURCE))
    app.volume.files["/train/x/last.pt"] = b"old"
    backend = app.ModalBackend("H100")
    paths = [PurePosixPath("train/x/last.pt"), PurePosixPath("train/x/best.pt")]
    backend.remove(paths)
    backend.remove(paths)  # already gone: no error
    assert app.volume.files == {}
