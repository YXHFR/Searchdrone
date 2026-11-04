"""``modal_app.py`` against a fake ``modal`` module: it imports in the container (where
Modal places it at ``/root/modal_app.py``) and here, the image holds only the listed
files, and a call is made with the GPU, the timeout and the deadline the client chose.
No network, no account and no real Modal."""

from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import sys
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


class FakeVolume:
    """Files on the volume, such as the marks in ``modal_app.CALLS`` that a started
    container leaves. ``log`` is shared with the function, to see the order."""

    def __init__(self, log: list[Any]) -> None:
        self.files: set[str] = set()
        self.log = log

    def listdir(self, path: str) -> list[types.SimpleNamespace]:
        self.log.append(("listdir", path))
        inside = [f for f in self.files if f.startswith(path.rstrip("/") + "/")]
        if not inside:  # as Modal: a folder that is not there
            raise NotFoundError(path)
        return [types.SimpleNamespace(path=f.lstrip("/")) for f in inside]

    def remove_file(self, path: str) -> None:
        if path not in self.files:
            raise FileNotFoundError(path)
        self.files.discard(path)

    def commit(self) -> None:
        pass


class FakeCall:
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
    mod.exception = types.SimpleNamespace(NotFoundError=NotFoundError)  # type: ignore[attr-defined]
    mod.images = images  # type: ignore[attr-defined]
    return mod, log


def load(monkeypatch: pytest.MonkeyPatch, local: bool, at: str) -> tuple[Any, Any, list[Any]]:
    """Import modal_app's source as if it lay at ``at``, with a fake modal."""
    mod, log = fake_modal(local)
    monkeypatch.setitem(sys.modules, "modal", mod)
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


def test_a_call_uses_the_gpu_timeout_and_budget_given(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _mod, log = load(monkeypatch, local=True, at=str(SOURCE))
    res = app.ModalBackend("H100").run(["a=1"], ["b=2"], 1234.5, 3600)
    assert res["status"] == "done"
    assert ("options", {"gpu": "H100", "timeout": 3600}) in log
    spawn = next(args for kind, args in log if kind == "spawn")
    assert spawn[:3] == (["a=1"], ["b=2"], 1234.5) and len(spawn[3]) == 32  # its token
    assert ("get", 3600 + cloud.START_SLACK_S) in log
    assert not any(kind == "cancel" for kind, _ in log)


@pytest.mark.parametrize(("started", "status"), [(True, "stopped"), (False, "no_gpu")])
def test_a_call_that_runs_over_is_cancelled(
    monkeypatch: pytest.MonkeyPatch, started: bool, status: str
) -> None:
    app, _mod, log = load(monkeypatch, local=True, at=str(SOURCE))
    monkeypatch.setattr(app.train_on_gpu, "late", True)
    if started:  # the container left its mark when it ran
        monkeypatch.setattr(app.uuid, "uuid4", lambda: types.SimpleNamespace(hex="t" * 32))
        app.volume.files.add(f"/{app.CALLS}/{'t' * 32}")
    app.volume.files.add(f"/{app.CALLS}/{'o' * 32}")  # another call's mark
    res = app.ModalBackend("H100").run(["a=1"], ["b=2"], 10.0, 60)
    assert res == {"status": status, "code": None}  # no "seconds": not known here
    kinds = [kind for kind, _ in log]
    assert kinds.index("listdir") < kinds.index("cancel")  # looked before cancelling
    assert ("listdir", f"/{app.CALLS}") in log  # the folder, not the file
    assert app.volume.files == {f"/{app.CALLS}/{'o' * 32}"}  # only its own mark removed


def test_removing_old_files_is_safe_to_repeat(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _mod, _log = load(monkeypatch, local=True, at=str(SOURCE))
    app.volume.files.add("/train/x/last.pt")
    backend = app.ModalBackend("H100")
    paths = [PurePosixPath("train/x/last.pt"), PurePosixPath("train/x/best.pt")]
    backend.remove(paths)
    backend.remove(paths)  # already gone: no error
    assert app.volume.files == set()
