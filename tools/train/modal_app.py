"""The Modal side of ``just train-detector-cloud`` (``cloud.py`` decides what is sent,
``gpu_call.py`` what a call does).

The container image is ``tools/train``'s own locked environment (``pyproject.toml`` and
``uv.lock``, PyTorch for CUDA 12.6) plus exactly the files ``cloud.image_files`` lists
(the training scripts, the vendored model code and what of ``tools/bench`` they import),
so the GPU runs the same ``train_detector.py`` and ``export_onnx.py`` as this computer.
The volume ``cloud.VOLUME`` is mounted as the data folder (``/data``); the runs, their
checkpoints and tile caches live there, and it is committed every few minutes.

No retries are asked of Modal: a call that fails ends the command, and the same command
carries on from the last save. Modal may still move a call to another machine after a
preemption; the client waits at most the call's timeout plus ``cloud.START_SLACK_S``,
then cancels it, and counts all of that against ``max_hours``, so the ceiling holds.
Each call's container keeps a mark on the volume (``CALLS``): written as soon as it runs,
then with the seconds it has run at every commit, and ``ended`` at its end. When the
client stops waiting it looks for the mark before it cancels the call, so it can tell
"stopped" from "never got a GPU" (``no_gpu``). A later command reads the mark, and asks
Modal whether the call has ended, to put the hours of a call it lost (Ctrl+C, a lost
line) right (``cloud.settle_calls``). Only the client removes the mark.

The container asks for exactly ``cloud.CPU_CORES`` and ``cloud.MEMORY_MIB`` (the same
request and limit), so the price per hour that ``cloud.cost`` prints holds.

This module is imported both here and in the container, where Modal places it at
``/root/modal_app.py``: nothing at import time may depend on where it lies. Needs
``modal`` (the ``cloud`` dependency group) and the owner's token, which Modal reads from
``~/.modal.toml`` or the environment; nothing here reads or prints it. CI never imports
this file with the real Modal.
"""

from __future__ import annotations

import contextlib
import io
import json
import threading
import time
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import modal
from cloud import (
    CALL_LIMIT_S,
    CPU_CORES,
    DEFAULT_GPU,
    IN_CONTAINER,
    MEMORY_MIB,
    REMOTE_DATA,
    START_SLACK_S,
    VOLUME,
    Upload,
    image_files,
)
from gpu_call import call

HERE = Path(__file__).resolve().parent
COMMIT_EVERY_S = 300
CALLS = PurePosixPath(".calls")  # in the volume: <token>.json per call whose container ran
# What a call may have run beyond its mark's seconds: since its last commit, and the
# container's start before the function began.
UNSEEN_S = COMMIT_EVERY_S + 5 * 60


def _uv_version(here: Path) -> str | None:
    """The repository's uv (``simulation/versions.lock``), when this file lies in the
    repository (``tools/train``); ``None`` anywhere else, such as in the container."""
    if len(here.parents) < 2:
        return None
    lock = here.parents[1] / "simulation" / "versions.lock"
    if not lock.is_file():
        return None
    return str(tomllib.loads(lock.read_text(encoding="utf-8"))["uv_version"])


def build_image(here: Path) -> Any:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_sync(
            str(here), extras=["cu126"], extra_options="--no-dev", uv_version=_uv_version(here)
        )
        .env(
            {
                "PYTHONPATH": f"{IN_CONTAINER}/tools:{IN_CONTAINER}/tools/train",
                "PYTHONUNBUFFERED": "1",
            }
        )
    )
    if modal.is_local():  # in the container the files are already there
        for local, remote in image_files(here.parent):
            image = image.add_local_file(local, str(remote))
    return image


app = modal.App("searchdrone-train")
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = build_image(HERE)


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={str(REMOTE_DATA): volume},
    timeout=CALL_LIMIT_S,
    cpu=(CPU_CORES, CPU_CORES),
    memory=(MEMORY_MIB, MEMORY_MIB),
)
def train_on_gpu(
    train: list[str], export: list[str], budget_s: float, token: str
) -> dict[str, Any]:
    started = time.monotonic()
    mark = Path(REMOTE_DATA / CALLS / f"{token}.json")

    def write_mark(ended: bool) -> None:
        mark.parent.mkdir(parents=True, exist_ok=True)
        tmp = mark.with_name(f".{mark.name}.tmp")
        body = {"seconds": round(time.monotonic() - started, 1), "ended": ended}
        tmp.write_text(json.dumps(body) + "\n", encoding="utf-8")
        tmp.replace(mark)

    write_mark(ended=False)
    volume.commit()
    stop = threading.Event()

    def commit_now_and_then() -> None:
        while not stop.wait(COMMIT_EVERY_S):
            try:
                write_mark(ended=False)
                volume.commit()
            except Exception as exc:  # the next one may work
                print(f"volymen sparades inte nu ({type(exc).__name__}); försöker igen", flush=True)

    keeper = threading.Thread(target=commit_now_and_then, daemon=True)
    keeper.start()
    try:
        result = call(train, export, budget_s, IN_CONTAINER / "tools" / "train")
    finally:
        stop.set()
        keeper.join()
        write_mark(ended=True)
        volume.commit()
    return {**result, "seconds": time.monotonic() - started}


class ModalBackend:
    """``cloud.Backend`` against the real Modal (the owner's account)."""

    def __init__(self, gpu: str) -> None:
        self.gpu = gpu

    def remote_sizes(self) -> dict[str, int]:
        from modal.volume import FileEntryType

        entries = volume.listdir("/", recursive=True)
        return {e.path.lstrip("/"): e.size for e in entries if e.type == FileEntryType.FILE}

    def upload(self, uploads: Sequence[Upload]) -> None:
        with volume.batch_upload(force=True) as batch:
            for u in uploads:
                batch.put_file(u.local, "/" + u.remote.as_posix())

    def run(
        self,
        train: Sequence[str],
        export: Sequence[str],
        budget_s: float,
        timeout_s: int,
        token: str,
        spawned: Callable[[str], None],
    ) -> dict[str, Any]:
        with modal.enable_output(), app.run():
            fn = train_on_gpu.with_options(gpu=self.gpu, timeout=timeout_s)
            handle = fn.spawn(list(train), list(export), budget_s, token)
            spawned(handle.object_id)
            try:
                return handle.get(timeout=timeout_s + START_SLACK_S)
            except TimeoutError:  # e.g. moved after a preemption: the ceiling holds
                started = self._mark(token) is not None  # before the cancel ends the call
                handle.cancel(terminate_containers=True)
                # no "seconds": the container did not return them (``call_state`` later)
                return {"status": "stopped" if started else "no_gpu", "code": None}

    def _mark(self, token: str) -> dict[str, Any] | None:
        """The call's mark in ``CALLS``, or ``None`` if its container never wrote one. The
        folder is listed rather than the file; a missing folder means no call has run."""
        name = f"{token}.json"
        try:
            entries = volume.listdir("/" + CALLS.as_posix())
        except (FileNotFoundError, modal.exception.NotFoundError):
            return None
        if not any(PurePosixPath(e.path).name == name for e in entries):
            return None
        buf = io.BytesIO()
        volume.read_file_into_fileobj("/" + (CALLS / name).as_posix(), buf)
        return dict(json.loads(buf.getvalue().decode("utf-8")))

    def call_state(self, token: str, call_id: str) -> dict[str, Any]:
        from modal.types import InputStatus

        mark = self._mark(token)
        if mark is not None and mark.get("ended"):
            return {"running": False, "seconds": float(mark["seconds"])}
        ended = False
        if call_id:
            try:
                graph = modal.FunctionCall.from_id(call_id).get_call_graph()
            except modal.exception.NotFoundError:  # Modal no longer knows: cannot say
                graph = []
            ended = bool(graph) and all(i.status != InputStatus.PENDING for i in graph)
        if not ended:
            return {"running": True, "seconds": None}
        # it ended without its last word (cancelled, stopped): at most UNSEEN_S more
        return {"running": False, "seconds": float(mark["seconds"] if mark else 0.0) + UNSEEN_S}

    def forget(self, token: str) -> None:
        with contextlib.suppress(FileNotFoundError, modal.exception.NotFoundError):
            volume.remove_file("/" + (CALLS / f"{token}.json").as_posix())

    def remove(self, paths: Sequence[PurePosixPath]) -> None:
        for path in paths:
            with contextlib.suppress(FileNotFoundError, modal.exception.NotFoundError):
                volume.remove_file("/" + path.as_posix())

    def fetch(self, remote: PurePosixPath, local: Path) -> None:
        with local.open("wb") as fh:
            volume.read_file_into_fileobj("/" + remote.as_posix(), fh)
