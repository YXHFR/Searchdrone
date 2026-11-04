"""The Modal side of ``just train-detector-cloud`` (``cloud.py`` decides what is sent,
``gpu_call.py`` what a call does).

The container image is ``tools/train``'s own locked environment (``pyproject.toml`` and
``uv.lock``, PyTorch for CUDA 12.6) plus exactly the files ``cloud.image_files`` lists
(the training scripts, the vendored model code and what of ``tools/bench`` they import),
so the GPU runs the same ``train_detector.py`` and ``export_onnx.py`` as this computer.
The volume ``cloud.VOLUME`` is mounted as the data folder (``/data``); the runs, their
checkpoints and tile caches live there, and it is committed every few minutes.

No Modal retries: a call that is lost ends the command, and the same command carries on
from the last save, so every GPU second is counted by the client against ``max_hours``.

This module is imported both here and in the container, where Modal places it at
``/root/modal_app.py``: nothing at import time may depend on where it lies. Needs
``modal`` (the ``cloud`` dependency group) and the owner's token, which Modal reads from
``~/.modal.toml`` or the environment; nothing here reads or prints it. CI never imports
this file with the real Modal.
"""

from __future__ import annotations

import threading
import time
import tomllib
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import modal
from cloud import (
    CALL_LIMIT_S,
    DEFAULT_GPU,
    IN_CONTAINER,
    REMOTE_DATA,
    VOLUME,
    Upload,
    image_files,
)
from gpu_call import call

HERE = Path(__file__).resolve().parent
COMMIT_EVERY_S = 300


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
    cpu=8.0,
    memory=32768,
)
def train_on_gpu(train: list[str], export: list[str], deadline: float) -> dict[str, Any]:
    started = time.monotonic()
    stop = threading.Event()

    def commit_now_and_then() -> None:
        while not stop.wait(COMMIT_EVERY_S):
            try:
                volume.commit()
            except Exception as exc:  # the next one may work
                print(f"volymen sparades inte nu ({type(exc).__name__}); försöker igen", flush=True)

    keeper = threading.Thread(target=commit_now_and_then, daemon=True)
    keeper.start()
    try:
        result = call(train, export, deadline, IN_CONTAINER / "tools" / "train")
    finally:
        stop.set()
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
        self, train: Sequence[str], export: Sequence[str], deadline: float, timeout_s: int
    ) -> dict[str, Any]:
        with modal.enable_output(), app.run():
            fn = train_on_gpu.with_options(gpu=self.gpu, timeout=timeout_s)
            return fn.remote(list(train), list(export), deadline)

    def fetch(self, remote: PurePosixPath, local: Path) -> None:
        with local.open("wb") as fh:
            volume.read_file_into_fileobj("/" + remote.as_posix(), fh)
