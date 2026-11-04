"""The Modal side of ``just train-detector-cloud`` (``cloud.py`` decides what is sent).

The container image is ``tools/train``'s own locked environment (``pyproject.toml`` and
``uv.lock``, PyTorch for CUDA 12.6) with ``tools/train`` and ``tools/bench`` copied in, so
the GPU runs the same ``train_detector.py`` and ``export_onnx.py`` as this computer. The
volume ``cloud.VOLUME`` is mounted as the data folder (``/data``); the runs, their
checkpoints and tile caches live there, and it is committed every few minutes, so a
call that ends early (24 hours, a lost GPU) loses at most that much.

Needs ``modal`` (the ``cloud`` dependency group) and the owner's token, which Modal reads
from ``~/.modal.toml`` or the environment; nothing here reads or prints it. CI never
imports this file.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import threading
import time
import tomllib
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import modal
from cloud import CALL_LIMIT_S, DEFAULT_GPU, REMOTE_DATA, VOLUME, Upload

HERE = Path(__file__).resolve().parent
IN_CONTAINER = PurePosixPath("/root/sd")
COMMIT_EVERY_S = 300
# After the stop signal, how long the training may take to stop before it is killed.
STOP_GRACE_S = 300


def _uv_version() -> str | None:
    """The repository's uv (``simulation/versions.lock``); in the container, unused."""
    lock = HERE.parents[1] / "simulation" / "versions.lock"
    if not lock.is_file():
        return None
    return str(tomllib.loads(lock.read_text(encoding="utf-8"))["uv_version"])


_IGNORE = [".venv", "**/__pycache__", "**/.pytest_cache", "tests"]

app = modal.App("searchdrone-train")
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(HERE), extras=["cu126"], extra_options="--no-dev", uv_version=_uv_version())
    .env(
        {
            "PYTHONPATH": f"{IN_CONTAINER}/tools:{IN_CONTAINER}/tools/train",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(HERE.parent / "bench", str(IN_CONTAINER / "tools" / "bench"), ignore=_IGNORE)
    .add_local_dir(HERE, str(IN_CONTAINER / "tools" / "train"), ignore=_IGNORE)
)


def _run(cmd: Sequence[str], deadline: float) -> int | None:
    """Run ``cmd``; at ``deadline`` stop it as Ctrl+C does (the training saves and
    exits). Returns its exit code, or ``None`` when the deadline stopped it."""
    proc = subprocess.Popen(list(cmd))  # noqa: S603 - our own scripts, our own arguments
    try:
        return proc.wait(timeout=max(1.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return None


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={str(REMOTE_DATA): volume},
    timeout=CALL_LIMIT_S,
    retries=modal.Retries(max_retries=2, initial_delay=30.0),
    cpu=8.0,
    memory=32768,
)
def train_on_gpu(train: list[str], export: list[str], budget_s: float) -> dict[str, Any]:
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
    scripts = IN_CONTAINER / "tools" / "train"
    try:
        deadline = started + budget_s
        code = _run([sys.executable, str(scripts / "train_detector.py"), *train], deadline)
        if code is None:
            status = "stopped"
        elif code == 0:
            code = _run([sys.executable, str(scripts / "export_onnx.py"), *export], deadline)
            status = "done" if code == 0 else ("stopped" if code is None else "failed")
        else:
            status = "failed"
    finally:
        stop.set()
        volume.commit()
    return {"status": status, "code": code, "seconds": time.monotonic() - started}


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

    def run(self, train: Sequence[str], export: Sequence[str], budget_s: float) -> dict[str, Any]:
        with modal.enable_output(), app.run():
            fn = train_on_gpu.with_options(gpu=self.gpu)
            return fn.remote(list(train), list(export), budget_s)

    def fetch(self, remote: PurePosixPath, local: Path) -> None:
        with local.open("wb") as fh:
            volume.read_file_into_fileobj("/" + remote.as_posix(), fh)
