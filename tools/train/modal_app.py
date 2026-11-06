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
then with the seconds it has run at every commit, and ``ended`` at its end; a container
Modal starts again after a preemption adds its seconds to the earlier attempt's. When the
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
import os
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
# The container's start before the function began, which no mark counts. A container
# that Modal moves before its first mark is saved loses this much from the count (the
# client's own wait for the call, see ``cloud.settle_calls``, still counts it then).
START_UP_S = 5 * 60
# How long one save (the mark and a commit of the volume, with what the training wrote
# since the last one: checkpoints, several GB of tile cache) may take. One that takes
# longer is taken to hang, and the container stops itself. A save that fails is tried
# again at the next commit; when that fails too, the container stops itself.
COMMIT_SLOW_S = 10 * 60
WATCH_S = 10  # how often the container looks at the save under way
# How long after the last mark saved was written the container stops itself at the
# latest: that save's commit, then the failed one and the one tried again, each started
# a commit's interval (or that save's time) after the one before.
SAVE_LIMIT_S = 2 * max(COMMIT_EVERY_S, COMMIT_SLOW_S) + COMMIT_SLOW_S + WATCH_S
# What a call may have run beyond a mark that is not ``ended``: its start-up and that.
UNSEEN_S = START_UP_S + SAVE_LIMIT_S
# The client waits for a call in steps of this many seconds (see ``ModalBackend.run``).
WAIT_STEP_S = 60.0


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
    try:
        return _train(train, export, budget_s, token)
    except Exception as exc:  # one known kind, so the client can tell it has ended
        raise CallFailedError(f"{type(exc).__name__}: {exc}") from None


class CallFailedError(Exception):
    """``train_on_gpu`` failed (its own error, as text). Modal hands it to the client as
    the call's result: the call has ended."""


def _train(train: list[str], export: list[str], budget_s: float, token: str) -> dict[str, Any]:
    started = time.monotonic()
    mark = Path(REMOTE_DATA / CALLS / f"{token}.json")
    # A mark already there is an earlier attempt of this call that Modal moved (a
    # preemption): its seconds count too, with its start-up and what it may have run
    # after its last mark. The mark adds up; it is never started over.
    before = 0.0
    if mark.exists():
        earlier = json.loads(mark.read_text(encoding="utf-8"))
        unseen = START_UP_S if earlier.get("ended") else UNSEEN_S
        before = float(earlier["seconds"]) + unseen

    def seconds() -> float:
        return before + time.monotonic() - started

    lock = threading.Lock()  # one writer of the mark at a time
    stop = threading.Event()  # the training is over: the keeper saves no more
    watched = threading.Event()  # the last mark is saved: the watchdog may go
    # busy: when the save under way began (None: none); failed: the error of the
    # keeper's last save, if it failed; final: the last mark is saved
    state: dict[str, Any] = {"busy": None, "failed": "", "final": False}

    def save(ended: bool) -> None:
        """Write the mark and commit the volume. After the last mark nothing is written
        again, and once the training is over only the last mark is."""
        with lock:
            if state["final"] or (stop.is_set() and not ended):
                return
            state["busy"] = time.monotonic()
            try:
                mark.parent.mkdir(parents=True, exist_ok=True)
                tmp = mark.with_name(f".{mark.name}.tmp")
                body = {"seconds": round(seconds(), 1), "ended": ended}
                tmp.write_text(json.dumps(body) + "\n", encoding="utf-8")
                tmp.replace(mark)
                volume.commit()
                state["final"] = ended
            finally:
                state["busy"] = None

    def keep_saving(next_at: float) -> None:
        while True:
            next_at += COMMIT_EVERY_S
            if stop.wait(max(0.0, next_at - time.monotonic())):
                return
            try:
                save(ended=False)
                state["failed"] = ""
            except Exception as exc:
                failed = state["failed"]
                if failed:  # the second in a row: nothing is saved, last.pt neither
                    give_up(
                        f"volymen gick inte att spara två gånger i rad ({failed}, "
                        f"{type(exc).__name__})"
                    )
                    return
                state["failed"] = type(exc).__name__
                print(f"volymen sparades inte nu ({state['failed']}); försöker igen", flush=True)

    def watch() -> None:
        # Apart from the keeper, so that a save that hangs is stopped too.
        while not watched.wait(WATCH_S):
            busy = state["busy"]
            if busy is not None and time.monotonic() - busy > COMMIT_SLOW_S:
                give_up(f"sparningen på volymen blev inte klar på {COMMIT_SLOW_S // 60} minuter")
                return

    watchdog = threading.Thread(target=watch, daemon=True)
    watchdog.start()
    try:
        first = time.monotonic()  # the keeper's commits are counted from this one
        save(ended=False)
        keeper = threading.Thread(target=keep_saving, args=(first,), daemon=True)
        keeper.start()
        try:
            result = call(train, export, budget_s, IN_CONTAINER / "tools" / "train")
        finally:
            stop.set()
            try:  # waits for a save under way (the watchdog bounds that)
                save(ended=True)
            except Exception as exc:  # a finished run is not lost on one failed commit,
                if state["failed"]:  # but never a third failed save in a row
                    raise
                print(f"volymen sparades inte nu ({type(exc).__name__}); försöker igen", flush=True)
                save(ended=True)
    finally:
        watched.set()
    return {**result, "seconds": seconds()}


def give_up(why: str) -> None:
    """End the container at once (the training too): Modal ends the call."""
    print(f"Anropet stoppas: {why}. Kör samma kommando igen senare.", flush=True)
    os._exit(3)


def _ended_errors() -> tuple[type[BaseException], ...]:
    """The errors ``FunctionCall.get`` raises for a call that is over: cancelled, out of
    time, its output gone, the container failed or stopped (``give_up``), or the
    function's own error (``CallFailedError``). Any other error (the line, the account,
    Modal's servers, a closed client) says nothing about the call."""
    e = modal.exception
    return (
        e.InputCancellation,
        e.FunctionTimeoutError,
        e.OutputExpiredError,
        e.InternalFailure,
        e.RemoteError,
        e.ExecutionError,
        CallFailedError,
    )


def _has_ended(call_id: str) -> bool:
    """Whether Modal says the call is over (``False``: running, or it cannot say). Its
    call graph is best effort (a call just cancelled may still show as pending), so ask
    for the result too, without waiting: a result, or one of ``_ended_errors``."""
    from modal.types import InputStatus

    handle = modal.FunctionCall.from_id(call_id)
    try:
        graph = handle.get_call_graph()
    except Exception:  # Modal no longer knows it, or cannot answer now: cannot say
        graph = []
    if graph and all(i.status != InputStatus.PENDING for i in graph):
        return True
    try:
        handle.get(timeout=0)
    except _ended_errors():
        return True
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return False  # not yet (TimeoutError), or cannot say
    return True


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
            # Wall clock, in short waits: a computer that slept past the deadline (the
            # monotonic clock stops then) cancels as soon as it wakes. From before the
            # send, as ``cloud.settle_calls`` counts it (from ``spawned("")``).
            deadline = time.time() + timeout_s + START_SLACK_S
            spawned("")  # being sent: from here on it may run, before its id is known
            handle = fn.spawn(list(train), list(export), budget_s, token)
            spawned(handle.object_id)
            while (left := deadline - time.time()) > 0:
                try:
                    return handle.get(timeout=min(left, WAIT_STEP_S))
                except TimeoutError:
                    continue
            # past the deadline (moved after a preemption, say): the ceiling holds
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
        """``running`` unless Modal says the call has ended (also when it cannot say, or
        has no id); ``seconds`` from the mark: an ``ended`` one plus its start-up,
        another plus ``UNSEEN_S``, no mark ``START_UP_S`` (no function ran). A mark that
        says ``ended`` is not trusted alone: after a preemption the earlier container
        may have written it while Modal runs the call again."""
        ended = bool(call_id) and _has_ended(call_id)
        mark = self._mark(token)
        if mark is None:
            seconds = float(START_UP_S)
        elif mark.get("ended"):
            seconds = float(mark["seconds"]) + START_UP_S
        else:
            seconds = float(mark["seconds"]) + UNSEEN_S
        return {"running": not ended, "seconds": seconds}

    def cancel(self, call_id: str) -> None:
        with contextlib.suppress(modal.exception.NotFoundError):
            modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)

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
