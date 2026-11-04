"""What one GPU call of ``just train-detector-cloud`` does, without Modal: run
``train_detector.py`` and then ``export_onnx.py`` until an absolute deadline, and say
how it ended. ``modal_app.train_on_gpu`` wraps this; the tests run it as it is.

The deadline is a wall-clock time (``time.time()``) set by the client before the call,
so the time the container takes to start counts against it too. At the deadline the
running script gets SIGINT, as Ctrl+C: ``train_detector.py`` then stops without saving,
so the work since its last save (every ``save_minutes``, default 10, and after each
epoch) is done again by the next call. A script that has not stopped ``grace_s`` later
is killed.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import PurePosixPath
from typing import Any

# After SIGINT, how long a script may take to stop before it is killed.
STOP_GRACE_S = 300.0

Runner = Callable[[Sequence[str], float], int | None]


def run_until(
    cmd: Sequence[str],
    deadline: float,
    grace_s: float = STOP_GRACE_S,
    clock: Callable[[], float] = time.time,
) -> int | None:
    """Run ``cmd``; at ``deadline`` (wall clock) stop it with SIGINT, and kill it if it
    is still running ``grace_s`` later. Returns its exit code, or ``None`` when the
    deadline stopped it."""
    proc = subprocess.Popen(list(cmd))  # noqa: S603 - our own scripts, our own arguments
    try:
        return proc.wait(timeout=max(0.0, deadline - clock()))
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return None


def call(
    train: Sequence[str],
    export: Sequence[str],
    deadline: float,
    scripts: PurePosixPath,
    run: Runner = run_until,
    python: str = sys.executable,
) -> dict[str, Any]:
    """Train, then export if the training finished. ``status``: ``done`` (both
    finished), ``stopped`` (the deadline came first; the next call carries on from the
    last save) or ``failed`` (a script exited with an error, ``code``)."""
    code = run([python, str(scripts / "train_detector.py"), *train], deadline)
    if code == 0:
        code = run([python, str(scripts / "export_onnx.py"), *export], deadline)
    if code is None:
        return {"status": "stopped", "code": None}
    return {"status": "done" if code == 0 else "failed", "code": code}
