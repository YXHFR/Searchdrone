"""What one GPU call does (``gpu_call.py``), without Modal: real child processes for the
deadline and the stop, and a fake runner for how a call ends."""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from pathlib import PurePosixPath

import gpu_call
import pytest

PY = sys.executable
SCRIPTS = PurePosixPath("/scripts")


def test_a_script_that_ends_in_time_gives_its_exit_code() -> None:
    assert gpu_call.run_until([PY, "-c", "raise SystemExit(3)"], time.monotonic() + 30) == 3
    assert gpu_call.run_until([PY, "-c", "pass"], time.monotonic() + 30) == 0


def test_the_deadline_stops_a_script_with_sigint() -> None:
    code = (
        "import time\n"
        "try:\n    time.sleep(30)\n"
        "except KeyboardInterrupt:\n    raise SystemExit(130)\n"
    )
    started = time.monotonic()
    assert gpu_call.run_until([PY, "-c", code], time.monotonic() + 1.0, grace_s=20) is None
    assert time.monotonic() - started < 15


def test_a_script_that_ignores_sigint_is_killed_after_the_grace() -> None:
    code = "import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\ntime.sleep(30)\n"
    started = time.monotonic()
    assert gpu_call.run_until([PY, "-c", code], time.monotonic() + 1.0, grace_s=1.0) is None
    assert time.monotonic() - started < 15


class Runner:
    def __init__(self, codes: list[int | None]) -> None:
        self.codes = codes
        self.cmds: list[tuple[list[str], float]] = []

    def __call__(self, cmd: Sequence[str], deadline: float) -> int | None:
        self.cmds.append((list(cmd), deadline))
        return self.codes[len(self.cmds) - 1]


@pytest.mark.parametrize(
    ("codes", "status", "code", "scripts"),
    [
        ([0, 0], "done", 0, 2),
        ([None], "stopped", None, 1),  # the deadline came during the training
        ([0, None], "stopped", None, 2),  # ... or during the export
        ([2], "failed", 2, 1),  # a failed training is never exported
        ([0, 1], "failed", 1, 2),
    ],
)
def test_how_a_call_ends(
    codes: list[int | None], status: str, code: int | None, scripts: int
) -> None:
    run = Runner(codes)
    res = gpu_call.call(
        ["model=yolox_s"], ["name=x"], 23.0, SCRIPTS, run=run, python="py", clock=lambda: 100.0
    )
    assert res == {"status": status, "code": code}
    assert len(run.cmds) == scripts
    assert run.cmds[0] == (["py", "/scripts/train_detector.py", "model=yolox_s"], 123.0)
    if scripts == 2:
        assert run.cmds[1] == (["py", "/scripts/export_onnx.py", "name=x"], 123.0)
