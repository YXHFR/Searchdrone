"""``executor_state.json``: what a restarted ``sd-onboard`` needs to know.

Written atomically (temporary file, fsync, rename) every second and at every
transition. On start, a vehicle that is armed or in the air is always HELD, whatever the
file says; the saved resume point is kept so that ``resume`` works after Resume.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any, Final

from pydantic import Field, TypeAdapter, ValidationError

from sd_obs.clock import Clock
from sd_schemas.base import Id, SdModel
from sd_schemas.executor import ExecutorState, ResumePoint

__all__ = ["SCHEMA", "SavedState", "StateStore"]

SCHEMA: Final = 2
_FILE: Final = "executor_state.json"
_RESUME: TypeAdapter[ResumePoint | None] = TypeAdapter(ResumePoint | None)


class SavedState(SdModel):
    schema_: int = Field(default=SCHEMA, alias="schema")
    vehicle_id: str
    mission_id: Id | None
    boot_id: int
    state: ExecutorState
    hold_reason: str | None
    held_by: str | None
    expected_mode: str | None
    charter_id: Id | None
    charter_version: int | None
    active_command_id: Id | None
    resume_point: dict[str, Any] | None
    saved_mono_ms: int
    saved_utc: str

    def resume(self) -> ResumePoint | None:
        try:
            return _RESUME.validate_python(self.resume_point)
        except ValidationError:
            return None


class StateStore:
    def __init__(self, directory: Path, clock: Clock) -> None:
        self.path = directory / _FILE
        self._clock = clock
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    def load(self) -> SavedState | None:
        """The saved state, or None when there is none or it cannot be read."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            saved = SavedState.model_validate(data)
        except (OSError, ValueError, ValidationError):
            return None
        return saved if saved.schema_ == SCHEMA else None

    def save(self, state: dict[str, Any]) -> None:
        body = {
            "schema": SCHEMA,
            **state,
            "saved_mono_ms": self._clock.monotonic_ms(),
            "saved_utc": self._clock.utc_now().isoformat().replace("+00:00", "Z"),
        }
        tmp = self.path.with_suffix(".json.tmp")
        data = json.dumps(body, separators=(",", ":"), default=str).encode("utf-8")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        tmp.replace(self.path)
        with contextlib.suppress(OSError):
            dfd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
