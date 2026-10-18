"""``runs/<run>/result.json`` for one scenario test.

Each criterion is ``passed`` (true/false) or not yet checkable: ``passed`` is then
``null`` and ``note`` says which part of the stack is missing. The scenario passes
when no criterion failed and the test rig did not fail.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sd_sim import paths


@dataclass
class Criterion:
    id: str
    text: str
    passed: bool | None
    measured: Any = None
    limit: Any = None
    unit: str = ""
    clock: str = ""
    note: str = ""
    # "max" or "min": which side of ``limit`` passes (``measure``); empty when not numeric.
    bound: str = ""


@dataclass
class Report:
    scenario: str
    run_dir: Path
    instance: int
    physics: str
    brain: str
    variant: str = "default"
    criteria: list[Criterion] = field(default_factory=list)
    param_mismatches: list[str] = field(default_factory=list)
    infra_failure: str | None = None
    error: str | None = None
    started_utc: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    _t0: float = field(default_factory=time.monotonic)

    def check(self, cid: str, text: str, passed: bool, **kw: Any) -> bool:
        if not isinstance(passed, bool):
            raise TypeError(f"{cid}: passed ska vara True eller False, inte {passed!r}")
        self.criteria.append(Criterion(cid, text, passed, **kw))
        return passed

    def not_built(self, cid: str, text: str, note: str) -> None:
        """A criterion whose part of the stack does not exist yet (``passed: null``)."""
        if not note.strip():
            raise ValueError(f"{cid}: ange vad som saknas")
        self.criteria.append(Criterion(cid, text, None, note=note))

    @property
    def passed(self) -> bool:
        return (
            self.infra_failure is None
            and self.error is None
            and any(c.passed is True for c in self.criteria)
            and all(c.passed is not False for c in self.criteria)
        )

    def failures(self) -> list[str]:
        out = [
            f"{c.id} {c.text}: {c.measured!r} (gräns {c.limit!r})"
            for c in self.criteria
            if c.passed is False
        ]
        if self.infra_failure:
            out.append(f"testriggen: {self.infra_failure}")
        if self.error:
            out.append(f"fel: {self.error}")
        if not any(c.passed is True for c in self.criteria):
            out.append("inget kriterium kontrollerades")
        return out

    def as_dict(self) -> dict[str, Any]:
        lock = paths.versions()
        artifacts = sorted(
            str(p.relative_to(self.run_dir))
            for p in self.run_dir.rglob("*")
            if p.is_file() and p.name != "result.json"
        )
        return {
            "schema": 1,
            "scenario": self.scenario,
            "variant": self.variant,
            "passed": self.passed,
            "started_utc": self.started_utc,
            "duration_s": round(time.monotonic() - self._t0, 1),
            "instance": self.instance,
            "physics": self.physics,
            "brain": self.brain,
            "seed": None,
            "versions": {
                "ardupilot": lock.get("ardupilot_tag"),
                "ardupilot_gazebo": str(lock.get("ardupilot_gazebo_commit", ""))[:7] or None,
                "gz_sim": None,
                "git_sha": git_sha(),
            },
            "criteria": [
                {k: v for k, v in vars(c).items() if v not in ("", None) or k == "passed"}
                for c in self.criteria
            ],
            "invariants": [],
            "param_mismatches": self.param_mismatches,
            "artifacts": artifacts,
            "infra_failure": self.infra_failure,
            "error": self.error,
            "not_built": sum(1 for c in self.criteria if c.passed is None),
        }

    def write(self) -> Path:
        path = self.run_dir / "result.json"
        path.write_text(json.dumps(self.as_dict(), indent=2, ensure_ascii=False) + "\n", "utf-8")
        return path


# What the running test measured (``measure``), written to its result.json by
# ``sd_sim.pytest_plugin``. One test runs at a time in each pytest process.
_measured: list[Criterion] = []


def measure(
    text: str,
    measured: float,
    limit: float,
    *,
    at_most: bool = True,
    unit: str = "s",
    passed: bool | None = None,
) -> bool:
    """Record one measured value against its limit for the running test's result.json.

    ``at_most``: the value passes when it is at most ``limit`` (a time to react); else
    when it is at least ``limit`` (a time something held, a coverage). ``passed`` gives
    the caller's own verdict instead, when its check is not the comparison (a wait whose
    condition held on the first look after the deadline still passed)."""
    if passed is None:
        passed = measured <= limit if at_most else measured >= limit
    cid = f"M{len(_measured) + 1}"
    _measured.append(
        Criterion(
            cid, text, passed, round(measured, 3), limit, unit, bound="max" if at_most else "min"
        )
    )
    return passed


def take_measured() -> list[Criterion]:
    """The values recorded since the last call (and forget them)."""
    out = list(_measured)
    _measured.clear()
    return out


def git_sha() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        out = subprocess.run(  # noqa: S603 - fixed arguments
            ["git", "-C", str(paths.REPO), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None
