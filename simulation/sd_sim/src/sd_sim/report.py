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
    started_utc: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    _t0: float = field(default_factory=time.monotonic)

    def check(self, cid: str, text: str, passed: bool | None, **kw: Any) -> bool:
        self.criteria.append(Criterion(cid, text, passed, **kw))
        return bool(passed)

    def not_built(self, cid: str, text: str, note: str) -> None:
        self.criteria.append(Criterion(cid, text, None, note=note))

    @property
    def passed(self) -> bool:
        return self.infra_failure is None and all(c.passed is not False for c in self.criteria)

    def failures(self) -> list[str]:
        out = [
            f"{c.id} {c.text}: {c.measured!r} (gräns {c.limit!r})"
            for c in self.criteria
            if c.passed is False
        ]
        if self.infra_failure:
            out.append(f"testriggen: {self.infra_failure}")
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
        }

    def write(self) -> Path:
        path = self.run_dir / "result.json"
        path.write_text(json.dumps(self.as_dict(), indent=2, ensure_ascii=False) + "\n", "utf-8")
        return path


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
