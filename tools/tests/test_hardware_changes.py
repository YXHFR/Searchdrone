"""The hardware gate end to end: path filter, allowlist and the hardware.yml jobs.

Mirrors the gate's test pull requests: a software-only change skips the hardware jobs,
a change to only hardware/drone.yaml runs them and passes the allowlist, and every
hardware file type without a check (KiCad, CAD, FEA, hardware/README.md) fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import check_hw_files as chf
import pytest
import yaml

if TYPE_CHECKING:
    from conftest import GitRepo

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "hardware.yml"


def load() -> dict[Any, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def changes(repo: GitRepo, capsys: pytest.CaptureFixture[str]) -> str:
    base = repo.git("rev-parse", "HEAD~1")
    assert chf.main(["--changes", "--base", base, "--head", "HEAD"]) == 0
    return capsys.readouterr().out.strip()


@pytest.mark.parametrize(
    ("rel", "run"),
    [
        ("software/panel/src/App.tsx", "false"),
        ("README.md", "false"),
        ("tools/other.py", "false"),
        ("justfile", "true"),
        ("uv.lock", "true"),
        ("tools/ci_aggregate.jq", "true"),
        (".github/actions/setup/action.yml", "true"),
        ("electronics/board.kicad_pcb", "true"),
        ("hardware/drone.yaml", "true"),
        ("hardware/electronics/test.kicad_sch", "true"),
        ("software/hwcalc/src/sd_hwcalc/calc.py", "true"),
        ("tools/check_hw_files.py", "true"),
        (".github/workflows/hardware.yml", "true"),
    ],
)
def test_path_filter(
    hw_repo: GitRepo,
    capsys: pytest.CaptureFixture[str],
    rel: str,
    run: str,
) -> None:
    hw_repo.write(rel, "changed\n")
    hw_repo.commit()
    assert changes(hw_repo, capsys) == f"run={run}"


def test_changes_writes_github_output(
    hw_repo: GitRepo,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    hw_repo.write("hardware/drone.yaml", "a: 9\n")
    hw_repo.commit()
    out = tmp_path_factory.mktemp("o") / "out"
    chf.main(["--changes", "--base", "HEAD~1", "--head", "HEAD", "--github-output", str(out)])
    assert out.read_text(encoding="utf-8") == "run=true\n"


@pytest.mark.parametrize(
    "rel",
    [
        "hardware/electronics/test.kicad_sch",
        "hardware/electronics/test.kicad_pcb",
        "hardware/mechanical/test.step",
        "hardware/mechanical/test.FCStd",
        "hardware/mechanical/fea/test.inp",
        "hardware/README.md",
    ],
)
def test_every_unchecked_type_runs_and_fails(
    hw_repo: GitRepo,
    capsys: pytest.CaptureFixture[str],
    rel: str,
) -> None:
    hw_repo.write(rel)
    hw_repo.commit()
    assert changes(hw_repo, capsys) == "run=true"
    assert chf.main(["--base", "HEAD~1", "--head", "HEAD"]) == 1


def test_workflow_runs_always_without_path_filter() -> None:
    on = load()[True]  # PyYAML reads the key "on" as True
    assert set(on) == {"pull_request", "push", "merge_group"}
    assert set(on["pull_request"]["types"]) == {"opened", "synchronize", "reopened", "edited"}
    assert on["push"] == {"branches": ["main"]}


def test_filter_matches_spec_paths_and_workflow_uses_script() -> None:
    assert set(chf.FILTER) >= {
        "hardware/**",
        "software/hwcalc/**",
        "tools/check_hw_files.py",
        ".github/workflows/hardware.yml",
    }
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "tools/check_hw_files.py --changes" in text


def test_jobs_and_aggregate() -> None:
    jobs = load()["jobs"]
    assert set(jobs) == {"changes", "hw-files", "hw-calc", "hw-cad", "hardware-ok"}
    for name in ("hw-files", "hw-calc", "hw-cad"):
        assert jobs[name]["needs"] == "changes" or jobs[name]["needs"] == ["changes"]
        assert jobs[name]["if"] == "needs.changes.outputs.run == 'true'"
    ok = jobs["hardware-ok"]
    assert ok["if"] == "always()"
    assert set(ok["needs"]) == {"changes", "hw-files", "hw-calc", "hw-cad"}
    assert any("ci_aggregate.jq" in str(s.get("run", "")) for s in ok["steps"])
    assert any("just hw-check" in str(s.get("run", "")) for s in jobs["hw-calc"]["steps"])
    cad_runs = [str(s.get("run", "")) for s in jobs["hw-cad"]["steps"]]
    assert "just cad-detailed" in cad_runs
    assert "just cad-test" in cad_runs
    assert jobs["changes"]["outputs"]["run"]


def test_concurrency() -> None:
    conc = load()["concurrency"]
    assert conc["group"] == "hardware-${{ github.event.pull_request.number || github.sha }}"
    assert conc["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"
