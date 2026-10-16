"""tools/check_hw_files.py: the hardware allowlist fails closed for every unchecked file."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import check_hw_files as chf
import pytest

if TYPE_CHECKING:
    from conftest import GitRepo


def run(repo: GitRepo, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    base = repo.git("rev-parse", "HEAD~1")
    code = chf.main(["--base", base, "--head", "HEAD"])
    return code, capsys.readouterr().out


def test_drone_yaml_changed_passes(hw_repo: GitRepo, capsys: pytest.CaptureFixture[str]) -> None:
    hw_repo.write("hardware/drone.yaml", "a: 2\n")
    hw_repo.commit()
    code, out = run(hw_repo, capsys)
    assert code == 0, out


def test_limits_yaml_changed_passes(hw_repo: GitRepo, capsys: pytest.CaptureFixture[str]) -> None:
    hw_repo.write("hardware/limits.yaml", "b: 2\n")
    hw_repo.commit()
    assert run(hw_repo, capsys)[0] == 0


@pytest.mark.parametrize(
    "rel",
    [
        "hardware/mechanical/frame.step",
        "hardware/mechanical/test.step",
        "hardware/mechanical/test.FCStd",
        "hardware/mechanical/arm.stl",
        "hardware/mechanical/fea/t.inp",
        "hardware/electronics/x.kicad_sch",
        "hardware/electronics/x.kicad_pcb",
        "hardware/electronics/x.kicad_pro",
        "hardware/README.md",
        "hardware/bom.csv",
        "hardware/sub/drone.yaml",  # only the exact top-level file is checked
    ],
)
def test_unchecked_file_fails(
    hw_repo: GitRepo, capsys: pytest.CaptureFixture[str], rel: str
) -> None:
    hw_repo.write(rel)
    hw_repo.commit()
    code, out = run(hw_repo, capsys)
    assert code == 1
    assert f"Ingen automatisk kontroll för {rel} ännu" in out


def test_deleted_file_passes(hw_repo: GitRepo, capsys: pytest.CaptureFixture[str]) -> None:
    (hw_repo.path / "hardware" / "limits.yaml").unlink()
    hw_repo.commit()
    assert run(hw_repo, capsys)[0] == 0


def test_rename_into_unchecked_name_fails(
    hw_repo: GitRepo, capsys: pytest.CaptureFixture[str]
) -> None:
    (hw_repo.path / "hardware" / "limits.yaml").rename(hw_repo.path / "hardware" / "limits.inp")
    hw_repo.commit()
    assert run(hw_repo, capsys)[0] == 1


def test_software_only_change_passes(hw_repo: GitRepo, capsys: pytest.CaptureFixture[str]) -> None:
    hw_repo.write("software/panel/x.ts", "x\n")
    hw_repo.commit()
    assert run(hw_repo, capsys)[0] == 0


def test_symlink_on_checked_name_fails(
    hw_repo: GitRepo, capsys: pytest.CaptureFixture[str]
) -> None:
    hw_repo.write("elsewhere.yaml", "a: 1\n")
    (hw_repo.path / "hardware" / "drone.yaml").unlink()
    (hw_repo.path / "hardware" / "drone.yaml").symlink_to("../elsewhere.yaml")
    hw_repo.commit()
    code, out = run(hw_repo, capsys)
    assert code == 1
    assert "inte en vanlig fil" in out


def test_newline_in_file_name_cannot_inject_workflow_commands(
    hw_repo: GitRepo, capsys: pytest.CaptureFixture[str]
) -> None:
    hw_repo.write("hardware/x\n::warning::y.step")
    hw_repo.commit()
    code, out = run(hw_repo, capsys)
    assert code == 1
    assert all(line.startswith(("::error::", "1 fil")) for line in out.splitlines())


def test_only_merge_base_diff_counts(hw_repo: GitRepo, capsys: pytest.CaptureFixture[str]) -> None:
    # A file added on the base branch after the fork point is not this PR's change.
    fork = hw_repo.git("rev-parse", "HEAD")
    hw_repo.git("checkout", "-q", "-b", "other")
    hw_repo.write("hardware/mechanical/x.step")
    base = hw_repo.commit()
    hw_repo.git("checkout", "-q", "-")
    hw_repo.write("hardware/drone.yaml", "a: 3\n")
    head = hw_repo.commit()
    assert fork != head
    assert chf.main(["--base", base, "--head", head]) == 0


def test_zero_base_checks_every_hardware_file(
    hw_repo: GitRepo, capsys: pytest.CaptureFixture[str]
) -> None:
    hw_repo.write("hardware/old.step")
    hw_repo.commit()
    assert chf.main(["--base", "0" * 40, "--head", "HEAD"]) == 1
    (hw_repo.path / "hardware" / "old.step").unlink()
    hw_repo.commit()
    assert chf.main(["--base", "0" * 40, "--head", "HEAD"]) == 0


@pytest.mark.parametrize(
    ("glob", "path", "hit"),
    [
        ("hardware/drone.yaml", "hardware/drone.yaml", True),
        ("hardware/drone.yaml", "hardware/drone.yamlx", False),
        ("hardware/mechanical/fea/*.inp", "hardware/mechanical/fea/a.inp", True),
        ("hardware/mechanical/fea/*.inp", "hardware/mechanical/fea/sub/a.inp", False),
        ("hardware/electronics/**/*.kicad_pro", "hardware/electronics/a.kicad_pro", True),
        ("hardware/electronics/**/*.kicad_pro", "hardware/electronics/b/c/a.kicad_pro", True),
        ("hardware/**", "hardware/a/b", True),
        ("hardware/**", "hardwarex/a", False),
        ("hardware/?.yaml", "hardware/a.yaml", True),
    ],
)
def test_glob(glob: str, path: str, hit: bool) -> None:
    assert chf.matches(path, [glob]) is hit


def test_every_checked_row_names_a_job_in_hardware_ok() -> None:
    import yaml

    root = Path(__file__).resolve().parents[2]
    jobs = yaml.safe_load((root / ".github/workflows/hardware.yml").read_text("utf-8"))["jobs"]
    for glob, job in chf.CHECKED.items():
        assert job in jobs, glob
        assert job in jobs["hardware-ok"]["needs"], glob
