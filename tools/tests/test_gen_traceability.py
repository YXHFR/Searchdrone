"""tools/gen_traceability.py: test discovery, the checks and the missing local list."""

from __future__ import annotations

from pathlib import Path

import gen_traceability as gt
import pytest
import yaml


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    write(tmp_path / "software/x/tests/test_a.py", "def test_one():\n    pass\n")
    write(
        tmp_path / "simulation/tests/test_b.py",
        "async def test_two():\n    pass\n\ndef helper():\n    pass\n",
    )
    write(tmp_path / "software/node_modules/test_c.py", "def test_hidden():\n    pass\n")
    return tmp_path


LIST = """
[meta]
scenarios = ["SC01", "SC02"]

[[id]]
id = "R-1"
typ = "A"
sc = ["SC01"]
tests = ["test_one"]

[[id]]
id = "R-2"
typ = "A"
sc = ["SC02"]
tests = ["test_two"]

[[id]]
id = "R-3"
typ = "M"
tests = []
"""


def test_finds_sync_and_async_tests_but_not_vendored_ones(repo: Path) -> None:
    found = gt.test_functions(repo)
    assert found == {
        "test_one": ["software/x/tests/test_a.py"],
        "test_two": ["simulation/tests/test_b.py"],
    }


def test_a_complete_list_passes_and_writes_the_map(repo: Path, tmp_path: Path) -> None:
    data = gt.tomllib.loads(LIST)
    tests = gt.test_functions(repo)
    assert gt.problems(data, tests) == []
    trace = gt.trace(data, tests)
    assert trace["R-1"]["tests"] == ["software/x/tests/test_a.py::test_one"]
    assert trace["R-3"]["tests"] == []


def test_every_problem_is_reported(repo: Path) -> None:
    bad = gt.tomllib.loads(
        """
[meta]
scenarios = ["SC01", "SC09"]

[[id]]
id = "R-1"
typ = "A"
sc = ["SC01"]
tests = ["test_missing"]

[[id]]
id = "R-1"
typ = "A"
tests = []
"""
    )
    found = gt.problems(bad, gt.test_functions(repo))
    assert found == [
        "R-1: testet test_missing finns inte",
        "R-1: finns två gånger",
        "R-1: automatiskt krav utan test",
        "SC09: inget krav nämner scenariot",
    ]


def test_main_without_the_local_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert gt.main(["--accept-ids", str(tmp_path / "none.toml")]) == 0
    assert gt.MISSING_TEXT in capsys.readouterr().out


def test_main_writes_the_map_and_fails_on_problems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    monkeypatch.setattr(gt, "REPO", repo)
    good = write(tmp_path / "local/accept-ids.toml", LIST)
    assert gt.main(["--accept-ids", str(good)]) == 0
    written = yaml.safe_load((good.parent / "traceability.yaml").read_text(encoding="utf-8"))
    assert written["R-2"]["tests"] == ["simulation/tests/test_b.py::test_two"]
    bad = write(tmp_path / "local/bad.toml", LIST.replace("test_two", "test_gone"))
    monkeypatch.setenv("SD_ACCEPT_IDS", str(bad))
    assert gt.main(["--out", str(tmp_path / "out.yaml")]) == 1
