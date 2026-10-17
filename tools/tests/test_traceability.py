"""tools/gen_traceability.py and tools/check_traceability.py (no ids in the repository:
the tests write their own made-up id file)."""

from __future__ import annotations

from pathlib import Path

import check_traceability
import gen_traceability
import pytest
import yaml

GOOD = """
["X-1"]
typ = "A"
dod_ref = "9"
sc = ["SC04"]
tests = ["tools/tests/test_traceability.py::test_good_file"]

["X-2"]
typ = "M"
dod_ref = "4"

["X-3"]
typ = "A"
dod_ref = "7"
planned = "a later ticket"
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "ids.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_good_file(tmp_path: Path) -> None:
    entries = gen_traceability.load(_write(tmp_path, GOOD))
    assert [e.id for e in entries] == ["X-1", "X-2", "X-3"]
    assert gen_traceability.problems(entries, range(4, 5)) == []
    out = yaml.safe_load(gen_traceability.to_yaml(entries))
    assert out["entries"][0]["tests"] == ["tools/tests/test_traceability.py::test_good_file"]
    assert out["entries"][2]["planned"] == "a later ticket"


def test_automatic_row_needs_a_test_and_every_scenario_is_traced(tmp_path: Path) -> None:
    text = GOOD.replace('planned = "a later ticket"', "")
    entries = gen_traceability.load(_write(tmp_path, text))
    found = gen_traceability.problems(entries, range(3, 6))
    assert "X-3: automatisk rad utan test" in found
    assert "SC03: inget id pekar på scenariot" in found
    assert "SC05: inget id pekar på scenariot" in found
    assert not any("SC04" in f for f in found)


@pytest.mark.parametrize(
    "bad",
    [
        '["X"]\ntyp = "B"\n',
        '["X"]\ntyp = "A"\nsc = ["SC4"]\n',
        '["X"]\ntyp = "A"\ntests = ["not a node"]\n',
        '["X"]\ntyp = "A"\ncolour = "red"\n',
        '["X"\n',
    ],
)
def test_bad_files_are_refused(tmp_path: Path, bad: str) -> None:
    with pytest.raises(gen_traceability.IdFileError):
        gen_traceability.load(_write(tmp_path, bad))


def test_missing_id_file_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert gen_traceability.main(["--ids", str(tmp_path / "none.toml")]) == 2
    assert "saknas" in capsys.readouterr().out


def test_main_writes_the_yaml(tmp_path: Path) -> None:
    ids = _write(tmp_path, GOOD)
    out = tmp_path / "trace.yaml"
    # Only SC04 is traced in the made-up file, so the scenario check fails (exit 1).
    assert gen_traceability.main(["--ids", str(ids), "--out", str(out)]) == 1
    assert yaml.safe_load(out.read_text())["entries"][0]["id"] == "X-1"


def test_missing_nodes() -> None:
    collected = {"a.py::t", "b.py::u[x]", "b.py::u[y]"}
    assert check_traceability.missing(["a.py::t", "b.py::u", "b.py::u[x]"], collected) == []
    assert check_traceability.missing(["a.py::gone", "b.py::u[z]"], collected) == [
        "a.py::gone",
        "b.py::u[z]",
    ]


def test_check_without_file_is_ok(tmp_path: Path) -> None:
    assert check_traceability.main([str(tmp_path / "none.yaml")]) == 0


def test_check_finds_a_missing_test(tmp_path: Path) -> None:
    trace = tmp_path / "t.yaml"
    trace.write_text(
        yaml.safe_dump(
            {
                "entries": [
                    {"id": "X", "tests": ["tools/tests/test_traceability.py::test_good_file"]},
                    {"id": "Y", "tests": ["tools/tests/test_traceability.py::test_nope"]},
                ]
            }
        )
    )
    assert check_traceability.main([str(trace)]) == 1


REAL = gen_traceability.ids_path(None)


@pytest.mark.skipif(not REAL.is_file(), reason="the id file lives only in the planning folder")
def test_traceability() -> None:
    """The real id file (local only): every automatic row has a test or names the ticket
    that adds it, SC00-SC18 are traced, and every listed test exists."""
    entries = gen_traceability.load(REAL)
    assert gen_traceability.problems(entries) == []
    listed = sorted({t for e in entries for t in e.tests})
    files = sorted({t.split("::")[0] for t in listed})
    assert all((gen_traceability.REPO / f).is_file() for f in files)
    assert check_traceability.missing(listed, check_traceability.nodes(files)) == []
