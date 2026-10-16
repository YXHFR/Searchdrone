import tomllib

import pytest
import versions


def test_known_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert versions.main(["just_version"]) == 0
    expected = tomllib.loads(versions.LOCK.read_text(encoding="utf-8"))["just_version"]
    assert capsys.readouterr().out.strip() == expected


@pytest.mark.parametrize("argv", [["no_such_key"], [], ["a", "b"]])
def test_bad_input_exits_2(argv: list[str]) -> None:
    assert versions.main(argv) == 2
