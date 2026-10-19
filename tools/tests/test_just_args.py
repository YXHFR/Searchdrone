import json

import just_args
import pytest


def test_dry_run_passes_args(capsys: pytest.CaptureFixture[str]) -> None:
    assert just_args.main(["dev", "physics=builtin", "gui=false", "dry=1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {"recipe": "dev", "args": ["physics=builtin", "gui=false", "dry=1"]}


def test_unknown_key_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert just_args.main(["dev", "foo=1"]) == 2
    assert "okänd nyckel 'foo'" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [[], ["nope"], ["dev", "--physics=gz"], ["dev", "=1"]])
def test_bad_input_exits_2(argv: list[str]) -> None:
    assert just_args.main(argv) == 2


def test_not_built(capsys: pytest.CaptureFixture[str]) -> None:
    assert just_args.main(["showcase", "stage=1"]) == 2
    assert "inte byggt än (B1-18)" in capsys.readouterr().out


def test_built_recipe_only_checks_its_keys(capsys: pytest.CaptureFixture[str]) -> None:
    assert just_args.main(["e2e", "runs=3", "offline=1"]) == 0
    assert just_args.main(["accept", "phase=pre"]) == 0
    assert just_args.main(["perf", "runs=2", "gui=false", "eval=0"]) == 0
    assert capsys.readouterr().out == ""
    assert just_args.main(["accept", "runs=1"]) == 2
    assert just_args.main(["perf", "foo=1"]) == 2
