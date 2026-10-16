import json

import pytest

from sd_sim import launch, testrun


def test_testrun_dry_echoes_args(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["physics=builtin", "sc=SC00", "dry=1"]
    assert testrun.main(args) == 0
    assert json.loads(capsys.readouterr().out) == {"args": args}


@pytest.mark.parametrize(
    "bad",
    [["foo=1"], ["--physics=builtin"], ["qgc=true"], ["qgc=yes"], ["qgc=on"], ["qgc=TRUE "]],
)
def test_testrun_rejects(bad: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        testrun.main(bad)
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "bad", [["up", "physics=unity"], ["up", "instance=9"], ["up", "gui=maybe"]]
)
def test_launch_rejects_bad_values(bad: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        launch.main(bad)
    assert exc.value.code == 2


def test_launch_dry(capsys: pytest.CaptureFixture[str]) -> None:
    assert launch.main(["up", "physics=builtin", "dry=1"]) == 0
    assert json.loads(capsys.readouterr().out)["args"] == ["physics=builtin", "dry=1"]


@pytest.mark.parametrize(
    "bad", [[], ["start"], ["up", "--lan"], ["up", "colour=red"], ["up", "record=true"]]
)
def test_launch_rejects(bad: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        launch.main(bad)
    assert exc.value.code == 2


def test_launch_demo_record_dry(capsys: pytest.CaptureFixture[str]) -> None:
    assert launch.main(["up", "--demo", "record=true", "dry=1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["flags"] == ["--demo"]
