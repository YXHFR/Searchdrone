"""Simulator settings and the demo's public-home rule."""

from pathlib import Path

import pytest

from sd_sim import config, launch, paths
from sd_sim.sitl import CMAC_HOME

EXAMPLE = (paths.REPO / "config" / "sim.example.toml").read_text(encoding="utf-8")
HOME = "-35.36,149.16,584,0"


def fake_repo(tmp_path: Path) -> Path:
    (tmp_path / "config" / "local").mkdir(parents=True)
    (tmp_path / "config" / "sim.example.toml").write_text(EXAMPLE, encoding="utf-8")
    return tmp_path


def test_defaults(tmp_path: Path) -> None:
    cfg = config.load({}, repo=fake_repo(tmp_path), environ={})
    assert (cfg.physics, cfg.gui, cfg.video, cfg.home) == ("gz", True, "camera", CMAC_HOME)
    assert not cfg.personal_home


def test_headless_env(tmp_path: Path) -> None:
    cfg = config.load({}, repo=fake_repo(tmp_path), environ={"SD_SIM__HEADLESS": "1"})
    assert (cfg.physics, cfg.gui, cfg.video) == ("builtin", False, "test")


def test_args_win_over_env(tmp_path: Path) -> None:
    cfg = config.load(
        {"link_profile": "dead"},
        repo=fake_repo(tmp_path),
        environ={"SD_SIM__LINK_PROFILE": "field_typical"},
    )
    assert cfg.link_profile == "dead"


def test_personal_home_outside_demo(tmp_path: Path) -> None:
    repo = fake_repo(tmp_path)
    (repo / "config" / "sim.toml").write_text(f'[sim]\nhome_override = "{HOME}"\n')
    cfg = config.load({}, repo=repo, environ={})
    assert cfg.personal_home and cfg.home == HOME


@pytest.mark.parametrize("where", ["sim.toml", "local/sim.toml", "env"])
def test_demo_uses_generic_home(tmp_path: Path, where: str) -> None:
    repo = fake_repo(tmp_path)
    env: dict[str, str] = {}
    if where == "env":
        env["SD_SIM__HOME_OVERRIDE"] = HOME
    else:
        (repo / "config" / where).write_text(f'[sim]\nhome_override = "{HOME}"\n')
    with pytest.raises(config.DemoHomeError, match="CMAC"):
        config.load({}, demo=True, repo=repo, environ=env)


def test_demo_exit_code_3(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SD_SIM__HOME_OVERRIDE", HOME)
    monkeypatch.setenv("SD_SIM__DRY_RUN", "1")
    assert launch.main(["up", "--demo"]) == 3
    assert config.DEMO_HOME_REFUSED in capsys.readouterr().err


def test_demo_ignores_local_settings(tmp_path: Path) -> None:
    repo = fake_repo(tmp_path)
    (repo / "config" / "sim.toml").write_text('[sim]\nphysics = "builtin"\n')
    assert config.load({}, demo=True, repo=repo, environ={}).physics == "gz"


@pytest.mark.parametrize("args", [{"speedup": "2"}, {"instance": "x"}, {"video": "4k"}])
def test_bad_values(tmp_path: Path, args: dict[str, str]) -> None:
    with pytest.raises(config.ConfigError):
        config.load(args, repo=fake_repo(tmp_path), environ={})


def test_bad_home(tmp_path: Path) -> None:
    with pytest.raises(config.ConfigError):
        config.load({}, repo=fake_repo(tmp_path), environ={"SD_SIM__HOME_OVERRIDE": "my house"})
