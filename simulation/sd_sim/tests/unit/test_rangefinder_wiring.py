"""``rangefinder=none|tf02|grf250`` through the launcher: the setting, SITL's serial6 and
parameter layer, the laser process, sd-onboard's settings and the map pack of the run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sd_onboard import config as onboard_config
from sd_onboard.io.params import effective_params
from sd_safety.floor_pack import load_floor_pack
from sd_sim import config, launch, paths, ports, rangefinder, services
from sd_sim.config import SimConfig
from sd_sim.errors import InfraError
from sd_sim.floor_pack import MapErrors
from sd_sim.obstacles import PowerLine, TreeStand
from sd_sim.sitl import SitlLayout, SitlOptions, build_sitl_argv

EXAMPLE = (paths.REPO / "config" / "sim.example.toml").read_text(encoding="utf-8")


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "config" / "local").mkdir(parents=True)
    (tmp_path / "config" / "sim.example.toml").write_text(EXAMPLE, encoding="utf-8")
    return tmp_path


def _layout(tmp_path: Path) -> SitlLayout:
    repo = tmp_path / "repo"
    for sub, name, text in (
        ("params", "searchdrone-safety.parm", ""),
        ("params", "sd_quad14-sim.parm", ""),
        ("frames", "sd_quad14.json", "{}"),
    ):
        folder = repo / "simulation" / "sitl" / sub
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(text)
    ap = tmp_path / "ardupilot"
    return SitlLayout(arducopter=ap / "arducopter", ardupilot_home=ap, repo=repo)


def test_the_setting() -> None:
    assert config.RANGEFINDERS == rangefinder.MODELS
    assert set(rangefinder.PROFILES) == set(rangefinder.MODELS) - {"none"}


def test_the_setting_is_read_and_checked(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    assert config.load({}, repo=repo, environ={}).rangefinder == "none"
    cfg = config.load({"rangefinder": "grf250", "physics": "builtin"}, repo=repo, environ={})
    assert cfg.rangefinder == "grf250"
    env = {"SD_SIM__RANGEFINDER": "tf02"}
    assert config.load({}, repo=repo, environ=env).rangefinder == "tf02"
    with pytest.raises(config.ConfigError):
        config.load({"rangefinder": "lidar"}, repo=repo, environ={})
    with pytest.raises(config.ConfigError, match="ArduPilot"):
        config.load(
            {"rangefinder": "tf02", "physics": "mock"}, repo=repo, environ={}, allow_mock=True
        )
    assert "rangefinder" in launch.ALLOWED_KEYS


def test_sitl_gets_serial6_only_with_a_laser(tmp_path: Path) -> None:
    p = ports.for_instance(1)
    plain = build_sitl_argv(SitlOptions(instance=1), p, _layout(tmp_path))
    assert "--serial6" not in plain
    argv = build_sitl_argv(SitlOptions(instance=1, rangefinder=True), p, _layout(tmp_path))
    assert argv[-2:] == ["--serial6", f"udpclient:127.0.0.1:{p['rangefinder_mavlink']}"]
    assert p["rangefinder_mavlink"] == 14575


def _stack(tmp_path: Path, model: str, **kwargs: object) -> launch.Stack:
    cfg = SimConfig(instance=3, physics="builtin", rangefinder=model)
    return launch.Stack(cfg, tmp_path / "run", repo=paths.REPO, **kwargs)  # type: ignore[arg-type]


def _paths(tmp_path: Path) -> services.StackPaths:
    return services.StackPaths(
        repo=paths.REPO,
        run_dir=tmp_path / "run",
        secrets_dir=tmp_path / "secrets",
        data_dir=tmp_path / "data",
        onboard_dir=tmp_path / "onboard",
    )


def test_the_launcher_starts_the_laser_with_the_runs_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand = TreeStand("stand", 80.0, 80.0, 15.0, 24.0)
    st = _stack(tmp_path, "tf02", obstacles=[stand], rangefinder_faults={"rf_spikes": 0.1})
    monkeypatch.setattr(st, "inst_dir", tmp_path / "inst")
    started: dict[str, list[str]] = {}
    monkeypatch.setattr(st.group, "start", lambda name, argv, **k: started.setdefault(name, argv))
    monkeypatch.setattr(st, "_wait", lambda *a, **k: None)
    st.start_rangefinder()
    argv = started["rangefinder"]
    assert argv[argv.index("--model") + 1] == "tf02"
    assert json.loads(argv[argv.index("--faults") + 1]) == {"rf_spikes": 0.1}
    scene_dir = Path(argv[argv.index("--scene") + 1])
    assert (scene_dir / "scene.json").is_file()
    assert st.rangefinder_layer.read_text() == rangefinder.param_layer("tf02", False)
    assert str(st.rangefinder_layer) in st.sitl_layout().extra_defaults[-1].as_posix()
    assert "--serial6" in st.sitl_argv()
    st.set_rangefinder_fault("stuck")
    assert json.loads(st.rangefinder_control.read_text()) == {"fault": "stuck"}
    with pytest.raises(ValueError):
        st.set_rangefinder_fault("melted")
    bad = _stack(tmp_path, "tf02", rangefinder_faults={"rf_typo": 1})
    monkeypatch.setattr(bad, "inst_dir", tmp_path / "inst2")
    with pytest.raises(ValueError):
        bad.start_rangefinder()


def test_sd_onboard_gets_the_laser_and_accepts_its_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    st = _stack(tmp_path, "grf250")
    monkeypatch.setattr(st, "inst_dir", tmp_path / "inst")
    monkeypatch.setattr(st, "stack_paths", lambda: _paths(tmp_path))
    monkeypatch.setattr(st, "_wait", lambda *a, **k: None)
    started: dict[str, dict[str, str]] = {}
    monkeypatch.setattr(st, "_start_service", lambda name, argv, env: started.setdefault(name, env))
    rangefinder.write_param_layer(st.inst_dir, "grf250", False)
    st.start_onboard()
    env = started["sd-onboard"]
    loaded = onboard_config.load(paths.REPO, {k: v for k, v in env.items() if k.startswith("SD_")})
    rf = loaded.config.rangefinder
    assert (rf.model, rf.min_tested_m, rf.max_tested_m) == ("other", 10.0, 250.0)
    files = loaded.param_files()
    assert files[-1] == st.rangefinder_layer and files[0].name == "searchdrone-safety.parm"
    params = effective_params(files)
    assert onboard_config.rangefinder_param_problems(rf, params) == []
    assert params["FS_GCS_ENABLE"] > 0  # the safety layer is still read back


def test_no_laser_no_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    st = _stack(tmp_path, "none")
    monkeypatch.setattr(st, "stack_paths", lambda: _paths(tmp_path))
    monkeypatch.setattr(st, "_wait", lambda *a, **k: None)
    started: dict[str, dict[str, str]] = {}
    monkeypatch.setattr(st, "_start_service", lambda name, argv, env: started.setdefault(name, env))
    st.start_onboard()
    assert not any("RANGEFINDER" in k or "PARAM_FILES" in k for k in started["sd-onboard"])
    assert "--serial6" not in st.sitl_argv()


def test_the_runs_map_pack_lands_where_the_drone_looks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    line = PowerLine("line", (100.0, -50.0), (100.0, 50.0), 18.0)
    st = _stack(tmp_path, "none", obstacles=[line])
    monkeypatch.setattr(st, "stack_paths", lambda: _paths(tmp_path))
    pack = st.write_floor_pack("sim-run", MapErrors(holes=((0.0, 50.0, 3.0),)))
    assert load_floor_pack(tmp_path / "data" / "floor" / "sim-run").sha256 == pack.sha256
    assert len(pack.lines) == 1
    bad = launch.Stack(SimConfig(world="sd_nowhere"), tmp_path / "r2")
    with pytest.raises(InfraError):
        bad.obstacle_scene()
