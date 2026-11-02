"""The tele camera (tele=true): a Raspberry Pi HQ Camera with a 20-degree M12 lens beside
the stills camera, looking straight down like it, with its own capture process and
folder, read by sd-onboard's perception as a second camera. Off by default."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from sd_sim import cameras, config, fpv7, launch, objects, ports, services, world
from sd_sim.config import ConfigError, SimConfig
from sd_sim.errors import InfraError

PERSON_M = 0.5  # a standing person seen from above (head and shoulders)


class _EmptyScene:
    def objects(self) -> Sequence[objects.WorldObject]:
        return []


def _paths(tmp_path: Path) -> services.StackPaths:
    return services.StackPaths(
        repo=tmp_path / "repo",
        run_dir=tmp_path / "repo/runs/r",
        secrets_dir=tmp_path / "repo/build/sim/i1/secrets",
        data_dir=tmp_path / "repo/runs/r/data",
        onboard_dir=tmp_path / "data/sim/i1/onboard",
    )


def _person_px(cam: cameras.StillsCamera, height_m: float) -> float:
    focal_px = (cam.width / 2) / math.tan(math.radians(cam.hfov_deg) / 2)
    return PERSON_M * focal_px / height_m


def _sensors(sdf: str) -> dict[str, ET.Element]:
    root = ET.fromstring(sdf)  # noqa: S314 - our own rendered file
    link = root.find("model/link[@name='camera_link']")
    assert link is not None
    return {s.get("name", ""): s for s in link.iter("sensor")}


def test_the_tele_camera_is_the_hq_camera_behind_a_20_degree_lens() -> None:
    tele = cameras.tele()
    assert tele.hfov_deg == 20.0
    sensor_w, sensor_h = cameras.TELE_SENSOR_PX
    assert (sensor_w, sensor_h) == (4056, 3040)
    assert tele.width / tele.height == pytest.approx(sensor_w / sensor_h, rel=0.005)
    # Rendered at the main camera's scale (1280 of Camera Module 3's 4608 px), so the zoom
    # between them is the real one: 4056 px / 20 deg against 4608 px / 66 deg, about 2.9x.
    real = (sensor_w / 20.0) / (4608 / 66.0)
    assert tele.px_per_deg / world.MAIN_STILLS.px_per_deg == pytest.approx(real, rel=0.01)


@pytest.mark.parametrize("height_m", [40.0, 60.0])
def test_a_person_is_about_three_times_larger_in_the_tele_still(height_m: float) -> None:
    main, tele = world.MAIN_STILLS, cameras.tele()
    ratio = _person_px(tele, height_m) / _person_px(main, height_m)
    assert ratio == pytest.approx(3.24, abs=0.05)  # 1128/tan(10) against 1280/tan(33)
    if height_m == 40.0:  # the scripted search height: tiny in the main still, clear in tele
        assert _person_px(main, height_m) < 16 <= 32 <= _person_px(tele, height_m)


def test_the_model_has_no_tele_camera_by_default() -> None:
    sensors = _sensors(world.model_sdf(0, stills=True, video=False))
    assert {"stills", "boxes"} <= set(sensors)
    assert not {"tele", "tele_boxes"} & set(sensors)
    assert [c["sensor"] for c in world.model_context(0, stills=True)["stills"]] == ["stills"]
    assert world.model_context(0)["stills"] == []


def test_the_tele_camera_sits_beside_the_stills_camera_and_looks_down() -> None:
    tele = cameras.tele(1000, 750, 14.0)
    sensors = _sensors(world.model_sdf(3, stills=True, video=False, tele=tele))
    main = sensors["stills"]
    for name, kind in (("tele", "camera"), ("tele_boxes", "boundingbox_camera")):
        s = sensors[name]
        assert s.get("type") == kind
        assert s.findtext("topic") == f"/sd_i3/{name}"
        assert s.findtext("pose") == main.findtext("pose") == "0 0 0 0 90 0"  # straight down
        assert s.findtext("camera/image/width") == "1000"
        assert s.findtext("camera/image/height") == "750"
        hfov = float(s.findtext("camera/horizontal_fov") or 0)
        assert hfov == pytest.approx(math.radians(14.0), abs=1e-6)
    assert sensors["tele_boxes"].findtext("camera/box_type") == "2d"
    assert main.findtext("camera/image/width") == "1280"  # the stills camera as before
    assert main.findtext("topic") == "/sd_i3/stills"


def test_the_tele_camera_needs_the_stills_camera_and_sd_quad14() -> None:
    with pytest.raises(ValueError, match="kräver"):
        world.model_context(0, tele=cameras.tele())
    with pytest.raises(ValueError, match="sd_fpv7"):
        world.model_context(0, vehicle=fpv7.MODEL_NAME, tele=cameras.tele())
    clash = cameras.StillsCamera("stills", "tele_boxes", 640, 480, 20.0)
    with pytest.raises(ValueError, match="egna sensornamn"):
        world.model_context(0, stills=True, tele=clash)
    for bad in ({"width": 8}, {"hfov_deg": 0.5}, {"hfov_deg": 150.0}):
        with pytest.raises(ValueError):
            cameras.tele(**bad)  # pyright: ignore[reportArgumentType]


def test_tele_is_off_unless_switched_on() -> None:
    cfg = config.load({})
    assert not cfg.tele and cfg.tele_camera() is None
    assert {"tele", "tele_hfov_deg", "tele_width", "tele_height"} <= set(launch.ALLOWED_KEYS)


def test_tele_settings_choose_the_camera() -> None:
    cfg = config.load({"tele": "true", "physics": "gz", "tele_hfov_deg": "14"})
    assert cfg.tele_camera() == cameras.tele(1128, 846, 14.0)
    env = {"SD_SIM__TELE": "1", "SD_SIM__TELE_WIDTH": "4056", "SD_SIM__TELE_HEIGHT": "3040"}
    full = config.load({"physics": "gz"}, environ=env)
    assert full.tele_camera() == cameras.tele(4056, 3040, 20.0)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"tele": "true", "physics": "builtin"}, "physics=gz"),
        ({"tele": "true", "physics": "gz", "vehicle": "fpv7"}, "sd_quad14"),
        ({"tele": "true", "physics": "gz", "tele_width": "wide"}, "heltal"),
        ({"tele": "true", "physics": "gz", "tele_hfov_deg": "200"}, "synfältet"),
        ({"tele": "maybe"}, "true eller false"),
    ],
)
def test_bad_tele_settings_stop_the_start(args: dict[str, str], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        config.load(args)


def test_without_a_scene_the_run_has_no_tele_camera(tmp_path: Path) -> None:
    cfg = SimConfig(tele=True)
    assert launch.Stack(cfg, tmp_path / "run").tele is None
    assert launch.Stack(cfg, tmp_path / "run", scene=_EmptyScene()).tele == cameras.tele()


def test_the_tele_camera_gets_its_own_capture_and_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sd_sim import gazebo

    st = launch.Stack(
        SimConfig(instance=2, tele=True),
        tmp_path / "run",
        scene=_EmptyScene(),
        stills_archive=tmp_path / "archive",
    )
    st.world = gazebo.World("sd_lake_forest", tmp_path, "sd_quad14")
    argv = st.stills_capture_argv(tele=True)
    assert argv[argv.index("--image-topic") + 1] == "/sd_i2/tele"
    assert argv[argv.index("--boxes-topic") + 1] == "/sd_i2/tele_boxes"
    assert argv[argv.index("--out") + 1] == str(tmp_path / "run" / "stills_tele")
    assert "--archive" not in argv  # the farm's archive keeps the stills camera only
    main = st.stills_capture_argv()
    assert main[main.index("--image-topic") + 1] == "/sd_i2/stills"
    assert "--archive" in main
    started: list[str] = []
    monkeypatch.setattr(st.group, "start", lambda name, *_a, **_k: started.append(name))
    st.start_stills_capture({})
    assert started == ["stills", "stills_tele"]
    assert (tmp_path / "run" / "stills_tele").is_dir()
    plain = launch.Stack(SimConfig(instance=2), tmp_path / "run2", scene=_EmptyScene())
    with pytest.raises(InfraError, match="tele=true"):
        plain.stills_capture_argv(tele=True)


class _RenderedError(Exception):
    pass


def test_the_world_is_rendered_with_the_tele_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = SimConfig(physics="gz", world=world.WORLD_NAME, vehicle=world.MODEL_NAME, tele=True)
    st = launch.Stack(cfg, tmp_path / "run", scene=_EmptyScene())
    seen: dict[str, Any] = {}

    def render(*_args: object, **kw: object) -> None:
        seen.update(kw)
        raise _RenderedError

    monkeypatch.setattr(world, "render_world", render)
    with pytest.raises(_RenderedError):
        st.start_gazebo()
    assert seen["stills"] is True and seen["tele"] == cameras.tele()


def test_sd_onboard_reads_the_tele_stills_as_a_second_camera(tmp_path: Path) -> None:
    p = ports.for_instance(1)
    tele_dir = tmp_path / "stills_tele"
    env = services.onboard_env(
        p, _paths(tmp_path), still_dir=tmp_path / "stills", tele=(cameras.tele(), tele_dir)
    )
    assert env["SD_ONBOARD__PERCEPTION__STILL_DIR"] == str(tmp_path / "stills")
    assert env["SD_ONBOARD__PERCEPTION__HFOV_DEG"] == "66.0"  # the main camera as before
    assert env["SD_ONBOARD__PERCEPTION__TELE__ENABLED"] == "true"
    assert env["SD_ONBOARD__PERCEPTION__TELE__STILL_DIR"] == str(tele_dir)
    assert env["SD_ONBOARD__PERCEPTION__TELE__WIDTH_PX"] == "1128"
    assert env["SD_ONBOARD__PERCEPTION__TELE__HEIGHT_PX"] == "846"
    assert env["SD_ONBOARD__PERCEPTION__TELE__HFOV_DEG"] == "20.0"
    plain = services.onboard_env(p, _paths(tmp_path), still_dir=tmp_path / "stills")
    assert not any("TELE" in k for k in plain)
    with pytest.raises(ValueError, match="still_dir"):
        services.onboard_env(p, _paths(tmp_path), tele=(cameras.tele(), tele_dir))


def test_the_launcher_hands_sd_onboard_the_tele_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    st = launch.Stack(SimConfig(instance=1, tele=True), tmp_path / "run", scene=_EmptyScene())
    started: dict[str, dict[str, str]] = {}
    monkeypatch.setattr(st, "stack_paths", lambda: _paths(tmp_path))
    monkeypatch.setattr(st, "_wait", lambda *a, **k: None)
    monkeypatch.setattr(st, "_start_service", lambda name, argv, env: started.setdefault(name, env))
    st.start_onboard()
    env = started["sd-onboard"]
    assert env["SD_ONBOARD__PERCEPTION__TELE__STILL_DIR"] == str(tmp_path / "run" / "stills_tele")
