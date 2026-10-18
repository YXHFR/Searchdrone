"""The sd_fpv7 model against the CAD model it is generated from (``cad.json``), its
builtin-physics frame, its parameter layer and the keys that pick it. No simulator, and no
CAD kernel: ``tools/sim_model_from_cad.py --check`` (CI job hw-cad) keeps ``cad.json`` and
the meshes equal to the CAD model; these tests keep the simulator equal to ``cad.json``."""

from __future__ import annotations

import importlib.util
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType

import pytest

from sd_sim import config, fpv7, paths, ports, smoke, world
from sd_sim.params import parse_parm
from sd_sim.sitl import CMAC_HOME, SitlLayout, SitlOptions, build_sitl_argv

PARAMS = paths.REPO / "simulation" / "sitl" / "params"
KNOWN_KEYS = paths.REPO / "simulation" / "sitl" / "frames" / "known_keys.txt"
KNOWN_PARAMS = PARAMS / "known_params_Copter-4.7.1.txt"
MAX_FILE_BYTES = 1024 * 1024  # pre-commit's check-added-large-files
INERTIA_KEYS = ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")


def load_tool(name: str) -> ModuleType:
    path = paths.REPO / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tools_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model() -> ET.Element:
    root = ET.fromstring(world.model_sdf(0, vehicle=fpv7.MODEL_NAME))  # noqa: S314 - ours
    found = root.find("model")
    assert found is not None
    return found


def floats(text: str | None) -> list[float]:
    return [float(v) for v in (text or "").split()]


def sim_layer() -> dict[str, float]:
    path = PARAMS / "sd_fpv7-sim.parm"
    return parse_parm(path.read_text(encoding="utf-8"), path.name)


# --- the Gazebo model against cad.json ----------------------------------------------------


def test_links_add_up_to_the_cad_mass_centre_and_inertia() -> None:
    """Every link's mass, centre and inertia from the rendered SDF, combined about the
    common centre of mass, give the CAD model's totals."""
    cad = fpv7.cad()
    masses, centres, inertias = [], [], []
    for link in model().iter("link"):
        link_pose = floats(link.findtext("pose")) or [0.0] * 6
        inertial = link.find("inertial")
        assert inertial is not None
        com = floats(inertial.findtext("pose")) or [0.0] * 6
        assert link_pose[3:] == [0, 0, 0] and com[3:] == [0, 0, 0]
        i = {k: float(inertial.findtext(f"inertia/{k}") or "nan") for k in INERTIA_KEYS}
        masses.append(float(inertial.findtext("mass") or "nan"))
        centres.append([a + b for a, b in zip(link_pose[:3], com[:3], strict=True)])
        inertias.append(
            [
                [i["ixx"], i["ixy"], i["ixz"]],
                [i["ixy"], i["iyy"], i["iyz"]],
                [i["ixz"], i["iyz"], i["izz"]],
            ]
        )
    total = sum(masses)
    cog = [sum(m * c[k] for m, c in zip(masses, centres, strict=True)) / total for k in range(3)]
    combined = [[0.0] * 3 for _ in range(3)]
    for m, c, inertia in zip(masses, centres, inertias, strict=True):
        d = [c[k] - cog[k] for k in range(3)]
        dd = sum(v * v for v in d)
        for r in range(3):
            for s in range(3):
                combined[r][s] += inertia[r][s] + m * ((dd if r == s else 0.0) - d[r] * d[s])
    assert total == pytest.approx(cad["mass_kg"], abs=1e-6)
    assert cog == pytest.approx(cad["cog_m"], abs=1e-6)
    for row, want in zip(combined, cad["inertia_kg_m2"], strict=True):
        assert row == pytest.approx(want, abs=1e-8)


def test_cad_values_are_a_seven_inch_quad() -> None:
    cad = fpv7.cad()
    assert 1.4 <= cad["mass_kg"] <= 1.8  # about 1.5 kg all up
    assert cad["prop"] == pytest.approx(
        {"diameter_m": 7 * 0.0254, "pitch_in": 4, "blades": 3, "mass_kg": cad["prop"]["mass_kg"]}
    )
    assert cad["source"].startswith("hardware/cad/fpv7")


def test_stands_on_its_legs() -> None:
    assert floats(model().findtext("pose"))[2] == pytest.approx(fpv7.cad()["stand_m"])
    assert world.stand_m(fpv7.MODEL_NAME) == fpv7.cad()["stand_m"]
    assert world.stand_m(world.MODEL_NAME) == world.STAND_M


def test_rotors_follow_ardupilot_quad_x_with_the_cad_directions() -> None:
    """Motor 1 front right and motor 2 rear left turn counter-clockwise, motors 3 front
    left and 4 rear right clockwise, at the CAD rotor positions."""
    cad_rotors = {tuple(r["position_m"]): r["clockwise"] for r in fpv7.cad()["rotors"]}
    found = model()
    for i, (sx, sy, ccw) in enumerate(fpv7.AP_QUAD_X):
        link = found.find(f"link[@name='rotor_{i}']")
        assert link is not None
        x, y, z, *_ = floats(link.findtext("pose"))
        assert (math.copysign(1, x), math.copysign(1, y)) == (sx, sy)
        assert cad_rotors[(x, y, z)] is not ccw
    multipliers = [float(c.findtext("multiplier") or "nan") for c in found.iter("control")]
    assert [m > 0 for m in multipliers] == [ccw for _, _, ccw in fpv7.AP_QUAD_X]
    assert {abs(m) for m in multipliers} == {fpv7.MAX_ROTOR_RAD_S}


def test_lift_surfaces_lift_the_cad_mass() -> None:
    """Two LiftDrag halves per rotor; at full speed the four rotors lift THRUST_TO_WEIGHT
    times the CAD weight (lift = 0.5 rho A cl v^2 per half, cl from a0 and CLA)."""
    found = model()
    plugins = [p for p in found.iter("plugin") if p.get("name") == "gz::sim::systems::LiftDrag"]
    assert len(plugins) == 8
    p = plugins[0]
    area, cp = float(p.findtext("area") or "nan"), abs(floats(p.findtext("cp"))[0])
    rho, a0, cla = (float(p.findtext(k) or "nan") for k in ("air_density", "a0", "cla"))
    lift = 0.5 * rho * area * cla * a0 * (fpv7.MAX_ROTOR_RAD_S * cp) ** 2
    weight = fpv7.cad()["mass_kg"] * 9.80665
    assert 8 * lift == pytest.approx(fpv7.THRUST_TO_WEIGHT * weight, rel=0.02)
    assert cp < fpv7.prop_radius_m()


def test_meshes_are_the_cad_meshes_and_small() -> None:
    uris = [u.text or "" for u in model().iter("uri")]
    cad = fpv7.cad()["meshes"]
    want = {f"model://{fpv7.MODEL_NAME}/{m}" for m in cad["body"]}
    want |= {f"model://{fpv7.MODEL_NAME}/{cad[k]}" for k in ("prop_cw", "prop_ccw")}
    assert set(uris) == want
    for uri in want:
        path = fpv7.MODEL_DIR / uri.removeprefix(f"model://{fpv7.MODEL_NAME}/")
        assert path.is_file() and path.stat().st_size < MAX_FILE_BYTES
        assert path.with_suffix(".mtl").is_file()


def test_camera_where_the_cad_camera_sits() -> None:
    link = model().find("link[@name='camera_link']")
    assert link is not None
    assert floats(link.findtext("pose"))[:3] == pytest.approx(fpv7.cad()["camera_m"])
    hfov = float(link.findtext("sensor/camera/horizontal_fov") or "nan")
    assert hfov == pytest.approx(math.radians(fpv7.cad()["camera_hfov_deg"]))
    assert link.findtext("sensor/update_rate") == str(fpv7.CAMERA["fps"])


def test_world_includes_the_chosen_model(tmp_path: Path) -> None:
    path = world.render_world(world.WORLD_NAME, 0, tmp_path, vehicle=fpv7.MODEL_NAME)
    root = ET.parse(path).getroot()  # noqa: S314 - our own rendered file
    assert [u.text for u in root.iter("uri")] == ["model://sd_fpv7_i0"]
    rendered = tmp_path / "build" / "sim" / "models" / "sd_fpv7_i0"
    assert (rendered / "model.sdf").is_file() and (rendered / "model.config").is_file()
    with pytest.raises(ValueError):
        world.model_context(0, vehicle="x8")


# --- builtin physics and parameters --------------------------------------------------------


def test_frame_file_matches_the_cad_model() -> None:
    frame = fpv7.frame_json()
    known = {
        line.strip()
        for line in KNOWN_KEYS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert set(frame) <= known
    cad = fpv7.cad()
    assert frame["mass"] == cad["mass_kg"]
    assert frame["moment_inertia"] == [cad["inertia_kg_m2"][i][i] for i in range(3)]
    x, y, _ = cad["rotors"][0]["position_m"]
    assert frame["diagonal_size"] == pytest.approx(2 * math.hypot(x, y), abs=1e-4)
    assert 15.0 <= frame["refCurrent"] <= 16.0
    assert frame["maxVoltage"] == pytest.approx(6 * 4.2)
    assert frame["num_motors"] == 4


def test_parameter_layer_has_the_seven_inch_values() -> None:
    layer = sim_layer()
    frame = fpv7.frame_json()
    assert (layer["FRAME_CLASS"], layer["FRAME_TYPE"]) == (1, 1)  # quad X
    assert layer["MOT_SPIN_MAX"] == 0.90 == frame["spin_max"]
    assert layer["ATC_ANGLE_MAX"] == 35
    assert layer["BATT_CAPACITY"] == 10000 == frame["battCapacityAh"] * 1000
    assert layer["SIM_BATT_VOLTAGE"] == frame["maxVoltage"]
    assert layer["MOT_THST_HOVER"] == frame["hoverThrOut"]
    known = set(KNOWN_PARAMS.read_text(encoding="utf-8").split())
    assert set(layer) <= known


def test_params_check_keeps_the_reserve_shares() -> None:
    check = load_tool("params_check")
    assert check.main([]) == 0
    safety = check.SAFETY.read_text(encoding="utf-8")
    good = "BATT_CAPACITY 10000\nBATT_LOW_MAH 1500\nBATT_CRT_MAH 700\n"
    assert check.battery_problems(good, safety, "x.parm") == []
    low = "BATT_CAPACITY 10000\nBATT_LOW_MAH 1000\nBATT_CRT_MAH 700\n"
    assert [p.split(":")[1].split()[0] for p in check.battery_problems(low, safety, "x")] == [
        "BATT_LOW_MAH"
    ]
    assert check.battery_problems("BATT_CAPACITY 20000\n", safety, "x")
    # sd_quad14's layer may not set what sd_fpv7's may.
    assert "ATC_ANGLE_MAX" in check.allowlist(vehicle="sd_fpv7")
    assert "ATC_ANGLE_MAX" not in check.allowlist()
    assert check.sim_layer_problems("ATC_ANGLE_MAX 35\n", check.allowlist())


def fake_layout(tmp_path: Path) -> SitlLayout:
    repo = tmp_path / "repo"
    params = repo / "simulation" / "sitl" / "params"
    params.mkdir(parents=True)
    for name in ("searchdrone-safety.parm", "sd_quad14-sim.parm", "sd_fpv7-sim.parm"):
        (params / name).write_text("")
    ap = tmp_path / "ardupilot"
    return SitlLayout(arducopter=ap / "arducopter", ardupilot_home=ap, repo=repo)


@pytest.mark.parametrize("physics", ["builtin", "gz"])
def test_sitl_loads_the_fpv7_layer_last(tmp_path: Path, physics: str) -> None:
    layout = fake_layout(tmp_path)
    opts = SitlOptions(physics=physics, vehicle=fpv7.MODEL_NAME)
    argv = build_sitl_argv(opts, ports.for_instance(0), layout)
    defaults = [Path(p).name for p in argv[argv.index("--defaults") + 1].split(",")]
    assert defaults[-2:] == ["searchdrone-safety.parm", "sd_fpv7-sim.parm"]
    model_arg = argv[argv.index("--model") + 1]
    assert model_arg == ("x:sd_fpv7.json" if physics == "builtin" else "JSON")


def test_frame_is_written_from_the_cad_model(tmp_path: Path) -> None:
    staged = fake_layout(tmp_path).stage_frame(tmp_path / "work", fpv7.MODEL_NAME)
    assert staged.name == "sd_fpv7.json"
    assert json.loads(staged.read_text(encoding="utf-8")) == fpv7.frame_json()


# --- picking the model ---------------------------------------------------------------------


@pytest.mark.parametrize("given", ["fpv7", "sd_fpv7"])
def test_vehicle_key_and_alias(given: str) -> None:
    assert config.load({"vehicle": given}, environ={}).vehicle == "sd_fpv7"
    assert config.load({}, environ={}).vehicle == "sd_quad14"  # the default stays
    with pytest.raises(config.ConfigError):
        config.load({"vehicle": "x8"}, environ={})


def _cfg(world_name: str) -> config.SimConfig:
    cfg = config.SimConfig()
    cfg.world, cfg.physics, cfg.gui = world_name, "gz", False
    return cfg


def test_smoke_flies_fpv7_at_cmac_or_in_the_site_world() -> None:
    cfg = smoke.harden(_cfg("iris_runway"), "fpv7")
    assert (cfg.vehicle, cfg.world, cfg.home) == ("sd_fpv7", config.TEST_WORLD, CMAC_HOME)
    cfg = smoke.harden(_cfg("sd_hellasgarden"), "fpv7")
    assert (cfg.vehicle, cfg.world) == ("sd_fpv7", "sd_hellasgarden")
    assert cfg.home != CMAC_HOME
    assert smoke.harden(_cfg("sd_lake_forest")).vehicle == "stock"  # unchanged without it
