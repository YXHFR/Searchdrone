"""The sd_quad14 model (Gazebo and builtin physics), its parameter layers, the world and
the demo fixture. No simulator needed."""

from __future__ import annotations

import importlib.util
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from sd_hwcalc.calc import compute
from sd_hwcalc.model import load_drone
from sd_sim import fixture, paths, ports, world
from sd_sim.sitl import CMAC_HOME

SIM = paths.REPO / "simulation"
FRAME_JSON = SIM / "sitl" / "frames" / "sd_quad14.json"
KNOWN_KEYS = SIM / "sitl" / "frames" / "known_keys.txt"
GAZEBO = SIM / "gazebo"
REMOTE_URI = re.compile(r"(?i)https?://|fuel\.")
WORLD_CHILDREN = {
    "physics",
    "plugin",
    "light",
    "scene",
    "spherical_coordinates",
    "include",
    "model",
    "gravity",
    "magnetic_field",
    "atmosphere",
    "gui",
}
MASS_TOLERANCE_KG = 0.05


def total_mass_kg() -> float:
    return compute(load_drone(world.DRONE_YAML)).total_mass_kg


def model_root(instance: int = 0) -> ET.Element:
    return ET.fromstring(world.model_sdf(instance))  # noqa: S314 - our own rendered file


def load_tool(name: str) -> ModuleType:
    path = paths.REPO / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tools_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pose(element: ET.Element | None) -> list[float]:
    assert element is not None
    return [float(v) for v in (element.findtext("pose") or "").split()]


# --- the Gazebo model --------------------------------------------------------------------


def test_model_mass_matches_drone_yaml() -> None:
    total = sum(float(m.text or "nan") for m in model_root().iter("mass"))
    assert abs(total - total_mass_kg()) <= MASS_TOLERANCE_KG
    assert abs(total - 2.30) <= MASS_TOLERANCE_KG


def test_model_rotors_x_frame_and_speed() -> None:
    model = model_root().find("model")
    assert model is not None
    rotors = [model.find(f"link[@name='rotor_{i}']") for i in range(4)]
    for rotor in rotors:
        x, y, *_ = pose(rotor)
        assert abs(abs(x) - 0.205) < 1e-3 and abs(abs(y) - 0.205) < 1e-3
        diagonal = 2 * math.hypot(x, y)
        assert abs(diagonal - 0.58) < 2e-3
    plugin = model.find("plugin[@name='ArduPilotPlugin']")
    assert plugin is not None
    multipliers = [float(c.findtext("multiplier") or "0") for c in plugin.iter("control")]
    assert [abs(m) for m in multipliers] == [650.0] * 4
    assert multipliers[0] > 0 and multipliers[2] < 0  # CCW pair, then CW pair


def test_every_rotor_lifts_with_both_blade_halves() -> None:
    # Two lift surfaces per rotor, each blade's forward along its own spin: lift surfaces
    # on the wrong rotor cancel that rotor's lift, and the vehicle cannot take off.
    model = model_root().find("model")
    assert model is not None
    plugin = model.find("plugin[@name='ArduPilotPlugin']")
    assert plugin is not None
    spin = [float(c.findtext("multiplier") or "0") for c in plugin.iter("control")]
    surfaces: dict[str, list[tuple[float, float]]] = {}
    for lift in model.iter("plugin"):
        if lift.get("name") == "gz::sim::systems::LiftDrag":
            cp_x = float((lift.findtext("cp") or "").split()[0])
            forward_y = float((lift.findtext("forward") or "").split()[1])
            surfaces.setdefault(lift.findtext("link_name") or "", []).append((cp_x, forward_y))
    assert sorted(surfaces) == [f"rotor_{i}" for i in range(4)]
    for i, rate in enumerate(spin):
        halves = surfaces[f"rotor_{i}"]
        assert len(halves) == 2
        # The blade moves along +y at +x when the joint spins positive (about +z).
        assert all(cp_x * forward_y * rate > 0 for cp_x, forward_y in halves)


@pytest.mark.parametrize("instance", [0, 3])
def test_model_ports_per_instance(instance: int) -> None:
    model = model_root(instance).find("model")
    assert model is not None and model.get("name") == f"sd_quad14_i{instance}"
    plugin = model.find("plugin[@name='ArduPilotPlugin']")
    assert plugin is not None
    assert plugin.findtext("fdm_addr") == "127.0.0.1"
    assert int(plugin.findtext("fdm_port_in") or 0) == 9002 + 10 * instance
    assert plugin.findtext("lock_step") == "1"
    gst = model.find(".//plugin[@name='GstCameraPlugin']")
    assert gst is not None
    assert gst.findtext("udp_host") == "127.0.0.1"
    assert int(gst.findtext("udp_port") or 0) == ports.for_instance(instance)["video_raw_udp"]
    assert gst.findtext("use_basic_pipeline") == "true"
    assert gst.findtext("use_cuda") == "false"


def test_camera_fixed_straight_down() -> None:
    model = model_root().find("model")
    assert model is not None
    joint = model.find("joint[@name='camera_joint']")
    assert joint is not None and joint.get("type") == "fixed"
    sensor = model.find("link[@name='camera_link']/sensor[@type='camera']")
    assert sensor is not None
    pose_el = sensor.find("pose")
    assert pose_el is not None and pose_el.get("degrees") == "true"
    roll, pitch, yaw = [float(v) for v in (pose_el.text or "").split()[3:]]
    # A Gazebo camera looks along its x axis; +90 degrees pitch points it straight down.
    assert (roll, pitch, yaw) == (0.0, 90.0, 0.0)
    assert sensor.findtext("update_rate") == "15"
    assert sensor.findtext("camera/image/width") == "640"
    assert sensor.findtext("camera/image/height") == "480"
    assert abs(math.degrees(float(sensor.findtext("camera/horizontal_fov") or 0)) - 80) < 0.01
    # No gimbal joints that could tilt it.
    assert not [j for j in model.iter("joint") if "gimbal" in (j.get("name") or "")]


def test_camera_tilt_is_a_stub() -> None:
    from sd_schemas.skills import SKILL_REGISTRY

    spec = next(s for s in SKILL_REGISTRY if s.name == "set_camera_tilt")
    assert spec.status == "stub"


def test_model_config_names_the_instance() -> None:
    config = ET.fromstring(world.model_config(2))  # noqa: S314 - our own rendered file
    assert config.findtext("name") == "sd_quad14_i2"
    assert config.findtext("sdf") == "model.sdf"


def test_render_model_writes_both_files(tmp_path: Path) -> None:
    out = world.render_model(1, tmp_path)
    assert out == tmp_path / "build" / "sim" / "models" / "sd_quad14_i1"
    assert (out / "model.sdf").is_file() and (out / "model.config").is_file()


def test_blade_area_gives_the_thrust_to_weight() -> None:
    mass = total_mass_kg()
    area = world.blade_area_m2(mass)
    v = world.MAX_ROTOR_RAD_S * world.BLADE_CP_M
    lift = 0.5 * world.AIR_DENSITY * area * world.CLA * world.BLADE_A0_RAD * v * v
    assert abs(8 * lift / (mass * 9.80665) - world.THRUST_TO_WEIGHT) < 1e-9


def test_model_needs_a_quad() -> None:
    drone = load_drone(world.DRONE_YAML)
    hexa = drone.model_copy(
        update={
            "motors": drone.motors.model_copy(update={"count": 6}),
            "props": drone.props.model_copy(update={"count": 6}),
        }
    )
    with pytest.raises(ValueError):
        world.model_context(0, hexa)


def test_model_is_self_contained() -> None:
    """The model draws its own shapes: no meshes or models from other packages."""
    assert ET.fromstring(world.model_sdf(0)).find(".//uri") is None  # noqa: S314 - our own rendered file


# --- the builtin-physics model -----------------------------------------------------------


def test_frame_json_keys() -> None:
    known = [
        line.strip()
        for line in KNOWN_KEYS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    patterns = [re.compile("^" + re.escape(k).replace(r"\*", r"[0-9]+") + "$") for k in known]
    frame = json.loads(FRAME_JSON.read_text(encoding="utf-8"))
    unknown = [key for key in frame if not any(p.match(key) for p in patterns)]
    assert unknown == []


def test_frame_json_matches_the_airframe() -> None:
    frame = json.loads(FRAME_JSON.read_text(encoding="utf-8"))
    drone = load_drone(world.DRONE_YAML)
    assert abs(frame["mass"] - total_mass_kg()) <= MASS_TOLERANCE_KG
    assert frame["num_motors"] == drone.motors.count
    assert frame["battCapacityAh"] == drone.battery.capacity_mah / 1000
    assert frame["diagonal_size"] == 0.58
    assert frame["moment_inertia"] == list(world.INERTIA_KG_M2)
    radius = drone.props.diameter_in * 0.0254 / 2
    assert abs(frame["disc_area"] - drone.props.count * math.pi * radius**2) < 0.005


# --- parameter layers --------------------------------------------------------------------


def test_sim_params_clean() -> None:
    check = load_tool("params_check")
    assert check.main([]) == 0
    sim = check.SIM.read_text(encoding="utf-8")
    assert check.sim_layer_problems(sim, check.allowlist()) == []
    assert check.allowlist() == {"MOT_THST_HOVER"}
    bad = "SIM_WIND_SPD 0\nBATT_CAPACITY 10000\nFENCE_ENABLE 0\n"
    found = check.sim_layer_problems(bad, check.allowlist())
    assert len(found) == 2 and "BATT_CAPACITY" in found[0]


def test_override_files_need_a_test_comment_and_safety_names(tmp_path: Path) -> None:
    check = load_tool("params_check")
    good = tmp_path / "good.parm"
    good.write_text("# test: test_sc06\nBATT_LOW_MAH 2000\n")
    bad = tmp_path / "bad.parm"
    bad.write_text("SIM_WIND_SPD 5\n")
    found = check.override_problems([good, bad], {"BATT_LOW_MAH"})
    assert len(found) == 2 and all(f.startswith("bad.parm") for f in found)


def test_shadow_check_flags_unset_safety_names(tmp_path: Path) -> None:
    check = load_tool("params_check")
    layer = tmp_path / "copter.parm"
    layer.write_text("FENCE_RADIUS 150\nFS_THR_ENABLE 1\nCOMPASS_OFS_X 5\nFLTMODE1 7\n")
    found = check.shadow_problems([layer], {"FENCE_RADIUS", "FS_THR_ENABLE"})
    assert found == ["copter.parm sätter FLTMODE1, som searchdrone-safety.parm inte sätter"]


# --- the world ---------------------------------------------------------------------------


def test_world_generates(tmp_path: Path) -> None:
    path = world.render_world(world.WORLD_NAME, 0, tmp_path)
    root = ET.parse(path).getroot()  # noqa: S314 - our own rendered file
    assert root.tag == "sdf" and root.get("version") == "1.9"
    world_el = root.find("world")
    assert world_el is not None and world_el.get("name") == world.WORLD_NAME
    assert {child.tag for child in world_el} <= WORLD_CHILDREN
    assert world_el.findtext("physics/max_step_size") == "0.00125"
    assert world_el.findtext("physics/real_time_factor") == "1.0"
    assert world_el.findtext("scene/shadows") == "false"
    sensors = world_el.find("plugin[@name='gz::sim::systems::Sensors']")
    assert sensors is not None and sensors.findtext("render_engine") == "ogre2"
    # Every model:// in the world exists locally.
    resource_dirs = [GAZEBO / "models", tmp_path / "build" / "sim" / "models"]
    all_uris = [u.text or "" for u in world_el.iter("uri")]
    uris = [u for u in all_uris if u.startswith("model://")]
    assert uris == ["model://sd_quad14_i0"]
    # The rest are the generated ground's files, on this disk.
    assert all(Path(u).is_file() for u in all_uris if u not in uris)
    # The include's pose replaces the model's own: the legs stand on the ground.
    assert pose(world_el.find("include"))[2] == world.STAND_M
    for uri in uris:
        name = uri.removeprefix("model://").split("/")[0]
        assert any((d / name / "model.config").is_file() for d in resource_dirs), uri
    # No remote URIs anywhere: Gazebo would download models and break offline use.
    assert not REMOTE_URI.search(path.read_text(encoding="utf-8"))
    assert (tmp_path / "build" / "sim" / "worlds" / "sd_lake_forest.geojson").is_file()


def test_no_remote_uris_in_templates() -> None:
    files = [p for p in GAZEBO.rglob("*") if p.is_file() and p.suffix != ".py"]
    assert files
    hits = [str(p) for p in files if REMOTE_URI.search(p.read_text(encoding="utf-8"))]
    assert hits == []
    assert not REMOTE_URI.search(world.model_sdf(0))


def test_world_home_is_the_fixture_home() -> None:
    root = ET.fromstring(world.world_sdf(world.WORLD_NAME, 0))  # noqa: S314 - our own rendered file
    coords = root.find("world/spherical_coordinates")
    assert coords is not None
    home = fixture.load().home
    assert float(coords.findtext("latitude_deg") or 0) == home.lat
    assert float(coords.findtext("longitude_deg") or 0) == home.lon
    assert coords.findtext("world_frame_orientation") == "ENU"


def test_world_is_deterministic() -> None:
    assert world.world_sdf(world.WORLD_NAME, 0) == world.world_sdf(world.WORLD_NAME, 0)
    assert world.trees() == world.trees(42)
    assert world.trees(1) != world.trees(42)


def test_forest_band_and_tree_heights() -> None:
    trees = world.trees()
    assert len(trees) == world.FOREST_TREES
    for t in trees:
        assert world.FOREST_X[0] <= t.x <= world.FOREST_X[1]
        assert world.FOREST_Y[0] <= t.y <= world.FOREST_Y[1]
        assert world.TREE_HEIGHT_M[0] <= t.height <= world.TREE_HEIGHT_M[1]


def test_forest_is_outside_every_charter() -> None:
    fix = fixture.load()
    east_edge = world.FOREST_X[1]
    for charter in fix.charters.values():
        west = min(x for x, _ in charter.area_enu) - charter.geofence_buffer_m
        assert west > east_edge, charter.name


def test_geojson_layers() -> None:
    data = world.world_geojson()
    kinds = [f["properties"]["kind"] for f in data["features"]]
    assert kinds == ["lake", "path", "forest"]
    lake = data["features"][0]["geometry"]["coordinates"][0]
    assert lake[0] == lake[-1]
    home = fixture.load().home
    for lon, lat in lake:
        assert abs(lat - home.lat) < 0.01 and abs(lon - home.lon) < 0.01


# --- the fixture -------------------------------------------------------------------------


def _limits() -> dict[str, float]:
    from sd_safety import limits

    return {
        "max_alt_agl_m": limits.ABSOLUTE_MAX_ALT_AGL_M,
        "max_speed_ms": limits.ABSOLUTE_MAX_SPEED_MS,
        "min_battery_rtl_pct": limits.ABSOLUTE_MIN_BATTERY_RTL_PCT,
    }


def _inside(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
            inside = not inside
    return inside


def _distance_to_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> float:
    px, py = point
    best = math.inf
    for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        dx, dy = x1 - x0, y1 - y0
        t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy)))
        best = min(best, math.hypot(px - (x0 + t * dx), py - (y0 + t * dy)))
    return best


def test_fixture_geometry() -> None:
    from sd_safety import limits
    from sd_safety.constants import FENCE_POLYGON_MARGIN_M

    fix = fixture.load(limits=_limits())
    test = fix.charters["charter_test"]
    wide = fix.charters["charter_wide"]
    area = list(test.area_enu)
    pilot = fix.pilot

    def dist(p: tuple[float, float]) -> float:
        return math.hypot(p[0] - pilot[0], p[1] - pilot[1])

    assert _inside(fix.pins["P1"].xy, area)
    assert all(_inside(c, area) for c in fix.survey_box.corners())
    out = fix.pins["OUT"].xy
    assert not _inside(out, area)
    assert _distance_to_polygon(out, area) > test.geofence_buffer_m + FENCE_POLYGON_MARGIN_M
    far = fix.pins["FAR"].xy
    wide_area = list(wide.area_enu)
    assert _inside(far, wide_area) or _distance_to_polygon(far, wide_area) <= wide.geofence_buffer_m
    assert dist(far) > limits.ABSOLUTE_MAX_RANGE_FROM_PILOT_M
    assert all(dist(c) <= 500 for c in [*test.area_enu, *wide.area_enu])
    toofar = fix.charters["charter_toofar"]
    assert any(dist(c) > limits.ABSOLUTE_MAX_RANGE_FROM_PILOT_M for c in toofar.area_enu)
    batt25 = fix.charters["charter_batt25"]
    assert batt25.battery_rtl_pct >= limits.ABSOLUTE_MIN_BATTERY_RTL_PCT + 5
    home = fix.home
    assert (home.lat, home.lon, home.alt_amsl_m, home.heading_deg) == (
        -35.363261,
        149.165230,
        584.0,
        353,
    )


def test_fixture_inheritance_and_overcap() -> None:
    caps = _limits()
    fix = fixture.load(limits=caps)
    base, batt25 = fix.charters["charter_test"], fix.charters["charter_batt25"]
    assert batt25.area_enu == base.area_enu and batt25.battery_rtl_pct == 25
    over = fix.charters["charter_overcap"]
    assert over.exceeds_limits and over.limits_applied
    assert over.alt_max_agl_m == caps["max_alt_agl_m"] + 30
    assert over.max_speed_ms == caps["max_speed_ms"] + 5
    assert over.battery_rtl_pct == caps["min_battery_rtl_pct"] - 5
    plain = fixture.load().charters["charter_overcap"]
    assert plain.exceeds_limits and not plain.limits_applied
    assert plain.alt_max_agl_m == base.alt_max_agl_m


def _raw() -> dict:
    return yaml.safe_load(fixture.DEFAULT_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(extra=1),
        lambda d: d["home"].update(street="x"),
        lambda d: d["pins"][0].update(colour="red"),
        lambda d: d["charters"]["charter_test"].update(max_alt=1),
        lambda d: d["charters"]["charter_batt25"].update(base="nope"),
        lambda d: d["charters"]["charter_batt25"].update(base="charter_toofar"),
        lambda d: d["survey_box"].update(center_pin="P9"),
        lambda d: d["pins"].append(dict(d["pins"][0])),
        lambda d: d["home"].update(lat="south"),
        lambda d: d["charters"]["charter_test"].pop("battery_rtl_pct"),
    ],
)
def test_fixture_errors(change) -> None:
    data = _raw()
    change(data)
    with pytest.raises(fixture.FixtureError):
        fixture.parse(data)


def test_fixture_load_errors(tmp_path: Path) -> None:
    broken = tmp_path / "f.yaml"
    broken.write_text("home: [")
    with pytest.raises(fixture.FixtureError):
        fixture.load(broken)
    with pytest.raises(fixture.FixtureError):
        fixture.load(tmp_path / "missing.yaml")
    with pytest.raises(fixture.FixtureError, match="exceed_limits"):
        fixture.load(limits={"max_alt_agl_m": 120.0})


def test_home_sitl_argument() -> None:
    assert fixture.load().home.sitl_arg() == CMAC_HOME
