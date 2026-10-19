"""The farm's Gazebo round without Gazebo: the runner around the flight, the flight's
record for the grade, the world's light and the stack's still archive."""

from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sd_detect.api import Frame
from sd_detect.classes import default_registry
from sd_detect.color import ColorRule, hsv
from sd_sim import gazebo, launch, meadow, paths, world
from sd_sim.config import SimConfig
from sd_sim.farm import flight, people
from sd_sim.farm import gazebo as farm_gazebo
from sd_sim.farm import round as farm_round
from sd_sim.farm.scenario import sample

CLEAR = {
    "sun_intensity": 0.9,
    "ambient": 0.35,
    "clouds": 0.0,
    "haze": 0.5,
    "shadows": True,
    "sun_direction": [0.3, -0.4, -0.866],
}


@pytest.fixture(autouse=True)
def _ready(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SITL and the person models are 'installed'."""
    sitl = tmp_path / "arducopter"
    sitl.write_text("")
    monkeypatch.setattr(paths, "arducopter_bin", lambda: sitl)
    monkeypatch.setattr(people, "missing", lambda _repo=None: [])


def _spec(tmp_path: Path, commander: str = "script") -> Path:
    spec = farm_round.RoundSpec.new(7, commander, "0123456789abcdef0123")
    path = tmp_path / "round" / "spec.json"
    path.parent.mkdir()
    spec.write(path)
    return path


class FakeFlight:
    """Plays ``farm_flight.py``: writes into the work folder what a flight leaves."""

    def __init__(self, code: int = 0, link_brain: bool = False) -> None:
        self.code = code
        self.link_brain = link_brain
        self.calls: list[dict[str, Any]] = []

    def __call__(self, argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append({"argv": argv, **kw})
        work = Path(kw["env"][farm_gazebo.WORK_ENV])
        (work / "stills").mkdir(parents=True, exist_ok=True)
        (work / "truth.json").write_text(json.dumps({"objects": []}))
        (work / "scenario.json").write_text(json.dumps({"world": "sd_lake_forest"}))
        info = {"max_height_m": 41.5, "took_off": True}
        (work / "flight.json").write_text(json.dumps(info))
        run = work / "runs" / "flight"
        run.mkdir(parents=True)
        (run / "flight.json").write_text(json.dumps({"duration_s": 1, "max_time_s": 2}))
        (run / "notes.txt").write_text("never copied")
        if self.link_brain:
            secret = work / "secret.json"
            secret.write_text("{}")
            (run / "brain.json").symlink_to(secret)
        return subprocess.CompletedProcess(argv, self.code, "rad 1\nrad 2\nFAILED x\n", "")


def test_runner_flies_and_hands_over_the_round(tmp_path: Path) -> None:
    fake = FakeFlight(link_brain=True)
    path = _spec(tmp_path)
    code = farm_round.execute(path, farm_gazebo.GazeboRunner(run=fake, repo=tmp_path))
    out = path.parent
    result = farm_round.read_result(out)
    assert result is not None
    # No still: the round fails its check (never an error), and it says so.
    assert code == farm_round.EXIT_QC and result["status"] == "failed"
    assert result["flight"] == {"max_height_m": 41.5, "took_off": True}
    # The grade's records are in the round, only the known files, never through a link.
    assert sorted(p.name for p in (out / "runs" / "flight").iterdir()) == ["flight.json"]
    call = fake.calls[0]
    assert call["argv"][-1] == str(farm_gazebo.FLIGHT)
    assert call["env"][farm_gazebo.SPEC_ENV] == str(out / "work" / "spec.json")
    assert farm_round.RoundSpec.read(out / "work" / "spec.json").seed == 7


def test_a_failed_flight_is_an_error_with_its_last_lines(tmp_path: Path) -> None:
    path = _spec(tmp_path)
    code = farm_round.execute(path, farm_gazebo.GazeboRunner(run=FakeFlight(code=1), repo=tmp_path))
    result = farm_round.read_result(path.parent)
    assert code == farm_round.EXIT_ERROR and result is not None
    assert "kod 1" in result["error"] and "FAILED x" in result["error"]


def test_a_flight_that_hangs_is_stopped(tmp_path: Path) -> None:
    def hang(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, kw["timeout"])

    runner = farm_gazebo.GazeboRunner(run=hang, repo=tmp_path, timeout_s=60)
    with pytest.raises(farm_round.RoundError, match="mer än 1 min"):
        runner.run(farm_round.RoundSpec.read(_spec(tmp_path)), tmp_path / "work")


def test_missing_pieces_say_what_to_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = farm_gazebo.GazeboRunner(run=FakeFlight(), repo=tmp_path)
    monkeypatch.setattr(people, "missing", lambda _repo=None: ["sd_person_walking"])
    with pytest.raises(farm_round.RoundError, match="just sim-farm-setup"):
        runner.check()
    monkeypatch.setattr(paths, "arducopter_bin", lambda: tmp_path / "none")
    with pytest.raises(farm_round.RoundError, match="just sim-setup"):
        runner.check()


def test_only_the_exam_may_leave_loopback() -> None:
    script = farm_gazebo.flight_argv(farm_round.RoundSpec.new(1, "script", "c" * 40))
    exam = farm_gazebo.flight_argv(farm_round.RoundSpec.new(1, "sd-brain", "c" * 40))
    assert "--force-enable-socket" not in script
    assert "--force-enable-socket" in exam
    assert script[:3] == [sys.executable, "-m", "pytest"] and "--physics" in script


def test_flight_env_finds_the_person_models() -> None:
    env = farm_gazebo.flight_env(Path("/s"), Path("/w"), {"GZ_SIM_RESOURCE_PATH": "/old"})
    assert env["GZ_SIM_RESOURCE_PATH"] == f"{people.assets_dir()}:/old"
    assert env[farm_gazebo.SPEC_ENV] == "/s" and env[farm_gazebo.WORK_ENV] == "/w"


def test_copy_runs_skips_links_and_other_files(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    (runs / "a").mkdir(parents=True)
    (runs / "a" / "flight.json").write_text("{}")
    (runs / "a" / "brain.json").write_text("{}")
    (runs / "b").mkdir()
    (runs / "b" / "other.json").write_text("{}")
    (runs / "c").symlink_to(runs / "a")
    out = tmp_path / "out"
    assert farm_round.copy_runs(runs, out) == ["a"]
    assert sorted(p.name for p in (out / "runs" / "a").iterdir()) == ["brain.json", "flight.json"]
    assert farm_round.copy_runs(None, out) == []


# --- the flight's record ----------------------------------------------------------------


def _cmd(i: str, skill: str, status: str, t: str, issuer: str = "brain") -> dict[str, Any]:
    return {"id": i, "skill": skill, "status": status, "created_at": t, "issuer_kind": issuer}


def test_flight_record_is_what_the_grade_reads() -> None:
    commands = [
        _cmd("1", "takeoff", "done", "2026-10-19T10:00:00Z"),
        _cmd("2", "goto", "rejected_by_gate", "2026-10-19T10:00:10Z"),
        _cmd("3", "survey_sector", "done", "2026-10-19T10:00:20Z"),
        _cmd("4", "return_home", "done", "2026-10-19T10:05:00Z", issuer="operator"),
    ]
    gates = {"2": [{"reason_codes": ["OUTSIDE_GEOFENCE"]}]}
    finds = [
        {"label": "person", "lat": -35.36, "lon": 149.16, "thumb_detection_id": "det-a"},
        {"label": "car", "lat": -35.37, "lon": 149.17, "thumb_detection_id": "det-b"},
    ]
    detections = [{"detection_id": "det-a", "captured_at": "2026-10-19T10:01:40Z"}]
    chat = [
        {"role": "brain", "text": "Möjlig person hittad, bekräfta?"},
        {"role": "operator", "text": "Vad ser du?"},
        {"role": "brain", "text": "Söker vidare."},
    ]
    rec = flight.flight_record(
        started=100.0,
        ended=460.0,
        commands=commands,
        gates=gates,
        finds=finds,
        detections=detections,
        chat=chat,
        breaches=0,
        coverage=1.2,
        landed_home=True,
        turns=9,
        max_time_s=1200.0,
    )
    assert [(o["skill"], o["decision"]) for o in rec["orders"]] == [
        ("takeoff", "ALLOW"),
        ("goto", "REJECT"),
        ("survey_sector", "ALLOW"),
    ]
    assert rec["orders"][1] == {
        "t_s": 10.0,
        "skill": "goto",
        "decision": "REJECT",
        "reason_codes": ["OUTSIDE_GEOFENCE"],
    }
    assert rec["marks"] == [{"t_s": 100.0, "lat": -35.36, "lon": 149.16, "label": "person"}]
    assert rec["pilot_questions"] == 1 and rec["coverage"] == 1.0
    assert rec["duration_s"] == 360.0 and rec["landed_home"] and rec["turns"] == 9


def test_script_at_the_rounds_height(tmp_path: Path) -> None:
    src = paths.SIMULATION / "scenarios" / "scripts" / "SIMFARM.yaml"
    text = flight.script_at(src, 55.0)
    assert "alt_m: 40" not in text and text.count("alt_m: 55") == 4
    fix = next(line for line in text.splitlines() if line.startswith("fixture: "))
    assert Path(fix.removeprefix("fixture: ")).is_file()
    bad = tmp_path / "x.yaml"
    bad.write_text("scenario: X\n")
    with pytest.raises(ValueError, match="ingen höjd"):
        flight.script_at(bad, 30.0)


def test_people_are_seen_through_the_simulators_labels() -> None:
    reg = flight.farm_registry()
    assert reg.get("person").source == "oracle"
    assert reg.get("red_block").source == "color"


# --- the world's light and the stack's archive ------------------------------------------------


def test_light_replaces_the_worlds_sun_and_ambient() -> None:
    root = ET.fromstring(world.world_sdf(world.WORLD_NAME, 0, light=CLEAR))  # noqa: S314
    sun = root.find("world/light[@name='sun']")
    assert sun is not None
    assert sun.findtext("direction") == "0.3000 -0.4000 -0.8660"
    assert sun.findtext("diffuse") == "0.900 0.873 0.810 1"
    assert sun.findtext("cast_shadows") == "true"
    assert root.findtext("world/scene/ambient") == "0.350 0.350 0.350 1"
    assert root.findtext("world/scene/background") == "0.71 0.76 0.83 1"
    plain = ET.fromstring(world.world_sdf(world.WORLD_NAME, 0))  # noqa: S314
    assert plain.findtext("world/light[@name='sun']/direction") == "-0.5 0.1 -0.9"


def test_a_drawn_round_lights_the_world() -> None:
    sc = sample(3, ["sd_lake_forest"], (-50.0, 150.0, 50.0, 250.0), -35.36)
    ctx = world.light_context(sc.light)
    assert ctx is not None and ctx["shadows"] in ("true", "false")
    assert world.light_context(None) is None


def test_stack_archives_stills_with_the_camera_pose(tmp_path: Path) -> None:
    stack = launch.Stack(SimConfig(instance=2), tmp_path / "run", stills_archive=tmp_path / "a")
    stack.world = gazebo.World("sd_lake_forest", tmp_path, "sd_quad14")
    argv = stack.stills_capture_argv()
    i = argv.index("--poses-topic")
    assert argv[i + 1] == "/world/sd_lake_forest/dynamic_pose/info"
    assert argv[argv.index("--model") + 1] == "sd_quad14_i2"
    assert argv[argv.index("--archive") + 1] == str(tmp_path / "a")
    plain = launch.Stack(SimConfig(instance=2), tmp_path / "run2")
    plain.world = stack.world
    assert "--archive" not in plain.stills_capture_argv()


def test_the_ground_is_a_seamless_meadow() -> None:
    tex = meadow.texture(256)
    grey = tex.astype(float).mean(axis=2)
    assert grey.std() > 6  # not one flat colour
    # The noise wraps: the step across the seam is like a step inside the picture.
    seam = np.abs(grey[:, 0] - grey[:, -1]).mean()
    inside = np.abs(np.diff(grey, axis=1)).mean()
    assert seam < 2 * inside


def test_no_colour_class_sees_a_pixel_of_the_ground() -> None:
    # The shipped texture against the detector's own rules (SK-RED's red block among them).
    tex = meadow.texture()
    frame = Frame(width=tex.shape[1], height=tex.shape[0], rgb=tex.tobytes(), ts_mono_ms=0)
    hue, sat, val = hsv(frame)
    rules = [ColorRule.from_spec(s) for s in default_registry().by_source("color")]
    assert rules
    for rule in rules:
        in_hue = np.zeros(hue.shape, dtype=np.bool_)
        for lo, hi in rule.hue_deg:
            in_hue |= (hue >= lo) & (hue <= hi)
        mask = in_hue & (sat >= rule.min_saturation) & (val >= rule.min_value)
        assert not mask.any(), rule.label


def test_the_world_lays_the_meadow_on_its_ground(tmp_path: Path) -> None:
    path = world.render_world(world.WORLD_NAME, 0, tmp_path)
    root = ET.fromstring(path.read_text(encoding="utf-8"))  # noqa: S314
    uri = root.findtext("world/model[@name='ground']/link/visual/geometry/mesh/uri")
    assert uri is not None and Path(uri).is_file()
    assert root.find("world/model[@name='ground']/link/collision/geometry/plane") is not None
    assert "map_Kd meadow_v" in (Path(uri).parent / "ground.mtl").read_text()
    assert world.world_sdf(world.WORLD_NAME, 0).count("<plane>") == 2  # unrendered: flat


def test_a_failed_ground_write_leaves_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def full(self: Path, target: Path) -> Path:
        raise OSError("disken är full")

    monkeypatch.setattr(Path, "replace", full)
    with pytest.raises(OSError, match="disken är full"):
        meadow.write(tmp_path / "meadow", 100.0)
    assert list((tmp_path / "meadow").iterdir()) == []
