"""The SITL command line against snapshots (builtin and gz, instance 0 and 1).

Set SD_UPDATE_SNAPSHOTS=1 to rewrite the snapshots after an intended change.
"""

import itertools
import json
import os
from pathlib import Path

import pytest

from sd_sim import ports
from sd_sim.errors import InfraError
from sd_sim.sitl import CMAC_HOME, FRAME_FILE, SitlLayout, SitlOptions, build_sitl_argv

SNAPSHOTS = Path(__file__).parent / "snapshots"


def fake_layout(tmp_path: Path) -> SitlLayout:
    repo = tmp_path / "repo"
    params = repo / "simulation" / "sitl" / "params"
    frames = repo / "simulation" / "sitl" / "frames"
    params.mkdir(parents=True)
    frames.mkdir(parents=True)
    (params / "searchdrone-safety.parm").write_text("")
    (params / "sd_quad14-sim.parm").write_text("")
    (frames / "sd_quad14.json").write_text("{}")
    ap = tmp_path / "ardupilot"
    return SitlLayout(arducopter=ap / "arducopter", ardupilot_home=ap, repo=repo)


def normalise(argv: list[str], tmp_path: Path) -> list[str]:
    return [
        a.replace(str(tmp_path / "repo"), "<REPO>").replace(
            str(tmp_path / "ardupilot"), "<ARDUPILOT>"
        )
        for a in argv
    ]


@pytest.mark.parametrize("physics", ["builtin", "gz"])
@pytest.mark.parametrize("instance", [0, 1])
def test_sitl_argv(tmp_path: Path, physics: str, instance: int) -> None:
    opts = SitlOptions(instance=instance, physics=physics)
    argv = normalise(
        build_sitl_argv(opts, ports.for_instance(instance), fake_layout(tmp_path)), tmp_path
    )
    snap = SNAPSHOTS / f"sitl_argv_{physics}_i{instance}.json"
    if os.environ.get("SD_UPDATE_SNAPSHOTS") == "1":
        snap.write_text(json.dumps(argv, indent=2) + "\n", encoding="utf-8")
    assert argv == json.loads(snap.read_text(encoding="utf-8"))


def test_serial_ports_are_loopback_udp_and_gps_untouched(tmp_path: Path) -> None:
    argv = build_sitl_argv(SitlOptions(qgc=True), ports.for_instance(0), fake_layout(tmp_path))
    pairs = dict(itertools.pairwise(argv))
    for flag in ("--serial0", "--serial1", "--serial2", "--serial5"):
        assert pairs[flag].startswith("udpclient:127.0.0.1:")
    # serial3/4 are SITL's simulated GPS receivers and serial9 does not exist.
    for flag in ("--serial3", "--serial4", "--serial9"):
        assert flag not in argv
    assert "--home" in argv and pairs["--home"] == CMAC_HOME
    assert "--wipe" in argv


def test_qgc_only_instance_0(tmp_path: Path) -> None:
    argv = build_sitl_argv(
        SitlOptions(instance=1, qgc=True), ports.for_instance(1), fake_layout(tmp_path)
    )
    assert "--serial5" not in argv


def test_stock_vehicle(tmp_path: Path) -> None:
    layout = fake_layout(tmp_path)
    argv = build_sitl_argv(SitlOptions(vehicle="stock"), ports.for_instance(0), layout)
    assert argv[argv.index("--model") + 1] == "quad"
    defaults = argv[argv.index("--defaults") + 1].split(",")
    assert [Path(p).name for p in defaults] == [
        "copter.parm",
        "searchdrone-safety.parm",
        "sd_quad14-sim.parm",
    ]


@pytest.mark.parametrize("missing", ["searchdrone-safety.parm", "sd_quad14-sim.parm"])
def test_missing_parameter_layer_is_an_error(tmp_path: Path, missing: str) -> None:
    layout = fake_layout(tmp_path)
    (layout.repo / "simulation" / "sitl" / "params" / missing).unlink()
    with pytest.raises(InfraError, match=missing):
        build_sitl_argv(SitlOptions(), ports.for_instance(0), layout)


def test_frame_is_staged_into_the_working_directory(tmp_path: Path) -> None:
    layout = fake_layout(tmp_path)
    argv = build_sitl_argv(SitlOptions(), ports.for_instance(0), layout)
    # SITL opens the frame file relative to its working directory.
    assert argv[argv.index("--model") + 1] == f"quad:{FRAME_FILE}"
    staged = layout.stage_frame(tmp_path / "work")
    assert staged == tmp_path / "work" / FRAME_FILE
    assert staged.read_text() == "{}"
    layout.frame_json.unlink()
    with pytest.raises(InfraError, match="sd_quad14"):
        layout.stage_frame(tmp_path / "work2")


def test_unknown_vehicle(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_sitl_argv(SitlOptions(vehicle="x8"), ports.for_instance(0), fake_layout(tmp_path))


def test_unknown_physics(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_sitl_argv(SitlOptions(physics="unity"), ports.for_instance(0), fake_layout(tmp_path))


def test_safety_layer_comes_after_generated_layers(tmp_path: Path) -> None:
    layout = fake_layout(tmp_path)
    extra = tmp_path / "qgc.parm"
    layout = SitlLayout(layout.arducopter, layout.ardupilot_home, layout.repo, (extra,))
    layers = layout.defaults("builtin")
    safety = next(i for i, p in enumerate(layers) if p.name == "searchdrone-safety.parm")
    assert layers.index(extra) < safety


def test_lua_scripts_are_staged_into_the_working_directory(tmp_path: Path) -> None:
    layout = fake_layout(tmp_path)
    scripts = layout.repo / "simulation" / "sitl" / "scripts"
    scripts.mkdir(parents=True)
    with pytest.raises(InfraError, match=r"sd_gcs_loiter_cap\.lua"):
        layout.stage_scripts(tmp_path / "work")
    (scripts / "sd_gcs_loiter_cap.lua").write_text("return")
    staged = layout.stage_scripts(tmp_path / "work")
    assert staged == [tmp_path / "work" / "scripts" / "sd_gcs_loiter_cap.lua"]
    assert staged[0].read_text() == "return"
