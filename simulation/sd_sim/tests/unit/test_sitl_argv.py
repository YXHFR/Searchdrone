"""The SITL command line against snapshots (builtin and gz, instance 0 and 1).

Set SD_UPDATE_SNAPSHOTS=1 to rewrite the snapshots after an intended change.
"""

import itertools
import json
import os
from pathlib import Path

import pytest

from sd_sim import ports
from sd_sim.sitl import CMAC_HOME, SitlLayout, SitlOptions, build_sitl_argv

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


def test_stock_vehicle_and_missing_files(tmp_path: Path) -> None:
    ap = tmp_path / "ap"
    layout = SitlLayout(arducopter=ap / "arducopter", ardupilot_home=ap, repo=tmp_path / "empty")
    argv = build_sitl_argv(SitlOptions(), ports.for_instance(0), layout)
    assert argv[argv.index("--model") + 1] == "quad"
    defaults = argv[argv.index("--defaults") + 1].split(",")
    assert defaults == [str(ap / "Tools" / "autotest" / "default_params" / "copter.parm")]


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
