"""The launcher's services (``sd_sim.services``), ``just e2e``'s runner, the RC pilot and
the MP4 probe, without starting anything."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from sd_sim import config, e2e, launch, mp4, rc, services


def _stack(tmp_path: Path, **args: str) -> launch.Stack:
    repo = tmp_path / "repo"
    (repo / "config").mkdir(parents=True)
    (repo / ".env").write_text(f"SD_DATA_DIR={tmp_path / 'data'}\n", encoding="utf-8")
    cfg = config.load({"physics": "builtin", **args}, repo=repo, environ={})
    run = tmp_path / "run"
    run.mkdir()
    return launch.Stack(cfg, run, repo=repo, with_services=True)


def test_ground_gets_this_instances_ports_and_files(tmp_path: Path) -> None:
    st = _stack(tmp_path, instance="2")
    lay = services.layout_for(st)
    env = services.ground_env(st, lay, video=False)
    p = st.ports
    assert env["SD_GROUND__MAVLINK__LISTEN"] == f"udpin:127.0.0.1:{p['ground_mavlink']}"
    assert env["SD_GROUND__HTTP__PANEL_BIND"] == f"127.0.0.1:{p['ground_panel']}"
    assert env["SD_GROUND__HTTP__COMMANDER_BIND"] == f"127.0.0.1:{p['ground_commander']}"
    assert env["SD_GROUND__SDLINK__BIND"] == f"127.0.0.1:{p['ground_sdlink']}"
    assert env["SD_GROUND__DB__URL"].endswith(f"@127.0.0.1:5433/{lay.db_name}")
    assert lay.db_name == "searchdrone_i2"
    # Another instance: its own data folder (its own control socket), under SD_DATA_DIR.
    assert lay.data_dir == tmp_path / "data" / "sim" / "i2"
    assert env["SD_DATA_DIR"] == env["SD_GROUND__DATA_DIR"] == str(lay.data_dir)
    assert env["SD_GROUND__SIM__DATAFLASH_DIR"] == str(st.run_dir / "sitl" / "logs")
    assert env["SD_GROUND__VIDEO__ENABLED"] == "false"
    assert env["SD_GROUND__LAN__ENABLED"] == "false" and "SD_GROUND__LAN__BIND" not in env
    # Only addresses, files and switches: no secret is ever in a variable.
    assert not [k for k in env if k.endswith(("PASSWORD", "TOKEN", "KEY"))]


def test_instance_0_is_the_operators_own_data_and_database(tmp_path: Path) -> None:
    st = _stack(tmp_path, lan="1")
    lay = services.layout_for(st)
    assert lay.data_dir == tmp_path / "data" and lay.db_name == "searchdrone"
    env = services.ground_env(st, lay, video=True)
    assert env["SD_GROUND__VIDEO__ENABLED"] == "true"
    assert env["SD_GROUND__LAN__ENABLED"] == "true"
    assert env["SD_GROUND__LAN__BIND"] == "0.0.0.0:8443"
    st.fresh_db = True  # the end-to-end test starts from an empty database
    assert services.layout_for(st).db_name == "searchdrone_i0"


def test_onboard_talks_through_mavtap_and_the_simulated_radio(tmp_path: Path) -> None:
    st = _stack(tmp_path)
    lay = services.layout_for(st)
    env = services.onboard_env(st, lay)
    assert env["SD_ONBOARD__MAV_URL"] == f"udpin:127.0.0.1:{st.ports['onboard_mavlink']}"
    assert env["SD_ONBOARD__SDLINK_URL"].startswith(f"ws://127.0.0.1:{st.ports['linksim_sdlink']}/")
    assert env["SD_ONBOARD__PROFILE"] == "sim" and env["SD_ONBOARD__REQUIRE_RC"] == "false"
    token = Path(env["SD_ONBOARD__LINK_TOKEN_FILE"])
    assert not token.is_relative_to(st.repo)  # sd-onboard refuses a token in the repository


@pytest.mark.parametrize(
    ("brain", "kind"),
    [("llm_local", "llm_local"), ("grammar", "grammar"), ("scripted", "scripted")],
)
def test_brain_kinds(tmp_path: Path, brain: str, kind: str) -> None:
    st = _stack(tmp_path, brain=brain, script="simulation/scenarios/scripts/SC01.yaml")
    env = services.brain_env(st, services.layout_for(st))
    assert env["SD_BRAIN__BACKEND__KIND"] == kind
    assert env["SD_BRAIN__GROUND__COMMANDER_URL"] == (
        f"http://127.0.0.1:{st.ports['ground_commander']}/commander/v1"
    )
    assert ("SD_BRAIN__BACKEND__SCRIPTED_FILE" in env) == (brain == "scripted")
    if brain == "scripted":
        assert env["SD_BRAIN__BACKEND__SCRIPTED_FILE"] == str(
            st.repo / "simulation/scenarios/scripts/SC01.yaml"
        )


def test_no_brain_and_the_health_parts(tmp_path: Path) -> None:
    st = _stack(tmp_path, brain="none")
    assert services.start_brain(st, services.layout_for(st)) is None
    assert "brain" not in services.wanted_parts(st)
    assert services.wanted_parts(_stack(tmp_path / "b")) == (
        "db",
        "mavlink",
        "sdlink",
        "onboard",
        "brain",
    )


def test_secrets_are_masked_in_logs(tmp_path: Path) -> None:
    st = _stack(tmp_path)
    launch.write_secret(st.secrets_dir, "brain_token", "s3cret-token-value")
    assert st.mask("token=s3cret-token-value") == "token=***"


# --- just e2e -------------------------------------------------------------------------


def test_e2e_runs_the_demo_alone_then_the_buttons_in_parallel() -> None:
    demo = e2e.pytest_argv(demo=True, workers=4)
    rest = e2e.pytest_argv(demo=False, workers=4)
    assert demo[demo.index("-m", 3) + 1] == "sitl and e2e"
    assert demo[demo.index("-k") + 1] == e2e.DEMO and "-n" not in demo
    assert rest[rest.index("-k") + 1] == f"not {e2e.DEMO}" and rest[rest.index("-n") + 1] == "4"
    for argv in (demo, rest):
        assert argv[argv.index("--only-rerun") + 1] == "InfraError"
        assert "--brain" in argv and argv[argv.index("--brain") + 1] == "scripted"


@pytest.mark.parametrize("argv", [["foo=1"], ["runs=0"], ["runs=x"], ["offline=2"]])
def test_e2e_bad_keys_exit_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        e2e.main(argv)
    assert exc.value.code == 2


def test_e2e_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    assert e2e.main(["runs=3", "offline=1", "dry=1"]) == 0
    assert json.loads(capsys.readouterr().out) == {"args": ["runs=3", "offline=1", "dry=1"]}


def test_e2e_offline_refuses_an_open_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(e2e, "internet_reachable", lambda: True)
    assert e2e.main(["offline=1"]) == 1


def test_e2e_counts_every_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    codes = iter([0, 0, 0, 1])

    class Done:
        def __init__(self) -> None:
            self.returncode = next(codes)

    monkeypatch.setattr("sd_sim.e2e.subprocess.run", lambda *a, **k: Done())
    monkeypatch.setattr("sd_sim.e2e.paths.runs_dir", lambda: tmp_path)
    assert e2e.main(["runs=2"]) == 1
    summary = json.loads((tmp_path / "e2e-summary.json").read_text(encoding="utf-8"))
    assert summary["runs"] == 2 and summary["passed"] == 1


# --- the RC pilot ---------------------------------------------------------------------


def test_rc_pilot_centres_the_throttle_only_in_the_air() -> None:
    assert rc.pilot_throttle(armed=False, alt_m=0.0, mode="GUIDED") == rc.THROTTLE_LOW
    assert rc.pilot_throttle(armed=True, alt_m=1.0, mode="GUIDED") == rc.THROTTLE_LOW
    assert rc.pilot_throttle(armed=True, alt_m=40.0, mode="LOITER") == rc.THROTTLE_CENTRE
    assert rc.pilot_throttle(armed=True, alt_m=40.0, mode="SMART_RTL") == rc.THROTTLE_LOW


# --- the MP4 probe --------------------------------------------------------------------


def _box(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(body), kind) + body


def _full(kind: bytes, body: bytes, flags: int = 0) -> bytes:
    return _box(kind, struct.pack(">I", flags) + body)


def test_mp4_probe_reads_codec_and_fragment_length(tmp_path: Path) -> None:
    tkhd = _full(b"tkhd", struct.pack(">IIII", 0, 0, 1, 0))  # times, track id 1
    mdhd = _full(b"mdhd", struct.pack(">IIII", 0, 0, 90000, 0))  # timescale 90 kHz, 0
    hdlr = _full(b"hdlr", struct.pack(">I4s", 0, b"vide") + b"\0" * 12)
    stsd = _full(b"stsd", struct.pack(">I", 1) + _box(b"avc1", b"\0" * 8))
    stbl = _box(b"stbl", stsd)
    trak = _box(b"trak", tkhd + _box(b"mdia", mdhd + hdlr + _box(b"minf", stbl)))
    trex = _full(b"trex", struct.pack(">IIIII", 1, 1, 3000, 0, 0))
    moov = _box(b"moov", trak + _box(b"mvex", trex))
    # Two fragments: 30 samples at the default 3000 (1 s), then 4 explicit durations.
    traf1 = _box(
        b"traf", _full(b"tfhd", struct.pack(">I", 1)) + _full(b"trun", struct.pack(">I", 30))
    )
    durations = struct.pack(">I", 4) + struct.pack(">4I", 45000, 45000, 45000, 45000)
    traf2 = _box(
        b"traf",
        _full(b"tfhd", struct.pack(">I", 1)) + _full(b"trun", durations, flags=0x100),
    )
    path = tmp_path / "rec.mp4"
    path.write_bytes(_box(b"ftyp", b"iso5") + moov + _box(b"moof", traf1) + _box(b"moof", traf2))
    video = mp4.probe(path).video()
    assert video is not None and video.codec == "avc1"
    assert video.seconds == pytest.approx(3.0)


def test_mp4_probe_survives_a_cut_file(tmp_path: Path) -> None:
    path = tmp_path / "cut.mp4"
    path.write_bytes(_box(b"ftyp", b"iso5")[:-2] + b"\x00\x00\x01\x00moov")
    assert mp4.probe(path).video() is None
