"""Start checks: every one stops sd-onboard with exit code 2; the token stays secret."""

from __future__ import annotations

import logging
import os
import shutil
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from sd_onboard import app
from sd_onboard.config import MIN_TOKEN_BYTES, check, load
from sd_onboard.logs import MASK, SecretFilter

REPO = Path(__file__).resolve().parents[4]
EXAMPLE = REPO / "config" / "onboard.example.toml"
PARM = REPO / "simulation/sitl/params/searchdrone-safety.parm"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A minimal repository root with the example config and the parameter file."""
    r = tmp_path / "repo"
    (r / "config").mkdir(parents=True)
    shutil.copy(EXAMPLE, r / "config" / "onboard.example.toml")
    (r / "simulation/sitl/params").mkdir(parents=True)
    shutil.copy(PARM, r / "simulation/sitl/params" / PARM.name)
    return r


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    data = tmp_path / "data"
    (data / "secrets").mkdir(parents=True)
    token = data / "secrets" / "onboard_link_token"
    token.write_text("t" * 43)
    token.chmod(0o600)
    return {"SD_DATA_DIR": str(data)}


def test_example_config_has_no_token() -> None:
    data = tomllib.loads(EXAMPLE.read_text())
    assert "token" not in data and "sdlink_token" not in data
    assert all("token" not in k or k == "link_token_file" for k in data)
    assert "${SD_DATA_DIR}" in data["link_token_file"]


def test_caps_not_configurable() -> None:
    import re

    keys = re.compile(r"max_alt|max_speed|max_range|battery_rtl")
    data = tomllib.loads(EXAMPLE.read_text())
    flat = list(data) + [k for v in data.values() if isinstance(v, dict) for k in v]
    assert not any(keys.search(k) for k in flat)
    from sd_onboard.config import OnboardConfig

    assert not any(keys.search(k) for k in OnboardConfig.model_fields)


def test_good_config_starts(root: Path, env: dict[str, str]) -> None:
    assert check(load(root, env)) == []


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("SD_ONBOARD__PROFILE", "field"),
        ("SD_ONBOARD__VEHICLE_ID", "Sim 01"),
        ("SD_ONBOARD__FC_SYSID", "250"),
        ("SD_ONBOARD__FC_SYSID", "255"),
        ("SD_ONBOARD__OWN_COMPID", "190"),
        ("SD_ONBOARD__SDLINK_URL", "ws://192.0.2.10:7600/sdlink/v1/control"),
        ("SD_ONBOARD__TEST_HOOKS__TAKEOFF_DELAY_AFTER_ARM_S", "5"),
    ],
)
def test_startup_checks(root: Path, env: dict[str, str], key: str, value: str) -> None:
    assert check(load(root, {**env, key: value}))


def test_perception_needs_a_still_folder(root: Path, env: dict[str, str]) -> None:
    on = {"SD_ONBOARD__PERCEPTION__ENABLED": "true"}
    assert any("still_dir" in e for e in check(load(root, {**env, **on})))
    folder = {**on, "SD_ONBOARD__PERCEPTION__STILL_DIR": "${SD_DATA_DIR}/stills"}
    loaded = load(root, {**env, **folder, "SD_ONBOARD__PERCEPTION__HFOV_DEG": "70"})
    assert check(loaded) == []
    assert loaded.config.perception.hfov_deg == 70.0
    link = SimpleNamespace(perception=lambda _r: None)
    made = app._perception(loaded, SimpleNamespace(), link, SimpleNamespace())  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    assert made is not None and made.camera.hfov_deg == 70.0
    off = app._perception(load(root, env), SimpleNamespace(), link, SimpleNamespace())  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    assert off is None


def test_the_wide_camera_takes_full_resolution_stills(root: Path, env: dict[str, str]) -> None:
    """The owner's decision of 2026-11-03: Camera Module 3 at its full 4608 x 2592 (66
    degrees), in the defaults and the example file alike; a smaller size still loads."""
    from sd_onboard.config import PerceptionConfig

    full = (4608, 2592, 66.0)
    p = load(root, env).config.perception
    assert (p.width_px, p.height_px, p.hfov_deg) == full
    d = PerceptionConfig()
    assert (d.width_px, d.height_px, d.hfov_deg) == full
    small = {"SD_ONBOARD__PERCEPTION__WIDTH_PX": "1280", "SD_ONBOARD__PERCEPTION__HEIGHT_PX": "720"}
    p = load(root, {**env, **small}).config.perception
    assert (p.width_px, p.height_px) == (1280, 720)


def test_the_tele_camera_is_off_unless_asked_for(root: Path, env: dict[str, str]) -> None:
    tele = load(root, env).config.perception.tele
    assert not tele.enabled
    # The owner's tele camera: Raspberry Pi HQ Camera (IMX477) with a 20-degree M12 lens.
    assert (tele.width_px, tele.height_px, tele.hfov_deg) == (4056, 3040, 20.0)


def test_the_tele_camera_needs_perception_and_its_own_folder(
    root: Path, env: dict[str, str]
) -> None:
    tele = {"SD_ONBOARD__PERCEPTION__TELE__ENABLED": "true"}
    assert any("perception.tele" in e for e in check(load(root, {**env, **tele})))
    on = {
        **tele,
        "SD_ONBOARD__PERCEPTION__ENABLED": "true",
        "SD_ONBOARD__PERCEPTION__STILL_DIR": "${SD_DATA_DIR}/stills",
    }
    assert any("still_dir" in e for e in check(load(root, {**env, **on})))
    same = {**on, "SD_ONBOARD__PERCEPTION__TELE__STILL_DIR": "${SD_DATA_DIR}/stills"}
    assert any("egen still_dir" in e for e in check(load(root, {**env, **same})))
    # The same folder written two ways is still the same folder.
    for main_dir, tele_dir in (("stills", "./stills"), ("stills", "x/../stills/")):
        spelled = {
            **on,
            "SD_ONBOARD__PERCEPTION__STILL_DIR": main_dir,
            "SD_ONBOARD__PERCEPTION__TELE__STILL_DIR": tele_dir,
        }
        assert any("egen still_dir" in e for e in check(load(root, {**env, **spelled})))
    good = {
        **on,
        "SD_ONBOARD__PERCEPTION__TELE__STILL_DIR": "${SD_DATA_DIR}/tele",
        "SD_ONBOARD__PERCEPTION__TELE__WIDTH_PX": "1128",
        "SD_ONBOARD__PERCEPTION__TELE__HEIGHT_PX": "846",
        "SD_ONBOARD__PERCEPTION__TELE__HFOV_DEG": "20",
    }
    loaded = load(root, {**env, **good})
    assert check(loaded) == []
    link = SimpleNamespace(perception=lambda _r: None)
    made = app._perception(loaded, SimpleNamespace(), link, SimpleNamespace())  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    assert made is not None and made.cameras == ["main", "tele"]
    (feed,) = made.extra
    assert (feed.camera.width_px, feed.camera.height_px, feed.camera.hfov_deg) == (1128, 846, 20.0)
    assert made.camera.hfov_deg == 66.0  # the main camera is as before


def test_a_wrong_detector_model_never_stops_the_start(
    root: Path, env: dict[str, str], tmp_path: Path
) -> None:
    """sd-onboard may restart in the air: a missing or changed model only turns perception
    off (in its own task, which tells the ground), never the executor and link."""
    on = {
        "SD_ONBOARD__PERCEPTION__ENABLED": "true",
        "SD_ONBOARD__PERCEPTION__STILL_DIR": "${SD_DATA_DIR}/stills",
        "SD_DETECTOR_MODEL": str(tmp_path / "gone.json"),
    }
    loaded = load(root, {**env, **on})
    assert check(loaded) == []
    events: list[tuple[str, str, dict[str, str]]] = []
    link = SimpleNamespace(
        perception=lambda _r: None,
        event=lambda *a: events.append(a),  # pyright: ignore[reportUnknownLambdaType]
    )
    made = app._perception(loaded, SimpleNamespace(), link, SimpleNamespace())  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    assert made is not None and made.off is not None
    made.off("sources_failed", "ModelError")
    assert made.on is not None
    made.on()
    assert events == [
        ("executor.perception_off", "warning", {"reason": "sources_failed", "error": "ModelError"}),
        ("executor.perception_on", "info", {}),
    ]


def test_test_hooks_rejected_outside_test(root: Path, env: dict[str, str]) -> None:
    hook = {"SD_ONBOARD__TEST_HOOKS__TAKEOFF_DELAY_AFTER_ARM_S": "5"}
    assert check(load(root, {**env, **hook}))
    assert check(load(root, {**env, **hook, "SD_TEST": "1"})) == []


def test_token_file_mode_checked(root: Path, env: dict[str, str]) -> None:
    token = Path(env["SD_DATA_DIR"]) / "secrets" / "onboard_link_token"
    token.chmod(0o640)
    assert any("600" in e for e in check(load(root, env)))


def test_token_min_length(root: Path, env: dict[str, str]) -> None:
    token = Path(env["SD_DATA_DIR"]) / "secrets" / "onboard_link_token"
    token.write_text("x" * (MIN_TOKEN_BYTES - 1))
    assert check(load(root, env))
    token.write_text("x" * MIN_TOKEN_BYTES)
    assert check(load(root, env)) == []


def test_token_outside_repo(root: Path, env: dict[str, str]) -> None:
    inside = root / "secrets" / "onboard_link_token"
    inside.parent.mkdir()
    inside.write_text("x" * 40)
    inside.chmod(0o600)
    errors = check(load(root, {**env, "SD_ONBOARD__LINK_TOKEN_FILE": "secrets/onboard_link_token"}))
    assert any("repot" in e for e in errors)
    state_inside = {**env, "SD_ONBOARD__STATE_DIR": "state"}
    assert any("state_dir" in e for e in check(load(root, state_inside)))


def test_token_on_a_windows_disk_is_refused(root: Path, env: dict[str, str]) -> None:
    errors = check(load(root, {**env, "SD_ONBOARD__LINK_TOKEN_FILE": "/mnt/c/token"}))
    assert any("Windows" in e for e in errors)


def test_unknown_key_is_an_error(root: Path) -> None:
    (root / "config" / "onboard.toml").write_text("max_alt_agl_m = 300\n")
    assert app.main.__module__ == "sd_onboard.app"
    cwd = Path.cwd()
    os.chdir(root)
    try:
        assert app.main() == 2
    finally:
        os.chdir(cwd)


def test_main_exits_2_on_a_failed_check(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(root)
    monkeypatch.setenv("SD_DATA_DIR", str(root.parent / "nothing"))
    assert app.main() == 2


def test_token_masked_in_logs() -> None:
    """The filter masks a configured value and keyed values. Records are built by hand
    and passed through the filter only; nothing is written to any log."""
    marker = "m4rk3r-" + "x" * 40
    masker = SecretFilter([marker])
    texts = []
    for msg, args in (
        ("dialling with %s", (marker,)),
        ("headers {'Authorization': 'Bearer abcdef123'}", None),
        ("token=zzz999", None),
    ):
        record = logging.LogRecord("sd_onboard.test", logging.INFO, __file__, 1, msg, args, None)
        assert masker.filter(record)
        texts.append(record.getMessage())
    text = "\n".join(texts)
    assert marker not in text and "abcdef123" not in text and "zzz999" not in text
    assert text.count(MASK) == 3


def test_token_file_may_not_be_a_link(root: Path, env: dict[str, str]) -> None:
    token = Path(env["SD_DATA_DIR"]) / "secrets" / "onboard_link_token"
    link = token.with_name("link")
    link.symlink_to(token)
    errors = check(load(root, {**env, "SD_ONBOARD__LINK_TOKEN_FILE": str(link)}))
    assert any("länk" in e for e in errors)


def test_log_file_is_private(tmp_path: Path) -> None:
    from sd_onboard.logs import setup_logging

    root_logger = logging.getLogger()
    before = list(root_logger.handlers)
    try:
        setup_logging(tmp_path / "state", secrets=("x" * 40,))
        assert (tmp_path / "state" / "onboard.jsonl").stat().st_mode & 0o077 == 0
    finally:
        for h in root_logger.handlers[:]:
            if h not in before:
                root_logger.removeHandler(h)
                h.close()


# --- the RC transmitter is required at real flight ------------------------------------

FIELD = REPO / "simulation/sitl/params/searchdrone-safety.field.parm"


def test_require_rc_is_on_by_default(root: Path, env: dict[str, str]) -> None:
    assert load(root, env).config.require_rc is True
    assert tomllib.loads(EXAMPLE.read_text())["require_rc"] is True


def test_require_rc_off_only_in_the_simulator(root: Path, env: dict[str, str]) -> None:
    off = {**env, "SD_ONBOARD__REQUIRE_RC": "false"}
    assert load(root, off).config.require_rc is False
    assert check(load(root, off)) == []
    errors = check(load(root, {**off, "SD_ONBOARD__PROFILE": "field"}))
    assert any("require_rc" in e for e in errors)


def test_field_params_require_the_rc_transmitter(root: Path, env: dict[str, str]) -> None:
    """Real flight: the flight controller itself refuses to arm without RC and flies home
    when the RC link is lost; the simulator file alone is refused for it."""
    field = {**env, "SD_ONBOARD__PROFILE": "field"}
    errors = check(load(root, field))
    assert any("ARMING_SKIPCHK" in e for e in errors) is False  # 0: every check on
    assert any("FS_THR_ENABLE" in e for e in errors)
    shutil.copy(FIELD, root / "simulation/sitl/params" / FIELD.name)
    files = f"simulation/sitl/params/{PARM.name},simulation/sitl/params/{FIELD.name}"
    errors = check(load(root, {**field, "SD_ONBOARD__EXPECTED_PARAM_FILES": files}))
    assert not any("ARMING_SKIPCHK" in e or "FS_THR_ENABLE" in e for e in errors)


def test_field_file_values() -> None:
    from sd_onboard.config import ARMING_SKIP_RC_BIT
    from sd_onboard.io.params import effective_params

    sim = effective_params([PARM])
    real = effective_params([PARM, FIELD])
    assert int(real["ARMING_SKIPCHK"]) & ARMING_SKIP_RC_BIT == 0
    assert real["FS_THR_ENABLE"] == 4.0  # RC lost in the air -> SmartRTL, else RTL
    assert sim["FS_THR_ENABLE"] == 0.0  # the simulator may leave it off
    # The field file only changes rows the safety file already has.
    assert set(effective_params([FIELD])) <= set(sim)


def test_rc_param_problems() -> None:
    from sd_onboard.config import _rc_param_problems  # pyright: ignore[reportPrivateUsage]

    assert _rc_param_problems({"ARMING_SKIPCHK": 0, "FS_THR_ENABLE": 4}) == []
    found = _rc_param_problems({"ARMING_SKIPCHK": 64, "FS_THR_ENABLE": 0})
    assert len(found) == 2
    assert len(_rc_param_problems({"ARMING_SKIPCHK": 0, "FS_THR_ENABLE": 2})) == 1  # continue
    assert len(_rc_param_problems({})) == 2  # unset counts as unsafe


# --- the downward laser ([rangefinder]) ----------------------------------------------------

PARAMS_DIR = REPO / "simulation/sitl/params"
LASER_FILES = {
    "tf02_pro": PARAMS_DIR / "searchdrone-rangefinder-tf02.parm",
    "grf250": PARAMS_DIR / "searchdrone-rangefinder-grf250.parm",
}


def test_no_laser_is_the_default_with_or_without_the_section(
    root: Path, env: dict[str, str]
) -> None:
    assert tomllib.loads(EXAMPLE.read_text())["rangefinder"]["model"] == "none"
    assert not load(root, env).config.rangefinder.fitted
    example = root / "config" / "onboard.example.toml"
    text = example.read_text()
    example.write_text(text[: text.index("[rangefinder]")])  # no section at all
    loaded = load(root, env)
    assert not loaded.config.rangefinder.fitted and check(loaded) == []


def test_a_laser_needs_a_tested_range_and_its_parameter_file(
    root: Path, env: dict[str, str]
) -> None:
    laser = {**env, "SD_ONBOARD__RANGEFINDER__MODEL": "tf02_pro"}
    errors = check(load(root, laser))
    assert any("min_tested_m < max_tested_m" in e for e in errors)
    assert any("RNGFND1_TYPE" in e for e in errors)  # the safety file alone has no laser
    assert any("WP_RFND_USE" in e for e in errors)
    tf02 = LASER_FILES["tf02_pro"]
    shutil.copy(tf02, root / "simulation/sitl/params")
    files = f"simulation/sitl/params/{PARM.name},simulation/sitl/params/{tf02.name}"
    good = {
        **laser,
        "SD_ONBOARD__RANGEFINDER__MIN_TESTED_M": "0.1",
        "SD_ONBOARD__RANGEFINDER__MAX_TESTED_M": "13",
        "SD_ONBOARD__EXPECTED_PARAM_FILES": files,
    }
    loaded = load(root, good)
    assert check(loaded) == []
    assert (loaded.config.rangefinder.min_tested_m, loaded.config.rangefinder.max_tested_m) == (
        0.1,
        13.0,
    )
    wrong = {**good, "SD_ONBOARD__RANGEFINDER__MODEL": "grf250"}  # the TF02 file
    assert any("RNGFND1_TYPE" in e for e in check(load(root, wrong)))


@pytest.mark.parametrize("model", ["tf02_pro", "grf250"])
def test_the_laser_files_keep_the_flight_controller_off_the_laser(model: str) -> None:
    from sd_onboard.config import RangefinderConfig, rangefinder_param_problems
    from sd_onboard.io.params import effective_params

    path = LASER_FILES[model]
    own = effective_params([path])
    assert own["WP_RFND_USE"] == 0 and own["SURFTRAK_MODE"] == 0
    assert own["EK3_RNG_USE_HGT"] == -1  # the EKF keeps its height from the barometer
    assert own["RNGFND1_ORIENT"] == 25
    assert not set(own) & set(effective_params([PARM]))  # changes nothing in the safety file
    known = set((PARAMS_DIR / "known_params_Copter-4.7.1.txt").read_text().split())
    assert set(own) <= known
    cfg = RangefinderConfig(model=model, min_tested_m=1.0, max_tested_m=10.0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    assert rangefinder_param_problems(cfg, effective_params([PARM, path])) == []
    if model == "grf250":
        assert own["RNGFND1_GRF_RET"] == 1  # first return: the treetop, not a gap


def test_rangefinder_param_problems() -> None:
    from sd_onboard.config import RangefinderConfig, rangefinder_param_problems

    good = {"RNGFND1_TYPE": 27.0, "RNGFND1_ORIENT": 25.0, "WP_RFND_USE": 0.0, "SURFTRAK_MODE": 0.0}
    tf02 = RangefinderConfig(model="tf02_pro", max_tested_m=13.0)
    assert rangefinder_param_problems(tf02, good) == []
    assert rangefinder_param_problems(RangefinderConfig(), {}) == []  # no laser, no rule
    other = RangefinderConfig(model="other", max_tested_m=40.0)
    assert rangefinder_param_problems(other, {**good, "RNGFND1_TYPE": 10.0}) == []  # MAVLink
    assert len(rangefinder_param_problems(other, {**good, "RNGFND1_TYPE": 0.0})) == 1
    assert len(rangefinder_param_problems(tf02, {})) == 4  # unset counts as unsafe
    for name, bad in (("RNGFND1_ORIENT", 0.0), ("WP_RFND_USE", 1.0), ("SURFTRAK_MODE", 1.0)):
        (found,) = rangefinder_param_problems(tf02, {**good, name: bad})
        assert name in found
