"""tools/check_safety_manifest.py: the real repository passes; each broken copy fails."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import check_safety_manifest as csm
import pytest

ROOT = Path(__file__).resolve().parents[2]
COPY = (
    "simulation/sitl/params",
    "simulation/sitl/scripts",
    "tools/params_allowlist.toml",
    "tools/safety_manifest.toml",
    "config/onboard.example.toml",
    "software/common/mav/src/sd_mav/denylist.py",
    "simulation/sd_sim/src/sd_sim/params.py",
    "software/common/safety/src/sd_safety/constants.py",
    "software/common/mav/src/sd_mav/allowlist.py",
    "simulation/sd_sim/src/sd_sim/sitl.py",
    "tools/params_check.py",
    "tools/check_forbidden_mavlink.py",
    "tools/check_safety_manifest.py",
)
SAFETY = "simulation/sitl/params/searchdrone-safety.parm"
FIELD = "simulation/sitl/params/searchdrone-safety.field.parm"
MANIFEST = "tools/safety_manifest.toml"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A copy of the files the manifest covers."""
    for rel in COPY:
        src, dst = ROOT / rel, tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)
    return tmp_path


def edit(root: Path, rel: str, old: str, new: str) -> None:
    path = root / rel
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def rules(problems: list[str]) -> set[str]:
    return {p.split(":", 1)[0] for p in problems}


def test_the_repository_matches_its_manifest() -> None:
    assert csm.check() == []


def test_the_copy_matches(root: Path) -> None:
    assert csm.check(root) == []


def test_a_comment_is_no_change(root: Path) -> None:
    edit(root, SAFETY, "FENCE_ENABLE 1", "FENCE_ENABLE 1  # a new comment")
    assert csm.check(root) == []


def test_a_changed_safety_value_needs_the_manifest(root: Path) -> None:
    edit(root, SAFETY, "FENCE_ENABLE 1", "FENCE_ENABLE 0")
    assert any(p.startswith(f"pinned: {SAFETY}") for p in csm.check(root))
    # Updating the digest is not enough: the fence must stay on.
    csm.update(root / MANIFEST, root)
    problems = csm.check(root)
    assert any(p.startswith("sim: FENCE_ENABLE") for p in problems), problems


def test_the_field_profile_needs_the_rc_failsafe(root: Path) -> None:
    edit(root, "simulation/sitl/params/searchdrone-safety.field.parm", "FS_THR_ENABLE 4", "")
    csm.update(root / MANIFEST, root)
    assert any(p.startswith("field: FS_THR_ENABLE") for p in csm.check(root))


def test_a_value_the_ground_station_reads_back(root: Path) -> None:
    edit(root, SAFETY, "BATT_LOW_MAH 1500", "BATT_LOW_MAH 1400")
    csm.update(root / MANIFEST, root)
    problems = csm.check(root)
    assert any("FC_SAFETY_EXPECTED BATT_LOW_MAH" in p for p in problems), problems


def test_a_new_failsafe_line_needs_a_rule(root: Path) -> None:
    edit(root, SAFETY, "FS_EKF_ACTION 1", "FS_EKF_ACTION 1\nFS_VIBE_ENABLE 0")
    csm.update(root / MANIFEST, root)
    assert csm.check(root) == [
        "ruled: searchdrone-safety.parm sätter FS_VIBE_ENABLE, som saknar regel i SAFETY_RULES"
    ]


def test_a_new_failsafe_line_in_the_field_layer_needs_a_rule(root: Path) -> None:
    # The layer real flights add on top: a digest bump alone must not switch a failsafe off.
    edit(root, FIELD, "FS_THR_ENABLE 4", "FS_THR_ENABLE 4\nFS_VIBE_ENABLE 0")
    csm.update(root / MANIFEST, root)
    assert csm.check(root) == [
        "ruled: searchdrone-safety.field.parm sätter FS_VIBE_ENABLE, som saknar regel i FIELD_RULES"
    ]


def test_a_weakened_rule_needs_the_manifest(root: Path) -> None:
    edit(root, "simulation/sd_sim/src/sd_sim/params.py", "RULED_PREFIXES", "RULED_PREFIXES ")
    assert rules(csm.check(root)) == {"pinned"}


def test_a_sim_layer_may_not_set_a_safety_parameter(root: Path) -> None:
    sim = root / "simulation/sitl/params/sd_quad14-sim.parm"
    sim.write_text(sim.read_text(encoding="utf-8") + "BATT_LOW_MAH 100\n", encoding="utf-8")
    assert "shadow: sd_quad14-sim.parm sätter säkerhetsparametern BATT_LOW_MAH" in csm.check(root)


@pytest.mark.parametrize(
    "name",
    [
        "FS_VIBE_ENABLE",
        "FENCE_AUTOENABLE",
        "ARMING_CHECK",
        "RC_OPTIONS",
        "RC7_OPTION",
        "BATT2_FS_LOW_ACT",
        "FLTMODE1",
        "EK3_SRC1_POSXY",
        "GPS_AUTO_SWITCH",
        "MOT_SAFE_DISARM",
        "BRD_SAFETY_DEFLT",
        "LAND_SPEED",
        "fs_thr_enable",
    ],
)
def test_a_sim_layer_may_set_no_safety_prefixed_parameter(root: Path, name: str) -> None:
    # Also a name the safety layers do not set themselves.
    sim = root / "simulation/sitl/params/sd_quad14-sim.parm"
    sim.write_text(sim.read_text(encoding="utf-8") + f"{name} 0\n", encoding="utf-8")
    assert csm.check(root) == [f"shadow: sd_quad14-sim.parm sätter säkerhetsparametern {name}"]


def test_a_sim_layer_may_set_other_parameters(root: Path) -> None:
    sim = root / "simulation/sitl/params/sd_quad14-sim.parm"
    sim.write_text(
        sim.read_text(encoding="utf-8") + "SIM_WIND_SPD 3\nEK3_DRAG_BCOEF_X 20\nRC7_TRIM 1500\n",
        encoding="utf-8",
    )
    assert csm.check(root) == []


def test_the_allowlist_may_not_name_a_safety_prefix(root: Path) -> None:
    edit(
        root,
        "tools/params_allowlist.toml",
        '"MOT_THST_HOVER",',
        '"MOT_THST_HOVER", "FS_VIBE_ENABLE",',
    )
    csm.update(root / MANIFEST, root)
    assert csm.check(root) == [
        "shadow: tools/params_allowlist.toml tillåter säkerhetsparametern FS_VIBE_ENABLE"
    ]


def test_the_allowlist_may_not_let_a_sim_layer_shadow_safety(root: Path) -> None:
    edit(
        root,
        "tools/params_allowlist.toml",
        '"MOT_THST_HOVER",',
        '"MOT_THST_HOVER", "FENCE_RADIUS",',
    )
    assert rules(csm.check(root)) == {"pinned"}
    csm.update(root / MANIFEST, root)
    problems = csm.check(root)
    assert problems == [
        "shadow: tools/params_allowlist.toml tillåter säkerhetsparametern FENCE_RADIUS"
    ]


def test_the_manifest_must_name_what_sitl_loads(root: Path) -> None:
    other = "simulation/sitl/params/other-safety.parm"
    shutil.copy(root / SAFETY, root / other)
    edit(
        root,
        MANIFEST,
        f'[profile.sim]\nlayers = ["{SAFETY}"]',
        f'[profile.sim]\nlayers = ["{other}"]',
    )
    edit(root, MANIFEST, "[[pinned]]", f'[[pinned]]\npath = "{other}"\nsha256 = ""\n\n[[pinned]]')
    csm.update(root / MANIFEST, root)
    problems = csm.check(root)
    assert any(p.startswith("loaded: SITL (sd_quad14) laddar") for p in problems), problems
    assert any(p.startswith("loaded: sd-onboard läser tillbaka") for p in problems), problems


def test_the_onboard_example_reads_back_the_safety_layer(root: Path) -> None:
    edit(
        root,
        "config/onboard.example.toml",
        f'expected_param_files = ["{SAFETY}"]',
        "expected_param_files = []",
    )
    assert csm.check(root) == [
        f"loaded: config/onboard.example.toml läser tillbaka [], manifestet ['{SAFETY}']"
    ]


def test_every_safety_layer_is_pinned(root: Path) -> None:
    text = (root / MANIFEST).read_text(encoding="utf-8")
    start = text.index("[[pinned]]")
    end = text.index("[[pinned]]", start + 1)
    (root / MANIFEST).write_text(text[:start] + text[end:], encoding="utf-8")
    assert f"pinned: säkerhetslagret {SAFETY} är inte låst i manifestet" in csm.check(root)


def unpin(root: Path, rel: str) -> None:
    path = root / MANIFEST
    block = rf'\[\[pinned\]\]\npath = "{re.escape(rel)}"\nsha256 = "[0-9a-f]*"\n*'
    text, n = re.subn(block, "", path.read_text(encoding="utf-8"))
    assert n == 1, rel
    path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("rel", csm.REQUIRED_PINS)
def test_a_required_pin_cannot_be_dropped(root: Path, rel: str) -> None:
    # Take the pin out and change the file: still caught.
    unpin(root, rel)
    (root / rel).write_text((root / rel).read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert f"pinned: {rel} måste vara låst i manifestet (REQUIRED_PINS)" in csm.check(root)


def test_a_pin_beyond_the_required_ones_may_be_dropped(root: Path) -> None:
    extra = "simulation/sitl/params/sd_quad14-sim.parm"
    edit(root, MANIFEST, "[[pinned]]", f'[[pinned]]\npath = "{extra}"\nsha256 = ""\n\n[[pinned]]')
    csm.update(root / MANIFEST, root)
    unpin(root, extra)
    assert csm.check(root) == []


def test_a_missing_pinned_file(root: Path) -> None:
    (root / "simulation/sitl/scripts/sd_gcs_loiter_cap.lua").unlink()
    assert "pinned: simulation/sitl/scripts/sd_gcs_loiter_cap.lua saknas" in csm.check(root)


def test_update_keeps_the_comments(root: Path) -> None:
    before = (root / MANIFEST).read_text(encoding="utf-8")
    assert csm.update(root / MANIFEST, root) == []
    assert (root / MANIFEST).read_text(encoding="utf-8") == before


def test_update_names_the_changed_files(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    edit(root, SAFETY, "BATT_LOW_MAH 1500", "BATT_LOW_MAH 1501")
    monkeypatch.setattr(csm, "REPO", root)
    monkeypatch.setattr(csm, "MANIFEST", root / MANIFEST)
    assert csm.main(["--update"]) == 1  # the new value breaks FC_SAFETY_EXPECTED
    out = capsys.readouterr().out
    assert f"--update: ny digest för {SAFETY}" in out
    assert "--update: ny digest för tools/" not in out
    assert csm.update(root / MANIFEST, root) == []
