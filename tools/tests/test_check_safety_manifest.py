"""tools/check_safety_manifest.py: the real repository passes; each broken copy fails."""

from __future__ import annotations

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
)
SAFETY = "simulation/sitl/params/searchdrone-safety.parm"
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


def test_a_sim_layer_may_not_set_a_safety_parameter(root: Path) -> None:
    sim = root / "simulation/sitl/params/sd_quad14-sim.parm"
    sim.write_text(sim.read_text(encoding="utf-8") + "BATT_LOW_MAH 100\n", encoding="utf-8")
    assert "shadow: sd_quad14-sim.parm sätter säkerhetsparametern BATT_LOW_MAH" in csm.check(root)


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


def test_a_missing_pinned_file(root: Path) -> None:
    (root / "simulation/sitl/scripts/sd_gcs_loiter_cap.lua").unlink()
    assert "pinned: simulation/sitl/scripts/sd_gcs_loiter_cap.lua saknas" in csm.check(root)


def test_update_keeps_the_comments(root: Path) -> None:
    before = (root / MANIFEST).read_text(encoding="utf-8")
    csm.update(root / MANIFEST, root)
    assert (root / MANIFEST).read_text(encoding="utf-8") == before
