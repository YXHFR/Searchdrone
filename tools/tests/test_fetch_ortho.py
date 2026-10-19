"""tools/terrain/fetch_ortho.py without the network: where the photo goes, and never in CI."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "terrain"))
_spec = importlib.util.spec_from_file_location(
    "fetch_ortho", ROOT / "tools" / "terrain" / "fetch_ortho.py"
)
assert _spec is not None and _spec.loader is not None
ortho: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ortho)


@pytest.fixture
def no_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("CI", "GITHUB_ACTIONS", "SD_LM_USER", "SD_LM_PASSWORD"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("key", ["CI", "GITHUB_ACTIONS"])
def test_refuses_to_run_in_ci(
    key: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(key, "true")
    monkeypatch.setenv("SD_LM_USER", "u")
    monkeypatch.setenv("SD_LM_PASSWORD", "p")
    monkeypatch.setattr(ortho.fetch_site, "fetch_ortho", pytest.fail)
    assert ortho.main(["site=hellasgarden"]) == 2
    assert "aldrig i CI" in capsys.readouterr().err


def test_needs_the_login(no_ci: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    site_dir = tmp_path / "simulation" / "gazebo" / "sites" / "x"
    site_dir.mkdir(parents=True)
    (site_dir / "site.toml").write_text(
        "[home]\nlat = 59.3\nlon = 18.2\n[area]\nhalf_size_m = 100\n[dem]\ncell_m = 20\n"
    )
    with pytest.raises(ortho.FetchError, match=r"geotorget\.example\.toml") as err:
        ortho.run(["site=x"], tmp_path)
    assert err.value.code == 2


def test_public_site_photo_goes_to_build_never_the_site_folder(
    no_ci: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_dir = tmp_path / "simulation" / "gazebo" / "sites" / "x"
    site_dir.mkdir(parents=True)
    (site_dir / "site.toml").write_text(
        "[home]\nlat = 59.3\nlon = 18.2\n[area]\nhalf_size_m = 100\n[dem]\ncell_m = 20\n"
    )
    monkeypatch.setenv("SD_LM_USER", "owneruser")
    monkeypatch.setenv("SD_LM_PASSWORD", "ownersecret")
    seen: dict[str, Any] = {}

    def fake(site: Any, login: Any, work: Path, dest: Path) -> dict[str, Any]:
        seen["login"] = login
        np.save(dest, np.zeros((4, 4, 3), np.uint8))  # synthetic: nothing real
        return {
            "collection": "orto-x",
            "year": 2025,
            "source_cell_m": 0.16,
            "cell_m": 0.16,
            "tiles": 1,
        }

    monkeypatch.setattr(ortho.fetch_site, "fetch_ortho", fake)
    assert ortho.run(["site=x"], tmp_path) == 0
    out = tmp_path / "build" / "ortho" / "x"
    assert (out / "ortho.npy").is_file() and not (out / "ortho.part.npy").exists()
    meta = (out / "ortho.toml").read_text(encoding="utf-8")
    assert "Lantmäteriet" in meta and "orto-x" in meta
    assert "owneruser" not in meta and "ownersecret" not in meta
    assert not list(site_dir.glob("ortho*"))
    assert seen["login"] == {"user": "owneruser", "password": "ownersecret"}
