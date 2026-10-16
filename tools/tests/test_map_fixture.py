"""Map files in the repository (B1-11): only the small synthetic test pack is committed, it is
up to date with its source, and it lies at the public SITL site."""

from __future__ import annotations

import importlib.util
import math
import subprocess
from pathlib import Path

from sd_ground.api.tiles import read_center

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "software" / "panel" / "e2e" / "fixtures"
FIXTURE = FIXTURES / "fixture.pmtiles"
CMAC_LAT, CMAC_LON = -35.363261, 149.165230


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000 * math.asin(math.sqrt(a))


def test_only_the_test_pack_is_committed() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "*.pmtiles"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    assert tracked == ["software/panel/e2e/fixtures/fixture.pmtiles"]
    assert FIXTURE.stat().st_size < 100 * 1024


def _ignored(path: str) -> bool:
    res = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT, check=False)
    return res.returncode == 0


def test_other_packs_are_ignored() -> None:
    assert _ignored("software/panel/e2e/fixtures/other.pmtiles")
    assert _ignored("demo.pmtiles")
    assert not _ignored("software/panel/e2e/fixtures/fixture.pmtiles")


def test_test_pack_matches_its_source() -> None:
    spec = importlib.util.spec_from_file_location(
        "make_fixture", FIXTURES / "make_fixture_pmtiles.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.build() == FIXTURE.read_bytes(), "run make_fixture_pmtiles.py and commit"


def test_test_pack_lies_at_the_public_sitl_site() -> None:
    center = read_center(FIXTURE)
    assert center is not None
    assert _distance_m(center.lat, center.lon, CMAC_LAT, CMAC_LON) < 3000
