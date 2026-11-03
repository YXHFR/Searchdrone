"""The closer look's facts on the ground without a database: the height floor from the
charter's map pack, overflight from the operation mode and the permit, and where a
candidate lies for the AI's card."""

from __future__ import annotations

import datetime as dt
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SAFETY_TESTS = Path(__file__).resolve().parents[3] / "common" / "safety" / "tests" / "unit"
sys.path.insert(0, str(SAFETY_TESTS))

from floor_fixtures import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    GROUND_M,
    at,
    make_pack,
    write_pack,
)
from gate_fixtures import CHARTER, ctx  # noqa: E402  # pyright: ignore[reportMissingImports]

from sd_ground.services import inspections  # noqa: E402
from sd_schemas.gate import GateContext  # noqa: E402
from sd_schemas.inspect import Permit  # noqa: E402

PACK = "testpack"


def flow(root: Path, permit: Permit | None = None, cfg: Permit | None = None) -> Any:
    loaded = SimpleNamespace(config=SimpleNamespace(permit=cfg or Permit()))
    return SimpleNamespace(
        data_root=lambda: root,
        now=lambda: dt.datetime(2026, 6, 1, 12, tzinfo=dt.UTC),
        permit=permit,
        g=SimpleNamespace(loaded=loaded),
    )


def over_the_pack(sha: str | None) -> GateContext:
    """The drone 60 m above the pack's flat ground, at its centre; the charter names it."""
    base = ctx()
    assert base.vehicle is not None
    here = at(0, 0)
    vehicle = base.vehicle.model_copy(
        update={"lat": here.lat, "lon": here.lon, "alt_amsl_m": GROUND_M + 60.0}
    )
    charter = CHARTER.model_copy(
        update={"floor_pack_id": None if sha is None else PACK, "floor_pack_sha256": sha}
    )
    return base.model_copy(update={"vehicle": vehicle, "charter": charter})


def target(east: float = 0.0, north: float = 20.0) -> dict[str, Any]:
    p = at(east, north)
    return {"candidate_id": "x", "lat": p.lat, "lon": p.lon}


@pytest.fixture
def pack_dir(tmp_path: Path) -> tuple[Path, str]:
    folder = write_pack(tmp_path / "floor" / PACK, make_pack(tree_m=20.0))
    inspections._load_pack.cache_clear()  # pyright: ignore[reportPrivateUsage]
    return tmp_path, hashlib.sha256((folder / "floor_pack.toml").read_bytes()).hexdigest()


def test_the_floor_comes_from_the_charters_pack(pack_dir: tuple[Path, str]) -> None:
    root, sha = pack_dir
    floor = inspections._floor(flow(root), over_the_pack(sha), target())  # pyright: ignore[reportPrivateUsage]
    assert floor is not None and floor.reason == "map"
    # Never lower than 30 m above the ground (the automatic floor), above the trees.
    assert floor.floor_amsl_m >= GROUND_M + 30.0


def test_without_the_pack_the_ground_leaves_the_floor_to_the_drone(
    pack_dir: tuple[Path, str],
) -> None:
    root, sha = pack_dir
    f = inspections._floor  # pyright: ignore[reportPrivateUsage]
    assert f(flow(root), over_the_pack(None), target()) is None  # the charter names none
    assert f(flow(root), over_the_pack("0" * 64), target()) is None  # another pack's hash
    assert f(flow(root / "elsewhere"), over_the_pack(sha), target()) is None  # not here
    assert f(flow(root), over_the_pack(sha), {"candidate_id": "x"}) is None  # no target
    no_vehicle = over_the_pack(sha).model_copy(update={"vehicle": None})
    assert f(flow(root), no_vehicle, target()) is None


def test_overflight_follows_the_mode_and_the_permit() -> None:
    root = Path("/nonexistent")
    assert inspections.overflight_allowed(flow(root), "sim")
    assert inspections.overflight_allowed(flow(root), "exercise")
    assert not inspections.overflight_allowed(flow(root), "live")
    assert not inspections.overflight_allowed(flow(root), None)
    granted = Permit(specific_category_ref="OP-1", allow_overflight_live=True)
    assert inspections.overflight_allowed(flow(root, cfg=granted), "live")
    # The pilot's change wins over the config until the ground restarts.
    assert not inspections.overflight_allowed(flow(root, permit=Permit(), cfg=granted), "live")
    assert inspections.permit(flow(root, cfg=granted)) == granted


def test_the_cards_direction() -> None:
    d = inspections._direction  # pyright: ignore[reportPrivateUsage]
    north = at(0, 140)
    dist, way = d(at(0, 0).lat, at(0, 0).lon, north.lat, north.lon)
    assert way == "N" and dist == pytest.approx(140, abs=1)
    for (east, north_m), want in {
        (100, 100): "NO",
        (100, 0): "O",
        (0, -100): "S",
        (-100, -100): "SV",
        (-100, 0): "V",
    }.items():
        p = at(east, north_m)
        assert d(at(0, 0).lat, at(0, 0).lon, p.lat, p.lon)[1] == want
