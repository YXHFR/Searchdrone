"""The closer look's facts on the ground without a database: the height floor from the
charter's map pack, overflight from the operation mode and the permit, and where a
candidate lies for the AI's card."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sd_ground.services import inspections
from sd_schemas.gate import GateContext, GateDecision
from sd_schemas.inspect import Permit

SAFETY_TESTS = Path(__file__).resolve().parents[3] / "common" / "safety" / "tests" / "unit"


def _fixtures(name: str) -> Any:
    """The safety tests' helper module, loaded by path (their folder is not put on
    ``sys.path``: its ``conftest`` would shadow this package's)."""
    spec = importlib.util.spec_from_file_location(f"safety_{name}", SAFETY_TESTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_floor_fx = _fixtures("floor_fixtures")
_gate_fx = _fixtures("gate_fixtures")
GROUND_M, at, make_pack, write_pack = (
    _floor_fx.GROUND_M,
    _floor_fx.at,
    _floor_fx.make_pack,
    _floor_fx.write_pack,
)
CHARTER, ctx = _gate_fx.CHARTER, _gate_fx.ctx


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


def test_a_revoked_permit_refuses_the_look_at_once() -> None:
    """A live charter approved with overflight, the permit revoked since: the gate's yes
    becomes a refusal (``OVERFLIGHT_NOT_ALLOWED``). Sim, exercise and a charter without
    overflight are not touched."""
    root = Path("/nonexistent")
    granted = Permit(specific_category_ref="OP-1", allow_overflight_live=True)
    live = CHARTER.model_copy(update={"operation_mode": "live", "overflight_allowed": True})
    with_live = ctx().model_copy(update={"charter": live})
    yes = GateDecision(
        decision="ALLOW", reason_codes=[], rule_ids=[], message_sv="", details={}, side="ground"
    )
    assert not inspections.overflight_revoked(flow(root, cfg=granted), live)
    assert inspections.refuse_revoked(flow(root, cfg=granted), with_live, yes) is yes
    revoked = flow(root, permit=Permit(), cfg=granted)
    assert inspections.overflight_revoked(revoked, live)
    no = inspections.refuse_revoked(revoked, with_live, yes)
    assert no.decision == "REJECT" and no.reason_codes == ["OVERFLIGHT_NOT_ALLOWED"]
    assert no.rule_ids[0] == "R-OVERFLIGHT" and no.details["permit_revoked"] is True
    exercise = live.model_copy(update={"operation_mode": "exercise"})
    assert not inspections.overflight_revoked(revoked, exercise)
    assert not inspections.overflight_revoked(revoked, CHARTER)
    assert not inspections.overflight_revoked(revoked, None)


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
