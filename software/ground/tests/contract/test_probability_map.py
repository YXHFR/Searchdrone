"""B2-08: the probability map end to end on the ground. ``sd-search``'s map loop (in
process, over the real search API) reads the inputs and the stills, writes versions, and
the panel and the AI read them back. The ground's checks on a version are tested on the
API directly."""

from __future__ import annotations

import base64
import struct
import sys
import zlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))

from flow_rig import BRAIN, SEARCH, VID, H, Rig, rig  # pyright: ignore[reportMissingImports]

from sd_ground.app import commander_asgi, create_commander_app
from sd_schemas.base import LatLon, new_id
from sd_schemas.link import DetectorInfo, PerceptionReport, Pose
from sd_search.client import GroundClient
from sd_search.mapper import MapService
from sd_search.probmap.terrain import NoTerrain
from sd_search.profile import GroundProfileSource

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]

AS_BRAIN = {"Authorization": f"Bearer {BRAIN}"}


@pytest.fixture
async def r(migrated_db: Db, tmp_path: Path) -> AsyncIterator[Rig]:
    async with rig(migrated_db.name, migrated_db.app_password, tmp_path) as built:
        yield built


async def approve_profile(r: Rig, mid: str, east: float = 0, north: float = 0) -> None:
    lat, lon = r.at(east, north)
    body = {
        "subject_category": "dementia",
        "garments": [{"part": "upper", "colors": ["red"], "certainty": "certain"}],
        "last_seen": {"lat": lat, "lon": lon, "at_utc": "2026-10-20T12:00:00Z"},
    }
    state = (await r.panel.get(f"/api/v1/missions/{mid}/search-profile")).json()
    version = 0 if state["approved"] is None else state["approved"]["profile_version"]
    res = await r.panel.post(
        f"/api/v1/missions/{mid}/search-profile",
        json={"profile": body, "base_version": version, "proposal_id": None},
        headers=H,
    )
    assert res.status_code == 201, res.text


def mapper(r: Rig) -> tuple[MapService, GroundClient]:
    transport = httpx.ASGITransport(app=commander_asgi(r.g, create_commander_app(r.g)))  # pyright: ignore[reportArgumentType]
    client = GroundClient("http://127.0.0.1:7700", SEARCH, transport=transport)
    return MapService(client, GroundProfileSource(client), NoTerrain()), client


async def still(r: Rig, east: float, north: float, half: float = 30.0) -> None:
    lat, lon = r.at(east, north)
    offsets = ((-half, half), (half, half), (half, -half), (-half, -half))
    corners = [r.at(east + e, north + n) for e, n in offsets]
    rep = PerceptionReport(
        still_id=new_id(),
        command_id=None,
        captured_utc=r.clock.utc_now(),
        sensor="rgb",
        pose=Pose(
            lat=lat,
            lon=lon,
            alt_amsl_m=624,
            agl_m=40,
            roll_deg=0,
            pitch_deg=0,
            yaw_deg=0,
            source_ts_ms=0,
        ),
        footprint=[LatLon(lat=la, lon=lo) for la, lo in corners],
        px_per_m=60.0,
        synthetic=True,
        detector=DetectorInfo(
            name="sd_detect:test",
            license="Apache-2.0",
            version="0",
            placement="onboard",
            ok=True,
            tiles=1,
            latency_ms=30,
            rate_hz=1,
        ),
        detections=[],
    )
    await r.onboard.session.send(rep, ttl_ms=30_000)
    q = r.flow.reports.get(VID)
    before = 0 if q is None else q.handled

    def done() -> bool:
        q = r.flow.reports.get(VID)
        return q is not None and q.handled > before and not q.running

    await r.until(done, 10, "report handled")


def f16(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype="<f2").astype(np.float64)


def z32(a: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(a.astype("<f4").tobytes())).decode()


async def test_the_map_reaches_the_panel_and_the_ai(r: Rig) -> None:
    mid = await r.ready_mission(half=300)
    service, client = mapper(r)
    try:
        assert await service.step([mid]) == 0  # no approved profile: no map
        none = await r.panel.get(f"/api/v1/missions/{mid}/probability-map")
        assert none.status_code == 404
        assert (await r.panel.get(f"/api/v1/missions/{mid}/sectors/poa")).json() == []

        await approve_profile(r, mid)
        assert await service.step([mid], force=True) == 1
        res = await r.panel.get(f"/api/v1/missions/{mid}/probability-map")
        assert res.status_code == 200, res.text
        m = res.json()
        assert m["version"] == 1 and m["reason"] == "prior" and not m["frozen"]
        grid = m["grid"]
        p = f16(m["p_b64"])
        assert p.size == grid["nx"] * grid["ny"]
        assert 64 <= grid["nx"] <= 66  # 600 m + 2 x 500 m, in whole cells around the IPP
        assert abs(p.sum() + m["row"] - 1) < 0.01
        assert f16(m["pod_b64"]).max() == 0
        assert "no_terrain" in m["notes"]
        assert grid["west"] < r.home[1] < grid["east"] and grid["south"] < r.home[0] < grid["north"]

        png = await r.panel.get(f"/api/v1/missions/{mid}/probability-map.png?layer=poa")
        assert png.status_code == 200 and png.headers["content-type"] == "image/png"
        assert png.content[:8] == b"\x89PNG\r\n\x1a\n"
        w, h = struct.unpack(">II", png.content[16:24])
        assert (w, h) == (grid["nx"], grid["ny"])
        bounds = [float(v) for v in png.headers["sd-bounds"].split(",")]
        assert bounds == pytest.approx([grid[k] for k in ("west", "south", "east", "north")])
        bad = await r.panel.get(f"/api/v1/missions/{mid}/probability-map.png?layer=x")
        assert bad.status_code == 422

        sectors = (await r.panel.get(f"/api/v1/missions/{mid}/sectors/poa")).json()
        assert len(sectors) >= 2
        assert [s["psr"] for s in sectors] == sorted((s["psr"] for s in sectors), reverse=True)
        assert all(s["covered_pct"] == 0 for s in sectors)

        res = await r.commander.get(f"/commander/v1/missions/{mid}/situation", headers=AS_BRAIN)
        line = res.json()["probability_line"]
        assert line.startswith("Sektorer (POA kvar / täckt / medel-POD / PSR): ")

        types = [e.type for e in r.g.bus.replay(r.g.bus.server_epoch, 0) or []]
        assert types.count("probability_map.updated") == 1

        # A still over the IPP: the next version counts its cells down.
        await still(r, 0, 0)
        feed = await client.stills(mid, 0, 10)
        assert len(feed) == 1 and feed[0].detector_ok and len(feed[0].footprint) == 4
        service.missions[mid].written_at = -1e9  # past the 30 s
        assert await service.step([mid], force=True) == 1
        m2 = (await r.panel.get(f"/api/v1/missions/{mid}/probability-map")).json()
        assert m2["version"] == 2 and m2["reason"] == "search" and m2["event_seq"] >= 1
        pod = f16(m2["pod_b64"])
        assert 0.3 < pod.max() <= 0.99
        assert f16(m2["p_b64"]).max() < p.max()  # the IPP's cells were the highest
        assert m2["row"] > m["row"]
    finally:
        await client.aclose()


async def test_the_ground_checks_every_version(r: Rig) -> None:
    mid = await r.ready_mission(half=300)
    await approve_profile(r, mid)
    service, client = mapper(r)
    try:
        assert await service.step([mid]) == 1
    finally:
        await client.aclose()
    assert r.search is not None
    good = (await r.panel.get(f"/api/v1/missions/{mid}/probability-map")).json()
    n = good["grid"]["nx"] * good["grid"]["ny"]
    p = np.full(n, (1 - 0.2) / n)
    base: dict[str, Any] = {
        "version": 2,
        "reason": "search",
        "grid": good["grid"],
        "row": 0.2,
        "event_seq": 0,
        "profile_version": 1,
        "p_z": z32(p),
        "pod_z": z32(np.zeros(n)),
    }
    url = f"/search/v1/missions/{mid}/probability-maps"

    async def post(**kw: Any) -> httpx.Response:
        return await r.search.post(url, json=base | kw)  # type: ignore[union-attr]

    stale = await post(version=1)
    assert stale.status_code == 409 and stale.json()["code"] == "MAP_VERSION_STALE"
    assert (await post(version=3)).status_code == 409
    old = await post(profile_version=7)
    assert old.status_code == 409 and old.json()["code"] == "MAP_PROFILE_STALE"
    short = await post(p_z=z32(p[:-1]))
    assert short.status_code == 422 and short.json()["code"] == "MAP_INVALID"
    big = await post(p_z=z32(np.concatenate([p, p])))
    assert big.status_code == 422
    assert (await post(p_z=z32(np.full(n, 2.0)))).status_code == 422
    assert (await post(p_z=z32(np.full(n, np.nan)))).status_code == 422
    assert (await post(row=0.5)).status_code == 422  # the sum is not 1
    assert (await post(p_z="not base64!")).status_code == 422
    moved = base["grid"] | {"ipp_lat": base["grid"]["ipp_lat"] + 0.001}
    assert (await post(grid=moved)).status_code == 422
    # bounds come from the grid, not from the writer
    lying = base["grid"] | {"west": 0.0, "south": 0.0, "east": 1.0, "north": 1.0}
    ok = await post(grid=lying)
    assert ok.status_code == 201, ok.text
    got = (await r.panel.get(f"/api/v1/missions/{mid}/probability-map")).json()
    assert got["version"] == 2 and got["grid"]["west"] == pytest.approx(good["grid"]["west"])

    frozen = await post(version=3, reason="confirmed", frozen=True)
    assert frozen.status_code == 201 and frozen.json() == {"version": 3, "frozen": True}
    after = await post(version=4)
    assert after.status_code == 409 and after.json()["code"] == "MAP_FROZEN"

    # Only the search token writes; the panel and the AI cannot.
    assert (await r.panel.post(url, json=base | {"version": 4}, headers=H)).status_code in (
        401,
        403,
        404,
    )
    assert (
        await r.commander.post(url, json=base | {"version": 4}, headers=AS_BRAIN)
    ).status_code == 403
    # The search token is refused on the panel's map.
    res = await r.search.get(f"/api/v1/missions/{mid}/probability-map")
    assert res.status_code in (401, 403, 404)


async def test_an_area_with_a_hint_reaches_the_map_inputs(r: Rig) -> None:
    mid = await r.open_mission()
    area = r.square(50, 100, 100)
    res = await r.panel.post(
        f"/api/v1/missions/{mid}/annotations",
        json={"kind": "area", "label": "Sjön", "geometry": area, "poa_hint": "excluded"},
        headers=H,
    )
    assert res.status_code == 201, res.text
    assert res.json()["poa_hint"] == "excluded"
    plain = await r.panel.post(
        f"/api/v1/missions/{mid}/annotations",
        json={"kind": "area", "label": "Skogen", "geometry": r.square(50, -100, 100)},
        headers=H,
    )
    assert plain.status_code == 201 and plain.json()["poa_hint"] is None
    lat, lon = r.at(0, 0)
    pin = await r.panel.post(
        f"/api/v1/missions/{mid}/annotations",
        json={
            "kind": "pin",
            "label": "M1",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "poa_hint": "high",
        },
        headers=H,
    )
    assert pin.status_code == 422
    assert r.search is not None
    inputs = (await r.search.get(f"/search/v1/missions/{mid}/probability-map/inputs")).json()
    assert inputs["area"] is None and inputs["latest"] is None
    assert [a["poa_hint"] for a in inputs["areas"]] == ["excluded"]
