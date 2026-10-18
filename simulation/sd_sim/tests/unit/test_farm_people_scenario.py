"""The farm's person models (catalog, fetch, SDF) and the drawn scenario."""

from __future__ import annotations

import hashlib
import io
import math
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from sd_sim import objects, world
from sd_sim.farm import people, scenario
from sd_sim.farm.people import CATALOG, Person, PersonModel

AREA = (-60.0, 20.0, 60.0, 140.0)


def test_catalog_is_pinned_and_licensed() -> None:
    assert len(CATALOG) >= 5
    assert {m.license for m in CATALOG} <= {"CC0-1.0", "CC-BY-4.0"}
    assert {m.pose for m in CATALOG} == {"standing", "sitting"}
    for m in CATALOG:
        assert len(m.sha256) == 64 and m.credit
        assert m.url.startswith("https://fuel.gazebosim.org/1.0/")
        x0, x1, y0, y1, z0, z1 = m.bounds_m
        assert x0 < x1 and y0 < y1 and z0 == 0.0 and 0.5 < z1 < 2.1


def test_person_sdf_passes_the_world_check() -> None:
    for tint in (None, (0.1, 0.2, 0.6)):
        for m in CATALOG:
            pose = "sitting" if m.pose == "sitting" else "lying"
            p = Person("person_3", m.id, pose, 20.0, 30.0, 45.0, tint=tint)
            text = world.checked_fragment(p)
            assert f"model://{m.id}/{m.mesh}" in text
            assert "<label>2</label>" in text
    objects.check([Person("person_0", "sd_person_walking", "standing", 20, 30, 0)])


def test_person_poses_are_checked() -> None:
    with pytest.raises(objects.ObjectError):
        Person("p", "sd_person_sitting", "standing", 10, 10, 0)
    with pytest.raises(objects.ObjectError):
        Person("p", "sd_person_walking", "sitting", 10, 10, 0)
    with pytest.raises(objects.ObjectError):
        Person("p", "nobody", "standing", 10, 10, 0)
    with pytest.raises(objects.ObjectError):
        Person("p", "sd_person_walking", "standing", 10, 10, 0, tint=(2.0, 0, 0))


def test_person_round_trips_through_the_truth() -> None:
    p = Person(
        "person_0", "sd_person_rescue_randy", "lying", 1.5, -2.0, 33.0, (0.2, 0.3, 0.4), True
    )
    row = {
        "name": p.name,
        "east_m": 1.5,
        "north_m": -2.0,
        "up_m": 0.0,
        "yaw_deg": 33.0,
        "truth": dict(p.truth),
    }
    assert Person.from_truth(row) == p
    assert p.truth["attribution"]["license"] == "CC-BY-4.0"


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def _model(data: bytes) -> PersonModel:
    return replace(CATALOG[0], sha256=hashlib.sha256(data).hexdigest(), mesh="meshes/m.dae")


def test_unpack_keeps_meshes_and_textures_only(tmp_path: Path) -> None:
    data = _zip(
        {
            "model.sdf": b"<sdf/>",
            "meshes/m.dae": b"<COLLADA/>",
            "materials/textures/t.png": b"png",
            "thumbnails/1.png": b"thumb",
        }
    )
    dest = tmp_path / "m"
    people.unpack(_model(data), data, dest)
    got = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    assert got == [".sha256", "materials/textures/t.png", "meshes/m.dae"]


def test_unpack_refuses_bad_archives(tmp_path: Path) -> None:
    data = _zip({"meshes/m.dae": b"x"})
    with pytest.raises(people.PeopleError, match="SHA-256"):
        people.unpack(replace(CATALOG[0], mesh="meshes/m.dae"), data, tmp_path / "a")
    evil = _zip({"meshes/../../evil.dae": b"x", "meshes/m.dae": b"x"})
    with pytest.raises(people.PeopleError, match="sökväg"):
        people.unpack(_model(evil), evil, tmp_path / "b")
    assert not (tmp_path / "evil.dae").exists()
    no_mesh = _zip({"meshes/other.dae": b"x"})
    with pytest.raises(people.PeopleError, match="saknas"):
        people.unpack(_model(no_mesh), no_mesh, tmp_path / "c")
    assert not (tmp_path / "c").exists()


def test_fetch_downloads_each_model_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archives = {}
    catalog = []
    for m in CATALOG[:2]:
        data = _zip({m.mesh: m.id.encode()})
        archives[m.url] = data
        catalog.append(replace(m, sha256=hashlib.sha256(data).hexdigest()))
    monkeypatch.setattr(people, "CATALOG", tuple(catalog))
    calls: list[str] = []

    def download(url: str) -> bytes:
        calls.append(url)
        return archives[url]

    assert people.fetch(tmp_path, download, say=lambda _m: None) == [m.id for m in catalog]
    assert people.missing(tmp_path) == []
    assert people.fetch(tmp_path, download, say=lambda _m: None) == []
    assert len(calls) == 2


def test_scenario_is_seeded_and_inside_the_area() -> None:
    a = scenario.sample(5, ["sd_lake_forest", "w2"], AREA, 59.3)
    b = scenario.sample(5, ["sd_lake_forest", "w2"], AREA, 59.3)
    assert a.as_dict() == b.as_dict()
    assert scenario.sample(6, ["sd_lake_forest"], AREA, 59.3).as_dict() != a.as_dict()
    for seed in range(40):
        s = scenario.sample(seed, ["sd_lake_forest"], AREA, 59.3)
        assert 1 <= len(s.people) <= scenario.MAX_PEOPLE
        assert s.people[0].is_target and not any(p.is_target for p in s.people[1:])
        assert scenario.HEIGHTS_M[0] <= s.height_m <= scenario.HEIGHTS_M[1]
        assert s.sun.elevation_deg >= scenario.MIN_SUN_ELEVATION_DEG
        assert s.weather in scenario.WEATHER
        objs = s.objects()
        objects.check(objs)
        for o in objs:
            assert AREA[0] < o.east_m < AREA[2] and AREA[1] < o.north_m < AREA[3]
        assert s.as_dict()["schema"] == scenario.SCHEMA


def test_scenario_varies_poses_and_clothes() -> None:
    poses, tints = set(), set()
    for seed in range(200):
        for p in scenario.sample(seed, ["w"], AREA, 59.3).people:
            poses.add(p.pose)
            tints.add(p.tint)
    assert poses == {"standing", "sitting", "lying"}
    assert None in tints and len(tints) > 5


def test_sun_position() -> None:
    # Spring equinox (day 80) at solar noon: elevation 90 - latitude, due south.
    noon = scenario.sun_position(59.3, 80, 12.0)
    assert noon.elevation_deg == pytest.approx(30.7, abs=1.5)
    assert noon.azimuth_deg == pytest.approx(180.0, abs=0.5)
    morning = scenario.sun_position(59.3, 172, 6.0)
    assert 60.0 < morning.azimuth_deg < 120.0  # in the east
    d = noon.direction()
    assert math.hypot(*d) == pytest.approx(1.0)
    assert d[1] > 0 and d[2] < 0  # southern sun: the light travels north and down
