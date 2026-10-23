"""sd_schemas.geo: the tangent plane the gate measures in."""

from __future__ import annotations

import math

from sd_schemas.geo import plane_distance_m, plane_offset, tangent_radii, wrap_lon


def test_offsets_are_metres_on_the_ellipsoid() -> None:
    lat0, lon0 = -35.363261, 149.165230
    east_r, north_r = tangent_radii(lat0)
    lat, lon = plane_offset(lat0, lon0, 100.0, 200.0)
    assert math.isclose(math.radians(lat - lat0) * north_r, 200.0, abs_tol=1e-9)
    assert math.isclose(math.radians(lon - lon0) * east_r, 100.0, abs_tol=1e-9)
    # About 111 km per degree of latitude, less per degree of longitude away from the equator.
    assert 110_500 < math.radians(1) * north_r < 111_500
    assert math.radians(1) * east_r < math.radians(1) * north_r


def test_longitude_wraps_at_the_date_line() -> None:
    assert wrap_lon(190.0) == -170.0
    assert wrap_lon(-190.0) == 170.0
    lat, lon = plane_offset(0.0, 179.9999, 1000.0, 0.0)
    assert lat == 0.0 and -180.0 <= lon < -179.98


def test_plane_distance_is_the_offset_back() -> None:
    lat0, lon0 = -35.363261, 149.165230
    lat, lon = plane_offset(lat0, lon0, 30.0, 40.0)
    assert math.isclose(plane_distance_m(lat0, lon0, lat, lon), 50.0, rel_tol=1e-4)
    assert plane_distance_m(lat0, lon0, lat0, lon0) == 0.0
    # across the date line it is the short way round, not most of the way round the earth
    east = plane_offset(0.0, 179.9999, 20.0, 0.0)
    assert math.isclose(plane_distance_m(0.0, 179.9999, *east), 20.0, rel_tol=1e-3)
