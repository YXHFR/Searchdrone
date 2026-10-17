"""sd_schemas.geo: the tangent plane the gate measures in."""

from __future__ import annotations

import math

from sd_schemas.geo import plane_offset, tangent_radii, wrap_lon


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
