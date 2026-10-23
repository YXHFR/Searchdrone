"""The ground client (``sd_search.client``): what the ground answers becomes data or a
``GroundError``, never a crash of the service."""

from __future__ import annotations

import httpx
import pytest

from sd_search.client import ConflictError, GroundClient, GroundError


def client(body: object, status: int = 200) -> GroundClient:
    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return GroundClient("http://127.0.0.1:7700", "t", transport=httpx.MockTransport(answer))


@pytest.mark.parametrize("body", [{"bogus": 1}, [{"seq": "x"}], None])
async def test_a_bad_body_is_a_ground_error(body: object) -> None:
    c = client(body)
    try:
        with pytest.raises(GroundError, match="bad body"):
            await c.detections("m", 0, 10)
        with pytest.raises(GroundError, match="bad body"):
            await c.tracks("m")
    finally:
        await c.aclose()


async def test_a_bad_mission_list_is_a_ground_error() -> None:
    c = client([{"id": "no mission_id"}])
    try:
        with pytest.raises(GroundError, match="bad body"):
            await c.missions()
    finally:
        await c.aclose()


async def test_status_codes() -> None:
    c = client({"code": "MISSION_CLOSED"}, 409)
    try:
        with pytest.raises(ConflictError):
            await c.tracks("m")
    finally:
        await c.aclose()
    c = client({"code": "PROFILE_NOT_FOUND"}, 404)
    try:
        assert await c.profile("m") is None
    finally:
        await c.aclose()
