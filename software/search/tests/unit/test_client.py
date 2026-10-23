"""The ground client (``sd_search.client``): what the ground answers becomes data or a
``GroundError``, never a crash of the service."""

from __future__ import annotations

import json

import httpx
import pytest

from sd_schemas.search import CandidateBatch
from sd_search.client import ConflictError, GroundClient, GroundError


def client(body: object, status: int = 200) -> GroundClient:
    def answer(request: httpx.Request) -> httpx.Response:
        content = body if isinstance(body, bytes) else json.dumps(body).encode()
        return httpx.Response(status, content=content)

    return GroundClient("http://127.0.0.1:7700", "t", transport=httpx.MockTransport(answer))


@pytest.mark.parametrize("body", [{"bogus": 1}, [{"seq": "x"}], None, b"not json"])
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
    c = client(None)  # no approved profile yet
    try:
        assert await c.profile("m") is None
    finally:
        await c.aclose()
    c = client({"code": "MISSION_NOT_FOUND"}, 404)
    try:
        with pytest.raises(GroundError):
            await c.profile("m")
    finally:
        await c.aclose()


async def test_a_bad_profile_or_write_answer_is_a_ground_error() -> None:
    c = client({"profile_version": "x"})
    try:
        with pytest.raises(GroundError, match="bad body"):
            await c.profile("m")
        with pytest.raises(GroundError, match="bad body"):
            await c.write("m", CandidateBatch.model_construct(revisions=[]))
    finally:
        await c.aclose()


async def test_a_body_that_is_not_json_is_a_ground_error() -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>proxy error</html>")

    c = GroundClient("http://127.0.0.1:7700", "t", transport=httpx.MockTransport(answer))
    try:
        with pytest.raises(GroundError, match="bad body"):
            await c.detections("m", 0, 10)
    finally:
        await c.aclose()
