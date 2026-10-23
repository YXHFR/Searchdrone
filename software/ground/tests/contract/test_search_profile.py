"""B2-05: the search profile on the ground, end to end against a migrated database.

The AI proposes over the commander API, a pilot approves over the panel API and the
search service reads over its own API. A small stand-in for ``sd-search``
(``MockSearch``) reads the profile the way the matching will: it assesses the profile's
fields only when an approved version exists (E10). Every description here is made up.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))

from flow_rig import BRAIN, H, Rig, rig  # pyright: ignore[reportMissingImports]

from sd_ground.app import commander_asgi, create_commander_app
from sd_ground.auth import ServiceToken, SessionRecord, new_token
from sd_ground.auth import hash_token as hash_session_token
from sd_ground.logging import setup_logging

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]

SEARCH = "search-token-for-profile-tests-0123456789"
MARKER = "PAHITTAD-PROFILMARKOR-5c1e"  # unique text that must never reach a log
AS_BRAIN = {"Authorization": f"Bearer {BRAIN}"}
AS_SEARCH = {"Authorization": f"Bearer {SEARCH}"}


@pytest.fixture
async def r(migrated_db: Db, tmp_path: Path) -> AsyncIterator[Rig]:
    async with rig(migrated_db.name, migrated_db.app_password, tmp_path) as built:
        built.g.search_token = ServiceToken(SEARCH)
        yield built


def draft(r: Rig, *, color: str = "red", note: str = "") -> dict[str, Any]:
    lat, lon = r.at(40, 60)
    return {
        "subject_category": "dementia",
        "mobility": "reduced",
        "age_band": "senior",
        "garments": [
            {"part": "outer", "kind": "jacka", "colors": [color], "certainty": "certain"},
            {"part": "lower", "kind": "byxor", "colors": ["black"], "certainty": "likely"},
        ],
        "items": [{"kind": "cane", "certainty": "unsure"}],
        "likely_state": ["walking"],
        "last_seen": {"lat": lat, "lon": lon, "at_utc": "2026-10-20T12:00:00Z"},
        "visual_notes": "vit mössa",
        "free_text": f"Påhittad person, går långsamt. {note}",
    }


def search_client(r: Rig) -> httpx.AsyncClient:
    app = commander_asgi(r.g, create_commander_app(r.g))
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),  # pyright: ignore[reportArgumentType]
        base_url="http://127.0.0.1:7700",
        headers=AS_SEARCH,
    )


async def observer_token(r: Rig) -> str:
    """An observer CLI token, as ``sd-ground token create --role observer`` makes it."""
    assert r.g.repo is not None
    token = new_token()
    now = dt.datetime.now(dt.UTC)
    rec = SessionRecord(
        id="",
        kind="token",
        role="observer",
        issuer_ref="operator:observer-test",
        auth_mode="cli",
        token_sha256=hash_session_token(token),
        created_at=now,
        last_used_at=now,
        expires_at=now + dt.timedelta(hours=1),
        name="observer-test",
    )
    await r.g.repo.save_session(rec)
    r.g.sessions.load(await r.g.repo.sessions())
    return token


async def propose(r: Rig, mid: str, body: dict[str, Any]) -> httpx.Response:
    return await r.commander.post(
        f"/commander/v1/missions/{mid}/profile-proposals", json={"profile": body}, headers=AS_BRAIN
    )


async def seen(r: Rig, mid: str) -> tuple[int, str | None]:
    """What the card shows: the approved version (0 for none) and the pending proposal."""
    state = (await r.panel.get(f"/api/v1/missions/{mid}/search-profile")).json()
    version = 0 if state["approved"] is None else state["approved"]["profile_version"]
    return version, state["proposal_id"]


async def approve(
    r: Rig,
    mid: str,
    body: dict[str, Any],
    *,
    base: tuple[int, str | None] | None = None,
    **headers: str,
) -> httpx.Response:
    """The card's approval, based on what it shows now unless ``base`` says otherwise."""
    version, proposal_id = base if base is not None else await seen(r, mid)
    return await r.panel.post(
        f"/api/v1/missions/{mid}/search-profile",
        json={"profile": body, "base_version": version, "proposal_id": proposal_id},
        headers=H | headers,
    )


class MockSearch:
    """Stands in for ``sd-search`` until it exists: it reads the profile in force over
    the search API and asks the profile's field questions only when one is approved."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def assess(self, mission_id: str) -> dict[str, Any]:
        res = await self.client.get(f"/search/v1/missions/{mission_id}/search-profile")
        assert res.status_code == 200, res.text
        profile = res.json()
        if profile is None:
            return {"profile_version": 0, "fields": []}  # only "is it a person"
        assert profile["approved_by"] is not None
        keys = [f"garment:{i}" for i in range(len(profile["garments"]))]
        keys += [f"item:{i}" for i in range(len(profile["items"]))]
        return {"profile_version": profile["profile_version"], "fields": [*keys, "state"]}


async def test_proposal_then_approved_versions(r: Rig) -> None:
    mid = await r.open_mission()
    res = await propose(r, mid, draft(r, color="orange"))
    assert res.status_code == 201, res.text
    proposal_id = res.json()["proposal_id"]

    state = (await r.panel.get(f"/api/v1/missions/{mid}/search-profile")).json()
    assert state["approved"] is None
    assert state["proposal_id"] == proposal_id
    assert state["proposed_by"] == "brain:ground"
    assert state["proposal"]["approved_by"] is None
    assert state["proposal"]["profile_version"] == 1
    assert state["proposal"]["garments"][0]["colors"] == ["orange"]

    # The pilot corrects the colour on the card and approves: version 1.
    fixed = draft(r, color="red")
    res = await approve(r, mid, fixed)
    assert res.status_code == 201, res.text
    v1 = res.json()
    assert v1["profile_version"] == 1 and v1["approved_by"] == "operator:ci-harness"
    assert v1["garments"][0]["colors"] == ["red"]
    state = (await r.panel.get(f"/api/v1/missions/{mid}/search-profile")).json()
    assert state["approved"]["profile_version"] == 1
    assert state["proposal"] is None and state["proposal_id"] is None
    profile_id = v1["profile_id"]

    # A change after approval is a new version; the id stays the same.
    res = await approve(r, mid, fixed | {"likely_state": ["sitting"]})
    assert res.status_code == 201, res.text
    v2 = res.json()
    assert (v2["profile_version"], v2["profile_id"]) == (2, profile_id)

    # A later proposal waits as the version it would become; the matching keeps v2.
    assert (await propose(r, mid, draft(r, color="blue"))).status_code == 201
    state = (await r.panel.get(f"/api/v1/missions/{mid}/search-profile")).json()
    assert state["approved"]["profile_version"] == 2
    assert state["proposal"]["profile_version"] == 3
    assert state["proposal"]["profile_id"] == profile_id
    versions = (await r.panel.get(f"/api/v1/missions/{mid}/search-profile/versions")).json()
    assert [v["profile_version"] for v in versions] == [1, 2]
    async with search_client(r) as client:
        got = (await client.get(f"/search/v1/missions/{mid}/search-profile")).json()
    assert got["profile_version"] == 2 and got["likely_state"] == ["sitting"]
    types = r.bus_types()
    assert types.count("search_profile.proposed") == 2
    assert types.count("search_profile.approved") == 2


async def test_profile_requires_pilot(r: Rig) -> None:
    """Only a pilot approves: an observer, the AI's token and the search token are refused."""
    mid = await r.open_mission()
    body = draft(r)
    observer = await observer_token(r)
    res = await approve(r, mid, body, Authorization=f"Bearer {observer}")
    assert res.status_code == 403, res.text
    # An observer still reads the card.
    res = await r.panel.get(
        f"/api/v1/missions/{mid}/search-profile", headers={"Authorization": f"Bearer {observer}"}
    )
    assert res.status_code == 200, res.text
    res = await approve(r, mid, body, Authorization=f"Bearer {BRAIN}")
    assert res.status_code == 403, res.text
    res = await approve(r, mid, body, Authorization=f"Bearer {SEARCH}")
    assert res.status_code in (401, 403), res.text
    # The commander API has no approval route, and its proposal is never an approval.
    res = await r.commander.post(
        f"/commander/v1/missions/{mid}/search-profile", json=body, headers=AS_BRAIN
    )
    assert res.status_code in (404, 405)
    res = await r.commander.post(
        f"/commander/v1/missions/{mid}/profile-proposals",
        json={"profile": body | {"approved_by": "operator:ci-harness"}},
        headers=AS_BRAIN,
    )
    assert res.status_code == 422, res.text
    # The search token reads, but proposes nothing.
    async with search_client(r) as client:
        res = await client.post(f"/commander/v1/missions/{mid}/profile-proposals", json={})
        assert res.status_code == 403
    state = (await r.panel.get(f"/api/v1/missions/{mid}/search-profile")).json()
    assert state == {"approved": None, "proposal": None, "proposal_id": None, "proposed_by": None}


async def test_unapproved_profile_not_used(r: Rig) -> None:
    """E10: with only the AI's proposal the matching asks no field question."""
    mid = await r.open_mission()
    assert (await propose(r, mid, draft(r))).status_code == 201
    async with search_client(r) as client:
        search = MockSearch(client)
        assert await search.assess(mid) == {"profile_version": 0, "fields": []}
        res = await client.get(f"/search/v1/missions/{mid}/search-profile")
        assert res.json() is None
        assert (await approve(r, mid, draft(r))).status_code == 201
        assert await search.assess(mid) == {
            "profile_version": 1,
            "fields": ["garment:0", "garment:1", "item:0", "state"],
        }
        # A newer proposal does not replace the approved version.
        assert (await propose(r, mid, draft(r, color="green"))).status_code == 201
        assert (await search.assess(mid))["profile_version"] == 1
        # Without the search token there is nothing to read.
        res = await client.get(
            f"/search/v1/missions/{mid}/search-profile", headers={"Authorization": ""}
        )
        assert res.status_code == 401


async def test_stale_approval_is_refused(r: Rig) -> None:
    """An approval covers only what the card showed: a newer proposal or version is 409."""
    mid = await r.open_mission()
    assert (await propose(r, mid, draft(r, color="orange"))).status_code == 201
    shown = await seen(r, mid)
    # The operator adds a detail while the pilot edits: the AI proposes again.
    assert (await propose(r, mid, draft(r, color="green"))).status_code == 201
    res = await approve(r, mid, draft(r), base=shown)
    assert res.status_code == 409 and res.json()["code"] == "PROFILE_STALE"
    # A new profile of the pilot's own does not skip a pending proposal either.
    res = await approve(r, mid, draft(r), base=(0, None))
    assert res.status_code == 409
    assert (await approve(r, mid, draft(r))).status_code == 201
    # Another approval in between: the old base is stale.
    res = await approve(r, mid, draft(r), base=(0, None))
    assert res.status_code == 409
    assert (await approve(r, mid, draft(r), base=(1, None))).status_code == 201
    state = (await r.panel.get(f"/api/v1/missions/{mid}/search-profile")).json()
    assert state["approved"]["profile_version"] == 2


async def test_closed_mission_takes_no_profile(r: Rig) -> None:
    mid = await r.open_mission()
    res = await r.panel.post(f"/api/v1/missions/{mid}/end", json={"reason": "test"}, headers=H)
    assert res.status_code == 200, res.text
    assert (await approve(r, mid, draft(r))).status_code == 409
    assert (await propose(r, mid, draft(r))).status_code == 409


async def test_invalid_profile_is_refused(r: Rig) -> None:
    mid = await r.open_mission()
    bad = draft(r) | {"garments": [{"part": "outer", "colors": ["magenta"], "certainty": "x"}]}
    res = await propose(r, mid, bad)
    assert res.status_code == 422 and res.json()["code"] == "VALIDATION"
    res = await approve(r, mid, draft(r) | {"face": "x"})
    assert res.status_code == 422
    res = await approve(r, mid, {k: v for k, v in draft(r).items() if k != "last_seen"})
    assert res.status_code == 422


async def test_profile_text_not_logged(r: Rig, tmp_path: Path) -> None:
    """The profile's text never reaches the log file, the event log or the live stream:
    those carry the profile's id and version only."""
    log_path = tmp_path / "logs" / "sd-ground.jsonl"
    setup_logging(log_path, level=logging.DEBUG)
    try:
        mid = await r.open_mission()
        body = draft(r, note=MARKER) | {"visual_notes": f"mössa {MARKER}"}
        assert (await propose(r, mid, body)).status_code == 201
        assert (await propose(r, mid, body | {"garments": "wrong"})).status_code == 422
        assert (await approve(r, mid, body)).status_code == 201
        assert (await approve(r, mid, body | {"hair": {"color": "x"}})).status_code == 422
        await r.panel.get(f"/api/v1/missions/{mid}/search-profile")
        async with search_client(r) as client:
            got = await client.get(f"/search/v1/missions/{mid}/search-profile")
            assert MARKER in got.text  # it is stored and served, just never logged
        for handler in logging.getLogger().handlers:
            handler.flush()
    finally:
        for handler in list(logging.getLogger().handlers):
            handler.close()
            logging.getLogger().removeHandler(handler)
    logged = log_path.read_text(encoding="utf-8")
    assert "search_profile.approved" in logged
    assert MARKER not in logged
    stream = json.dumps(
        [e.data for e in r.g.bus.replay(r.g.bus.server_epoch, 0) or []], ensure_ascii=False
    )
    assert "search_profile.proposed" in json.dumps(r.bus_types()) and MARKER not in stream
    await r.g.flush_events()
    assert r.g.repo is not None
    async with r.g.repo.engine.connect() as conn:
        rows = (
            await conn.execute(
                text("SELECT type, payload::text FROM events WHERE mission_id = :m"), {"m": mid}
            )
        ).all()
    assert {"search_profile.proposed", "search_profile.approved"} <= {t for t, _ in rows}
    assert not [t for t, p in rows if MARKER in p]
