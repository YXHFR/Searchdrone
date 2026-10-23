"""B2-06: cloud matching v1 (cascade steps 1-2) against a fake cloud server.

What leaves the ground (the approved profile as text, at most two clean crops, never data
collection), the answer format and its single repair, the flags for claims a still cannot
show, the pause when the cloud or the budget fails, and the service writing the fast
model's assessment with the fields merged over the code's.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path
from typing import Any

import pytest
from fake_cloud import (  # pyright: ignore[reportMissingImports]
    FakeBudget,
    FakeCloud,
    answer,
    runtime,
)
from fake_ground import FakeGround, utc  # pyright: ignore[reportMissingImports]
from PIL import Image
from search_helpers import (  # pyright: ignore[reportMissingImports]
    MISSION,
    RED,
    at,
    crop_jpeg,
    profile,
)

from sd_brain.runtime.base import ModelError
from sd_schemas.base import new_id
from sd_schemas.link import BBox
from sd_search.app import SearchService, merge_fields
from sd_search.config import SearchConfigError, cloud_key, load
from sd_search.match.cascade import Cascade, skip_cloud
from sd_search.match.engines import CloudMatchEngine, Crop, MatchRequest, claim_flags
from sd_search.match.prompt import (
    UnapprovedProfileError,
    answer_format,
    build_messages,
    context_line,
    profile_text,
)
from sd_search.profile import GroundProfileSource

REPO = Path(__file__).resolve().parents[4]
CID = "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a99"


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _full_profile():
    marker = "PÅHITTAD-MARKÖR-7f3a"
    p = profile(
        free_text=f"Dement, påhittad person. {marker}",
        visual_notes="röd mössa med vit tofs",
        excluded=["inte gul jacka"],
        hair={"color": "gray", "length": "short", "certainty": "unsure"},
        subject_category="dementia",
    )
    return p, marker


def _request(p: Any = None, n: int = 3, size: int = 96) -> MatchRequest:
    crops = [
        Crop(str(new_id()), 40.0 - i, crop_jpeg(RED, size=size, box=size // 3)) for i in range(n)
    ]
    return MatchRequest(mission_id=MISSION, candidate_id=CID, profile=p, crops=crops)


def _parts(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [p for m in body["messages"] if isinstance(m["content"], list) for p in m["content"]]


# --- what is sent -------------------------------------------------------------------------


def test_cloud_payload_fields() -> None:
    """The approved description goes as text (only what an image can show); the ground
    keeps who approved it, ids, the place last seen, the bands and every source."""
    p, marker = _full_profile()
    messages = build_messages(p, [], agl_m=None, px_size=32)
    assert messages[0]["role"] == "system"  # the static part first
    parts = _parts({"messages": messages})
    assert parts and all(part["type"] == "text" for part in parts)
    text = parts[0]["text"].split("SÖKPROFIL: ", 1)[1]
    sent = json.loads(text)
    ground_only = {"approved_by", "approved_at", "mission_id", "profile_id"}
    ground_only |= {"last_seen", "age_band", "height_band"}
    expected = p.model_dump(mode="json", exclude=ground_only)
    for entry in [*expected["garments"], *expected["items"]]:
        entry.pop("source", None)
    if expected.get("hair"):
        expected["hair"].pop("source", None)
    assert {k: v for k, v in sent.items() if k != "field_keys"} == expected
    assert "source" not in text and "last_seen" not in text
    assert sent["field_keys"] == ["garment:0", "hair", "state"]
    assert marker in text and "dementia" in text
    blob = json.dumps(messages, ensure_ascii=False)
    assert "operator:" not in blob and str(p.mission_id) not in blob


def test_cloud_payload_refuses_a_proposal() -> None:
    """E10: a profile no pilot approved is never sent."""
    p = profile(approved_by=None, approved_at=None)
    with pytest.raises(UnapprovedProfileError):
        profile_text(p)
    with pytest.raises(UnapprovedProfileError):
        build_messages(p, [], agl_m=None, px_size=32)


def test_context_line() -> None:
    assert context_line(40, 31.6, 2) == (
        "seen from above at 40 m, person about 32 px, 2 crops from different moments"
    )
    assert context_line(None, 32, 1).startswith("seen from above, person about 32 px")


def test_answer_format_has_only_the_model_fields() -> None:
    fmt = answer_format(["garment:0", "state"])
    schema = fmt["json_schema"]["schema"]
    assert fmt["type"] == "json_schema"
    assert set(schema["properties"]) == {
        "person",
        "person_p",
        "fields",
        "overall",
        "explanation_sv",
    }
    assert schema["properties"]["fields"]["required"] == ["garment:0", "state"]
    assert answer_format([])["json_schema"]["schema"]["properties"]["fields"]["properties"] == {}


async def test_the_request_on_the_wire() -> None:
    """Fast tier: temperature 0, 300 tokens, no reasoning, the answer schema, two clean
    crops (the largest), and never data collection."""
    p, marker = _full_profile()
    cloud, budget = FakeCloud(reply=answer(fields={"garment:0": "match"})), FakeBudget()
    engine = CloudMatchEngine(runtime(cloud, budget))
    a = await engine.assess(_request(p, n=3, size=640))  # too large: scaled to fit 512
    (body,) = cloud.requests
    assert body["provider"]["data_collection"] == "deny"
    assert body["temperature"] == 0 and body["max_tokens"] == 300
    assert "reasoning" not in body
    assert body["response_format"]["type"] == "json_schema"
    images = [part for part in _parts(body) if part["type"] == "image_url"]
    assert len(images) == 2
    for part in images:
        url = part["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as img:
            assert max(img.size) <= 512
            assert not img.getexif()
    assert marker in json.dumps(body, ensure_ascii=False)
    assert a.engine == "fast" and a.fields == {"garment:0": "match"}
    assert len(a.crops_used) == 2
    assert a.cost_sek == pytest.approx(0.01)
    assert budget.reserved[0].purpose == "match" and str(budget.reserved[0].mission_id) == MISSION
    assert len(budget.committed) == 1


async def test_without_a_profile_only_the_person_question() -> None:
    cloud = FakeCloud(reply=answer(fields={"garment:0": "match"}))  # ignored: no such key
    a = await CloudMatchEngine(runtime(cloud, FakeBudget())).assess(_request(None, n=1))
    text = _parts(cloud.requests[0])[0]["text"]
    assert "SÖKPROFIL" not in text and "only whether this is a person" in text
    assert a.fields == {} and a.profile_version == 0


# --- the answer -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Personen rör sig mot skogen.",
        "Någon vinkar.",
        "Personen rörde på sig.",
        "Den går mot vägen.",
        "Someone is moving.",
        "A person waving.",
    ],
)
def test_motion_claim_flag(text: str) -> None:
    assert claim_flags(text) == ["motion_claim"]


@pytest.mark.parametrize(
    "text", ["Jag känner igen personen.", "Samma person som förut.", "Ansiktet syns."]
)
def test_identity_claim_flag(text: str) -> None:
    assert claim_flags(text) == ["identity_claim"]


def test_plain_text_has_no_flag() -> None:
    assert claim_flags("En person i röd jacka står still på gräset.") == []


async def test_motion_claim_reaches_the_assessment() -> None:
    cloud = FakeCloud(reply=answer(explanation_sv="Personen vinkar mot drönaren."))
    a = await CloudMatchEngine(runtime(cloud, FakeBudget())).assess(_request(profile(), n=1))
    assert a.flags == ["motion_claim"]


async def test_format_repaired_once() -> None:
    replies = iter(["inte json", json.dumps(answer())])
    cloud = FakeCloud(reply=lambda _: next(replies))
    budget = FakeBudget()
    a = await CloudMatchEngine(runtime(cloud, budget)).assess(_request(profile(), n=1))
    assert a.flags == ["format_repaired"]
    assert len(cloud.requests) == 2 and len(budget.committed) == 2


async def test_broken_twice_is_a_failed_call() -> None:
    cloud = FakeCloud(reply="{")
    with pytest.raises(ModelError) as err:
        await CloudMatchEngine(runtime(cloud, FakeBudget())).assess(_request(profile(), n=1))
    assert err.value.kind == "parse_error"
    assert len(cloud.requests) == 2


# --- the cascade ------------------------------------------------------------------------------


def test_skip_rule() -> None:
    assert skip_cloud(0.05, 0.3, 1)
    assert not skip_cloud(0.05, 0.3, 2)
    assert not skip_cloud(0.05, 0.4, 1)
    assert not skip_cloud(0.2, 0.3, 1)
    assert not skip_cloud(None, 0.3, 1)


def test_merge_fields() -> None:
    code = {"garment:0": "match", "hair": "conflict", "state": "not_visible"}
    model = {"garment:0": "not_visible", "hair": "match", "state": "match"}
    assert merge_fields(code, model) == {"garment:0": "match", "hair": "match", "state": "match"}  # type: ignore[arg-type]
    assert merge_fields({"hair": "match"}, {"hair": "conflict"}) == {"hair": "conflict"}  # type: ignore[arg-type]


@pytest.mark.parametrize("fail", ["503", "timeout", "budget"])
async def test_cascade_offline(fail: str) -> None:
    """No network, a server error or a refused budget: matching pauses (heartbeat down),
    nothing more is sent, and the cloud is tried again after 60 s."""
    clock = Clock()
    cloud = FakeCloud(fail=None if fail == "budget" else fail)
    budget = FakeBudget(cap_sek=0.0 if fail == "budget" else 100.0)
    cascade = Cascade(CloudMatchEngine(runtime(cloud, budget)), monotonic=clock)
    assert cascade.state == "ok"
    assert cascade.submit(_request(profile(), n=1), (1, ("a",)))
    await cascade.drain()
    assert cascade.state == "down"
    (r,) = cascade.take()
    assert r.assessment is None  # to be asked again
    sent = len(cloud.requests)
    assert sent == (0 if fail == "budget" else 1)
    assert not cascade.submit(_request(profile(), n=1), (1, ("b",)))
    assert len(cloud.requests) == sent
    clock.t += 61
    cloud.fail, budget.cap_sek = None, 100.0
    assert cascade.state == "ok"
    assert cascade.submit(_request(profile(), n=1), (1, ("c",)))
    await cascade.drain()
    (r,) = cascade.take()
    assert r.assessment is not None and r.assessment.engine == "fast"


async def test_a_bad_answer_does_not_pause() -> None:
    cascade = Cascade(CloudMatchEngine(runtime(FakeCloud(reply="nej"), FakeBudget())))
    cascade.submit(_request(profile(), n=1), (1, ("a",)))
    await cascade.drain()
    assert cascade.state == "ok" and cascade.take() == []


async def test_queue_and_concurrency_limits() -> None:
    gate = asyncio.Event()
    running = 0
    peak = 0

    class Slow:
        engine = "fast"
        model_id = "slow"

        async def assess(self, req: MatchRequest) -> Any:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await gate.wait()
            running -= 1
            raise ModelError("x")  # http_error: pauses, fine for this test

    cascade = Cascade(Slow(), max_concurrent=4, queue_max=8)  # pyright: ignore[reportArgumentType]
    taken = 0
    for i in range(12):
        req = MatchRequest(mission_id=MISSION, candidate_id=str(new_id()), profile=None)
        taken += cascade.submit(req, (0, (str(i),)))
    assert taken == 8 and not cascade.accepting
    await asyncio.sleep(0)
    assert peak == 4
    gate.set()
    await cascade.drain()


# --- the service ------------------------------------------------------------------------------


def detect(g: FakeGround, east: float, t: float, crop: bytes, **kw: Any) -> None:
    lat, lon = at(east, 0)
    did = str(new_id())
    g.crops[did] = crop
    g.add_detection(
        detection_id=did,
        still_id=str(new_id()),
        lat=lat,
        lon=lon,
        captured_utc=utc(t),
        has_crop=True,
        bbox=BBox(x=132, y=132, w=32, h=32),
        crop_box=BBox(x=100, y=100, w=96, h=96),
        **kw,
    )


async def test_service_writes_the_fast_assessment() -> None:
    """SK01's candidate gets an ``engine = fast`` assessment after the code's; its fields
    override the code's (but not a code match it could not see), its person_p sets the
    priority, every call is booked, and the heartbeat says ok."""
    g = FakeGround(MISSION, profile_body=profile())
    cloud = FakeCloud(
        reply=answer(fields={"garment:0": "not_visible", "state": "match"}, person_p=0.95)
    )
    budget = FakeBudget()
    s = SearchService(
        g, GroundProfileSource(g), cascade=Cascade(CloudMatchEngine(runtime(cloud, budget)))
    )
    detect(g, 0, 0, crop_jpeg(RED))
    await s.step()
    (cid,) = g.revs
    code = g.latest(cid)
    assert g.assessments[-1].engine == "code"
    await s.cascade.drain()
    await s.step()
    fast = g.assessments[-1]
    assert fast.engine == "fast" and fast.model_id == "anthropic/claude-haiku-5.5"
    c = g.latest(cid)
    assert c.rev == code.rev + 1 and c.state.value == "ready"
    assert c.priority > code.priority  # person_p 0.95 over the detector's score
    assert len(cloud.requests) == 1
    assert len(budget.committed) == len(budget.reserved) == 1
    await s.beat(force=True)
    assert g.beats[-1].cloud == "ok"
    # Nothing new to look at: not asked again.
    await s.step()
    await s.cascade.drain()
    assert len(cloud.requests) == 1


async def test_service_skips_a_clear_non_person() -> None:
    g = FakeGround(MISSION, profile_body=profile())
    cloud = FakeCloud()
    s = SearchService(
        g, GroundProfileSource(g), cascade=Cascade(CloudMatchEngine(runtime(cloud, FakeBudget())))
    )
    detect(g, 0, 0, crop_jpeg(RED), score=0.2)
    await s.step()
    (cid,) = g.revs
    s.missions[MISSION].fusion.tracks[cid].refined_person_score = 0.05
    await s.step()
    await s.cascade.drain()
    # The first round offered it before the refinement was known; after that nothing.
    assert len(cloud.requests) <= 1


async def test_service_heartbeat_without_cloud_is_off() -> None:
    g = FakeGround(MISSION, profile_body=profile())
    s = SearchService(g, GroundProfileSource(g))
    await s.beat(force=True)
    assert g.beats[-1].cloud == "off"


async def test_service_heartbeat_down_when_paused() -> None:
    g = FakeGround(MISSION, profile_body=profile())
    cloud = FakeCloud(fail="503")
    s = SearchService(
        g, GroundProfileSource(g), cascade=Cascade(CloudMatchEngine(runtime(cloud, FakeBudget())))
    )
    detect(g, 0, 0, crop_jpeg(RED))
    await s.step()
    await s.cascade.drain()
    await s.step()
    await s.beat(force=True)
    assert g.beats[-1].cloud == "down"
    assert g.assessments[-1].engine == "code"  # the code still assesses


# --- configuration --------------------------------------------------------------------------


def test_example_config_loads() -> None:
    cfg = load(REPO)
    assert cfg.cloud.fast_model == "anthropic/claude-haiku-5.5"
    assert cfg.cloud.provider.data_collection == "deny"
    rt = cfg.cloud.runtime_config()
    assert rt.reasoning_effort == "" and rt.fallback_model == "" and rt.call_timeout_s == 8
    assert rt.provider.zdr is True  # whatever SD_SIM says


def test_data_collection_cannot_be_allowed(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "search.toml").write_text(
        '[cloud.provider]\ndata_collection = "allow"\n'
    )
    with pytest.raises(SearchConfigError) as err:
        load(tmp_path)
    assert "data_collection" in str(err.value)


def test_cloud_key(tmp_path: Path) -> None:
    assert cloud_key(tmp_path, {}) is None
    (tmp_path / ".env").write_text("SD_CLOUD_API_KEY=abc\n")
    assert cloud_key(tmp_path, {}) == "abc"
    assert cloud_key(tmp_path, {"SD_CLOUD_API_KEY": "env"}) == "env"
