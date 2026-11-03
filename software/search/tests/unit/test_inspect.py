"""A closer look (``inspect_detection``) as the search service sees it: the look's pictures
join its target, the target is assessed again after ``inspect.done``, its position (and
so its keep-out cylinder) tightens, and the look counts as a pass in the map."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import Any

from fake_cloud import FakeBudget, FakeCloud, runtime  # pyright: ignore[reportMissingImports]
from fake_ground import FakeGround, utc  # pyright: ignore[reportMissingImports]
from search_helpers import (  # pyright: ignore[reportMissingImports]
    MISSION,
    RED,
    at,
    crop_jpeg,
    profile,
    sighting,
)

from sd_schemas.base import new_id
from sd_schemas.inspect import InspectDone
from sd_schemas.link import BBox
from sd_schemas.search import CandidateState, SearchDetection
from sd_search.app import (
    INSPECT_POLL_S,
    INSPECT_SETTLE_S,
    SearchService,
    close_ups,
    inspect_target,
)
from sd_search.fusion import Fusion, Track
from sd_search.mapper import map_candidate
from sd_search.match.cascade import Cascade
from sd_search.match.engines import CloudMatchEngine
from sd_search.probmap.build import MapBuilder, MapInputs
from sd_search.probmap.pod import Still
from sd_search.probmap.terrain import NoTerrain
from sd_search.profile import GroundProfileSource

MAIN_PX_PER_M, TELE_PX_PER_M = 20.0, 60.0  # the tele camera: about three times the detail


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def detect(g: FakeGround, east: float, north: float, t: float, **kw: Any) -> SearchDetection:
    lat, lon = at(east, north)
    did = str(new_id())
    g.crops[did] = crop_jpeg(RED)
    base: dict[str, Any] = {
        "detection_id": did,
        "still_id": str(new_id()),
        "lat": lat,
        "lon": lon,
        "captured_utc": utc(t),
        "has_crop": True,
        "bbox": BBox(x=132, y=132, w=32, h=32),
        "crop_box": BBox(x=100, y=100, w=96, h=96),
    }
    return g.add_detection(**(base | kw))


class Look:
    """One closer look at ``target``: its stills carry the order and the target."""

    def __init__(self, g: FakeGround, target: str, sigma95_m: float) -> None:
        self.g, self.target, self.sigma = g, target, sigma95_m
        self.command_id = str(new_id())
        self.stills: list[tuple[str, float, float]] = []  # (still id, px/m, seconds)

    def shot(self, east: float, north: float, t: float, *, tele: bool, **kw: Any) -> str:
        """A detection in one of the look's stills (main or tele camera)."""
        still = str(new_id())
        px_per_m = TELE_PX_PER_M if tele else MAIN_PX_PER_M
        self.stills.append((still, px_per_m, t))
        d = detect(
            self.g,
            east,
            north,
            t,
            still_id=still,
            command_id=self.command_id,
            still_px_per_m=px_per_m,
            inspect_candidate_id=self.target,
            inspect_sigma95_m=self.sigma,
            **kw,
        )
        return str(d.detection_id)

    def done(self) -> InspectDone:
        return InspectDone.model_validate(
            {
                "command_id": self.command_id,
                "candidate_id": self.target,
                "end": "clear",
                "steps": 2,
                "lowest_above_ground_m": 21.0,
                "still_ids": [s for s, _, _ in self.stills],
                "clear": True,
            }
        )


def inspecting(g: FakeGround, cid: str) -> None:
    """The pilot sends the drone to look (the ground's decision, TN-7)."""
    c = g.latest(cid)
    g.revs[cid].append(c.model_copy(update={"rev": c.rev + 1, "state": CandidateState.INSPECTING}))


async def steps(s: SearchService, n: int = 2) -> None:
    for _ in range(n):  # a 409 after the ground's own revision reads the mission back
        await s.step()
        await s.cascade.drain()


async def test_inspect_stills_join_target() -> None:
    """Acceptance 23 (and TN19 on the search side): the look's pictures of its target
    join it, beyond the fusion radius, and start no candidate; a second person 25 m away
    in the same stills gets a candidate of their own. The close-ups tighten the target's
    position, so its keep-out cylinder (30 m + sigma) shrinks."""
    g = FakeGround(MISSION, profile_body=profile())
    s = SearchService(g, GroundProfileSource(g))
    detect(g, 0, 0, 0, sigma95_m=3.0)  # seen in the survey, 10 m off the truth
    await s.step()
    (target,) = g.revs
    before = g.latest(target)
    inspecting(g, target)
    look = Look(g, target, before.sigma95_m)
    joined = [
        look.shot(10, 0, 10, tele=False, sigma95_m=1.5, px_size=60.0),
        look.shot(10.2, 0, 10.1, tele=True, sigma95_m=0.8, px_size=180.0),
    ]
    look.shot(35, 0, 10, tele=False, sigma95_m=1.5)  # someone else, 25 m away
    await steps(s)
    assert len(g.revs) == 2
    assert all(g.observed[d] == target for d in joined)
    (other,) = set(g.revs) - {target}
    assert g.latest(other).state is CandidateState.READY
    after = g.latest(target)
    assert after.state is CandidateState.INSPECTING  # the pilot's state stays
    assert after.n_seen == 3
    assert after.sigma95_m < before.sigma95_m  # the ground's cylinder: 30 m + sigma
    lat, lon = at(10, 0)
    assert abs(after.lat - lat) < 1e-5 and abs(after.lon - lon) < 1e-5  # within about 1 m


async def test_only_the_target_within_the_match_radius_is_forced() -> None:
    """A person detection in the look's stills beyond ``match_radius_m`` (15 m here) of
    the target is fused as usual, even though the still names the target."""
    g = FakeGround(MISSION, profile_body=profile())
    s = SearchService(g, GroundProfileSource(g))
    detect(g, 0, 0, 0, sigma95_m=3.0)
    await s.step()
    (target,) = g.revs
    inspecting(g, target)
    look = Look(g, target, 3.0)
    look.shot(16, 0, 10, tele=False, sigma95_m=1.5)
    await steps(s)
    assert len(g.revs) == 2


async def test_reassess_after_inspect() -> None:
    """Acceptance 24: while the look flies, the code assesses each step and the cloud
    waits; ``inspect.done`` gives a new code and cloud assessment of the target within
    10 s, on the look's best crop from each camera, through the budget, without
    changing the pilot's state. Nothing more is sent for the same look."""
    assert INSPECT_POLL_S + INSPECT_SETTLE_S < 10
    g = FakeGround(MISSION, profile_body=profile())
    cloud, budget, clock = FakeCloud(), FakeBudget(), Clock()
    s = SearchService(
        g,
        GroundProfileSource(g),
        cascade=Cascade(CloudMatchEngine(runtime(cloud, budget))),
        monotonic=clock,
    )
    detect(g, 0, 0, 0, sigma95_m=3.0, px_size=24.0)
    await steps(s)
    (target,) = g.revs
    assert len(cloud.requests) == 1  # the survey's sighting, as before
    inspecting(g, target)
    look = Look(g, target, 3.0)
    for k, (main_px, tele_px) in enumerate(((40.0, 120.0), (60.0, 180.0))):
        t0 = 10.0 + 5 * k
        look.shot(1, 0, t0, tele=False, px_size=main_px, sigma95_m=1.5)
        look.shot(1, 0, t0 + 0.1, tele=True, px_size=tele_px, sigma95_m=0.8)
        codes = sum(a.engine == "code" for a in g.assessments)
        await steps(s)
        clock.t += 5
        assert sum(a.engine == "code" for a in g.assessments) > codes  # each step
    assert len(cloud.requests) == 1  # the cloud waits for the end of the look
    best_main, best_tele = [str(d.detection_id) for d in g.stored[-2:]]

    g.looks.append(look.done())
    clock.t += INSPECT_POLL_S
    await steps(s, 1)  # read; its last step's pictures may still be on their way
    assert len(cloud.requests) == 1
    clock.t += INSPECT_SETTLE_S
    written = len(g.assessments)
    await steps(s)
    new = g.assessments[written:]
    assert [a.engine for a in new] == ["code", "fast"]
    assert {str(c) for c in new[0].crops_used} == {best_tele, best_main}
    assert len(cloud.requests) == 2 and len(budget.reserved) == 2
    images = [
        p
        for m in cloud.requests[-1]["messages"]
        if isinstance(m["content"], list)
        for p in m["content"]
        if p["type"] == "image_url"
    ]
    assert len(images) == 2  # one crop from each camera
    assert g.latest(target).state is CandidateState.INSPECTING

    for _ in range(3):  # the same look is not assessed again
        clock.t += INSPECT_POLL_S + INSPECT_SETTLE_S
        await steps(s, 1)
    assert len(cloud.requests) == 2


async def test_a_look_without_close_ups_is_not_assessed_again() -> None:
    g = FakeGround(MISSION, profile_body=profile())
    clock = Clock()
    s = SearchService(g, GroundProfileSource(g), monotonic=clock)
    detect(g, 0, 0, 0)
    await s.step()
    (target,) = g.revs
    inspecting(g, target)
    g.looks.append(Look(g, target, 3.0).done())
    await steps(s)
    clock.t += INSPECT_POLL_S + INSPECT_SETTLE_S
    written = len(g.assessments)
    await steps(s)
    assert len(g.assessments) == written


async def test_probmap_inspect_pass() -> None:
    """Acceptance 25: the look's stills are one pass (its order) and count the ground
    around the target as searched; the target's own cell is not counted down while the
    pilot has not decided on it, and is once it is rejected."""
    g = FakeGround(MISSION, profile_body=profile())
    s = SearchService(g, GroundProfileSource(g))
    detect(g, 0, 0, 0, sigma95_m=3.0)
    await s.step()
    (target,) = g.revs
    inspecting(g, target)
    look = Look(g, target, 3.0)
    for k in range(3):
        look.shot(0, 0, 10 + 5 * k, tele=False, sigma95_m=1.5)
        look.shot(0, 0, 10.1 + 5 * k, tele=True, sigma95_m=0.8)
    await steps(s)
    (track,) = [t for t in await g.tracks(MISSION) if str(t.candidate.candidate_id) == target]
    pending = map_candidate(track)
    assert {sid for sid, _, _ in look.stills} <= pending.still_ids

    def square(half: float) -> tuple[tuple[float, float], ...]:
        pts = [at(-half, -half), at(half, -half), at(half, half), at(-half, half)]
        return tuple((lon, lat) for lat, lon in pts)

    stills = [
        Still(
            sid,
            n + 1,
            look.command_id,
            square(60.0 if px == MAIN_PX_PER_M else 20.0),
            px,
            True,
            utc(0) + timedelta(seconds=t),
        )
        for n, (sid, px, t) in enumerate(look.stills)
    ]
    area = [[lon, lat] for lon, lat in square(600.0)]

    def build(state: str) -> tuple[float, float]:
        c = replace(pending, state=state)
        built = MapBuilder(NoTerrain(), motion=False).build(
            MapInputs(
                profile=profile(subject_category="dementia"),
                area=[*area, area[0]],
                sectors=[],
                areas=[],
                stills=stills,
                candidates=[c],
                alt_m=40.0,
                speed_ms=5.0,
            )
        )
        grid = built.state.grid
        own = grid.cell_of(*grid.xy(*at(0, 0)))
        near = grid.cell_of(*grid.xy(*at(50, 0)))  # in the main camera's view, not held
        assert own is not None and near is not None
        return float(built.state.pod_cum[own]), float(built.state.pod_cum[near])

    held_own, held_near = build("inspecting")
    assert held_own == 0.0 and held_near > 0.0
    rejected_own, _ = build("rejected")
    assert rejected_own > 0.0


def test_close_ups_take_one_crop_from_each_camera() -> None:
    """The largest crop, and beside it only a crop the other camera took at the same
    moment: never two of one camera, nor of two moments, nor of an unknown camera."""
    look = str(new_id())

    def s(px: float, px_per_m: float | None, t: float, *, crop: bool = True, order: str = look):
        return replace(sighting(0, 0, px=px, t=t, crop=crop), command_id=order, px_per_m=px_per_m)

    tele = s(180, TELE_PX_PER_M, 10.1)
    main = s(60, MAIN_PX_PER_M, 10.0)
    track = Track(
        candidate_id="k",
        label="person",
        state=CandidateState.INSPECTING,
        lat=tele.lat,
        lon=tele.lon,
        sigma95_m=2.0,
        first_seen_utc=tele.captured_utc,
        last_seen_utc=tele.captured_utc,
    )
    track.sightings = [
        s(170, TELE_PX_PER_M, 10.1),  # the same camera
        s(150, MAIN_PX_PER_M, 4.0),  # another moment
        s(140, None, 10.0),  # an unknown camera
        s(130, MAIN_PX_PER_M, 10.0, crop=False),  # no crop
        s(500, MAIN_PX_PER_M, 10.0, order=str(new_id())),  # another order
        main,
        tele,
    ]
    assert close_ups(track, look) == [tele, main]
    track.sightings.remove(main)
    assert close_ups(track, look) == [tele]
    assert close_ups(track, str(new_id())) == []


def test_inspect_target_needs_a_known_candidate_within_the_radius() -> None:
    f = Fusion()
    step = f.add(sighting(0, 0, sigma=10.0))
    assert step is not None
    lat, lon = at(18, 0)
    base: dict[str, Any] = {
        "detection_id": str(new_id()),
        "seq": 1,
        "still_id": str(new_id()),
        "label": "person",
        "score": 0.6,
        "lat": lat,
        "lon": lon,
        "sigma95_m": 1.0,
        "px_size": 40.0,
        "bbox": BBox(x=1, y=1, w=10, h=10),
        "has_crop": False,
        "captured_utc": utc(0),
    }
    d = SearchDetection.model_validate(base)
    assert inspect_target(f, d) is None  # not a look's still
    look = d.model_copy(update={"inspect_candidate_id": step.candidate_id})
    assert inspect_target(f, look) == step.candidate_id  # 2 x 10 m (the track's sigma)
    then = look.model_copy(update={"inspect_sigma95_m": 3.0})
    assert inspect_target(f, then) is None  # 15 m with the order's sigma
    other = look.model_copy(update={"inspect_candidate_id": new_id()})
    assert inspect_target(f, other) is None
