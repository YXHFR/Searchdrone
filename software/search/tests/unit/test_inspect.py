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
from sd_schemas.geo import plane_distance_m
from sd_schemas.inspect import InspectDone
from sd_schemas.link import BBox
from sd_schemas.search import CandidateState, SearchDetection
from sd_search.app import (
    INSPECT_POLL_S,
    INSPECT_SETTLE_S,
    SearchService,
    close_ups,
    look_joins,
)
from sd_search.fusion import LOOK_JOIN_MAX_M, Fusion, Track
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
    """One closer look at ``target``: its stills carry the order, the target and where
    the target was (``anchor``, east and north metres) with its sigma when it went out."""

    def __init__(
        self,
        g: FakeGround,
        target: str,
        sigma95_m: float,
        anchor: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.g, self.target, self.sigma = g, target, sigma95_m
        self.anchor = at(*anchor)
        self.command_id = str(new_id())
        self.stills: list[tuple[str, float, float]] = []  # (still id, px/m, seconds)

    def shot(
        self,
        east: float,
        north: float,
        t: float,
        *,
        tele: bool,
        still: str | None = None,
        **kw: Any,
    ) -> str:
        """A detection in one of the look's stills (main or tele camera); a new still
        unless ``still`` names one already taken."""
        px_per_m = TELE_PX_PER_M if tele else MAIN_PX_PER_M
        if still is None:
            still = str(new_id())
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
            inspect_lat=self.anchor[0],
            inspect_lon=self.anchor[1],
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


def covered(g: FakeGround) -> None:
    """Every candidate's 95 % radius reaches every place it was seen, so its keep-out
    cylinder (30 m + sigma) keeps the drone at least 30 m from each of them."""
    for cid in g.revs:
        c = g.latest(cid)
        for d in g.stored:
            if g.observed.get(str(d.detection_id)) == cid:
                assert plane_distance_m(c.lat, c.lon, d.lat, d.lon) <= c.sigma95_m, cid


async def survey_target(g: FakeGround, s: SearchService, **kw: Any) -> str:
    """A candidate from the survey at (0, 0) that the pilot sends the drone to look at."""
    detect(g, 0, 0, 0, **kw)
    await s.step()
    (target,) = g.revs
    inspecting(g, target)
    return target


async def test_inspect_stills_join_target() -> None:
    """Acceptance 23 (and TN19 on the search side): the look's pictures of its target
    join it, beyond the fusion radius, and start no candidate; a second person 25 m away
    in the same still gets a candidate of their own. The close-ups tighten the target's
    position, so its keep-out cylinder (30 m + sigma) shrinks."""
    g = FakeGround(MISSION, profile_body=profile())
    s = SearchService(g, GroundProfileSource(g))
    target = await survey_target(g, s, sigma95_m=8.0)  # seen from high up, 3 m off
    before = g.latest(target)
    look = Look(g, target, before.sigma95_m)
    joined = [
        look.shot(3, 0, 10, tele=False, sigma95_m=1.5, px_size=60.0),
        look.shot(3.2, 0, 10.1, tele=True, sigma95_m=0.8, px_size=180.0),
    ]
    still = look.stills[0][0]
    look.shot(28, 0, 10, tele=False, still=still, sigma95_m=1.5)  # someone else, 25 m away
    await steps(s)
    assert len(g.revs) == 2
    assert all(g.observed[d] == target for d in joined)
    (other,) = set(g.revs) - {target}
    assert g.latest(other).state is CandidateState.READY
    after = g.latest(target)
    assert after.state is CandidateState.INSPECTING  # the pilot's state stays
    assert after.n_seen == 3
    assert after.sigma95_m < before.sigma95_m  # the ground's cylinder: 30 m + sigma
    assert plane_distance_m(after.lat, after.lon, *at(3, 0)) < 0.5
    covered(g)


async def test_only_the_target_within_the_match_radius_is_forced() -> None:
    """A person detection in the look's stills beyond ``match_radius_m`` (15 m here) of
    the target is fused as usual, even though the still names the target."""
    g = FakeGround(MISSION, profile_body=profile())
    s = SearchService(g, GroundProfileSource(g))
    target = await survey_target(g, s)
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


async def test_two_people_in_one_look_get_a_candidate_each() -> None:
    """Two people 12 m apart, both within the 15 m match radius: in a still that shows
    both, only the one nearest where the order sent the drone joins the target; the other
    gets a candidate of their own. A still that shows only the other one joins it to the
    target, but the target's radius then reaches both, so no cylinder leaves anyone out."""
    g = FakeGround(MISSION, profile_body=profile())
    s = SearchService(g, GroundProfileSource(g))
    target = await survey_target(g, s, sigma95_m=2.0, score=0.8)
    look = Look(g, target, 2.0)
    near = look.shot(0.5, 0, 10, tele=True, sigma95_m=0.8)
    other = look.shot(12, 0, 10, tele=True, still=look.stills[0][0], sigma95_m=0.8)
    await steps(s)
    assert g.observed[near] == target
    second = g.observed[other]
    assert second != target and len(g.revs) == 2
    assert g.latest(target).sigma95_m < 3.0  # the close-up agrees with the survey
    covered(g)

    alone = look.shot(12.3, 0, 15, tele=True, sigma95_m=0.8)  # the tele sees only them
    await steps(s)
    assert g.observed[alone] == target
    after = g.latest(target)
    assert after.sigma95_m >= plane_distance_m(after.lat, after.lon, *at(0, 0))
    covered(g)


async def test_a_far_person_in_a_large_order_radius_is_not_forced() -> None:
    """A target known only to 40 m (match radius 80 m on board): a person 70 m away in
    the look's stills is beyond the 30 m cap and gets a candidate of their own; a person
    25 m away is forced on, and the target's radius grows to reach them."""
    g = FakeGround(MISSION, profile_body=profile())
    s = SearchService(g, GroundProfileSource(g))
    target = await survey_target(g, s, sigma95_m=40.0)
    look = Look(g, target, 40.0)
    a = look.shot(0, 0, 10, tele=False, sigma95_m=2.0)
    b = look.shot(70, 0, 10, tele=False, still=look.stills[0][0], sigma95_m=2.0)
    far = look.shot(70, 0, 12, tele=False, sigma95_m=2.0)  # a still with only them
    await steps(s)
    assert g.observed[a] == target
    assert g.observed[b] != target and g.observed[far] == g.observed[b]
    covered(g)
    c = look.shot(25, 0, 14, tele=False, sigma95_m=2.0)
    await steps(s)
    assert g.observed[c] == target
    after = g.latest(target)
    assert after.sigma95_m >= plane_distance_m(after.lat, after.lon, *at(25, 0))
    covered(g)


async def test_a_chain_of_pictures_cannot_walk_the_target_away() -> None:
    """The match radius is measured from where the order sent the drone, not from the
    target's moving estimate: pictures 13 m apart join only while within 15 m of that
    point, and each farther one starts a candidate of its own."""
    g = FakeGround(MISSION, profile_body=profile())
    s = SearchService(g, GroundProfileSource(g))
    target = await survey_target(g, s, sigma95_m=2.0)
    look = Look(g, target, 2.0)
    shots: list[str] = []
    for k in range(1, 7):  # one still at a time, each fused before the next is taken
        shots.append(look.shot(13 * k, 0, 10 + k, tele=True, sigma95_m=0.8))
        await steps(s, 1)
    assert g.observed[shots[0]] == target
    assert len({g.observed[d] for d in shots[1:]} - {target}) == 5
    assert g.latest(target).n_seen == 2
    covered(g)


def test_look_joins_one_picture_per_still_near_the_order() -> None:
    f = Fusion()
    step = f.add(sighting(0, 0, sigma=40.0))
    assert step is not None
    cid = step.candidate_id
    lat, lon = at(0, 0)
    still = str(new_id())

    def det(east: float, **kw: Any) -> SearchDetection:
        dlat, dlon = at(east, 0)
        base: dict[str, Any] = {
            "detection_id": str(new_id()),
            "seq": 1,
            "still_id": still,
            "label": "person",
            "score": 0.6,
            "lat": dlat,
            "lon": dlon,
            "sigma95_m": 1.0,
            "px_size": 40.0,
            "bbox": BBox(x=1, y=1, w=10, h=10),
            "has_crop": False,
            "captured_utc": utc(0),
            "inspect_candidate_id": cid,
            "inspect_lat": lat,
            "inspect_lon": lon,
            "inspect_sigma95_m": 40.0,
        }
        return SearchDetection.model_validate(base | kw)

    near, nearer, sure = det(20), det(10), det(10, score=0.9)
    assert look_joins(f, [near, nearer, sure]) == {str(sure.detection_id): cid}
    assert look_joins(f, [det(LOOK_JOIN_MAX_M + 0.5)]) == {}  # capped below 2 x 40 m
    assert look_joins(f, [det(16, inspect_sigma95_m=3.0)]) == {}  # 15 m with sigma 3
    assert look_joins(f, [det(1, inspect_candidate_id=None)]) == {}  # not a look's still
    assert look_joins(f, [det(1, inspect_lat=None)]) == {}  # the order's position unknown
    assert look_joins(f, [det(1, inspect_candidate_id=new_id())]) == {}  # unknown target
    assert look_joins(f, [det(1, label="red_block")]) == {}
    other_still = det(1, still_id=str(new_id()))
    f.add(sighting(0, 0), force_candidate_id=cid)
    f.tracks[cid].sightings[-1] = replace(f.tracks[cid].sightings[-1], still_id=still)
    assert look_joins(f, [det(1)]) == {}  # this still already gave the target its picture
    assert look_joins(f, [other_still]) == {str(other_still.detection_id): cid}
    seen = det(1, still_id=str(new_id()))
    f.seen.add(str(seen.detection_id))
    assert look_joins(f, [seen]) == {}  # fused before
    f.tracks[cid].state = CandidateState.REJECTED
    assert look_joins(f, [other_still]) == {}
