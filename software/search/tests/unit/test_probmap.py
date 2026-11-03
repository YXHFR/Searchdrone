"""The probability map's logic: rings, terrain, POD, Bayes and the sectors."""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from search_helpers import HOME, at, profile  # pyright: ignore[reportMissingImports]

from sd_schemas.probmap import PSR_MAX, situation_line, suggest_sector_order
from sd_schemas.search import SubjectCategory
from sd_search.probmap.bayes import HeadingPrior, MapState, OperatorArea, RingTerrainMap
from sd_search.probmap.build import MapBuilder, MapCandidate, MapInputs
from sd_search.probmap.grid import Grid, grid_around
from sd_search.probmap.pod import (
    DETECTORS,
    PERSON_WIDTH_M,
    SAME_MOMENT_S,
    Camera,
    Still,
    pass_pod,
    same_moment_groups,
    still_pod,
)
from sd_search.probmap.rings import ring_mass, rings_for
from sd_search.probmap.sectors import PlanSector, sector_scores, spacing_m
from sd_search.probmap.terrain import NoTerrain, SimTerrainSource

CATEGORIES: list[SubjectCategory] = [
    "dementia",
    "hiker",
    "despondent",
    "gatherer",
    "hunter",
    "child_1_6",
    "autism",
    "other",
]
DET = DETECTORS["det_oracle_v0"]
MODEL = RingTerrainMap()


def grid(half: float = 2000.0) -> Grid:
    n = round(2 * half / 25)
    return Grid(HOME[0], HOME[1], -half, -half, n, n)


def square(east: float, north: float, half: float) -> tuple[tuple[float, float], ...]:
    """A footprint (lon, lat corners) of side ``2 half`` around a point."""
    pts = [
        at(east - half, north - half),
        at(east + half, north - half),
        at(east + half, north + half),
        at(east - half, north + half),
    ]
    return tuple((lon, lat) for lat, lon in pts)


def ring(east: float, north: float, half: float) -> list[list[float]]:
    c = [list(p) for p in square(east, north, half)]
    return [*c, c[0]]


def still(
    east: float, north: float, *, seq: int, order: str = "p1", px: float = 120.0, ok: bool = True
) -> Still:
    return Still(f"s{seq}", seq, order, square(east, north, 30), px, ok)


def plane(g: Grid, s: Still) -> Still:
    return Still(
        s.still_id,
        s.seq,
        s.pass_id,
        tuple(g.xy(lat, lon) for lon, lat in s.footprint),
        s.px_per_m,
        s.detector_ok,
    )


def inputs(**kw: object) -> MapInputs:
    base: dict[str, object] = {
        "profile": profile(subject_category="dementia"),
        "area": ring(0, 0, 700),
        "sectors": [("A", ring(-350, 0, 340)), ("B", ring(350, 0, 340))],
        "areas": [],
        "stills": [],
        "candidates": [],
        "alt_m": 40.0,
        "speed_ms": 5.0,
    }
    return MapInputs(**(base | kw))  # type: ignore[arg-type]


# --- rings -----------------------------------------------------------------------------


@pytest.mark.parametrize("category", CATEGORIES)
def test_half_the_mass_lies_within_the_median_radius(category: SubjectCategory) -> None:
    rings = rings_for(category)
    half = 2 * rings.radii_m[3] + 500  # the whole distribution fits on this grid
    g = Grid(HOME[0], HOME[1], -half, -half, round(2 * half / 50), round(2 * half / 50), 50.0)
    mass = ring_mass(g, rings)
    cx, cy = g.centers
    assert mass.sum() == pytest.approx(1.0, abs=0.01)
    inside = float(mass[np.hypot(cx, cy) <= rings.radii_m[1]].sum())
    assert inside == pytest.approx(0.5, abs=0.02)


def test_row_holds_at_least_the_prior_and_the_mass_outside() -> None:
    state = MODEL.prior(HeadingPrior("hiker"), NoTerrain(), [], grid())
    assert state.row >= 0.10
    assert state.row == pytest.approx(1 - float(ring_mass(grid(), rings_for("hiker")).sum()))
    assert state.total() == pytest.approx(1.0, abs=1e-9)


# --- terrain ---------------------------------------------------------------------------


def _path_terrain(north: float) -> SimTerrainSource:
    a, b = at(-2000, north), at(2000, north)
    return SimTerrainSource(
        [
            {
                "type": "Feature",
                "properties": {"kind": "path"},
                "geometry": {"type": "LineString", "coordinates": [[a[1], a[0]], [b[1], b[0]]]},
            }
        ]
    )


def test_cells_near_a_path_weigh_more_for_dementia() -> None:
    g = grid()
    state = MODEL.prior(HeadingPrior("dementia"), _path_terrain(500), [], g)
    cx, cy = g.centers
    r = np.hypot(cx, cy)
    band = (r > 450) & (r < 650)
    near = band & (np.abs(cy - 500) <= 25)
    far = band & (np.abs(cy - 500) >= 300)
    assert near.any() and far.any()
    assert state.p[near].mean() >= 2 * state.p[far].mean()


def test_no_terrain_is_the_rings_alone_and_says_so() -> None:
    built = MapBuilder(NoTerrain()).build(inputs())
    assert "no_terrain" in built.notes
    p = built.state.p
    g = built.state.grid
    cx, cy = g.centers
    r = np.hypot(cx, cy)
    # rotationally symmetric: two cells at the same distance carry the same mass
    a = p[g.cell_of(312.5, 12.5)]  # type: ignore[index]
    b = p[g.cell_of(12.5, 312.5)]  # type: ignore[index]
    assert a == pytest.approx(b, rel=1e-6)
    assert r.shape == p.shape


def test_children_get_the_general_rings_and_a_note() -> None:
    built = MapBuilder(NoTerrain()).build(inputs(profile=profile(subject_category="child_7_12")))
    assert "child_statistics" in built.notes
    assert rings_for("child_7_12") == rings_for("other")


def test_forest_canopy_lowers_the_pod() -> None:
    lat0, lon0 = at(-100, -100)
    lat1, lon1 = at(100, 100)
    forest = SimTerrainSource(
        [
            {
                "type": "Feature",
                "properties": {"kind": "forest"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]
                    ],
                },
            }
        ]
    )
    g = grid(500)
    open_pod = pass_pod(g, [plane(g, still(0, 0, seq=1))], NoTerrain().canopy(g), DET)
    wood_pod = pass_pod(g, [plane(g, still(0, 0, seq=1))], forest.canopy(g), DET)
    cell = g.cell_of(0, 0)
    assert cell is not None
    assert wood_pod[cell] == pytest.approx(open_pod[cell] * 0.4)


# --- heading and the operator's areas ---------------------------------------------------


def test_a_heading_tilts_the_map_ahead() -> None:
    g = grid()
    state = MODEL.prior(HeadingPrior("hiker", 90.0), NoTerrain(), [], g)
    east = state.p[g.cell_of(512.5, 12.5)]  # type: ignore[index]
    west = state.p[g.cell_of(-512.5, 12.5)]  # type: ignore[index]
    assert east > 1.5 * west


def test_an_excluded_area_holds_nothing_and_a_high_one_more() -> None:
    g = grid()
    sq = ((100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0))
    mirror = tuple((-x, y) for x, y in sq)
    base = MODEL.prior(HeadingPrior("hiker"), NoTerrain(), [], g)
    hinted = MODEL.prior(
        HeadingPrior("hiker"),
        NoTerrain(),
        [OperatorArea("excluded", sq), OperatorArea("high", mirror)],
        g,
    )
    inside = g.cell_of(200, 200)
    other = g.cell_of(-200, 200)
    assert inside is not None and other is not None
    assert hinted.p[inside] == 0.0
    assert hinted.p[other] > 2.5 * base.p[other]
    assert hinted.total() == pytest.approx(1.0, abs=1e-9)


# --- POD ---------------------------------------------------------------------------------


def test_pod_follows_the_detector_curve_and_the_overlap() -> None:
    g = grid(500)
    canopy = NoTerrain().canopy(g)
    got = still_pod(g, plane(g, still(0, 0, seq=1, px=120)), canopy, DET)
    assert got is not None
    win, p = got
    full = g.cell_of(0, 0)
    assert full is not None
    assert p[full[0] - win[0].start, full[1] - win[1].start] == pytest.approx(0.889)
    assert p.min() >= 0 and p.max() <= 0.889 + 1e-9
    small = still_pod(g, plane(g, still(0, 0, seq=1, px=20)), canopy, DET)  # 10 px people
    assert small is not None and small[1].max() == pytest.approx(DET.p_det(10))


def test_a_failed_detector_sees_nothing() -> None:
    g = grid(500)
    assert still_pod(g, plane(g, still(0, 0, seq=1, ok=False)), NoTerrain().canopy(g), DET) is None


def test_a_pass_counts_at_most_two_looks() -> None:
    g = grid(500)
    canopy = NoTerrain().canopy(g)
    many = [plane(g, still(0, 0, seq=i)) for i in range(1, 8)]
    pod = pass_pod(g, many, canopy, DET)
    cell = g.cell_of(0, 0)
    assert cell is not None
    assert pod[cell] == pytest.approx(1 - (1 - 0.889) ** 2)
    one = pass_pod(g, many[:1], canopy, DET)
    assert 1 - (1 - one[cell]) ** 2 == pytest.approx(pod[cell])


# --- Bayes, properties --------------------------------------------------------------------


@st.composite
def pods(draw: st.DrawFn) -> list[tuple[float, float, float]]:
    """Up to five passes as (east, north, POD) squares on a small grid."""
    return draw(
        st.lists(
            st.tuples(
                st.floats(-400, 400), st.floats(-400, 400), st.floats(0.0, 1.0, allow_nan=False)
            ),
            max_size=5,
        )
    )


@settings(max_examples=60, deadline=None)
@given(passes=pods(), category=st.sampled_from(CATEGORIES))
def test_bayes_keeps_the_sum_and_lowers_only_the_searched_cells(
    passes: list[tuple[float, float, float]], category: SubjectCategory
) -> None:
    g = grid(500)
    state = MODEL.prior(HeadingPrior(category), NoTerrain(), [], g)
    assert state.total() == pytest.approx(1.0, abs=1e-6)
    cx, cy = g.centers
    for east, north, value in passes:
        pod = np.where((np.abs(cx - east) < 60) & (np.abs(cy - north) < 60), value, 0.0)
        new = MODEL.update_negative(state, pod)
        assert new.total() == pytest.approx(1.0, abs=1e-6)
        assert (new.p >= 0).all()
        assert (new.pod_cum >= state.pod_cum - 1e-12).all()  # never less seen than before
        searched = pod > 0
        if value > 1e-6 and float((state.p * pod).sum()) > 1e-9:
            positive = state.p > 0
            assert (new.p[searched & positive] < state.p[searched & positive]).all()
            assert (new.p[~searched & positive] > state.p[~searched & positive]).all()
            assert new.row > state.row
        state = new


@settings(max_examples=20, deadline=None)
@given(order=st.permutations(list(range(4))))
def test_the_same_passes_give_the_same_map(order: list[int]) -> None:
    """Determinism: the map is the prior times what the passes missed, in any order."""
    g = grid(500)
    cx, cy = g.centers
    pods_ = [np.where(np.hypot(cx - 100 * i, cy) < 80, 0.5, 0.0) for i in range(4)]
    a = MODEL.prior(HeadingPrior("hiker"), NoTerrain(), [], g)
    b = a
    for i in range(4):
        a = MODEL.update_negative(a, pods_[i])
    for i in order:
        b = MODEL.update_negative(b, pods_[i])
    np.testing.assert_allclose(a.p, b.p, rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(a.pod_cum, b.pod_cum, rtol=1e-12)


def test_a_frozen_map_takes_no_update() -> None:
    g = grid(500)
    state = MODEL.freeze(MODEL.prior(HeadingPrior("hiker"), NoTerrain(), [], g))
    assert MODEL.update_negative(state, np.full(g.shape, 0.9)) is state


# --- the builder: candidates hold the map back ---------------------------------------------


def _track(seq_from: int, n: int, order: str = "p1") -> list[Still]:
    return [still(-350, -300 + 40 * k, seq=seq_from + k, order=order) for k in range(n)]


def test_pending_candidate_not_decremented() -> None:
    stills = _track(1, 10)
    lat, lon = at(-350, -100)
    cand = MapCandidate("c1", "ready", lat, lon, 3.0, frozenset({"s6"}))
    b = MapBuilder(NoTerrain())
    held = b.build(inputs(stills=stills, candidates=[cand]))
    g = held.state.grid
    cell = g.cell_of(*g.xy(lat, lon))
    prior = b.build(inputs()).state
    assert cell is not None
    assert held.state.pod_cum[cell] == 0.0
    assert held.state.p[cell] >= prior.p[cell]  # not counted down, only lifted a little
    rejected = MapCandidate("c1", "rejected", lat, lon, 3.0, frozenset({"s6"}))
    after = b.build(inputs(stills=stills, candidates=[rejected]))
    assert after.state.pod_cum[cell] > 0.5
    assert after.state.p[cell] < prior.p[cell]


def test_a_merged_candidate_holds_while_its_target_waits() -> None:
    stills = _track(1, 10)
    lat, lon = at(-350, -100)
    other = at(-350, 60)
    merged = MapCandidate("c1", "merged", lat, lon, 3.0, frozenset({"s6"}), merged_into="c2")
    target = MapCandidate("c2", "inspecting", *other, 3.0, frozenset({"s10"}))
    built = MapBuilder(NoTerrain()).build(inputs(stills=stills, candidates=[merged, target]))
    g = built.state.grid
    for la, lo in ((lat, lon), other):
        cell = g.cell_of(*g.xy(la, lo))
        assert cell is not None and built.state.pod_cum[cell] == 0.0


def test_a_confirmed_candidate_is_reported() -> None:
    lat, lon = at(0, 0)
    built = MapBuilder(NoTerrain()).build(
        inputs(candidates=[MapCandidate("c1", "confirmed", lat, lon, 3.0, frozenset())])
    )
    assert built.confirmed


def test_a_surveyed_sector_loses_poa_and_gets_a_pod_between_03_and_095() -> None:
    """Sector A (west) is flown at 40 m with the planned spacing; B is left alone."""
    b = MapBuilder(NoTerrain())
    before = {s.sector_id: s for s in b.build(inputs()).sectors}
    spacing = spacing_m(40)
    px_per_m = 1280 / (spacing / 0.8)  # the plan camera at 40 m
    width = spacing / 0.8
    stills: list[Still] = []
    seq = 0
    x = -690 + spacing / 2
    while x < -10:
        y = -340.0
        while y < 340:
            seq += 1
            corners = square(x, y, width / 2)
            stills.append(Still(f"s{seq}", seq, "survey-A", corners, px_per_m, True))
            y += width * 0.7
        x += spacing
    after = {s.sector_id: s for s in b.build(inputs(stills=stills)).sectors}
    assert after["A"].poa < 0.75 * before["A"].poa
    assert after["B"].poa > before["B"].poa
    assert after["A"].covered_pct > 90 and 0.3 <= after["A"].mean_pod <= 0.95
    assert after["B"].covered_pct == 0
    built = b.build(inputs(stills=stills))
    g = built.state.grid
    cx, cy = g.centers
    seen = built.state.pod_cum[(cx > -650) & (cx < -50) & (cy > -300) & (cy < 300)]
    assert seen.min() >= 0.3 and seen.max() <= 0.95


def test_sector_order_and_the_situation_line() -> None:
    built = MapBuilder(NoTerrain()).build(inputs())
    order = suggest_sector_order(built.sectors)
    assert [s.psr for s in order] == sorted((s.psr for s in order), reverse=True)
    line = situation_line(built.sectors)
    assert line.startswith("Sektorer (POA kvar / täckt / medel-POD / PSR / min@stråk m): ")
    assert " A " in f" {line} " and " B " in f" {line} "


def test_a_tiny_sector_holding_all_the_mass_stays_within_psr_max() -> None:
    """Under 0.005 min of flight est_min rounds to 0; psr still divides by 0.01 at least."""
    grid = Grid(lat0=HOME[0], lon0=HOME[1], x0=0.0, y0=0.0, nx=4, ny=4, cell=1.0)
    p = np.zeros((4, 4))
    p[0, 0] = 1.0
    state = MapState(grid=grid, p=p, row=0.0, pod_cum=np.zeros((4, 4)))
    tiny = PlanSector("A", ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)))
    [score] = sector_scores(state, [tiny], np.zeros((4, 4)), DET, alt_m=40, speed_ms=8)
    assert score.est_min == 0.0 and score.pos_plan > 0
    assert score.psr == pytest.approx(score.pos_plan / 0.01) and score.psr <= PSR_MAX


def test_the_grid_covers_the_area_plus_the_margin_and_at_most_4_km() -> None:
    small = grid_around(*HOME, [(-200, -200), (200, 200)])
    assert small.nx == small.ny == 56  # 1 400 m / 25 m
    big = grid_around(*HOME, [(-9000, -100), (9000, 100)])
    assert big.nx == 160 and big.ny == 48
    assert big.size <= 25_600


# --- performance -------------------------------------------------------------------------


def test_the_prior_is_quick_and_an_update_quicker() -> None:
    g = grid(2000)
    assert g.size == 25_600
    t = time.perf_counter()
    state = MODEL.prior(HeadingPrior("dementia", 45.0), _path_terrain(300), [], g)
    assert time.perf_counter() - t < 1.0
    pod = pass_pod(g, [plane(g, still(0, 40 * k, seq=k)) for k in range(30)], g.centers[0] * 0, DET)
    t = time.perf_counter()
    MODEL.update_negative(state, pod)
    assert time.perf_counter() - t < 0.05


def test_a_tele_still_counts_as_searched_ground() -> None:
    """The owner's decision (2026-11-02): a still from the tele camera (Raspberry Pi HQ
    Camera, 20-degree lens) counts as searched ground like a main still, over its own
    narrower footprint and with the detail its pixels give there."""
    g = grid(500)
    canopy = NoTerrain().canopy(g)
    alt_m = 40.0
    main, tele = Camera(width_px=1280, hfov_deg=66.0), Camera(width_px=1128, hfov_deg=20.0)

    def look(foot: Camera, px: Camera, seq: int) -> Still:
        """A still over ``foot``'s footprint with ``px``'s detail, straight down."""
        half = foot.ground_width_m(alt_m) / 2
        corners = square(0, 0, half)
        return plane(g, Still(f"s{seq}", seq, "p1", corners, px.px_per_m(alt_m), True))

    cell = g.cell_of(0, 0)
    assert cell is not None

    def at_cell(s: Still) -> float:
        got = still_pod(g, s, canopy, DET)
        assert got is not None
        win, p = got
        return float(p[cell[0] - win[0].start, cell[1] - win[1].start])

    tele_still = look(tele, tele, 1)
    tele_p = at_cell(tele_still)
    assert tele_p > 0  # searched ground
    # Over the same (tele) footprint, the tele camera's detail counts, not the main one's.
    blurred_p = at_cell(look(tele, main, 2))
    detail = DET.p_det(PERSON_WIDTH_M * tele.px_per_m(alt_m))
    assert detail == pytest.approx(0.889, abs=0.001)  # a person about 40 px across
    assert tele_p / blurred_p == pytest.approx(
        detail / DET.p_det(PERSON_WIDTH_M * main.px_per_m(alt_m))
    )
    pod = pass_pod(g, [tele_still], canopy, DET)
    assert pod[cell] == pytest.approx(tele_p)
    far = g.cell_of(100, 0)  # outside the tele footprint (about 14 m wide at 40 m)
    assert far is not None and pod[far] == 0.0


def test_a_main_and_a_tele_still_of_one_moment_are_one_look() -> None:
    """The main and the tele camera take a still of the same ground at the same moment:
    one look, each camera with its own p (both inside the tele footprint, the main
    camera alone outside it), not two independent looks at the tele camera's p."""
    g = grid(500)
    canopy = NoTerrain().canopy(g)
    alt_m = 60.0
    main_cam, tele_cam = Camera(width_px=1280, hfov_deg=66.0), Camera(1128, 20.0)
    t0 = datetime(2026, 11, 2, 12, 0, 0, tzinfo=UTC)

    def shot(cam: Camera, seq: int, when: datetime | None) -> Still:
        corners = square(0, 0, cam.ground_width_m(alt_m) / 2)
        s = Still(f"s{seq}", seq, "p1", corners, cam.px_per_m(alt_m), True, when)
        return replace(plane(g, s), captured_utc=when)

    main = shot(main_cam, 1, t0)
    tele = shot(tele_cam, 2, t0 + timedelta(seconds=0.1))  # polled right after the main
    assert [[s.still_id for s in grp] for grp in same_moment_groups([tele, main])] == [["s1", "s2"]]

    def p_at(s: Still, cell: tuple[int, int]) -> float:
        got = still_pod(g, s, canopy, DET)
        assert got is not None
        (rows, cols), p = got
        j, i = cell
        if not (rows.start <= j < rows.stop and cols.start <= i < cols.stop):
            return 0.0
        return float(p[j - rows.start, i - cols.start])

    inside, outside = g.cell_of(0, 0), g.cell_of(30, 0)
    assert inside is not None and outside is not None
    pm, pt = p_at(main, inside), p_at(tele, inside)
    assert pm > 0 and pt > 0  # both cameras see this cell (the tele one a part of it)
    assert tele.px_per_m > 2.5 * main.px_per_m  # with much more detail where it looks
    pod = pass_pod(g, [main, tele], canopy, DET)
    assert pod[inside] == pytest.approx(1 - (1 - pm) * (1 - pt))
    assert p_at(tele, outside) == 0.0
    assert pod[outside] == pytest.approx(p_at(main, outside))  # the main camera alone
    # Not two independent looks at the better camera's p, as before.
    assert pod[inside] < 1 - (1 - max(pm, pt)) ** 2
    # Stills of two moments (or of unknown time) stay two looks.
    later = replace(tele, captured_utc=t0 + timedelta(seconds=SAME_MOMENT_S + 0.2))
    for other in (later, replace(tele, captured_utc=None)):
        two = pass_pod(g, [main, other], canopy, DET)
        assert two[inside] == pytest.approx(1 - (1 - max(pm, pt)) ** 2)
    # Two main stills of one moment never pair: one camera, the same detail.
    assert len(same_moment_groups([main, replace(main, still_id="s3")])) == 2
    # A tele still near two main stills pairs with the closer one in time, and the
    # other main still stays a look of its own.
    closer = replace(main, still_id="s4", captured_utc=t0 + timedelta(seconds=0.15))
    groups = same_moment_groups([main, tele, closer])
    assert sorted([s.still_id for s in grp] for grp in groups) == [["s1"], ["s4", "s2"]]
