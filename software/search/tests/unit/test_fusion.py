"""Fusion of detections into candidates (``sd_search.fusion``)."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest
from search_helpers import at, sighting  # pyright: ignore[reportMissingImports]

from sd_safety.constants import PERSON_KEEPOUT_M
from sd_schemas.geo import plane_distance_m as distance_m
from sd_schemas.search import CandidateState
from sd_search.fusion import (
    LOOK_JOIN_MAX_M,
    SIGMA_MIN_M,
    Fusion,
    Track,
    fusion_radius_m,
    look_radius_m,
)


def test_fusion_joins_within_the_radius_and_starts_a_new_one_beyond() -> None:
    f = Fusion()
    first = f.add(sighting(0, 0, sigma=3.0))
    assert first is not None and first.created
    near = f.add(sighting(6.5, 0, sigma=3.0))  # 4 + 3 = 7 m
    assert near is not None and not near.created and near.candidate_id == first.candidate_id
    assert near.distance_m == pytest.approx(6.5, abs=0.05)
    far = f.add(sighting(30, 0, sigma=3.0))
    assert far is not None and far.created
    assert len(f.tracks) == 2


def test_fusion_radius_is_capped_at_twelve_metres() -> None:
    assert fusion_radius_m(3.0, 1.0) == 7.0
    assert fusion_radius_m(50.0, 1.0) == 12.0
    f = Fusion()
    f.add(sighting(0, 0, sigma=40.0))
    step = f.add(sighting(13, 0, sigma=40.0))
    assert step is not None and step.created


def test_fusion_never_drops_a_single_sighting() -> None:
    f = Fusion()
    step = f.add(sighting(0, 0, score=0.01))
    assert step is not None and step.created
    t = f.tracks[step.candidate_id]
    assert t.n_seen == 1 and not t.multi_frame and t.state is CandidateState.NEW


def test_fusion_multi_frame_counts_stills_not_detections() -> None:
    f = Fusion()
    a = f.add(sighting(0, 0, still="s1"))
    assert a is not None
    f.add(sighting(1, 0, still="s1"))  # the same still twice: still one frame
    t = f.tracks[a.candidate_id]
    assert t.n_seen == 1 and not t.multi_frame
    f.add(sighting(0.5, 0.5, still="s2"))
    assert t.n_seen == 2 and t.multi_frame


def test_fusion_weighted_position_and_shrinking_sigma() -> None:
    f = Fusion()
    a = f.add(sighting(0, 0, sigma=2.0, score=0.9))
    assert a is not None
    f.add(sighting(4, 0, sigma=4.0, score=0.9))  # a quarter of the weight: 0.8 m east
    t = f.tracks[a.candidate_id]
    assert distance_m(*at(0, 0), t.lat, t.lon) == pytest.approx(0.8, abs=0.05)
    assert t.sigma95_m == SIGMA_MIN_M  # 1/sqrt(1/4 + 1/16) = 1.79 m, floored at 2 m
    f2 = Fusion()
    b = f2.add(sighting(0, 0, sigma=8.0))
    assert b is not None
    f2.add(sighting(0, 0, sigma=8.0))
    assert f2.tracks[b.candidate_id].sigma95_m == pytest.approx(8.0 / 2**0.5, abs=0.01)


def test_fusion_keeps_classes_apart() -> None:
    f = Fusion()
    f.add(sighting(0, 0, label="person"))
    step = f.add(sighting(0, 0, label="red_block"))
    assert step is not None and step.created


def test_fusion_closed_candidates_take_no_sightings() -> None:
    f = Fusion()
    a = f.add(sighting(0, 0))
    assert a is not None
    f.tracks[a.candidate_id].state = CandidateState.REJECTED
    step = f.add(sighting(1, 0))
    assert step is not None and step.created and step.candidate_id != a.candidate_id
    assert len(f.tracks[a.candidate_id].sightings) == 1
    f.tracks[step.candidate_id].state = CandidateState.CONFIRMED
    again = f.add(sighting(1, 0))
    assert again is not None and again.created


def test_fusion_merges_candidates_that_drift_together_older_survives() -> None:
    f = Fusion()
    older = f.add(sighting(0, 0, sigma=2.0, t=0))
    younger = f.add(sighting(9, 0, sigma=2.0, t=1))  # beyond 6 m: its own candidate
    assert older is not None and younger is not None and younger.created
    step = f.add(sighting(4.5, 0, sigma=2.0, t=2))  # joins the older; it moves towards 9 m
    assert step is not None and step.candidate_id == older.candidate_id
    step = f.add(sighting(5.5, 0, sigma=2.0, t=3))
    assert step is not None
    young = f.tracks[younger.candidate_id]
    assert young.state is CandidateState.MERGED and young.merged_into == older.candidate_id
    assert younger.candidate_id in step.merged
    assert len(young.sightings) == 1  # observations are not moved
    assert f.tracks[older.candidate_id].state is CandidateState.NEW


def test_fusion_does_not_merge_what_a_person_holds() -> None:
    for held in ("decided", CandidateState.INSPECTING):
        f = Fusion()
        older = f.add(sighting(0, 0, sigma=2.0, t=0))
        younger = f.add(sighting(9, 0, sigma=2.0, t=1))
        assert older is not None and younger is not None
        if held == "decided":
            f.tracks[younger.candidate_id].decided = True
        else:
            f.tracks[younger.candidate_id].state = CandidateState.INSPECTING
        f.add(sighting(4.5, 0, sigma=2.0, t=2))
        f.add(sighting(5.5, 0, sigma=2.0, t=3))
        assert f.tracks[younger.candidate_id].state is not CandidateState.MERGED


def test_fusion_same_detection_is_fused_once() -> None:
    f = Fusion()
    s = sighting(0, 0)
    assert f.add(s) is not None
    assert f.add(s) is None
    assert len(f.tracks) == 1 and len(next(iter(f.tracks.values())).sightings) == 1


def test_fusion_is_deterministic_for_the_same_order() -> None:
    rng = random.Random(7)  # noqa: S311 - test data
    sightings = [
        sighting(rng.uniform(-40, 40), rng.uniform(-40, 40), sigma=rng.uniform(1, 8), t=i)
        for i in range(200)
    ]

    def run() -> list[tuple[str, str, float, float, str | None]]:
        f = Fusion()
        for s in sightings:
            f.add(s)
        return [
            (t.candidate_id, t.state.value, t.lat, t.lon, t.merged_into) for t in f.tracks.values()
        ]

    assert run() == run()


def test_fusion_resumes_from_loaded_tracks() -> None:
    f = Fusion()
    s1, s2 = sighting(0, 0, t=0), sighting(1, 0, t=1)
    f.add(s1)
    f.add(s2)
    again = Fusion(list(f.tracks.values()))
    assert again.add(s1) is None and again.add(s2) is None
    step = again.add(sighting(0.5, 0, t=2))
    assert step is not None and not step.created


def test_a_closer_looks_picture_joins_its_target_whatever_the_radius() -> None:
    """test_inspect_stills_join_target (fusion part): forced onto the target, no new
    candidate and no merge; the close-up's small sigma pulls the position to it, and
    the radius grows to reach the earlier sighting 14 m away (it may be someone else)."""
    f = Fusion()
    target = f.add(sighting(0, 0, sigma=6.0, t=0))
    other = f.add(sighting(30, 0, sigma=6.0, t=1))
    assert target is not None and other is not None
    t = f.tracks[target.candidate_id]
    t.state = CandidateState.INSPECTING
    far = sighting(14, 0, sigma=1.0, t=2)  # beyond the 10 m fusion radius
    step = f.add(far, force_candidate_id=target.candidate_id)
    assert step is not None and step.candidate_id == target.candidate_id
    assert not step.created and step.merged == ()
    assert step.distance_m == pytest.approx(14, abs=0.05)
    assert len(f.tracks) == 2 and f.tracks[other.candidate_id].state is CandidateState.NEW
    assert distance_m(*at(14, 0), t.lat, t.lon) < 1.0  # the close-up weighs 36 times more
    assert t.sigma95_m >= distance_m(*at(0, 0), t.lat, t.lon) > 13.0  # not 1/sqrt(1/36 + 1)
    assert t.sightings[-1].look_of == t.candidate_id


def test_a_forced_picture_is_fused_as_usual_when_its_target_cannot_take_it() -> None:
    f = Fusion()
    a = f.add(sighting(0, 0))
    assert a is not None
    f.tracks[a.candidate_id].state = CandidateState.REJECTED
    closed = f.add(sighting(1, 0), force_candidate_id=a.candidate_id)
    assert closed is not None and closed.created
    other_class = f.add(sighting(1, 0, label="red_block"), force_candidate_id=closed.candidate_id)
    assert other_class is not None and other_class.created
    unknown = f.add(sighting(1.5, 0), force_candidate_id="no-such-candidate")
    assert unknown is not None and unknown.candidate_id == closed.candidate_id


def test_close_ups_that_agree_shrink_the_radius_to_their_spread() -> None:
    """Pictures of one person tighten the estimate as before: the radius is the larger
    of the statistical one and the farthest sighting, rounded up, so the cylinder always
    covers every sighting."""
    f = Fusion()
    target = f.add(sighting(0, 0, sigma=8.0))
    assert target is not None
    t = f.tracks[target.candidate_id]
    for east in (1.0, 1.3, 0.7):
        f.add(sighting(east, 0, sigma=0.8), force_candidate_id=t.candidate_id)
    assert SIGMA_MIN_M == t.sigma95_m < 8.0
    for s in t.sightings:
        assert distance_m(s.lat, s.lon, t.lat, t.lon) <= t.sigma95_m


def test_a_restart_keeps_the_spread_rule() -> None:
    """A candidate loaded back with its look's pictures (``look_of``) is refitted the same
    way as when they were forced on; one without keeps the statistical radius."""
    a, b = sighting(0, 0, sigma=2.0), sighting(9, 0, sigma=2.0)
    t = Track(
        candidate_id="k",
        label="person",
        state=CandidateState.INSPECTING,
        lat=0.0,
        lon=0.0,
        sigma95_m=2.0,
        first_seen_utc=a.captured_utc,
        last_seen_utc=a.captured_utc,
        sightings=[a, b],
    )
    t.refit()
    assert t.sigma95_m == SIGMA_MIN_M  # two survey sightings: as before
    t.sightings[1] = replace(b, look_of="k")
    t.refit()
    assert t.sigma95_m == pytest.approx(4.5, abs=0.01)
    t.sightings[1] = replace(b, look_of="another")
    t.refit()
    assert t.sigma95_m == SIGMA_MIN_M


def test_look_radius_is_the_drones_match_radius_capped_at_the_keep_out() -> None:
    assert LOOK_JOIN_MAX_M == PERSON_KEEPOUT_M
    assert look_radius_m(2.0) == 15.0
    assert look_radius_m(12.0) == 24.0
    assert look_radius_m(40.0) == LOOK_JOIN_MAX_M
