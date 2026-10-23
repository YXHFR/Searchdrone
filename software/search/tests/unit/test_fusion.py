"""Fusion of detections into candidates (``sd_search.fusion``)."""

from __future__ import annotations

import random

import pytest
from search_helpers import at, sighting  # pyright: ignore[reportMissingImports]

from sd_schemas.search import CandidateState
from sd_search.fusion import SIGMA_MIN_M, Fusion, fusion_radius_m
from sd_search.geo import distance_m


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
