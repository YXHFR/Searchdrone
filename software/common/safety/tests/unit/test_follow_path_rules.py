"""The gate on a drawn path (follow_path): inside the charter, not too long, and the
battery lasts for the path and the flight home with the reserve left."""

from __future__ import annotations

from typing import Any

import pytest
from gate_fixtures import ctx, onboard_ctx, point, req, vehicle

from sd_safety import evaluate
from sd_safety.constants import ROUTE_BATTERY_PCT_PER_MIN


def path(*corners: tuple[float, float], alt_m: float = 40.0, **extra: Any) -> dict[str, Any]:
    pts = [point(e, n).model_dump() for e, n in corners]
    return {"points": pts, "alt_m": alt_m, "label": "pa-1", **extra}


# Five points inside the L-shaped charter, about 560 m in all.
FIVE = path((0, 60), (-100, 100), (-200, 0), (-200, -150), (0, -200))


def test_a_path_inside_the_charter_needs_the_pilots_click() -> None:
    d = evaluate(req("follow_path", FIVE), ctx())
    assert d.decision == "PENDING_APPROVAL", d
    assert d.reason_codes == ["NEEDS_APPROVAL_SKILL"]


def test_the_pilots_click_passes_and_onboard_never_asks_again() -> None:
    assert evaluate(req("follow_path", FIVE, phase="approve"), ctx()).decision == "ALLOW"
    assert evaluate(req("follow_path", FIVE), onboard_ctx()).decision == "ALLOW"


def test_a_path_outside_the_charter_is_refused_in_swedish() -> None:
    # The second leg cuts through the L's notch (east 100..300, north 100..300).
    d = evaluate(req("follow_path", path((50, 60), (250, 250), (0, 280))), ctx())
    assert d.decision == "REJECT"
    assert d.reason_codes[0] == "OUTSIDE_CHARTER"
    assert d.message_sv.startswith("Vägen går utanför sökområdet (")
    assert "Rita om den inom uppdragsramen." in d.message_sv


def test_every_corner_of_the_path_is_checked_not_only_the_last() -> None:
    d = evaluate(req("follow_path", path((0, 60), (0, 400), (0, 0))), ctx())
    assert "OUTSIDE_CHARTER" in d.reason_codes


def test_a_too_long_path_is_refused_with_a_battery_explanation() -> None:
    # Back and forth inside the charter: 10 x 500 m = 5 km at 5 m/s is about 17 min.
    zigzag = [(-250 if i % 2 else 250, -250 + 50 * i) for i in range(10)]
    # 80 % battery: the path, the flight home and the landing need about 15 + 87 %.
    d = evaluate(req("follow_path", path(*zigzag)), ctx())
    assert d.decision == "REJECT"
    assert "ROUTE_BATTERY" in d.reason_codes
    assert d.details["needed_pct"] > 80
    assert d.details["battery_pct"] == 80
    assert d.details["rtl_pct"] == 15
    assert d.message_sv.startswith("Batteriet räcker inte för hela vägen och flygningen hem: ")
    assert "inklusive reserven på 15 %, men batteriet har 80 %." in d.message_sv
    assert "Rita en kortare väg eller byt batteri." in d.message_sv


def test_the_battery_need_follows_the_model() -> None:
    # From 50 m north of home to home, then 250 m south: 300 m, then 250 m back home.
    p = path((0, 0), (0, -250))
    seconds = 300 / 5 + 250 / 5 + 40 / 1.0
    need = 15 + seconds / 60 * ROUTE_BATTERY_PCT_PER_MIN
    low = vehicle(battery_pct=need - 0.5)
    enough = vehicle(battery_pct=need + 0.5)
    assert "ROUTE_BATTERY" in evaluate(req("follow_path", p), ctx(vehicle=low)).reason_codes
    assert "ROUTE_BATTERY" not in evaluate(req("follow_path", p), ctx(vehicle=enough)).reason_codes


def test_a_slower_path_needs_more_battery() -> None:
    p = path((0, -250), (-250, -250), (-250, 250))
    fast = evaluate(req("follow_path", {**p, "speed_ms": 10}), ctx(vehicle=vehicle(battery_pct=40)))
    slow = evaluate(req("follow_path", {**p, "speed_ms": 2}), ctx(vehicle=vehicle(battery_pct=40)))
    assert "ROUTE_BATTERY" not in fast.reason_codes
    assert "ROUTE_BATTERY" in slow.reason_codes


def test_unknown_battery_rejects_a_path() -> None:
    d = evaluate(req("follow_path", FIVE), ctx(vehicle=vehicle(battery_pct=None)))
    assert "BATTERY_UNKNOWN" in d.reason_codes and d.decision == "REJECT"


def test_a_path_over_five_km_is_too_long() -> None:
    laps = [(-250 if i % 2 else 250, -250 + 25 * i) for i in range(12)]  # 11 x ~500 m
    d = evaluate(req("follow_path", path(*laps)), ctx(vehicle=vehicle(battery_pct=100)))
    assert d.reason_codes[0] == "PATH_TOO_LONG"
    assert d.details["route_m"] > 5_000
    assert d.message_sv.startswith("Vägen är för lång: 5,")
    assert "km, högst 5,0 km." in d.message_sv


def test_the_length_cap_counts_the_drawn_line_not_the_way_there() -> None:
    """The cap is on the drawn line (as the panel and the ground measure it); the way to
    its start counts only for the battery."""
    # 10 legs of about 490 m: 4,9 km drawn, about 5,1 km with the way to its start.
    laps = [(-245 if i % 2 else 245, -250 + 10 * i) for i in range(11)]
    d = evaluate(req("follow_path", path(*laps)), ctx(vehicle=vehicle(battery_pct=100)))
    assert "PATH_TOO_LONG" not in d.reason_codes


def test_the_path_is_checked_against_height_and_speed_too() -> None:
    assert (
        "ALT_ABOVE_LIMIT"
        in evaluate(req("follow_path", path((0, 0), (0, -50), alt_m=130)), ctx()).reason_codes
    )
    assert (
        "SPEED_ABOVE_LIMIT"
        in evaluate(req("follow_path", path((0, 0), (0, -50), speed_ms=16)), ctx()).reason_codes
    )


def test_other_moves_are_not_held_to_the_route_battery_rule() -> None:
    far_goto = {**point(0, 250).model_dump(), "alt_m": 40}
    d = evaluate(req("goto", far_goto), ctx(vehicle=vehicle(battery_pct=21)))
    assert d.reason_codes == []


@pytest.mark.parametrize("bad", [{"points": []}, {"points": "pa-1", "alt_m": 40}])
def test_coordinates_are_required_never_a_name(bad: dict[str, Any]) -> None:
    d = evaluate(req("follow_path", bad), ctx())
    assert d.decision == "REJECT" and d.reason_codes[0] == "SCHEMA_INVALID"
