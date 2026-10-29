"""The person detector's weights: only licences the repository allows (the image
bank's data has its own rule, tools/tests/test_bench_datasets.py)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import check_licenses as cl

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "tools" / "train"


def test_no_forbidden_detector_licenses() -> None:
    rows = tomllib.loads((TRAIN / "models.toml").read_text(encoding="utf-8"))["model"]
    assert rows
    for row in rows:
        assert cl.license_allowed(row["licence"]), row["name"]
        assert row["attribution"], row["name"]


def test_the_model_card_credits_every_measured_set() -> None:
    card = (ROOT / "software" / "detector" / "MODEL_CARD.md").read_text(encoding="utf-8")
    for name, licence in (("HERIDAL", "CC BY 3.0"), ("KIOS", "CC BY 4.0"), ("WiSARD", "MIT")):
        line = next(x for x in card.splitlines() if x.startswith(f"| {name}"))
        assert licence in line
