"""The brain's simulator exams price an image crop as the ground's cloud budget does."""

from __future__ import annotations

from sd_brain.simeval.budget import IMAGE_TOKENS_PER_CROP
from sd_ground.services.cloud_budget import ModelPrice


def test_the_exam_prices_a_crop_as_the_ground() -> None:
    default = ModelPrice(in_usd_per_mtok=1, out_usd_per_mtok=1).image_tokens_per_crop
    assert default == IMAGE_TOKENS_PER_CROP
