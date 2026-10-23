"""Candidates from the search service (sd-search).

* ``detections.crop_box``: the square of the still the crop was cut from (``CropData``'s
  ``bbox_in_still``), so the detector's box can be found inside the crop. Empty for a
  detection without a crop and for rows stored before this revision.
* One observation per detection: the same detection never joins two candidates, and a
  restart of sd-search that writes it again is refused instead of counted twice.
* One display number per mission: ``seq`` is unique among a mission's first revisions.

The tables stay append-only (the triggers from 0003_search).

Revision ID: 0005_candidates
Revises: 0004_detections
Create Date: 2026-10-23
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0005_candidates"
down_revision: str | None = "0004_detections"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

STATEMENTS = (
    "ALTER TABLE detections ADD COLUMN crop_box jsonb "
    "CHECK (crop_box IS NULL OR jsonb_typeof(crop_box) = 'object')",
    "CREATE UNIQUE INDEX candidate_observations_one_per_detection "
    "ON candidate_observations(detection_id)",
    "CREATE UNIQUE INDEX candidates_one_seq_per_mission ON candidates(mission_id, seq) "
    "WHERE rev = 1",
)


def upgrade() -> None:
    for statement in STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP INDEX candidates_one_seq_per_mission")
    op.execute("DROP INDEX candidate_observations_one_per_detection")
    op.execute("ALTER TABLE detections DROP COLUMN crop_box")
