"""Classes and finds on the detections from 0003_search.

* ``detections.label``: the class from the detector's class register (``sd_detect``),
  ``kind`` (``target`` or ``context``) and ``source`` (``oracle`` for the simulator's own
  labels). Empty for a detection from a detector without classes.
* ``detections.find_id`` groups detections of the same class at the same place into one
  find (the first detection's ``detection_id``); the ground station sets it when the row
  is written.
* ``mission_files.kind`` may be ``crop`` (a detection's thumbnail).

The table stays append-only (the triggers from 0003_search).

Revision ID: 0004_detections
Revises: 0003_search
Create Date: 2026-10-18
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004_detections"
down_revision: str | None = "0003_search"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

STATEMENTS = (
    "ALTER TABLE detections "
    "ADD COLUMN label text CHECK (label ~ '^[a-z][a-z0-9_]{1,31}$'), "
    "ADD COLUMN kind text CHECK (kind IN ('target','context')), "
    "ADD COLUMN source text CHECK (source IN ('model','color','oracle')), "
    "ADD COLUMN find_id uuid, "
    "ADD CONSTRAINT detections_class_complete "
    "CHECK ((label IS NULL) = (kind IS NULL) AND (label IS NULL) = (source IS NULL))",
    "CREATE INDEX detections_find_idx ON detections(mission_id, find_id)",
    "ALTER TABLE mission_files DROP CONSTRAINT mission_files_kind_check",
    "ALTER TABLE mission_files ADD CONSTRAINT mission_files_kind_check "
    "CHECK (kind IN ('video','tlog','dataflash','log','export','brain','onboard','crop'))",
)


def upgrade() -> None:
    for statement in STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DELETE FROM mission_files WHERE kind = 'crop'")
    op.execute("ALTER TABLE mission_files DROP CONSTRAINT mission_files_kind_check")
    op.execute(
        "ALTER TABLE mission_files ADD CONSTRAINT mission_files_kind_check "
        "CHECK (kind IN ('video','tlog','dataflash','log','export','brain','onboard'))"
    )
    op.execute("DROP INDEX detections_find_idx")
    op.execute(
        "ALTER TABLE detections DROP CONSTRAINT detections_class_complete, "
        "DROP COLUMN find_id, DROP COLUMN source, DROP COLUMN kind, DROP COLUMN label"
    )
