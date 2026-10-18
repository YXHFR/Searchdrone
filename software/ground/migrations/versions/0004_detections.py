"""Detections from the vehicle's perception reports.

* ``detections``: one row per detection (append-only): the class, the source (``oracle``
  for the simulator's own labels), score, box, place on the ground and its 95 % radius,
  size, the still it came from and, for targets, the thumbnail's file in the mission
  folder. ``find_id`` groups detections of the same class at the same place into one find
  (the first detection's id); the ground station sets it when the row is written.
* ``mission_files.kind`` may be ``crop`` (a detection's thumbnail).

Revision ID: 0004_detections
Revises: 0003_path_annotations
Create Date: 2026-10-18
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004_detections"
down_revision: str | None = "0003_path_annotations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE = """
CREATE TABLE detections (
  id           uuid PRIMARY KEY,                         -- the vehicle's detection_id
  mission_id   uuid NOT NULL REFERENCES missions(id),
  vehicle_id   text NOT NULL REFERENCES vehicles(id),
  find_id      uuid NOT NULL,
  still_id     uuid NOT NULL,
  captured_at  timestamptz NOT NULL,
  label        text NOT NULL CHECK (label ~ '^[a-z][a-z0-9_]{1,31}$'),
  kind         text NOT NULL CHECK (kind IN ('target','context')),
  source       text NOT NULL CHECK (source IN ('model','color','oracle')),
  score        double precision NOT NULL CHECK (score BETWEEN 0 AND 1),
  bbox         jsonb NOT NULL,
  lat          double precision NOT NULL CHECK (lat BETWEEN -90 AND 90),
  lon          double precision NOT NULL CHECK (lon BETWEEN -180 AND 180),
  sigma95_m    double precision NOT NULL CHECK (sigma95_m >= 0),
  size_m       double precision NOT NULL CHECK (size_m >= 0),
  synthetic    boolean NOT NULL,
  crop_path    text CHECK (crop_path !~ '(^/|\\.\\.)'),
  created_at   timestamptz NOT NULL DEFAULT now()
)"""

STATEMENTS = (
    TABLE,
    "CREATE INDEX detections_mission_idx ON detections(mission_id, created_at)",
    "CREATE INDEX detections_find_idx ON detections(mission_id, find_id)",
    "CREATE TRIGGER detections_no_update_delete BEFORE UPDATE OR DELETE ON detections "
    "FOR EACH ROW EXECUTE FUNCTION sd_forbid_change()",
    "CREATE TRIGGER detections_no_truncate BEFORE TRUNCATE ON detections "
    "FOR EACH STATEMENT EXECUTE FUNCTION sd_forbid_change()",
    "GRANT SELECT, INSERT ON detections TO sd_app",
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
    op.execute("DROP TABLE detections")
