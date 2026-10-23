"""The large views a person opened before deciding on a candidate.

* ``candidate_views``: one row each time a session asks for a detection's picture in
  large size. A confirmation needs a row from the same session for one of the
  candidate's detections, so the server checks it in its own log, never trusting the
  panel to say so.

Append-only like the other stage 2 tables (``sd_forbid_change()``).

Revision ID: 0006_candidate_views
Revises: 0005_candidates
Create Date: 2026-10-23
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0006_candidate_views"
down_revision: str | None = "0005_candidates"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

STATEMENTS = (
    "CREATE TABLE candidate_views ("
    "  id            uuid PRIMARY KEY DEFAULT uuidv7(),"
    "  mission_id    uuid NOT NULL REFERENCES missions(id),"
    "  detection_id  uuid NOT NULL REFERENCES detections(detection_id),"
    "  operator_ref  text NOT NULL CHECK (operator_ref ~ '^operator:.+'),"
    "  session_id    text NOT NULL,"
    "  created_at    timestamptz NOT NULL DEFAULT now()"
    ")",
    "CREATE INDEX candidate_views_session_idx ON candidate_views(session_id, detection_id)",
    "CREATE TRIGGER candidate_views_no_update_delete BEFORE UPDATE OR DELETE ON candidate_views "
    "FOR EACH ROW EXECUTE FUNCTION sd_forbid_change()",
    "CREATE TRIGGER candidate_views_no_truncate BEFORE TRUNCATE ON candidate_views "
    "FOR EACH STATEMENT EXECUTE FUNCTION sd_forbid_change()",
    "GRANT SELECT, INSERT ON candidate_views TO sd_app",
)


def upgrade() -> None:
    for statement in STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP TABLE candidate_views")
