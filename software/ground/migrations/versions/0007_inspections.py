"""The closer look ("Titta närmare", ``inspect_detection`` v2) on the ground.

* ``inspections``: one row when a look starts and one when it ends, per order
  (``command_id``). A look is open while it has no end row; at most one is open per
  mission. The end row says how it ended (``outcome``) and what the drone reported
  (``steps``, ``lowest_above_ground_m``, ``clear``). Append-only like the other stage 2
  tables (``sd_forbid_change()``).
* ``charters.operation_mode`` and ``charters.overflight_allowed``: set when the pilot
  approves a charter (sim, exercise or live, and whether a look may fly straight over its
  candidate), laid over the stored body like ``autonomy_mode``.
* ``candidate_decisions.decision`` may be ``resume`` ("Fortsätt söka nu").

Revision ID: 0007_inspections
Revises: 0006_candidate_views
Create Date: 2026-11-03
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0007_inspections"
down_revision: str | None = "0006_candidate_views"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OUTCOMES = (
    "'confirmed','rejected','unsure','resume','timeout','resumed_by_pilot','aborted',"
    "'refused_onboard','failed'"
)

STATEMENTS = (
    "CREATE TABLE inspections ("
    "  id                     uuid PRIMARY KEY DEFAULT uuidv7(),"
    "  command_id             uuid NOT NULL REFERENCES commands(id),"
    "  candidate_id           uuid NOT NULL,"
    "  mission_id             uuid NOT NULL REFERENCES missions(id),"
    "  row_kind               text NOT NULL CHECK (row_kind IN ('start','end')),"
    "  started_at             timestamptz,"
    "  started_by             text CHECK (started_by IS NULL OR started_by ~ '^operator:.+'),"
    "  operation_mode         text CHECK (operation_mode IN ('sim','exercise','live')),"
    "  overflight             boolean,"
    "  ended_at               timestamptz,"
    f"  outcome                text CHECK (outcome IN ({_OUTCOMES})),"
    "  detail                 text CHECK (detail IS NULL OR length(detail) <= 64),"
    "  steps                  int CHECK (steps IS NULL OR steps >= 0),"
    "  lowest_above_ground_m  real,"
    "  clear                  boolean,"
    "  created_at             timestamptz NOT NULL DEFAULT now(),"
    "  CHECK (row_kind <> 'start' OR (started_at IS NOT NULL AND started_by IS NOT NULL"
    "         AND operation_mode IS NOT NULL AND overflight IS NOT NULL AND outcome IS NULL)),"
    "  CHECK (row_kind <> 'end' OR (ended_at IS NOT NULL AND outcome IS NOT NULL))"
    ")",
    # One start and one end per order.
    "CREATE UNIQUE INDEX inspections_one_row_kind ON inspections(command_id, row_kind)",
    "CREATE INDEX inspections_mission_idx ON inspections(mission_id, created_at)",
    "CREATE TRIGGER inspections_no_update_delete BEFORE UPDATE OR DELETE ON inspections "
    "FOR EACH ROW EXECUTE FUNCTION sd_forbid_change()",
    "CREATE TRIGGER inspections_no_truncate BEFORE TRUNCATE ON inspections "
    "FOR EACH STATEMENT EXECUTE FUNCTION sd_forbid_change()",
    "GRANT SELECT, INSERT ON inspections TO sd_app",
    "ALTER TABLE charters ADD COLUMN operation_mode text "
    "CHECK (operation_mode IN ('sim','exercise','live'))",
    "ALTER TABLE charters ADD COLUMN overflight_allowed boolean NOT NULL DEFAULT false",
    "ALTER TABLE charters ADD CONSTRAINT charters_overflight_needs_mode "
    "CHECK (NOT overflight_allowed OR operation_mode IS NOT NULL)",
    "ALTER TABLE candidate_decisions DROP CONSTRAINT candidate_decisions_decision_check",
    "ALTER TABLE candidate_decisions ADD CONSTRAINT candidate_decisions_decision_check "
    "CHECK (decision IN ('confirm','reject','inspect','unsure','reopen','resume'))",
)


def upgrade() -> None:
    for statement in STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    # The decisions are append-only: a "resume" row cannot be rewritten, so the old check
    # comes back NOT VALID (it holds for every new row).
    op.execute("ALTER TABLE candidate_decisions DROP CONSTRAINT candidate_decisions_decision_check")
    op.execute(
        "ALTER TABLE candidate_decisions ADD CONSTRAINT candidate_decisions_decision_check "
        "CHECK (decision IN ('confirm','reject','inspect','unsure','reopen')) NOT VALID"
    )
    op.execute("ALTER TABLE charters DROP CONSTRAINT charters_overflight_needs_mode")
    op.execute("ALTER TABLE charters DROP COLUMN overflight_allowed")
    op.execute("ALTER TABLE charters DROP COLUMN operation_mode")
    op.execute("DROP TABLE inspections")
