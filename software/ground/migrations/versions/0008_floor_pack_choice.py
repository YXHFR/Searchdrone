"""The pilot's height-floor map pack on the charter card.

* ``charters.floor_pack_id`` and ``charters.floor_pack_sha256``: set when the pilot
  approves a charter (the pack the closer look comes down with, or none), laid over the
  stored body like ``operation_mode``. Both or neither; the name and the hash in the
  same form as ``MissionCharter``'s. Charters approved before keep what their body says.

Revision ID: 0008_floor_pack_choice
Revises: 0007_inspections
Create Date: 2026-11-03
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0008_floor_pack_choice"
down_revision: str | None = "0007_inspections"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

STATEMENTS = (
    "ALTER TABLE charters ADD COLUMN floor_pack_id text "
    "CHECK (floor_pack_id ~ '^[a-z0-9][a-z0-9_-]{0,63}$')",
    "ALTER TABLE charters ADD COLUMN floor_pack_sha256 text "
    "CHECK (floor_pack_sha256 ~ '^[0-9a-f]{64}$')",
    "ALTER TABLE charters ADD CONSTRAINT charters_floor_pack_both "
    "CHECK ((floor_pack_id IS NULL) = (floor_pack_sha256 IS NULL))",
    # Whether the columns hold the pilot's pick: set on every approval from now on.
    "ALTER TABLE charters ADD COLUMN floor_pack_set boolean NOT NULL DEFAULT false",
    "ALTER TABLE charters ADD CONSTRAINT charters_floor_pack_set "
    "CHECK (floor_pack_set OR floor_pack_id IS NULL)",
)


def upgrade() -> None:
    for statement in STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("ALTER TABLE charters DROP CONSTRAINT charters_floor_pack_set")
    op.execute("ALTER TABLE charters DROP COLUMN floor_pack_set")
    op.execute("ALTER TABLE charters DROP CONSTRAINT charters_floor_pack_both")
    op.execute("ALTER TABLE charters DROP COLUMN floor_pack_sha256")
    op.execute("ALTER TABLE charters DROP COLUMN floor_pack_id")
