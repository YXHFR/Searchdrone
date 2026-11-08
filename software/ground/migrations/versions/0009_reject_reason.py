"""Why a person rejected a candidate.

* ``candidate_decisions.reason``: ``not_a_person`` ("Ingen person här") on a rejection
  releases the candidate's keep-out cylinder; NULL (every older row, and "Fel person /
  osäker") keeps it when the candidate may be a person. Only a rejection has a reason.

Revision ID: 0009_reject_reason
Revises: 0008_floor_pack_choice
Create Date: 2026-11-08
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0009_reject_reason"
down_revision: str | None = "0008_floor_pack_choice"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

STATEMENTS = (
    "ALTER TABLE candidate_decisions ADD COLUMN reason text "
    "CHECK (reason IS NULL OR (reason = 'not_a_person' AND decision = 'reject'))",
)


def upgrade() -> None:
    for statement in STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("ALTER TABLE candidate_decisions DROP COLUMN reason")
