"""Mission flow: the "rule" backend and the charter card's default autonomy mode.

* ``commands.backend`` may be ``rule`` (an order the AI's rules gave without a language
  model, e.g. the rule take-off after the operator asked for a start).
* ``approvals.default_autonomy_mode``: what the charter card shows pre-selected (the
  active charter's mode, or the profile's default for a first charter). The pilot's
  choice at the click is still what the charter gets.

Revision ID: 0002_mission_flow
Revises: 0001_initial
Create Date: 2026-10-17
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002_mission_flow"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE commands DROP CONSTRAINT commands_backend_check")
    op.execute(
        "ALTER TABLE commands ADD CONSTRAINT commands_backend_check "
        "CHECK (backend IN ('llm','grammar','scripted','rule','cloud'))"
    )
    op.execute(
        "ALTER TABLE approvals ADD COLUMN default_autonomy_mode text "
        "CHECK (default_autonomy_mode IN ('APPROVE_EACH','WITHIN_CHARTER'))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE approvals DROP COLUMN default_autonomy_mode")
    op.execute("UPDATE commands SET backend = NULL WHERE backend = 'rule'")
    op.execute("ALTER TABLE commands DROP CONSTRAINT commands_backend_check")
    op.execute(
        "ALTER TABLE commands ADD CONSTRAINT commands_backend_check "
        "CHECK (backend IN ('llm','grammar','scripted','cloud'))"
    )
