"""Drawn paths: map annotations of kind ``path`` (a LineString, ``pa-<n>``).

The pilot draws a path on the map (along a trail or a stream); the AI can propose to fly
it with ``follow_path``.

Revision ID: 0003_path_annotations
Revises: 0002_mission_flow
Create Date: 2026-10-18
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003_path_annotations"
down_revision: str | None = "0002_mission_flow"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE map_annotations DROP CONSTRAINT map_annotations_kind_check")
    op.execute(
        "ALTER TABLE map_annotations ADD CONSTRAINT map_annotations_kind_check "
        "CHECK (kind IN ('pin','area','path'))"
    )
    op.execute("ALTER TABLE map_annotations DROP CONSTRAINT map_annotations_geom_check")
    op.execute(
        "ALTER TABLE map_annotations ADD CONSTRAINT map_annotations_geom_check "
        "CHECK (GeometryType(geom) IN ('POINT','POLYGON','LINESTRING'))"
    )


def downgrade() -> None:
    op.execute("DELETE FROM map_annotations WHERE kind = 'path'")
    op.execute("ALTER TABLE map_annotations DROP CONSTRAINT map_annotations_geom_check")
    op.execute(
        "ALTER TABLE map_annotations ADD CONSTRAINT map_annotations_geom_check "
        "CHECK (GeometryType(geom) IN ('POINT','POLYGON'))"
    )
    op.execute("ALTER TABLE map_annotations DROP CONSTRAINT map_annotations_kind_check")
    op.execute(
        "ALTER TABLE map_annotations ADD CONSTRAINT map_annotations_kind_check "
        "CHECK (kind IN ('pin','area'))"
    )
