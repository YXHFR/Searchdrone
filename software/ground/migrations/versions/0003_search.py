"""Stage 2 search: profiles, perception, candidates, the probability map and cloud spend.

Every new table is history and append-only: sd_app may only read and add, and the
``sd_forbid_change()`` triggers from 0001 stop changes by the owner too. A change to a
candidate is a new row with the next ``rev``; a person's decision is its own row; a
cloud reservation that is committed or released is a new row with the new ``status``.

Also ``missions.found_candidate_id`` (set when a person confirms a find; the mission
state does not change) and ``map_annotations.poa_hint`` (an operator's hint for an
area on the probability map).

Revision ID: 0003_search
Revises: 0002_mission_flow
Create Date: 2026-10-18
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy.util import await_only

revision: str = "0003_search"
down_revision: str | None = "0002_mission_flow"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Drop order (children first). Every table here is append-only.
TABLES = (
    "candidate_decisions",
    "candidate_assessments",
    "candidate_observations",
    "candidates",
    "detections",
    "perception_stills",
    "probability_maps",
    "cloud_usage",
    "search_profiles",
)

SCHEMA = """
-- A proposal from the AI has no approved_by. The profile in force is the highest approved
-- profile_version of the mission.
CREATE TABLE search_profiles (
  id               uuid PRIMARY KEY DEFAULT uuidv7(),
  mission_id       uuid NOT NULL REFERENCES missions(id),
  profile_id       uuid NOT NULL,
  profile_version  int  NOT NULL CHECK (profile_version >= 1),
  body             jsonb NOT NULL,                        -- SearchProfile
  proposed_by      text NOT NULL,
  approved_by      text CHECK (approved_by IS NULL OR approved_by ~ '^operator:.+'),
  approved_at      timestamptz,
  created_at       timestamptz NOT NULL DEFAULT now(),
  CHECK ((approved_by IS NULL) = (approved_at IS NULL))
);
CREATE UNIQUE INDEX search_profiles_one_approved_version
  ON search_profiles(mission_id, profile_version) WHERE approved_by IS NOT NULL;
CREATE INDEX search_profiles_mission_idx ON search_profiles(mission_id, created_at);

CREATE TABLE perception_stills (
  id            uuid PRIMARY KEY DEFAULT uuidv7(),
  seq           bigint GENERATED ALWAYS AS IDENTITY UNIQUE,  -- after_seq paging
  mission_id    uuid NOT NULL REFERENCES missions(id),
  vehicle_id    text NOT NULL REFERENCES vehicles(id),
  still_id      uuid NOT NULL UNIQUE,
  command_id    uuid,
  captured_utc  timestamptz NOT NULL,
  sensor        text NOT NULL CHECK (sensor IN ('rgb')),
  pose          jsonb NOT NULL,                           -- Pose
  footprint     geometry(Polygon, 4326) NOT NULL,
  px_per_m      real NOT NULL CHECK (px_per_m > 0),
  detector      jsonb NOT NULL,                           -- DetectorInfo
  synthetic     boolean NOT NULL DEFAULT false,
  created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX perception_stills_mission_idx ON perception_stills(mission_id, seq);
CREATE INDEX perception_stills_footprint_gist ON perception_stills USING gist(footprint);

CREATE TABLE detections (
  id            uuid PRIMARY KEY DEFAULT uuidv7(),
  seq           bigint GENERATED ALWAYS AS IDENTITY UNIQUE,
  mission_id    uuid NOT NULL REFERENCES missions(id),
  vehicle_id    text NOT NULL REFERENCES vehicles(id),
  detection_id  uuid NOT NULL UNIQUE,
  still_id      uuid NOT NULL REFERENCES perception_stills(still_id),
  score         real NOT NULL CHECK (score BETWEEN 0 AND 1),
  bbox          jsonb NOT NULL,                           -- BBox in still pixels
  geom          geometry(Point, 4326) NOT NULL,
  sigma95_m     real NOT NULL CHECK (sigma95_m >= 0),
  px_size       real NOT NULL CHECK (px_size > 0),
  crop_path     text CHECK (crop_path !~ '(^/|\\.\\.)'),  -- relative to the mission folder
  crop_sha256   text CHECK (crop_sha256 ~ '^[0-9a-f]{64}$'),
  created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX detections_mission_idx ON detections(mission_id, seq);
CREATE INDEX detections_geom_gist ON detections USING gist(geom);

-- One row per revision; the candidate in force is the highest rev.
CREATE TABLE candidates (
  id                    uuid PRIMARY KEY DEFAULT uuidv7(),
  candidate_id          uuid NOT NULL,
  rev                   int  NOT NULL CHECK (rev >= 1),
  mission_id            uuid NOT NULL REFERENCES missions(id),
  seq                   int  NOT NULL CHECK (seq >= 1),  -- display number K1, K2, ...
  state                 text NOT NULL CHECK (state IN ('new','assessing','ready','inspecting',
                          'confirmed','rejected','merged')),
  geom                  geometry(Point, 4326) NOT NULL,
  sigma95_m             real NOT NULL CHECK (sigma95_m >= 0),
  n_seen                int  NOT NULL CHECK (n_seen >= 0),
  n_in_view             int  NOT NULL CHECK (n_in_view >= 0),
  best_detector_score   real NOT NULL CHECK (best_detector_score BETWEEN 0 AND 1),
  refined_person_score  real CHECK (refined_person_score BETWEEN 0 AND 1),
  priority              real NOT NULL CHECK (priority BETWEEN 0 AND 1),
  merged_into           uuid,
  latest_assessment_id  uuid,
  first_seen_utc        timestamptz NOT NULL,
  last_seen_utc         timestamptz NOT NULL,
  created_at            timestamptz NOT NULL DEFAULT now(),
  UNIQUE (candidate_id, rev),
  CHECK ((state = 'merged') = (merged_into IS NOT NULL))
);
CREATE INDEX candidates_mission_idx ON candidates(mission_id, candidate_id, rev);
CREATE INDEX candidates_geom_gist ON candidates USING gist(geom);

CREATE TABLE candidate_observations (
  id            uuid PRIMARY KEY DEFAULT uuidv7(),
  candidate_id  uuid NOT NULL,
  detection_id  uuid NOT NULL REFERENCES detections(detection_id),
  distance_m    real NOT NULL CHECK (distance_m >= 0),
  created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX candidate_observations_cand_idx ON candidate_observations(candidate_id);

CREATE TABLE candidate_assessments (
  id               uuid PRIMARY KEY DEFAULT uuidv7(),
  candidate_id     uuid NOT NULL,
  profile_version  int  NOT NULL CHECK (profile_version >= 0),  -- 0 = no approved profile
  engine           text NOT NULL CHECK (engine IN ('code','fast','strong','oracle')),
  model_id         text NOT NULL,
  body             jsonb NOT NULL,                        -- CandidateAssessment
  latency_ms       int  NOT NULL CHECK (latency_ms >= 0),
  cost_sek         real NOT NULL DEFAULT 0 CHECK (cost_sek >= 0),
  created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX candidate_assessments_cand_idx ON candidate_assessments(candidate_id, created_at);

-- Only a named person decides. The server also refuses service tokens (E1).
CREATE TABLE candidate_decisions (
  id            uuid PRIMARY KEY DEFAULT uuidv7(),
  candidate_id  uuid NOT NULL,
  decision      text NOT NULL CHECK (decision IN ('confirm','reject','inspect','unsure','reopen')),
  note          text NOT NULL DEFAULT '' CHECK (length(note) <= 300),
  subject_rev   int  NOT NULL CHECK (subject_rev >= 1),
  operator_ref  text NOT NULL CHECK (operator_ref ~ '^operator:.+'),
  session_id    text NOT NULL,
  created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX candidate_decisions_cand_idx ON candidate_decisions(candidate_id, created_at);

-- p and pod_cum: float32 per cell, zstd-compressed. row: the rest-of-world (ROW) cell.
CREATE TABLE probability_maps (
  id          uuid PRIMARY KEY DEFAULT uuidv7(),
  mission_id  uuid NOT NULL REFERENCES missions(id),
  version     int  NOT NULL CHECK (version >= 1),
  reason      text NOT NULL,
  grid        jsonb NOT NULL,
  p           bytea NOT NULL,
  pod_cum     bytea NOT NULL,
  row         real NOT NULL CHECK (row BETWEEN 0 AND 1),
  frozen      boolean NOT NULL DEFAULT false,
  event_seq   bigint NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now(),
  UNIQUE (mission_id, version)
);

-- A reservation's life is several rows: reserved, then committed / released / expired.
CREATE TABLE cloud_usage (
  id                 uuid PRIMARY KEY DEFAULT uuidv7(),
  reservation_id     uuid NOT NULL,
  mission_id         uuid REFERENCES missions(id),
  process            text NOT NULL CHECK (process IN ('commander','search')),
  purpose            text NOT NULL,
  model_id           text NOT NULL,
  est_sek            real NOT NULL CHECK (est_sek >= 0),
  actual_sek         real CHECK (actual_sek >= 0),
  prompt_tokens      int  CHECK (prompt_tokens >= 0),
  completion_tokens  int  CHECK (completion_tokens >= 0),
  status             text NOT NULL CHECK (status IN ('reserved','committed','released','expired')),
  created_at         timestamptz NOT NULL DEFAULT now(),
  UNIQUE (reservation_id, status)
);
CREATE INDEX cloud_usage_day_idx ON cloud_usage(created_at);
CREATE INDEX cloud_usage_mission_idx ON cloud_usage(mission_id);

ALTER TABLE missions ADD COLUMN found_candidate_id uuid;
ALTER TABLE map_annotations ADD COLUMN poa_hint text
  CHECK (poa_hint IN ('high','low','excluded'));
ALTER TABLE map_annotations ADD CONSTRAINT map_annotations_poa_hint_area
  CHECK (poa_hint IS NULL OR kind = 'area');
"""

GRANTS = """
GRANT SELECT, INSERT ON search_profiles, perception_stills, detections, candidates,
      candidate_observations, candidate_assessments, candidate_decisions, probability_maps,
      cloud_usage TO sd_app;
GRANT USAGE ON SEQUENCE perception_stills_seq_seq, detections_seq_seq TO sd_app;
"""


def _append_only_triggers() -> str:
    lines = []
    for t in TABLES:
        lines.append(
            f"CREATE TRIGGER {t}_no_update_delete BEFORE UPDATE OR DELETE ON {t} "
            "FOR EACH ROW EXECUTE FUNCTION sd_forbid_change();"
        )
        lines.append(
            f"CREATE TRIGGER {t}_no_truncate BEFORE TRUNCATE ON {t} "
            "FOR EACH STATEMENT EXECUTE FUNCTION sd_forbid_change();"
        )
    return "\n".join(lines)


def _run_script(sql: str) -> None:
    """Run a multi-statement SQL script inside the migration's transaction (see 0001)."""
    bind = op.get_bind()
    bind.exec_driver_sql("SELECT 1")
    driver = bind.connection.driver_connection
    if driver is None:
        raise RuntimeError("no database connection")
    if not driver.is_in_transaction():
        raise RuntimeError("migration script must run inside a transaction")
    await_only(driver.execute(sql))


def upgrade() -> None:
    _run_script(SCHEMA)
    _run_script(_append_only_triggers())
    _run_script(GRANTS)


def downgrade() -> None:
    # Dropping a table drops its triggers, indexes and grants. Names come from the fixed
    # TABLES tuple, never from input.
    _run_script(
        "\n".join(
            [
                "ALTER TABLE map_annotations DROP CONSTRAINT map_annotations_poa_hint_area;",
                "ALTER TABLE map_annotations DROP COLUMN poa_hint;",
                "ALTER TABLE missions DROP COLUMN found_candidate_id;",
                *(f"DROP TABLE {t};" for t in TABLES),
            ]
        )
    )
