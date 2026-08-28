-- ─────────────────────────────────────────────────────────────────
-- 011_unknown_person_id.sql
--
-- Make the history of unenrolled people QUERYABLE.
--
-- WHY THIS FILE EXISTS
-- The data already exists. Since the unknown registry landed, every
-- entry/exit of a stranger with a stable identity has been recorded:
-- the FK guard in main.py cannot put that id in `person_id` (there is
-- no `persons` row, so the foreign key would reject it), so it moves
-- the id into the event's JSON metadata instead.
--
-- That made it write-only. A JSON key cannot be indexed usefully here,
-- `journey_service` takes a `person_id`, and the incident pack falls
-- back to correlating `unknown_faces.first_seen` within a time window
-- -- which misses every stranger whose FIRST EVER sighting was on some
-- other day, i.e. most repeat visitors.
--
-- Promoting the id to a real, indexed column is what turns
--   "who was in Classroom 204 at 14:00?"
-- from a heuristic into a query.
--
-- NOT A FOREIGN KEY. It points at `unknown_faces.id`, but an unknown
-- row can be enrolled (becoming a person) or dismissed, and neither
-- should cascade into or block the permanent event log. The id stays
-- meaningful as a historical record either way.
--
-- APPLY WITH (from backend/, uses the app's own DB settings):
--   python apply_migration.py app/migrations/011_unknown_person_id.sql
--
-- `psql` is not on PATH on a standard Windows Postgres install; the runner
-- reads app.config.settings so it always targets the same database the
-- backend talks to. Check drift at any time with:
--   python apply_migration.py --check
--
-- Idempotent: safe to run more than once.
-- Additive only: no column is dropped, no existing value is changed.
-- ─────────────────────────────────────────────────────────────────

BEGIN;

-- ── 1. The column ────────────────────────────────────────────────
ALTER TABLE detection_events
    ADD COLUMN IF NOT EXISTS unknown_person_id VARCHAR(36);


-- ── 2. Backfill from whichever JSON column this database has ─────
--
-- The metadata column's NAME is genuinely ambiguous in this repo:
-- 001_initial_schema.sql creates `metadata JSONB`, while db/models.py
-- declares `metadata_json` -- and a table created by
-- Base.metadata.create_all() (which is what init_db does) follows the
-- MODEL, not the SQL file. No migration renames one to the other, so
-- which one exists depends on how this particular database was first
-- created.
--
-- Rather than guess and fail, detect it. Anything not matching a
-- 36-character id is ignored: `anon:<camera>:<track>` handles were
-- never identities, and Step 1.5 stopped emitting them.
DO $$
DECLARE
    json_col TEXT;
    updated  BIGINT;
BEGIN
    SELECT column_name INTO json_col
      FROM information_schema.columns
     WHERE table_name = 'detection_events'
       AND column_name IN ('metadata_json', 'metadata')
     ORDER BY CASE column_name WHEN 'metadata_json' THEN 0 ELSE 1 END
     LIMIT 1;

    IF json_col IS NULL THEN
        RAISE NOTICE 'No metadata column found; nothing to backfill.';
        RETURN;
    END IF;

    -- `->>` and NOT NULL, deliberately NOT the `?` key-existence operator:
    -- `?` is jsonb-only, and this column is plain `json` (it was created by
    -- Base.metadata.create_all() from models.py's Column(JSON), not by
    -- 001_initial_schema.sql, which declared JSONB). `->>` works on both.
    EXECUTE format(
        'UPDATE detection_events
            SET unknown_person_id = %I->>''unknown_person_id''
          WHERE unknown_person_id IS NULL
            AND %I->>''unknown_person_id'' IS NOT NULL
            AND length(%I->>''unknown_person_id'') = 36',
        json_col, json_col, json_col
    );
    GET DIAGNOSTICS updated = ROW_COUNT;
    RAISE NOTICE 'Backfilled % rows from %.', updated, json_col;
END $$;


-- ── 3. Indexes ───────────────────────────────────────────────────
--
-- "Where has this stranger been?" -- the unknown-person journey.
-- Partial: the overwhelming majority of rows are NULL here (every
-- known person, every alert), and indexing those is dead weight.
-- Mirrors ix_events_person_created from 009, which does the same for
-- the identified half.
CREATE INDEX IF NOT EXISTS ix_events_unknown_created
    ON detection_events (unknown_person_id, created_at)
    WHERE unknown_person_id IS NOT NULL;

-- "Who was on THIS camera between T1 and T2?" -- the incident query.
-- It must return known and unknown subjects together, so it cannot be
-- served by the partial index above.
CREATE INDEX IF NOT EXISTS ix_events_camera_created_all
    ON detection_events (camera_id, created_at);

COMMIT;


-- ── Verification ─────────────────────────────────────────────────
-- How many events now carry a stable unknown identity:
--   SELECT COUNT(*) FROM detection_events WHERE unknown_person_id IS NOT NULL;
--
-- The busiest strangers, and where they were seen:
--   SELECT unknown_person_id,
--          COUNT(*)                  AS sightings,
--          COUNT(DISTINCT camera_id) AS cameras,
--          MIN(created_at)           AS first_seen,
--          MAX(created_at)           AS last_seen
--     FROM detection_events
--    WHERE unknown_person_id IS NOT NULL
--    GROUP BY unknown_person_id
--    ORDER BY sightings DESC
--    LIMIT 20;
--
-- Column and JSON must agree on rows written after this migration:
--   SELECT COUNT(*) FROM detection_events
--    WHERE unknown_person_id IS NOT NULL
--      AND metadata_json->>'unknown_person_id' IS DISTINCT FROM unknown_person_id;
--   -- expect 0
--
-- ── Rollback (DOWN) ──────────────────────────────────────────────
--   DROP INDEX IF EXISTS ix_events_camera_created_all;
--   DROP INDEX IF EXISTS ix_events_unknown_created;
--   ALTER TABLE detection_events DROP COLUMN IF EXISTS unknown_person_id;
--   -- The id survives in the JSON metadata, which is still written.
-- ─────────────────────────────────────────────────────────────────
