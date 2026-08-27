-- ─────────────────────────────────────────────────────────────────
-- 009_college_features.sql
--
-- Foundation for the college-management feature set:
--   Faculty Attendance · Currently on Campus · Person Journey ·
--   Daily Campus Summary · Incident Evidence Pack
--
-- WHY THIS FILE EXISTS
-- init_db() runs Base.metadata.create_all(), which creates missing
-- TABLES but never adds COLUMNS or INDEXES to tables that already
-- exist. There is no Alembic. So the new `incident_reviews` table is
-- created automatically on boot, but everything below must be applied
-- by hand -- exactly like 001..008.
--
-- APPLY WITH:
--   psql -U postgres -d sentinelai -f app/migrations/009_college_features.sql
--
-- Idempotent: safe to run more than once.
-- ─────────────────────────────────────────────────────────────────

BEGIN;

-- ── 1. persons.person_type ───────────────────────────────────────
--
-- Faculty vs student is currently encoded only in the free-text
-- `role` column, whose documented values (models/person.py: employee/
-- visitor/vip/banned) do NOT match what the seed scripts actually
-- write (assistant_professor / student). Every attendance and
-- presence query filters on this distinction, so string-matching
-- against `role` would silently break the first time somebody typed
-- "Asst. Professor" into the UI.
--
-- `role` is kept as the human-facing designation (Assistant
-- Professor, HOD, ...). `person_type` is the machine-facing class.

ALTER TABLE persons ADD COLUMN IF NOT EXISTS person_type VARCHAR(20);

-- Backfill from the designations the seed scripts and UI have used.
UPDATE persons SET person_type = 'faculty'
 WHERE person_type IS NULL
   AND role IN (
       'professor', 'associate_professor', 'assistant_professor',
       'hod', 'lecturer', 'teacher'
   );

UPDATE persons SET person_type = 'student'
 WHERE person_type IS NULL AND role = 'student';

UPDATE persons SET person_type = 'staff'
 WHERE person_type IS NULL AND role IN ('employee', 'staff');

-- Everything unclassified (including 'vip' and 'banned', which are
-- access flags rather than roles) falls back to visitor.
UPDATE persons SET person_type = 'visitor'
 WHERE person_type IS NULL;

-- Attendance rollups are always "faculty in department X".
CREATE INDEX IF NOT EXISTS ix_persons_type_dept
    ON persons (person_type, department);


-- ── 2. detection_events (person_id, created_at) ──────────────────
--
-- Every attendance, journey and presence query has the shape
--   WHERE person_id = ? AND created_at BETWEEN ? AND ?
-- The existing indexes are (event_type, created_at) and
-- (camera_id, created_at) plus a bare person_id -- none of which
-- serve that shape. Partial, because ~all unknown-person rows carry
-- person_id IS NULL (the FK guard in main.py moves the id into
-- metadata_json), and indexing those would be dead weight.

CREATE INDEX IF NOT EXISTS ix_events_person_created
    ON detection_events (person_id, created_at)
    WHERE person_id IS NOT NULL;

COMMIT;


-- ── Verification ─────────────────────────────────────────────────
-- SELECT person_type, role, COUNT(*)
--   FROM persons GROUP BY person_type, role ORDER BY person_type, role;
--
-- Expect after seeding: faculty/assistant_professor, student/student.
