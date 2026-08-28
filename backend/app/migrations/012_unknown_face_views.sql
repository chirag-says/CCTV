-- ─────────────────────────────────────────────────────────────────
-- 012_unknown_face_views.sql
--
-- Let an unknown identity keep MORE THAN ONE face view, so it can be
-- re-identified on another camera or after a restart.
--
-- THE MEASURED PROBLEM
-- Cross-camera re-identification of unenrolled people was measured at
-- exactly 0.0% -- 415 strangers across 7 live cameras, not one matched
-- on two of them. Cross-day was 0.3-0.5%.
--
-- The matching logic was not at fault. `UnknownRegistry` already does
-- best-of-K matching over several views per identity, precisely so a
-- stranger first seen frontally can still be matched in profile. But
-- those extra views existed ONLY in the camera thread's memory:
-- `unknown_faces` has a single `face_encoding` column, holding the
-- first view ever seen. So `maybe_refresh()` rebuilt every identity
-- as a single frozen frontal vector, and best-of-K silently degraded
-- to best-of-1 everywhere except inside one camera's uptime.
--
-- That is the exact failure unknown_registry.py's own docstring says
-- multi-view was introduced to fix. It was fixed in memory and never
-- on disk.
--
-- WHY A TABLE AND NOT A COLUMN
-- Mirrors `face_encodings`, which is what enrolled people already use
-- for the same reason. Per-view pose and quality can then be recorded,
-- which is what the Gallery Optimizer's diversity gate reads when an
-- unknown is eventually promoted to a person.
--
-- APPLY WITH (from backend/):
--   python apply_migration.py app/migrations/012_unknown_face_views.sql
--
-- Idempotent. Additive: `unknown_faces.face_encoding` keeps its meaning
-- as the identity's canonical first view and is still written.
-- ─────────────────────────────────────────────────────────────────

BEGIN;

-- ── 1. The table ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS unknown_face_views (
    id                VARCHAR(36) PRIMARY KEY,
    unknown_face_id   VARCHAR(36) NOT NULL
                      REFERENCES unknown_faces(id) ON DELETE CASCADE,
    encoding          BYTEA NOT NULL,
    pose_bucket       VARCHAR(20),
    quality           FLOAT,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

-- ON DELETE CASCADE, unlike detection_events.unknown_person_id in 011:
-- a view is not a historical record, it is matching material. When the
-- identity goes, its views are meaningless.

-- The only access pattern: "give me every view of these identities",
-- run on every registry refresh.
CREATE INDEX IF NOT EXISTS ix_unknown_views_face
    ON unknown_face_views (unknown_face_id);


-- ── 2. Seed from the canonical first view ────────────────────────
--
-- Every pending identity that has an encoding gets view #1, so the
-- table is immediately complete rather than only covering identities
-- seen after this migration. Additional views accumulate from here as
-- the registry learns them.
--
-- gen_random_uuid() is pgcrypto/PG13+; md5() is used instead so this
-- does not depend on an extension being installed.
INSERT INTO unknown_face_views (id, unknown_face_id, encoding, created_at)
SELECT
    md5(uf.id || ':seed')::uuid::text,
    uf.id,
    uf.face_encoding,
    COALESCE(uf.first_seen, NOW())
FROM unknown_faces uf
WHERE uf.face_encoding IS NOT NULL
  AND NOT EXISTS (
      SELECT 1 FROM unknown_face_views v WHERE v.unknown_face_id = uf.id
  );

COMMIT;


-- ── Verification ─────────────────────────────────────────────────
-- Identities and how many views each now has:
--   SELECT views, COUNT(*) AS identities FROM (
--     SELECT unknown_face_id, COUNT(*) AS views
--       FROM unknown_face_views GROUP BY 1) t
--   GROUP BY 1 ORDER BY 1;
--
-- Immediately after this migration every identity has exactly 1. The
-- number to watch is whether identities with 2+ views appear once the
-- cameras have run: that is the registry persisting what it learns.
--
-- Then re-run the re-identification measurement that motivated this:
--   SELECT COUNT(*) FILTER (WHERE cams > 1)::float / COUNT(*) AS cross_camera
--     FROM (SELECT unknown_person_id, COUNT(DISTINCT camera_id) AS cams
--             FROM detection_events
--            WHERE unknown_person_id IS NOT NULL
--              AND created_at >= NOW() - INTERVAL '1 day'
--            GROUP BY 1) s;
--   -- was 0.000
--
-- ── Rollback (DOWN) ──────────────────────────────────────────────
--   DROP TABLE IF EXISTS unknown_face_views;
--   -- unknown_faces.face_encoding is untouched, so matching degrades
--   -- to the pre-012 single-view behaviour rather than breaking.
-- ─────────────────────────────────────────────────────────────────
