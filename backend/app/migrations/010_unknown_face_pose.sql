-- ============================================================================
-- Migration 010: Real pose metadata on unknown faces + member-level gallery
--                updates (Gallery Optimizer)
--
-- Why:
--   1. unknown_faces had no pose columns, so the Gallery Optimizer's
--      pose-rarity weight (15% of the priority score) and its coverage
--      delta were both computing against a hardcoded "frontal" constant.
--   2. Approving a review candidate wrote ONE averaged cluster centroid to
--      the gallery. Member-level approval needs each gallery_updates row to
--      point back at the individual unknown face it came from, and to record
--      why that member was selected.
--
-- Safety: purely additive. Every column is nullable with no default, so
-- existing rows keep a NULL pose rather than a fabricated 0-degree yaw.
-- Application code MUST treat NULL pose as "unknown", never as "frontal".
-- ============================================================================

-- ── 1. Pose + quality metadata on unknown faces ─────────────────
-- Mirrors the columns face_encodings and suggested_encodings already carry,
-- so an unknown face and a gallery encoding are described identically.
ALTER TABLE unknown_faces
    ADD COLUMN IF NOT EXISTS yaw         FLOAT,
    ADD COLUMN IF NOT EXISTS pitch       FLOAT,
    ADD COLUMN IF NOT EXISTS roll        FLOAT,
    ADD COLUMN IF NOT EXISTS pose_bucket VARCHAR(20),
    ADD COLUMN IF NOT EXISTS quality     FLOAT,
    ADD COLUMN IF NOT EXISTS blur_score  FLOAT,
    ADD COLUMN IF NOT EXISTS det_score   FLOAT,
    ADD COLUMN IF NOT EXISTS lighting    VARCHAR(20);

-- Partial index: the optimizer only ever scans faces that HAVE a pose.
CREATE INDEX IF NOT EXISTS idx_unknown_faces_pose_bucket
    ON unknown_faces(pose_bucket)
    WHERE pose_bucket IS NOT NULL;

-- ── 2. Member-level provenance on gallery updates ───────────────
-- source_face_id: which individual unknown face produced this embedding
--                 (was: the cluster centroid, which came from no single face)
-- selection_reason: why the diversity gate accepted it ("New Extreme Left profile")
ALTER TABLE gallery_updates
    ADD COLUMN IF NOT EXISTS source_face_id   VARCHAR(36),
    ADD COLUMN IF NOT EXISTS selection_reason VARCHAR(255),
    ADD COLUMN IF NOT EXISTS encoding_id      VARCHAR(36),
    ADD COLUMN IF NOT EXISTS coverage_before  FLOAT,
    ADD COLUMN IF NOT EXISTS coverage_after   FLOAT;

CREATE INDEX IF NOT EXISTS idx_gallery_updates_source_face
    ON gallery_updates(source_face_id);

-- ============================================================================
-- Rollback (DOWN):
--   DROP INDEX IF EXISTS idx_gallery_updates_source_face;
--   DROP INDEX IF EXISTS idx_unknown_faces_pose_bucket;
--   ALTER TABLE gallery_updates
--       DROP COLUMN IF EXISTS source_face_id,
--       DROP COLUMN IF EXISTS selection_reason,
--       DROP COLUMN IF EXISTS encoding_id,
--       DROP COLUMN IF EXISTS coverage_before,
--       DROP COLUMN IF EXISTS coverage_after;
--   ALTER TABLE unknown_faces
--       DROP COLUMN IF EXISTS yaw,
--       DROP COLUMN IF EXISTS pitch,
--       DROP COLUMN IF EXISTS roll,
--       DROP COLUMN IF EXISTS pose_bucket,
--       DROP COLUMN IF EXISTS quality,
--       DROP COLUMN IF EXISTS blur_score,
--       DROP COLUMN IF EXISTS det_score,
--       DROP COLUMN IF EXISTS lighting;
-- ============================================================================
