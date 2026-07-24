-- ============================================================================
-- Migration 003: Face Encoding Metadata + Encoding Versioning
--
-- Adds pose/quality/context metadata to face_encodings,
-- creates encoding_versions table for incremental cache sync,
-- and adds cluster_id to unknown_faces for unknown person grouping.
-- ============================================================================

-- ── Face Encodings: Add metadata columns ────────────────────────

-- Pose metadata (from InsightFace 5-point landmarks)
ALTER TABLE face_encodings ADD COLUMN IF NOT EXISTS yaw FLOAT;
ALTER TABLE face_encodings ADD COLUMN IF NOT EXISTS pitch FLOAT;
ALTER TABLE face_encodings ADD COLUMN IF NOT EXISTS roll FLOAT;

-- Quality metadata
ALTER TABLE face_encodings ADD COLUMN IF NOT EXISTS det_score FLOAT;
ALTER TABLE face_encodings ADD COLUMN IF NOT EXISTS blur_score FLOAT;

-- Capture context
ALTER TABLE face_encodings ADD COLUMN IF NOT EXISTS camera_id VARCHAR(36);
ALTER TABLE face_encodings ADD COLUMN IF NOT EXISTS capture_distance FLOAT;
ALTER TABLE face_encodings ADD COLUMN IF NOT EXISTS lighting VARCHAR(20);

-- Versioning
ALTER TABLE face_encodings ADD COLUMN IF NOT EXISTS model_version VARCHAR(50) DEFAULT 'arcface_r50';
ALTER TABLE face_encodings ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1;

-- ── Convert existing pickle blobs to raw float32 ────────────────
-- NOTE: This must be run from Python after the column changes.
-- Existing data uses pickle.dumps(numpy_array) format.
-- New data uses numpy_array.astype(np.float32).tobytes() format.
-- The application's deserializer handles both formats automatically
-- (tries np.frombuffer first, falls back to pickle.loads).
-- No SQL-level conversion is needed.

-- ── Unknown Faces: Add cluster_id ───────────────────────────────

ALTER TABLE unknown_faces ADD COLUMN IF NOT EXISTS cluster_id VARCHAR(36);
CREATE INDEX IF NOT EXISTS idx_unknown_cluster ON unknown_faces(cluster_id);

-- ── Encoding Versions (incremental cache sync) ──────────────────

CREATE TABLE IF NOT EXISTS encoding_versions (
    id              INTEGER PRIMARY KEY DEFAULT 1,
    version         INTEGER NOT NULL DEFAULT 0,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Seed initial version row
INSERT INTO encoding_versions (id, version, updated_at)
VALUES (1, 0, NOW())
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- Migration complete!
-- ============================================================================
