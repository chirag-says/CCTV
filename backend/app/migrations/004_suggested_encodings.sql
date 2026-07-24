-- ============================================================================
-- Migration 004: Suggested Encodings + Encoding Health Tracking
--
-- Adds the suggested_encodings table for admin-reviewed embedding suggestions,
-- and health tracking columns to face_encodings for smart replacement decisions.
-- ============================================================================

-- ── Face Encodings: Add health tracking columns ─────────────────

ALTER TABLE face_encodings ADD COLUMN IF NOT EXISTS usage_count INTEGER DEFAULT 0;
ALTER TABLE face_encodings ADD COLUMN IF NOT EXISTS success_count INTEGER DEFAULT 0;
ALTER TABLE face_encodings ADD COLUMN IF NOT EXISTS last_matched TIMESTAMPTZ;
ALTER TABLE face_encodings ADD COLUMN IF NOT EXISTS pose_bucket VARCHAR(20);
ALTER TABLE face_encodings ADD COLUMN IF NOT EXISTS appearance_tags VARCHAR(255);

-- ── Suggested Encodings (admin approval queue) ──────────────────

CREATE TABLE IF NOT EXISTS suggested_encodings (
    id              VARCHAR(36) PRIMARY KEY,
    person_id       VARCHAR(36) NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    encoding_data   BYTEA NOT NULL,
    snapshot_url    VARCHAR(500),
    camera_id       VARCHAR(36),

    -- Quality & pose
    quality         FLOAT,
    yaw             FLOAT,
    pitch           FLOAT,
    roll            FLOAT,
    det_score       FLOAT,
    blur_score      FLOAT,
    lighting        VARCHAR(20),
    pose_bucket     VARCHAR(20),
    appearance_tags VARCHAR(255),

    -- Match context
    similarity      FLOAT,
    reason          VARCHAR(255),

    -- Status
    status          VARCHAR(20) DEFAULT 'pending',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_suggested_person ON suggested_encodings(person_id);
CREATE INDEX IF NOT EXISTS idx_suggested_status ON suggested_encodings(status);
CREATE INDEX IF NOT EXISTS idx_suggested_person_status ON suggested_encodings(person_id, status);

-- ============================================================================
-- Migration complete!
-- ============================================================================
