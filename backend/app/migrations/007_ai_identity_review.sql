-- ============================================================================
-- Migration 007: AI Identity Review (Background Gallery Learning)
--
-- Adds tables for the AI-powered identity review system:
--   1. unknown_clusters    — Groups of unknown faces belonging to same person
--   2. identity_review_candidates — Proposed matches between clusters and known persons
--   3. gallery_updates     — Approved learning events (pending embedding insertion)
--   4. identity_review_activity  — Audit trail for all review actions
--
-- All tables use UUID primary keys consistent with the rest of the schema.
-- ============================================================================

-- ── 1. Unknown Clusters ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS unknown_clusters (
    id VARCHAR(36) PRIMARY KEY,
    representative_snapshot VARCHAR(500),
    representative_face_id VARCHAR(36),
    embedding_centroid BYTEA,
    member_count INTEGER DEFAULT 0,
    first_seen TIMESTAMP WITH TIME ZONE,
    last_seen TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'active',  -- active/reviewed/merged/ignored
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_clusters_status
    ON unknown_clusters(status);

CREATE INDEX IF NOT EXISTS idx_clusters_last_seen
    ON unknown_clusters(last_seen DESC);

-- Link unknown_faces to clusters (column already exists, add FK constraint)
-- ALTER TABLE unknown_faces ADD CONSTRAINT fk_unknown_faces_cluster
--     FOREIGN KEY (cluster_id) REFERENCES unknown_clusters(id) ON DELETE SET NULL;
-- Note: FK constraint commented out because cluster_id already exists as a plain column.
-- Adding a hard FK would break existing inserts that use arbitrary cluster IDs.
-- The relationship is enforced at the application layer instead.

-- ── 2. Identity Review Candidates ───────────────────────────────
CREATE TABLE IF NOT EXISTS identity_review_candidates (
    id VARCHAR(36) PRIMARY KEY,
    cluster_id VARCHAR(36) NOT NULL,
    person_id VARCHAR(36) NOT NULL,
    embedding_similarity FLOAT,
    verifier_similarity FLOAT,
    vlm_confidence FLOAT,
    image_quality_score FLOAT,
    pose_quality_score FLOAT,
    cluster_consistency FLOAT,
    overall_score FLOAT,
    vlm_explanation TEXT,
    recommended_action VARCHAR(20) DEFAULT 'review',  -- approve/reject/review
    status VARCHAR(20) DEFAULT 'pending',  -- pending/approved/rejected/skipped
    reviewed_by VARCHAR(36),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_candidates_status
    ON identity_review_candidates(status);

CREATE INDEX IF NOT EXISTS idx_candidates_cluster
    ON identity_review_candidates(cluster_id);

CREATE INDEX IF NOT EXISTS idx_candidates_person
    ON identity_review_candidates(person_id);

CREATE INDEX IF NOT EXISTS idx_candidates_score
    ON identity_review_candidates(overall_score DESC)
    WHERE status = 'pending';

-- ── 3. Gallery Updates ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gallery_updates (
    id VARCHAR(36) PRIMARY KEY,
    person_id VARCHAR(36) NOT NULL,
    candidate_id VARCHAR(36),
    cluster_id VARCHAR(36),
    encoding_data BYTEA,
    snapshot_url VARCHAR(500),
    pose_bucket VARCHAR(20),
    quality FLOAT,
    action VARCHAR(20) DEFAULT 'append',  -- append/replace
    replaced_encoding_id VARCHAR(36),
    status VARCHAR(20) DEFAULT 'pending',  -- pending/applied/rolled_back
    applied_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gallery_updates_status
    ON gallery_updates(status);

CREATE INDEX IF NOT EXISTS idx_gallery_updates_person
    ON gallery_updates(person_id);

-- ── 4. Identity Review Activity (Audit Trail) ──────────────────
CREATE TABLE IF NOT EXISTS identity_review_activity (
    id VARCHAR(36) PRIMARY KEY,
    candidate_id VARCHAR(36),
    cluster_id VARCHAR(36),
    action VARCHAR(50) NOT NULL,  -- approved/rejected/skipped/merged_cluster/created_person/ignored_forever
    actor VARCHAR(36),
    details JSON,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_review_activity_created
    ON identity_review_activity(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_review_activity_candidate
    ON identity_review_activity(candidate_id);

-- ============================================================================
-- Rollback:
--   DROP TABLE IF EXISTS identity_review_activity;
--   DROP TABLE IF EXISTS gallery_updates;
--   DROP TABLE IF EXISTS identity_review_candidates;
--   DROP TABLE IF EXISTS unknown_clusters;
-- ============================================================================
