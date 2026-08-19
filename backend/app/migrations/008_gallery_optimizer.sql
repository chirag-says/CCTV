-- ============================================================================
-- Migration 008: Gallery Optimizer
--
-- Enriches the identity review system with gallery improvement scoring,
-- coverage tracking, review prioritization, and versioned gallery updates.
--
-- Part of the Gallery Optimizer feature that continuously improves
-- biometric gallery coverage using local VLM (Ollama Qwen2.5-VL-7B).
-- ============================================================================

-- ── 1. Enrich identity_review_candidates ─────────────────────────
ALTER TABLE identity_review_candidates
    ADD COLUMN IF NOT EXISTS gallery_improvement_value VARCHAR(20),
    ADD COLUMN IF NOT EXISTS gallery_improvement_reason VARCHAR(255),
    ADD COLUMN IF NOT EXISTS coverage_before FLOAT,
    ADD COLUMN IF NOT EXISTS coverage_after FLOAT,
    ADD COLUMN IF NOT EXISTS vlm_detected_pose VARCHAR(20),
    ADD COLUMN IF NOT EXISTS vlm_quality_assessment VARCHAR(20),
    ADD COLUMN IF NOT EXISTS review_priority VARCHAR(20) DEFAULT 'low';

-- Priority index for the review queue (critical first, then by score)
CREATE INDEX IF NOT EXISTS idx_candidates_priority
    ON identity_review_candidates(review_priority, overall_score DESC)
    WHERE status = 'pending';

-- ── 2. Gallery version tracking ──────────────────────────────────
ALTER TABLE gallery_updates
    ADD COLUMN IF NOT EXISTS gallery_version INTEGER;

-- ============================================================================
-- Rollback:
--   ALTER TABLE identity_review_candidates
--       DROP COLUMN IF EXISTS gallery_improvement_value,
--       DROP COLUMN IF EXISTS gallery_improvement_reason,
--       DROP COLUMN IF EXISTS coverage_before,
--       DROP COLUMN IF EXISTS coverage_after,
--       DROP COLUMN IF EXISTS vlm_detected_pose,
--       DROP COLUMN IF EXISTS vlm_quality_assessment,
--       DROP COLUMN IF EXISTS review_priority;
--   DROP INDEX IF EXISTS idx_candidates_priority;
--   ALTER TABLE gallery_updates DROP COLUMN IF EXISTS gallery_version;
-- ============================================================================
