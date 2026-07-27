-- ============================================================================
-- Migration 005: Embedding V2 (Color-Correct BGR Input)
--
-- Fixes: InsightFace expects BGR input but the pipeline was converting to RGB
-- before calling app.get(), double-swapping channels. All stored embeddings
-- were computed on color-inverted images.
--
-- This migration adds a V2 embedding column so both versions can coexist
-- during the transition period. After migration, V1 embeddings are kept
-- for one week as rollback insurance, then dropped.
-- ============================================================================

-- Add V2 embedding column (will hold correctly-computed embeddings)
ALTER TABLE face_encodings ADD COLUMN IF NOT EXISTS encoding_data_v2 BYTEA;

-- Track which version each row has been migrated to
-- 1 = original (color-swapped BGR→RGB before InsightFace)
-- 2 = corrected (BGR passed directly, InsightFace handles swapRB internally)
ALTER TABLE face_encodings ADD COLUMN IF NOT EXISTS embedding_version INTEGER DEFAULT 1;

-- Index for finding un-migrated rows efficiently
CREATE INDEX IF NOT EXISTS idx_face_encodings_version
    ON face_encodings(embedding_version)
    WHERE embedding_version < 2;

-- ============================================================================
-- After running migrate_embeddings.py:
--   1. Verify accuracy on test set with V2 embeddings
--   2. UPDATE face_encodings SET encoding_data = encoding_data_v2 WHERE embedding_version = 2;
--   3. Wait one week, then:
--      ALTER TABLE face_encodings DROP COLUMN encoding_data_v2;
-- ============================================================================
