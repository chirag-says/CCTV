-- ============================================================================
-- Migration 006: Stream Health Events
--
-- Stream health monitoring is entirely in-memory (StreamHealthMonitor).
-- Stream offline/recovered events are persisted via the existing
-- detection_events table using:
--   event_type = 'security_alert'
--   subtype = 'stream_offline' | 'stream_recovered'
--
-- No new tables required. This migration adds an index to speed up
-- stream health event queries and ensures the subtype column can be
-- efficiently filtered for stream events on the security alerts page.
-- ============================================================================

-- Index for filtering stream health events specifically
CREATE INDEX IF NOT EXISTS idx_events_stream_health
    ON detection_events(subtype, created_at)
    WHERE subtype IN ('stream_offline', 'stream_recovered');

-- ============================================================================
-- Rollback:
--   DROP INDEX IF EXISTS idx_events_stream_health;
-- ============================================================================
