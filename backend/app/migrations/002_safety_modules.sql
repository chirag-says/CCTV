-- ============================================================================
-- Safety & Security Modules — Schema Update
-- Run this in your Supabase SQL Editor AFTER 001_initial_schema.sql
-- ============================================================================

-- 1. Expand the event_type CHECK constraint to include security_alert
--    The existing constraint only allows: 'entry', 'exit', 'detection', 'unknown'
--    We need to add: 'security_alert'

ALTER TABLE detection_events
    DROP CONSTRAINT IF EXISTS detection_events_event_type_check;

ALTER TABLE detection_events
    ADD CONSTRAINT detection_events_event_type_check
    CHECK (event_type IN ('entry', 'exit', 'detection', 'unknown', 'security_alert'));

-- 2. Add index for security alerts (useful for dashboard queries)
CREATE INDEX IF NOT EXISTS idx_events_security_alerts
    ON detection_events(event_type, created_at)
    WHERE event_type = 'security_alert';

-- 3. Create a view for security alerts with metadata details
CREATE OR REPLACE VIEW security_alerts AS
SELECT
    de.id,
    de.camera_id,
    c.name AS camera_name,
    c.location AS camera_location,
    de.event_type,
    de.metadata->>'subtype' AS alert_subtype,
    de.metadata->>'person_count' AS person_count,
    de.metadata->>'person_id' AS alert_person_id,
    de.metadata->>'person_name' AS alert_person_name,
    de.metadata->>'threat_class' AS threat_class,
    de.metadata->>'duration_sec' AS duration_sec,
    de.confidence,
    de.snapshot_url,
    de.metadata,
    de.created_at
FROM detection_events de
LEFT JOIN cameras c ON c.id = de.camera_id
WHERE de.event_type = 'security_alert'
ORDER BY de.created_at DESC;

-- ============================================================================
-- Migration complete!
-- ============================================================================
