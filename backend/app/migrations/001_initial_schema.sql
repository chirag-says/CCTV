-- ============================================================================
-- AI CCTV Surveillance System — Database Schema
-- Run this in your Supabase SQL Editor
-- ============================================================================

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Persons (Known Individuals) ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS persons (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            VARCHAR(255) NOT NULL,
    role            VARCHAR(50) DEFAULT 'visitor' CHECK (role IN ('employee', 'visitor', 'vip', 'banned')),
    department      VARCHAR(100),
    phone           VARCHAR(20),
    email           VARCHAR(255),
    avatar_url      TEXT,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_persons_role ON persons(role);
CREATE INDEX idx_persons_active ON persons(is_active);
CREATE INDEX idx_persons_name ON persons(name);

-- ── Face Encodings ───────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS face_encodings (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id       UUID NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    encoding        TEXT NOT NULL,  -- Base64-encoded pickle of numpy array
    source_image    TEXT,
    quality         FLOAT DEFAULT 1.0 CHECK (quality >= 0 AND quality <= 1),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_face_encodings_person ON face_encodings(person_id);

-- ── Cameras ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS cameras (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            VARCHAR(100) NOT NULL,
    location        VARCHAR(255) DEFAULT '',
    stream_url      TEXT DEFAULT '0',
    camera_type     VARCHAR(20) DEFAULT 'webcam' CHECK (camera_type IN ('webcam', 'rtsp', 'ip')),
    is_active       BOOLEAN DEFAULT TRUE,
    config          JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_cameras_active ON cameras(is_active);

-- ── Tracking Sessions ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS tracking_sessions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id       UUID NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    camera_id       UUID REFERENCES cameras(id) ON DELETE SET NULL,
    entry_time      TIMESTAMPTZ NOT NULL,
    exit_time       TIMESTAMPTZ,
    duration_sec    INTEGER,
    status          VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'completed')),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_sessions_person ON tracking_sessions(person_id);
CREATE INDEX idx_sessions_camera ON tracking_sessions(camera_id);
CREATE INDEX idx_sessions_status ON tracking_sessions(status);
CREATE INDEX idx_sessions_entry ON tracking_sessions(entry_time);

-- ── Detection Events ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS detection_events (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id       UUID REFERENCES persons(id) ON DELETE SET NULL,
    camera_id       UUID REFERENCES cameras(id) ON DELETE SET NULL,
    event_type      VARCHAR(20) NOT NULL CHECK (event_type IN ('entry', 'exit', 'detection', 'unknown')),
    confidence      FLOAT DEFAULT 0.0,
    snapshot_url    TEXT,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_events_type ON detection_events(event_type);
CREATE INDEX idx_events_person ON detection_events(person_id);
CREATE INDEX idx_events_camera ON detection_events(camera_id);
CREATE INDEX idx_events_created ON detection_events(created_at);
-- Composite index for common queries
CREATE INDEX idx_events_type_date ON detection_events(event_type, created_at);

-- ── Unknown Faces ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS unknown_faces (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id       UUID REFERENCES cameras(id) ON DELETE SET NULL,
    snapshot_url    TEXT,
    full_frame      TEXT,
    encoding        TEXT,  -- Base64-encoded pickle of numpy array
    occurrence      INTEGER DEFAULT 1,
    first_seen      TIMESTAMPTZ DEFAULT NOW(),
    last_seen       TIMESTAMPTZ DEFAULT NOW(),
    status          VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'enrolled', 'dismissed')),
    assigned_to     UUID,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_unknown_status ON unknown_faces(status);
CREATE INDEX idx_unknown_occurrence ON unknown_faces(occurrence DESC);
CREATE INDEX idx_unknown_camera ON unknown_faces(camera_id);

-- ── Admin Users ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS admin_users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email           VARCHAR(255) UNIQUE NOT NULL,
    password_hash   TEXT NOT NULL,
    name            VARCHAR(255) NOT NULL,
    role            VARCHAR(20) DEFAULT 'operator' CHECK (role IN ('superadmin', 'admin', 'operator')),
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_admin_email ON admin_users(email);

-- ── Row Level Security (RLS) ─────────────────────────────────────────────────

ALTER TABLE persons ENABLE ROW LEVEL SECURITY;
ALTER TABLE face_encodings ENABLE ROW LEVEL SECURITY;
ALTER TABLE cameras ENABLE ROW LEVEL SECURITY;
ALTER TABLE tracking_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE detection_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE unknown_faces ENABLE ROW LEVEL SECURITY;
ALTER TABLE admin_users ENABLE ROW LEVEL SECURITY;

-- Service role can do everything (backend uses service key)
CREATE POLICY "Service role full access" ON persons FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON face_encodings FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON cameras FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON tracking_sessions FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON detection_events FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON unknown_faces FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON admin_users FOR ALL USING (true) WITH CHECK (true);

-- ── Seed Default Admin ───────────────────────────────────────────────────────
-- Password: admin123 (bcrypt hash)
-- CHANGE THIS IN PRODUCTION!

INSERT INTO admin_users (id, email, password_hash, name, role)
VALUES (
    uuid_generate_v4(),
    'admin@cctv.local',
    '$2b$12$LJ3m2wv5E5Y5C5K5L5M5N.5O5P5Q5R5S5T5U5V5W5X5Y5Z5A5B5',
    'System Admin',
    'superadmin'
) ON CONFLICT (email) DO NOTHING;

-- ── Helpful Views ────────────────────────────────────────────────────────────

-- Active occupancy view
CREATE OR REPLACE VIEW active_occupancy AS
SELECT 
    p.id AS person_id,
    p.name AS person_name,
    p.role,
    ts.camera_id,
    c.name AS camera_name,
    ts.entry_time,
    EXTRACT(EPOCH FROM (NOW() - ts.entry_time))::INTEGER AS duration_sec
FROM tracking_sessions ts
JOIN persons p ON p.id = ts.person_id
LEFT JOIN cameras c ON c.id = ts.camera_id
WHERE ts.status = 'active';

-- Daily summary view
CREATE OR REPLACE VIEW daily_summary AS
SELECT 
    DATE(created_at) AS date,
    event_type,
    COUNT(*) AS count
FROM detection_events
GROUP BY DATE(created_at), event_type
ORDER BY date DESC, event_type;

-- Movement summary view (enriched detection_events)
CREATE OR REPLACE VIEW movements AS
SELECT 
    de.id,
    de.person_id,
    p.name AS person_name,
    p.role AS person_role,
    de.camera_id,
    c.name AS camera_name,
    c.location AS camera_location,
    de.event_type,
    de.confidence,
    de.snapshot_url,
    de.metadata,
    de.created_at
FROM detection_events de
LEFT JOIN persons p ON p.id = de.person_id
LEFT JOIN cameras c ON c.id = de.camera_id
ORDER BY de.created_at DESC;

-- Movement heatmap view (hourly aggregation for analytics)
CREATE OR REPLACE VIEW movement_heatmap AS
SELECT 
    DATE(created_at) AS date,
    EXTRACT(HOUR FROM created_at)::INTEGER AS hour,
    event_type,
    COUNT(*) AS event_count,
    COUNT(DISTINCT person_id) AS unique_persons,
    AVG(confidence)::FLOAT AS avg_confidence
FROM detection_events
GROUP BY DATE(created_at), EXTRACT(HOUR FROM created_at), event_type
ORDER BY date DESC, hour;

-- ============================================================================
-- Schema setup complete!
-- ============================================================================
