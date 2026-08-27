'use client';

/**
 * JourneyMetrics — the four numbers that summarise a person's day.
 *
 *   FIRST DETECTED    LAST DETECTED    CAMERAS    TIME OBSERVED
 *   08:42             17:21            4          8h 39m
 *
 * "Time observed" is deliberately not "time on campus": it is the span
 * between the first and last detection, and the person was out of camera
 * view for much of it (SPEC §1.3).
 */

export function hhmm(iso) {
    if (!iso) return '—';
    try {
        return new Date(iso).toLocaleTimeString([], {
            hour: '2-digit', minute: '2-digit', hour12: false,
        });
    } catch { return '—'; }
}

export function humanDuration(seconds) {
    if (!seconds || seconds < 60) return seconds ? `${seconds}s` : '—';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    if (h && m) return `${h}h ${m}m`;
    if (h) return `${h}h`;
    return `${m}m`;
}

function Metric({ label, value, hint }) {
    return (
        <div style={{ flex: '1 1 120px', minWidth: '110px' }}>
            <div style={{
                fontSize: '0.625rem', fontWeight: 700, letterSpacing: '0.07em',
                textTransform: 'uppercase', color: 'var(--text-muted)',
            }}>
                {label}
            </div>
            <div style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: '1.375rem', fontWeight: 700,
                color: 'var(--text-primary)', marginTop: '4px', lineHeight: 1.1,
            }}>
                {value}
            </div>
            {hint && (
                <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                    {hint}
                </div>
            )}
        </div>
    );
}

export default function JourneyMetrics({ journey }) {
    const { first_seen, last_seen, camera_count, observed_sec, stop_count } = journey;
    return (
        <div className="card" style={{
            display: 'flex', gap: '18px', flexWrap: 'wrap', padding: '18px 20px',
        }}>
            <Metric label="First detected" value={hhmm(first_seen)} />
            <Metric label="Last detected" value={hhmm(last_seen)} />
            <Metric
                label="Cameras"
                value={camera_count ?? 0}
                hint={stop_count ? `${stop_count} stop${stop_count === 1 ? '' : 's'}` : null}
            />
            <Metric
                label="Time observed"
                value={humanDuration(observed_sec)}
                hint="first to last detection"
            />
        </div>
    );
}
