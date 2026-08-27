'use client';

/**
 * Presence state rendering — shared by the on-campus page and the
 * "Where is someone?" search card so both phrase presence identically.
 *
 * Three visible states over four internal ones: ON_CAMPUS and SEEN_EARLIER
 * look the same, because the honest difference between them is *how long
 * ago*, and that is always shown as text. What separates them is only
 * whether they count toward the headline number.
 *
 * Labelling rule (SPEC §1.3): a location is only asserted as *current* while
 * a camera is actually looking at the person. Otherwise it is "last seen at".
 */

export const PRESENCE_CONFIG = {
    LIVE: { dot: 'var(--success)', label: 'Visible now', pulse: true },
    ON_CAMPUS: { dot: 'var(--primary)', label: 'On campus', pulse: false },
    SEEN_EARLIER: { dot: 'var(--primary)', label: 'On campus', pulse: false },
    DEPARTED: { dot: 'var(--text-muted)', label: 'Departed', pulse: false },
};

/** "4 min ago" / "2h 14m ago". Never rounds away a stale claim. */
export function agoLabel(minutes) {
    if (minutes === null || minutes === undefined) return null;
    if (minutes < 1) return 'just now';
    if (minutes < 60) return `${minutes} min ago`;
    const h = Math.floor(minutes / 60);
    const m = minutes % 60;
    return m ? `${h}h ${m}m ago` : `${h}h ago`;
}

export function timeLabel(iso) {
    if (!iso) return '—';
    try {
        return new Date(iso).toLocaleTimeString([], {
            hour: '2-digit', minute: '2-digit', hour12: false,
        });
    } catch { return '—'; }
}

export function durationLabel(minutes) {
    if (!minutes) return '—';
    const h = Math.floor(minutes / 60);
    const m = minutes % 60;
    return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

export function PresenceStateDot({ state, size = 8 }) {
    const cfg = PRESENCE_CONFIG[state] || PRESENCE_CONFIG.DEPARTED;
    return (
        <span
            aria-hidden="true"
            style={{
                width: `${size}px`, height: `${size}px`, borderRadius: '50%',
                background: cfg.dot, flexShrink: 0, display: 'inline-block',
                animation: cfg.pulse ? 'pulse-dot 2s infinite' : 'none',
            }}
        />
    );
}

/**
 * Full status line: dot + label + freshness.
 *
 * The freshness suffix is what keeps a two-hour-old sighting honest while
 * still showing it as "On campus" — the reader can judge it themselves.
 */
export function PresenceStatus({ state, minutesSince, departedAt }) {
    const cfg = PRESENCE_CONFIG[state] || PRESENCE_CONFIG.DEPARTED;
    const ago = agoLabel(minutesSince);

    let suffix = null;
    if (state === 'DEPARTED') suffix = departedAt ? timeLabel(departedAt) : null;
    else if (state !== 'LIVE') suffix = ago;

    return (
        <span style={{
            display: 'inline-flex', alignItems: 'center', gap: '6px',
            fontSize: '0.75rem', fontWeight: 600, color: cfg.dot,
            whiteSpace: 'nowrap',
        }}>
            <PresenceStateDot state={state} />
            {cfg.label}
            {suffix && (
                <span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>
                    · {suffix}
                </span>
            )}
        </span>
    );
}
