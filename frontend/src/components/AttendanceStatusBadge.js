'use client';

/**
 * AttendanceStatusBadge — present / late / absent, using the existing
 * .badge classes so attendance looks like the rest of the app.
 *
 * When somebody is absent but WAS seen, the shortfall is shown next to the
 * cross ("Absent · 1 sighting"). An absence is close to an accusation, so it
 * must never be a bare mark the admin cannot explain — the rule needs two
 * independent confirmations, and this says which one was missing.
 */

const CONFIG = {
    present: { cls: 'active', mark: '✓', label: 'Present' },
    late: { cls: 'pending', mark: '⚠', label: 'Late' },
    absent: { cls: 'offline', mark: '✕', label: 'Absent' },
};

export default function AttendanceStatusBadge({
    status,
    sightings = 0,
    hasEncodings = true,
    showReason = true,
}) {
    // Somebody with no enrolled face can never be recognised, so they are
    // absent every single day. That is an enrolment gap, not an attendance
    // record, and saying so prevents a standing false accusation.
    if (!hasEncodings) {
        return (
            <span
                className="badge pending"
                title="No face enrolled — this person cannot be recognised by the cameras"
            >
                No face enrolled
            </span>
        );
    }

    const cfg = CONFIG[status] || CONFIG.absent;
    const shortfall =
        showReason && status === 'absent' && sightings > 0
            ? `${sightings} sighting${sightings === 1 ? '' : 's'}`
            : null;

    return (
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
            <span className={`badge ${cfg.cls}`}>
                <span aria-hidden="true" style={{ marginRight: '4px' }}>{cfg.mark}</span>
                {cfg.label}
            </span>
            {shortfall && (
                <span
                    style={{ fontSize: '0.6875rem', color: 'var(--text-muted)' }}
                    title="Seen, but not enough to confirm attendance under the current rule"
                >
                    · {shortfall}
                </span>
            )}
        </span>
    );
}
