'use client';

/**
 * DepartmentCard — "CSE  18 / 22" with a proportional bar.
 *
 * Clicking selects the department; clicking the selected one clears it.
 * Built from the existing .card tokens rather than a new visual language.
 */

export default function DepartmentCard({ dept, selected = false, onSelect }) {
    const { name, attending, total, pct, present, late, absent } = dept;

    // Late still counts as attending, but is shown separately in the bar so a
    // department carrying a lot of late arrivals is visible at a glance.
    const presentPct = total ? (present / total) * 100 : 0;
    const latePct = total ? (late / total) * 100 : 0;

    const tone =
        pct >= 85 ? 'var(--success)'
            : pct >= 70 ? 'var(--warning)'
                : 'var(--danger)';

    return (
        <button
            type="button"
            className="card"
            onClick={() => onSelect?.(selected ? null : name)}
            aria-pressed={selected}
            id={`dept-card-${name.toLowerCase().replace(/\s+/g, '-')}`}
            style={{
                textAlign: 'left', cursor: 'pointer', width: '100%',
                padding: '16px', border: '1px solid',
                borderColor: selected ? 'var(--primary)' : 'var(--border-color)',
                boxShadow: selected ? '0 0 0 3px var(--primary-glow)' : undefined,
                transition: 'border-color 0.15s ease, box-shadow 0.15s ease',
                font: 'inherit', color: 'inherit',
            }}
        >
            <div style={{
                display: 'flex', alignItems: 'baseline',
                justifyContent: 'space-between', marginBottom: '10px', gap: '8px',
            }}>
                <span style={{
                    fontWeight: 700, fontSize: '0.9375rem', color: 'var(--text-primary)',
                }}>
                    {name}
                </span>
                <span style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: '1.0625rem', fontWeight: 700, color: tone,
                }}>
                    {attending}<span style={{ color: 'var(--text-muted)', fontWeight: 500 }}> / {total}</span>
                </span>
            </div>

            <div
                role="progressbar"
                aria-valuenow={Math.round(pct)}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label={`${name} attendance`}
                style={{
                    display: 'flex', height: '6px', borderRadius: '3px',
                    overflow: 'hidden', background: 'var(--border-color)',
                }}
            >
                <div style={{ width: `${presentPct}%`, background: 'var(--success)' }} />
                <div style={{ width: `${latePct}%`, background: 'var(--warning)' }} />
            </div>

            <div style={{
                display: 'flex', justifyContent: 'space-between',
                marginTop: '8px', fontSize: '0.6875rem', color: 'var(--text-muted)',
            }}>
                <span>{pct}%</span>
                <span>
                    {present} present{late > 0 ? ` · ${late} late` : ''}
                    {absent > 0 ? ` · ${absent} absent` : ''}
                </span>
            </div>
        </button>
    );
}
