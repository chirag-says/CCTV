'use client';

/**
 * DatePicker — compact day navigator: ‹ 27 Aug 2026 ›
 *
 * Shared by attendance, journey and the daily summary so every date-scoped
 * page steps through days the same way. Forward navigation stops at today:
 * none of these features can report on a day that hasn't happened.
 */

import { ChevronLeftIcon, ChevronRightIcon } from './Icons';

/** Local YYYY-MM-DD. toISOString() would shift the date across the UTC boundary. */
export function toDateKey(d) {
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

export function todayKey() {
    return toDateKey(new Date());
}

function shiftDays(dateKey, delta) {
    const [y, m, d] = dateKey.split('-').map(Number);
    const dt = new Date(y, m - 1, d);
    dt.setDate(dt.getDate() + delta);
    return toDateKey(dt);
}

function labelFor(dateKey) {
    const [y, m, d] = dateKey.split('-').map(Number);
    const dt = new Date(y, m - 1, d);
    if (dateKey === todayKey()) return 'Today';
    return dt.toLocaleDateString('en-GB', {
        day: 'numeric', month: 'short', year: 'numeric',
    });
}

export default function DatePicker({ value, onChange, id = 'date-picker' }) {
    const today = todayKey();
    const atToday = value >= today;

    return (
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <button
                className="btn btn-secondary btn-sm"
                onClick={() => onChange(shiftDays(value, -1))}
                aria-label="Previous day"
                id={`${id}-prev`}
                style={{ padding: '4px 8px' }}
            >
                <ChevronLeftIcon size={14} />
            </button>

            <label
                htmlFor={`${id}-input`}
                style={{
                    position: 'relative', display: 'inline-flex', alignItems: 'center',
                    justifyContent: 'center', minWidth: '132px', height: '30px',
                    padding: '0 10px', cursor: 'pointer',
                    borderRadius: '8px', border: '1px solid var(--border-color)',
                    background: 'var(--bg-tertiary, transparent)',
                    fontSize: '0.8125rem', fontWeight: 600,
                    color: 'var(--text-primary)',
                }}
            >
                {labelFor(value)}
                {/* The native picker sits invisibly on top so the whole pill is
                    clickable while the label stays formatted the way we want. */}
                <input
                    id={`${id}-input`}
                    type="date"
                    value={value}
                    max={today}
                    onChange={(e) => e.target.value && onChange(e.target.value)}
                    style={{
                        position: 'absolute', inset: 0, opacity: 0,
                        width: '100%', height: '100%', cursor: 'pointer',
                    }}
                />
            </label>

            <button
                className="btn btn-secondary btn-sm"
                onClick={() => onChange(shiftDays(value, 1))}
                disabled={atToday}
                aria-label="Next day"
                id={`${id}-next`}
                style={{ padding: '4px 8px', opacity: atToday ? 0.4 : 1 }}
            >
                <ChevronRightIcon size={14} />
            </button>

            {!atToday && (
                <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => onChange(today)}
                    style={{ marginLeft: '2px' }}
                    id={`${id}-today`}
                >
                    Today
                </button>
            )}
        </div>
    );
}
