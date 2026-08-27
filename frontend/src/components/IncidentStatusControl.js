'use client';

/**
 * IncidentStatusControl — Open / Investigating / Resolved.
 *
 * Optimistic: the button flips immediately and reverts if the request fails,
 * so triaging a queue of alerts does not stall on the network. Admin-only on
 * the backend, so a non-admin sees the failure surfaced rather than a silent
 * no-op.
 */

import { useState } from 'react';
import api from '@/lib/api';
import { useToast } from '@/lib/ToastContext';

const OPTIONS = [
    { value: 'open', label: 'Open' },
    { value: 'investigating', label: 'Investigating' },
    { value: 'resolved', label: 'Resolved' },
];

export default function IncidentStatusControl({ eventId, status, onChange }) {
    const [current, setCurrent] = useState(status || 'open');
    const [saving, setSaving] = useState(false);
    const { showToast } = useToast();

    async function select(next) {
        if (next === current || saving) return;
        const previous = current;
        setCurrent(next);
        setSaving(true);
        try {
            const result = await api.setIncidentStatus(eventId, next);
            onChange?.(result);
            showToast({ type: 'success', message: `Marked ${next}` });
        } catch (e) {
            setCurrent(previous);
            showToast({ type: 'error', message: e.message || 'Could not update status' });
        } finally {
            setSaving(false);
        }
    }

    return (
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
            {OPTIONS.map((o) => (
                <button
                    key={o.value}
                    className={`btn btn-sm ${current === o.value ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => select(o.value)}
                    disabled={saving}
                    aria-pressed={current === o.value}
                    id={`incident-status-${o.value}`}
                >
                    {o.label}
                </button>
            ))}
        </div>
    );
}
