'use client';

/**
 * Incidents — the triage queue.
 *
 * Complements /security-alerts, which stays as the live monitoring wall.
 * This page is for working through alerts: filter by status, open one, act
 * on it. Camera stream-health events are excluded by default because they
 * are plumbing, not security.
 */

import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import AppShell from '@/components/AppShell';
import DatePicker, { todayKey } from '@/components/DatePicker';
import api from '@/lib/api';
import {
    SecurityIcon, AlertIcon, RefreshIcon, CameraIcon, ClockIcon, ArrowRightIcon,
} from '@/components/Icons';

const STATUS_TABS = [
    { value: '', label: 'All' },
    { value: 'open', label: 'Open' },
    { value: 'investigating', label: 'Investigating' },
    { value: 'resolved', label: 'Resolved' },
];

const SEVERITY_TONE = {
    high: 'var(--danger-light)',
    medium: 'var(--warning-light)',
    low: 'var(--accent-light)',
};

function when(iso) {
    if (!iso) return '—';
    try {
        return new Date(iso).toLocaleString('en-GB', {
            day: 'numeric', month: 'short',
            hour: '2-digit', minute: '2-digit', hour12: false,
        });
    } catch { return '—'; }
}

export default function IncidentsPage() {
    const [status, setStatus] = useState('open');
    const [severity, setSeverity] = useState('');
    const [date, setDate] = useState('');
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const router = useRouter();

    const load = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            setData(await api.getIncidents({ status, severity, date, limit: 100 }));
        } catch (e) {
            setError(e.message || 'Could not load incidents.');
            setData(null);
        } finally {
            setLoading(false);
        }
    }, [status, severity, date]);

    useEffect(() => { load(); }, [load]);

    const counts = data?.counts_by_status || {};
    const rows = data?.data || [];

    return (
        <AppShell>
            <div className="page-header">
                <div className="toolbar-responsive" style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px',
                }}>
                    <div>
                        <h1>Incidents</h1>
                        <p>Security alerts, with evidence and review status</p>
                    </div>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                        {date && <DatePicker value={date} onChange={setDate} id="incident-date" />}
                        <button
                            className={`btn btn-sm ${date ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={() => setDate(date ? '' : todayKey())}
                            id="incident-date-toggle"
                        >
                            {date ? 'All dates' : 'Filter by date'}
                        </button>
                        <button className="btn btn-secondary btn-sm" onClick={load} id="incident-refresh">
                            <RefreshIcon size={14} />
                        </button>
                    </div>
                </div>
            </div>

            {/* Status counts */}
            <div className="security-stats">
                <div className="security-stat total">
                    <div className="stat-number">{data?.total ?? 0}</div>
                    <div className="stat-label">Matching</div>
                </div>
                <div className="security-stat danger">
                    <div className="stat-number">{counts.open ?? 0}</div>
                    <div className="stat-label">Open</div>
                </div>
                <div className="security-stat warning">
                    <div className="stat-number">{counts.investigating ?? 0}</div>
                    <div className="stat-label">Investigating</div>
                </div>
                <div className="security-stat info">
                    <div className="stat-number">{counts.resolved ?? 0}</div>
                    <div className="stat-label">Resolved</div>
                </div>
            </div>

            {/* Filters */}
            <div className="card" style={{ marginBottom: '20px' }}>
                <div className="toolbar-responsive" style={{
                    display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center',
                }}>
                    {STATUS_TABS.map((t) => (
                        <button
                            key={t.value || 'all'}
                            className={`btn btn-sm ${status === t.value ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={() => setStatus(t.value)}
                            id={`incident-filter-${t.value || 'all'}`}
                        >
                            {t.label}
                        </button>
                    ))}
                    <span style={{ width: '1px', height: '20px', background: 'var(--border-color)' }} />
                    {['', 'high', 'medium', 'low'].map((s) => (
                        <button
                            key={s || 'any'}
                            className={`btn btn-sm ${severity === s ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={() => setSeverity(s)}
                        >
                            {s ? s.charAt(0).toUpperCase() + s.slice(1) : 'Any severity'}
                        </button>
                    ))}
                </div>
            </div>

            {loading && !data ? (
                <div className="card" style={{ padding: '60px 20px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    <div style={{
                        width: '36px', height: '36px', margin: '0 auto 14px',
                        border: '3px solid var(--border-color)', borderTopColor: 'var(--danger)',
                        borderRadius: '50%', animation: 'spin 1s linear infinite',
                    }} />
                    Loading incidents…
                </div>
            ) : error ? (
                <div className="card" style={{ padding: '50px 20px', textAlign: 'center' }}>
                    <div style={{ opacity: 0.3, marginBottom: '10px' }}><AlertIcon size={44} /></div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: '16px' }}>{error}</div>
                    <button className="btn btn-primary btn-sm" onClick={load}>Retry</button>
                </div>
            ) : rows.length === 0 ? (
                <div className="card">
                    <div className="empty-state">
                        <div className="empty-state-icon"><SecurityIcon size={32} /></div>
                        <div className="empty-state-title">No incidents match</div>
                        <div className="empty-state-text">
                            {status === 'open'
                                ? 'Nothing is waiting for review.'
                                : 'Try a different status or severity filter.'}
                        </div>
                    </div>
                </div>
            ) : (
                <div className="card" style={{ padding: 0 }}>
                    <div className="table-responsive-wrapper">
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Severity</th>
                                    <th>Incident</th>
                                    <th>Camera</th>
                                    <th>When</th>
                                    <th>Status</th>
                                    <th></th>
                                </tr>
                            </thead>
                            <tbody>
                                {rows.map((r) => (
                                    <tr
                                        key={r.event_id}
                                        onClick={() => router.push(`/incidents/${r.event_id}`)}
                                        style={{ cursor: 'pointer' }}
                                    >
                                        <td>
                                            <span style={{
                                                display: 'inline-flex', alignItems: 'center', gap: '6px',
                                                color: SEVERITY_TONE[r.severity] || 'var(--text-muted)',
                                                fontWeight: 600, fontSize: '0.75rem',
                                                textTransform: 'uppercase', letterSpacing: '0.04em',
                                            }}>
                                                <span style={{
                                                    width: '7px', height: '7px', borderRadius: '50%',
                                                    background: 'currentColor',
                                                }} />
                                                {r.severity}
                                            </span>
                                        </td>
                                        <td>
                                            <div style={{ fontWeight: 600 }}>
                                                {r.seq ? `#${r.seq} · ` : ''}{r.label}
                                            </div>
                                            <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)' }}>
                                                {r.summary}
                                            </div>
                                        </td>
                                        <td>
                                            <span style={{
                                                display: 'inline-flex', alignItems: 'center', gap: '5px',
                                                fontSize: '0.8125rem',
                                            }}>
                                                <CameraIcon size={12} />
                                                {r.zone || r.camera_name}
                                            </span>
                                        </td>
                                        <td style={{
                                            fontFamily: "'JetBrains Mono', monospace",
                                            fontSize: '0.75rem', whiteSpace: 'nowrap',
                                        }}>
                                            <ClockIcon size={11} style={{ verticalAlign: '-1px', marginRight: '4px' }} />
                                            {when(r.created_at)}
                                        </td>
                                        <td>
                                            <span className={`badge ${
                                                r.status === 'resolved' ? 'active'
                                                    : r.status === 'investigating' ? 'pending' : 'offline'
                                            }`}>
                                                {r.status}
                                            </span>
                                        </td>
                                        <td style={{ color: 'var(--text-muted)' }}>
                                            <ArrowRightIcon size={14} />
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </AppShell>
    );
}
