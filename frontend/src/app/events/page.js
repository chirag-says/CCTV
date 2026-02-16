'use client';

/**
 * Events Log Page — Detection events timeline with filtering.
 * All data fetched from backend API — no mock/dummy data.
 */

import { useState, useEffect } from 'react';
import AppShell from '@/components/AppShell';
import api from '@/lib/api';
import { EntryIcon, ExitIcon, DetectionIcon, UnknownFaceIcon, EventLogIcon } from '@/components/Icons';

export default function EventsPage() {
    const [events, setEvents] = useState([]);
    const [total, setTotal] = useState(0);
    const [typeFilter, setTypeFilter] = useState('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        loadEvents();
    }, [typeFilter]);

    async function loadEvents() {
        setLoading(true);
        setError(null);
        try {
            const params = {};
            if (typeFilter) params.event_type = typeFilter;
            const result = await api.getEvents(params);
            setEvents(result.data || []);
            setTotal(result.total || 0);
        } catch (e) {
            setError('Could not load events. Make sure the backend is running.');
            setEvents([]);
            setTotal(0);
        } finally {
            setLoading(false);
        }
    }

    const formatDate = (dateStr) => {
        try {
            return new Date(dateStr).toLocaleString('en-US', {
                month: 'short', day: 'numeric',
                hour: '2-digit', minute: '2-digit', second: '2-digit',
                hour12: false,
            });
        } catch { return '—'; }
    };

    return (
        <AppShell>
            <div className="page-header">
                <h1>Events Log</h1>
                <p>Complete detection event timeline</p>
            </div>

            {/* Filters */}
            <div className="card" style={{ marginBottom: '20px' }}>
                <div className="toolbar-responsive" style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                    {['', 'entry', 'exit', 'detection', 'unknown'].map((type) => (
                        <button
                            key={type || 'all'}
                            className={`btn btn-sm ${typeFilter === type ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={() => setTypeFilter(type)}
                        >
                            {type ? (
                                <span className={`badge ${type}`} style={{ padding: '2px 8px' }}>
                                    {type.charAt(0).toUpperCase() + type.slice(1)}
                                </span>
                            ) : 'All Events'}
                        </button>
                    ))}
                    <span className="ml-auto" style={{
                        fontSize: '0.813rem',
                        color: 'var(--text-muted)',
                        display: 'flex',
                        alignItems: 'center',
                    }}>
                        {total} events
                    </span>
                </div>
            </div>

            {/* Events Table */}
            <div className="card">
                {loading ? (
                    <div style={{ padding: '60px 20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                        Loading events...
                    </div>
                ) : error ? (
                    <div style={{ padding: '60px 20px', textAlign: 'center' }}>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: '16px' }}>{error}</div>
                        <button className="btn btn-primary btn-sm" onClick={loadEvents}>Retry</button>
                    </div>
                ) : events.length === 0 ? (
                    <div className="empty-state">
                        <div className="empty-state-icon"><EventLogIcon size={32} /></div>
                        <div className="empty-state-title">No events recorded</div>
                        <div className="empty-state-text">
                            {typeFilter
                                ? `No "${typeFilter}" events found. Try a different filter.`
                                : 'Events will appear here once the system starts detecting activity.'}
                        </div>
                    </div>
                ) : (
                    <div className="table-responsive-wrapper">
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Type</th>
                                    <th>Person</th>
                                    <th>Camera</th>
                                    <th>Confidence</th>
                                </tr>
                            </thead>
                            <tbody>
                                {events.map((event) => (
                                    <tr key={event.id}>
                                        <td>
                                            <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.813rem' }}>
                                                {formatDate(event.created_at)}
                                            </span>
                                        </td>
                                        <td>
                                            <span className={`badge ${event.event_type}`} style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                                                {event.event_type === 'entry' && <EntryIcon size={12} />}
                                                {event.event_type === 'exit' && <ExitIcon size={12} />}
                                                {event.event_type === 'unknown' && <UnknownFaceIcon size={12} />}
                                                {event.event_type === 'detection' && <DetectionIcon size={12} />}
                                                {event.event_type}
                                            </span>
                                        </td>
                                        <td>
                                            {event.person_name ? (
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                    <div className="avatar sm">{event.person_name.charAt(0)}</div>
                                                    <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{event.person_name}</span>
                                                </div>
                                            ) : (
                                                <span style={{ color: 'var(--warning)', fontStyle: 'italic' }}>Unknown Person</span>
                                            )}
                                        </td>
                                        <td>{event.camera_name || event.camera_id}</td>
                                        <td>
                                            {event.confidence > 0 ? (
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                    <div style={{
                                                        width: '60px', height: '6px',
                                                        background: 'var(--bg-tertiary)',
                                                        borderRadius: '3px',
                                                        overflow: 'hidden',
                                                    }}>
                                                        <div style={{
                                                            width: `${event.confidence * 100}%`,
                                                            height: '100%',
                                                            background: event.confidence > 0.8
                                                                ? 'var(--success)'
                                                                : event.confidence > 0.6
                                                                    ? 'var(--warning)'
                                                                    : 'var(--danger)',
                                                            borderRadius: '3px',
                                                        }}></div>
                                                    </div>
                                                    <span style={{
                                                        fontFamily: "'JetBrains Mono', monospace",
                                                        fontSize: '0.75rem',
                                                        color: 'var(--text-secondary)',
                                                    }}>
                                                        {(event.confidence * 100).toFixed(0)}%
                                                    </span>
                                                </div>
                                            ) : (
                                                <span style={{ color: 'var(--text-muted)' }}>—</span>
                                            )}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </AppShell>
    );
}
