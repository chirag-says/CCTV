'use client';

/**
 * Events Log Page — Detection events timeline with filtering.
 * All data fetched from backend API — no mock/dummy data.
 */

import React, { useState, useEffect } from 'react';
import AppShell from '@/components/AppShell';
import api, { API_BASE } from '@/lib/api';
import { useLightbox } from '@/components/Lightbox';
import { EntryIcon, ExitIcon, DetectionIcon, UnknownFaceIcon, EventLogIcon, SecurityIcon, CrowdIcon, LoiterIcon, HazardIcon, VehicleIcon, PlateIcon } from '@/components/Icons';

export default function EventsPage() {
    const [events, setEvents] = useState([]);
    const [total, setTotal] = useState(0);
    const [typeFilter, setTypeFilter] = useState('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [expandedId, setExpandedId] = useState(null);
    const { openLightbox, LightboxComponent } = useLightbox();

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
                    {['', 'entry', 'exit', 'detection', 'unknown', 'security_alert', 'vehicle_entry'].map((type) => (
                        <button
                            key={type || 'all'}
                            className={`btn btn-sm ${typeFilter === type ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={() => setTypeFilter(type)}
                        >
                            {type ? (
                                <span className={`badge ${type === 'security_alert' ? 'security-alert' : type === 'vehicle_entry' ? 'vehicle-entry' : type}`} style={{ padding: '2px 8px' }}>
                                    {type === 'security_alert' ? 'Security Alert' : type === 'vehicle_entry' ? 'Vehicle Entry' : type.charAt(0).toUpperCase() + type.slice(1)}
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
                                    <th style={{ width: '50px' }}>Snap</th>
                                </tr>
                            </thead>
                            <tbody>
                                {events.map((event) => (
                                    <React.Fragment key={event.id}>
                                    <tr
                                        style={{ cursor: 'pointer' }}
                                        onClick={() => setExpandedId(expandedId === event.id ? null : event.id)}
                                    >
                                        <td>
                                            <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.813rem' }}>
                                                {formatDate(event.created_at)}
                                            </span>
                                        </td>
                                        <td>
                                            <span className={`badge ${event.event_type === 'security_alert' ? 'security-alert' : event.event_type}`} style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                                                {event.event_type === 'entry' && <EntryIcon size={12} />}
                                                {event.event_type === 'exit' && <ExitIcon size={12} />}
                                                {event.event_type === 'unknown' && <UnknownFaceIcon size={12} />}
                                                {event.event_type === 'detection' && <DetectionIcon size={12} />}
                                                {event.event_type === 'security_alert' && <SecurityIcon size={12} />}
                                                {event.event_type === 'security_alert'
                                                    ? (event.metadata?.subtype || 'alert').toUpperCase()
                                                    : event.event_type}
                                            </span>
                                        </td>
                                        <td>
                                            {event.event_type === 'security_alert' ? (
                                                <span style={{ color: 'var(--text-secondary)', fontSize: '0.813rem' }}>
                                                    {event.metadata?.person_name || event.metadata?.threat_class || event.metadata?.person_count ? `${event.metadata.person_count} people` : '—'}
                                                </span>
                                            ) : event.person_name ? (
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
                                        <td>
                                            {event.snapshot_path ? (
                                                <img
                                                    src={`${API_BASE}${event.snapshot_path}`}
                                                    alt="snap"
                                                    style={{
                                                        width: 36, height: 36,
                                                        borderRadius: '6px',
                                                        objectFit: 'cover',
                                                        cursor: 'pointer',
                                                        border: '1px solid var(--border-color)',
                                                        transition: 'transform 0.15s',
                                                    }}
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        openLightbox(`${API_BASE}${event.snapshot_path}`, `${event.event_type} - ${event.person_name || 'Unknown'}`);
                                                    }}
                                                    onMouseEnter={(e) => e.target.style.transform = 'scale(1.15)'}
                                                    onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
                                                />
                                            ) : (
                                                <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>—</span>
                                            )}
                                        </td>
                                    </tr>
                                    {/* Expandable detail row */}
                                    {expandedId === event.id && (
                                        <tr>
                                            <td colSpan={6} style={{ padding: '16px 20px', background: 'var(--bg-tertiary)' }}>
                                                <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
                                                    {event.snapshot_path && (
                                                        <img
                                                            src={`${API_BASE}${event.snapshot_path}`}
                                                            alt="Event snapshot"
                                                            style={{
                                                                width: 160, height: 120,
                                                                borderRadius: '10px',
                                                                objectFit: 'cover',
                                                                cursor: 'pointer',
                                                                border: '1px solid var(--border-color)',
                                                            }}
                                                            onClick={() => openLightbox(`${API_BASE}${event.snapshot_path}`, event.person_name || 'Event')}
                                                        />
                                                    )}
                                                    <div style={{ flex: 1, minWidth: '200px' }}>
                                                        <div style={{ fontSize: '0.8125rem', fontWeight: 600, marginBottom: '8px', color: 'var(--text-primary)' }}>
                                                            Event Details
                                                        </div>
                                                        <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '4px 16px', fontSize: '0.75rem' }}>
                                                            <span style={{ color: 'var(--text-muted)' }}>ID</span>
                                                            <span style={{ fontFamily: "'JetBrains Mono', monospace", color: 'var(--text-secondary)' }}>{event.id?.slice(0, 12)}...</span>
                                                            <span style={{ color: 'var(--text-muted)' }}>Type</span>
                                                            <span>{event.event_type}</span>
                                                            <span style={{ color: 'var(--text-muted)' }}>Person</span>
                                                            <span>{event.person_name || 'Unknown'}</span>
                                                            <span style={{ color: 'var(--text-muted)' }}>Camera</span>
                                                            <span>{event.camera_name || event.camera_id}</span>
                                                            <span style={{ color: 'var(--text-muted)' }}>Time</span>
                                                            <span>{formatDate(event.created_at)}</span>
                                                            {event.confidence > 0 && (<>
                                                                <span style={{ color: 'var(--text-muted)' }}>Confidence</span>
                                                                <span>{(event.confidence * 100).toFixed(1)}%</span>
                                                            </>)}
                                                        </div>
                                                    </div>
                                                </div>
                                            </td>
                                        </tr>
                                    )}
                                    </React.Fragment>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
            {LightboxComponent}
        </AppShell>
    );
}
