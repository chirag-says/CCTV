'use client';

/**
 * PersonTimeline — Shows chronological activity history for a person.
 * Slide-out panel that loads data from /api/movements/person/{id}/timeline.
 */

import { useState, useEffect } from 'react';
import api, { API_BASE } from '@/lib/api';
import { EntryIcon, ExitIcon, DetectionIcon, UnknownFaceIcon, XIcon } from './Icons';

const EVENT_ICONS = {
    entry: { Icon: EntryIcon, color: '#10b981', label: 'Entry' },
    exit: { Icon: ExitIcon, color: '#ef4444', label: 'Exit' },
    detection: { Icon: DetectionIcon, color: '#6366f1', label: 'Detection' },
    unknown: { Icon: UnknownFaceIcon, color: '#f59e0b', label: 'Unknown' },
};

export default function PersonTimeline({ person, onClose }) {
    const [timeline, setTimeline] = useState([]);
    const [loading, setLoading] = useState(true);
    const [days, setDays] = useState(7);

    useEffect(() => {
        if (person?.id) loadTimeline();
    }, [person?.id, days]);

    async function loadTimeline() {
        setLoading(true);
        try {
            const data = await api.request(`/api/movements/person/${person.id}/timeline?days=${days}&limit=200`);
            setTimeline(data.timeline || []);
        } catch {
            setTimeline([]);
        } finally {
            setLoading(false);
        }
    }

    const formatTime = (dateStr) => {
        try {
            return new Date(dateStr).toLocaleString('en-US', {
                month: 'short', day: 'numeric',
                hour: '2-digit', minute: '2-digit',
                hour12: false,
            });
        } catch { return '—'; }
    };

    // Group events by date
    const grouped = {};
    timeline.forEach(event => {
        const dateKey = new Date(event.created_at || event.timestamp).toLocaleDateString('en-US', {
            weekday: 'short', month: 'short', day: 'numeric',
        });
        if (!grouped[dateKey]) grouped[dateKey] = [];
        grouped[dateKey].push(event);
    });

    return (
        <div
            style={{
                position: 'fixed',
                top: 0,
                right: 0,
                bottom: 0,
                width: '420px',
                maxWidth: '95vw',
                background: 'var(--bg-secondary, #1a1b2e)',
                borderLeft: '1px solid var(--border-color)',
                zIndex: 9000,
                display: 'flex',
                flexDirection: 'column',
                boxShadow: '-10px 0 40px rgba(0,0,0,0.3)',
                animation: 'slideInFromRight 0.25s ease',
            }}
        >
            {/* Header */}
            <div style={{
                padding: '20px',
                borderBottom: '1px solid var(--border-color)',
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
            }}>
                <div style={{
                    width: '40px', height: '40px', borderRadius: '10px',
                    background: 'var(--primary-glow)', color: 'var(--primary-light)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontWeight: 700, fontSize: '1rem',
                }}>
                    {person?.name?.charAt(0) || '?'}
                </div>
                <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 700, fontSize: '0.9375rem', color: 'var(--text-primary)' }}>
                        {person?.name || 'Unknown'}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        Activity Timeline • {timeline.length} events
                    </div>
                </div>
                <button
                    onClick={onClose}
                    style={{
                        background: 'none', border: 'none', cursor: 'pointer',
                        color: 'var(--text-muted)', padding: '4px',
                    }}
                >
                    <XIcon size={20} />
                </button>
            </div>

            {/* Period selector */}
            <div style={{
                padding: '12px 20px',
                display: 'flex', gap: '6px',
                borderBottom: '1px solid var(--border-color)',
            }}>
                {[{ d: 1, l: '24h' }, { d: 7, l: '7d' }, { d: 30, l: '30d' }, { d: 90, l: '90d' }].map(opt => (
                    <button
                        key={opt.d}
                        className={`btn btn-sm ${days === opt.d ? 'btn-primary' : 'btn-secondary'}`}
                        onClick={() => setDays(opt.d)}
                        style={{ minWidth: 'auto', padding: '4px 12px' }}
                    >
                        {opt.l}
                    </button>
                ))}
            </div>

            {/* Timeline content */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px' }}>
                {loading ? (
                    <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '40px 0', fontSize: '0.8125rem' }}>
                        Loading timeline...
                    </div>
                ) : timeline.length === 0 ? (
                    <div style={{ textAlign: 'center', padding: '40px 0' }}>
                        <div style={{ fontSize: '2rem', marginBottom: '8px', opacity: 0.3 }}>📭</div>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
                            No activity in the last {days} days
                        </div>
                    </div>
                ) : (
                    Object.entries(grouped).map(([dateKey, events]) => (
                        <div key={dateKey} style={{ marginBottom: '20px' }}>
                            {/* Date header */}
                            <div style={{
                                fontSize: '0.6875rem',
                                fontWeight: 700,
                                color: 'var(--text-muted)',
                                textTransform: 'uppercase',
                                letterSpacing: '0.08em',
                                marginBottom: '10px',
                                paddingBottom: '6px',
                                borderBottom: '1px solid var(--border-color)',
                            }}>
                                {dateKey}
                            </div>

                            {/* Events */}
                            {events.map((event, idx) => {
                                const config = EVENT_ICONS[event.event_type] || EVENT_ICONS.detection;
                                const { Icon } = config;
                                return (
                                    <div
                                        key={event.id || idx}
                                        style={{
                                            display: 'flex',
                                            gap: '12px',
                                            padding: '8px 0',
                                            borderLeft: `2px solid ${config.color}20`,
                                            paddingLeft: '14px',
                                            marginLeft: '8px',
                                        }}
                                    >
                                        <div style={{
                                            width: '28px', height: '28px', borderRadius: '50%',
                                            background: `${config.color}15`,
                                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                                            color: config.color, flexShrink: 0,
                                        }}>
                                            <Icon size={13} />
                                        </div>
                                        <div style={{ flex: 1, minWidth: 0 }}>
                                            <div style={{
                                                fontSize: '0.8125rem', fontWeight: 600,
                                                color: 'var(--text-primary)',
                                            }}>
                                                {config.label}
                                            </div>
                                            <div style={{
                                                fontSize: '0.6875rem', color: 'var(--text-muted)',
                                                display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '2px',
                                            }}>
                                                <span>{event.camera_name || event.camera_id}</span>
                                                {event.confidence > 0 && (
                                                    <span>• {(event.confidence * 100).toFixed(0)}%</span>
                                                )}
                                            </div>
                                        </div>
                                        <div style={{
                                            fontSize: '0.6875rem',
                                            fontFamily: "'JetBrains Mono', monospace",
                                            color: 'var(--text-muted)',
                                            flexShrink: 0,
                                            whiteSpace: 'nowrap',
                                        }}>
                                            {formatTime(event.created_at || event.timestamp)}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}
