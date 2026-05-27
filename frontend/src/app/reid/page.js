'use client';

/**
 * Person Re-Identification Page — Cross-camera person tracking.
 * Shows persons tracked across multiple cameras with their movement trail.
 */

import { useState, useEffect, useCallback } from 'react';
import AppShell from '@/components/AppShell';
import api from '@/lib/api';
import { UsersIcon, CameraIcon, SignalIcon } from '@/components/Icons';

function ReIDIcon({ size = 18 }) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
            <circle cx="8.5" cy="7" r="4" />
            <path d="M20 8v6" />
            <path d="M23 11h-6" />
        </svg>
    );
}

function CrossCameraIcon({ size = 18 }) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4" />
            <polyline points="10 17 15 12 10 7" />
            <line x1="15" y1="12" x2="3" y2="12" />
        </svg>
    );
}

function ClockIcon({ size = 14 }) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
        </svg>
    );
}

export default function ReIDPage() {
    const [persons, setPersons] = useState([]);
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [filter, setFilter] = useState('all'); // 'all' or 'cross-camera'
    const [expandedPerson, setExpandedPerson] = useState(null);
    const [sinceMinutes, setSinceMinutes] = useState(30);

    const loadData = useCallback(async () => {
        try {
            const result = await api.request(
                `/api/reid/persons?cross_camera_only=${filter === 'cross-camera'}&since_minutes=${sinceMinutes}`
            );
            setPersons(result.persons || []);
            setStats(result.stats || null);
        } catch (e) {
            console.error('ReID load error:', e);
            setPersons([]);
        } finally {
            setLoading(false);
        }
    }, [filter, sinceMinutes]);

    useEffect(() => {
        loadData();
        const interval = setInterval(loadData, 10000); // Refresh every 10s
        return () => clearInterval(interval);
    }, [loadData]);

    const formatTime = (isoStr) => {
        if (!isoStr) return '—';
        try {
            return new Date(isoStr).toLocaleString('en-US', {
                month: 'short', day: 'numeric',
                hour: '2-digit', minute: '2-digit', second: '2-digit',
                hour12: false,
            });
        } catch { return '—'; }
    };

    const timeSince = (isoStr) => {
        if (!isoStr) return '';
        const diff = (Date.now() - new Date(isoStr).getTime()) / 1000;
        if (diff < 60) return `${Math.floor(diff)}s ago`;
        if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
        if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
        return `${Math.floor(diff / 86400)}d ago`;
    };

    const cameraColors = {};
    const colorPalette = [
        '#10b981', '#6366f1', '#f59e0b', '#ef4444', '#8b5cf6',
        '#ec4899', '#14b8a6', '#f97316', '#06b6d4', '#84cc16',
    ];
    let colorIdx = 0;
    const getCameraColor = (camId) => {
        if (!cameraColors[camId]) {
            cameraColors[camId] = colorPalette[colorIdx % colorPalette.length];
            colorIdx++;
        }
        return cameraColors[camId];
    };

    return (
        <AppShell>
            <div className="page-header">
                <h1>Person Re-Identification</h1>
                <p>Track persons across multiple cameras using body appearance</p>
            </div>

            {/* Stats Bar */}
            <div className="card" style={{
                marginBottom: '20px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: '12px',
            }}>
                <div style={{ display: 'flex', gap: '32px' }}>
                    <div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                            Gallery Size
                        </div>
                        <div style={{ fontSize: '1.5rem', fontWeight: 800 }}>
                            {stats?.gallery_size ?? '—'}
                        </div>
                    </div>
                    <div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                            Cross-Camera
                        </div>
                        <div style={{ fontSize: '1.5rem', fontWeight: 800, color: stats?.cross_camera_persons > 0 ? 'var(--accent)' : 'var(--text-primary)' }}>
                            {stats?.cross_camera_persons ?? '—'}
                        </div>
                    </div>
                    <div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                            Total Sightings
                        </div>
                        <div style={{ fontSize: '1.5rem', fontWeight: 800 }}>
                            {stats?.total_sightings ?? '—'}
                        </div>
                    </div>
                </div>

                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <select
                        className="input"
                        value={sinceMinutes}
                        onChange={(e) => setSinceMinutes(Number(e.target.value))}
                        style={{ width: 'auto', minWidth: '100px', padding: '6px 10px', fontSize: '0.813rem' }}
                        id="reid-time-filter"
                    >
                        <option value={5}>Last 5 min</option>
                        <option value={15}>Last 15 min</option>
                        <option value={30}>Last 30 min</option>
                        <option value={60}>Last 1 hour</option>
                        <option value={360}>Last 6 hours</option>
                        <option value={1440}>Last 24 hours</option>
                    </select>

                    <div style={{
                        display: 'flex',
                        borderRadius: 'var(--radius-md)',
                        overflow: 'hidden',
                        border: '1px solid var(--border-color)',
                    }}>
                        <button
                            className={`btn btn-sm ${filter === 'all' ? 'btn-primary' : ''}`}
                            onClick={() => setFilter('all')}
                            style={{
                                borderRadius: 0,
                                border: 'none',
                                fontSize: '0.75rem',
                                padding: '6px 12px',
                                background: filter === 'all' ? '' : 'var(--bg-tertiary)',
                                color: filter === 'all' ? '' : 'var(--text-secondary)',
                            }}
                            id="reid-filter-all"
                        >
                            All
                        </button>
                        <button
                            className={`btn btn-sm ${filter === 'cross-camera' ? 'btn-primary' : ''}`}
                            onClick={() => setFilter('cross-camera')}
                            style={{
                                borderRadius: 0,
                                border: 'none',
                                borderLeft: '1px solid var(--border-color)',
                                fontSize: '0.75rem',
                                padding: '6px 12px',
                                background: filter === 'cross-camera' ? '' : 'var(--bg-tertiary)',
                                color: filter === 'cross-camera' ? '' : 'var(--text-secondary)',
                            }}
                            id="reid-filter-cross"
                        >
                            <CrossCameraIcon size={12} /> Cross-Camera
                        </button>
                    </div>
                </div>
            </div>

            {/* Person List */}
            {loading ? (
                <div className="card" style={{ padding: '60px 20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                    Loading Re-ID data...
                </div>
            ) : persons.length === 0 ? (
                <div className="card">
                    <div className="empty-state">
                        <div className="empty-state-icon"><ReIDIcon size={32} /></div>
                        <div className="empty-state-title">No Persons Tracked Yet</div>
                        <div className="empty-state-text">
                            {filter === 'cross-camera'
                                ? 'No persons have been seen on multiple cameras yet. Try the "All" filter.'
                                : 'Person Re-Identification will start tracking when cameras detect people. Make sure cameras are running.'
                            }
                        </div>
                    </div>
                </div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    {persons.map((person) => {
                        const isExpanded = expandedPerson === person.reid_id;
                        const isCrossCamera = person.camera_count >= 2;

                        return (
                            <div
                                key={person.reid_id}
                                className="card"
                                style={{
                                    cursor: 'pointer',
                                    transition: 'all 0.2s ease',
                                    borderLeft: isCrossCamera ? '4px solid var(--accent)' : '4px solid transparent',
                                }}
                                onClick={() => setExpandedPerson(isExpanded ? null : person.reid_id)}
                                id={`reid-person-${person.reid_id.slice(0, 8)}`}
                            >
                                {/* Person Header */}
                                <div style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center',
                                    marginBottom: isExpanded ? '16px' : 0,
                                }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                        <div style={{
                                            width: 40,
                                            height: 40,
                                            borderRadius: '50%',
                                            background: isCrossCamera
                                                ? 'linear-gradient(135deg, var(--accent), #6366f1)'
                                                : 'var(--bg-tertiary)',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            color: isCrossCamera ? '#fff' : 'var(--text-muted)',
                                            fontWeight: 700,
                                            fontSize: '0.875rem',
                                        }}>
                                            {isCrossCamera ? <CrossCameraIcon size={18} /> : <UsersIcon size={18} />}
                                        </div>
                                        <div>
                                            <div style={{ fontWeight: 600, fontSize: '0.875rem' }}>
                                                {person.person_name}
                                                {person.face_person_id && (
                                                    <span style={{
                                                        marginLeft: '8px',
                                                        fontSize: '0.625rem',
                                                        padding: '2px 6px',
                                                        borderRadius: '4px',
                                                        background: 'rgba(16, 185, 129, 0.15)',
                                                        color: 'var(--success)',
                                                    }}>
                                                        Face Matched
                                                    </span>
                                                )}
                                            </div>
                                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', gap: '12px', marginTop: '2px' }}>
                                                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                                    <CameraIcon size={12} /> {person.camera_count} camera{person.camera_count !== 1 ? 's' : ''}
                                                </span>
                                                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                                    <SignalIcon size={12} /> {person.sighting_count} sightings
                                                </span>
                                            </div>
                                        </div>
                                    </div>

                                    <div style={{ textAlign: 'right' }}>
                                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                            {timeSince(person.last_seen)}
                                        </div>
                                        {/* Camera pills */}
                                        <div style={{ display: 'flex', gap: '4px', marginTop: '4px', justifyContent: 'flex-end' }}>
                                            {person.cameras_seen.map((cam) => (
                                                <span
                                                    key={cam}
                                                    style={{
                                                        fontSize: '0.625rem',
                                                        padding: '1px 6px',
                                                        borderRadius: '4px',
                                                        background: getCameraColor(cam) + '22',
                                                        color: getCameraColor(cam),
                                                        fontWeight: 600,
                                                    }}
                                                >
                                                    {cam.length > 12 ? cam.slice(0, 12) + '…' : cam}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                </div>

                                {/* Expanded Trail */}
                                {isExpanded && person.trail && person.trail.length > 0 && (
                                    <div style={{
                                        borderTop: '1px solid var(--border-color)',
                                        paddingTop: '16px',
                                    }}>
                                        <div style={{ fontSize: '0.75rem', fontWeight: 600, marginBottom: '12px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                            Movement Trail
                                        </div>

                                        <div style={{ position: 'relative', paddingLeft: '24px' }}>
                                            {/* Timeline line */}
                                            <div style={{
                                                position: 'absolute',
                                                left: '7px',
                                                top: '8px',
                                                bottom: '8px',
                                                width: '2px',
                                                background: 'var(--border-color)',
                                                borderRadius: '1px',
                                            }} />

                                            {person.trail.map((sighting, idx) => {
                                                const camColor = getCameraColor(sighting.camera_id);
                                                const isFirst = idx === 0;
                                                const isLast = idx === person.trail.length - 1;

                                                return (
                                                    <div
                                                        key={idx}
                                                        style={{
                                                            display: 'flex',
                                                            alignItems: 'center',
                                                            gap: '12px',
                                                            marginBottom: idx < person.trail.length - 1 ? '12px' : 0,
                                                            position: 'relative',
                                                        }}
                                                    >
                                                        {/* Timeline dot */}
                                                        <div style={{
                                                            position: 'absolute',
                                                            left: '-20px',
                                                            width: '12px',
                                                            height: '12px',
                                                            borderRadius: '50%',
                                                            background: isLast ? camColor : 'var(--bg-secondary)',
                                                            border: `2px solid ${camColor}`,
                                                            boxShadow: isLast ? `0 0 8px ${camColor}44` : 'none',
                                                        }} />

                                                        <div style={{
                                                            display: 'flex',
                                                            alignItems: 'center',
                                                            justifyContent: 'space-between',
                                                            width: '100%',
                                                            padding: '6px 10px',
                                                            borderRadius: 'var(--radius-sm)',
                                                            background: isLast ? camColor + '11' : 'var(--bg-tertiary)',
                                                            border: `1px solid ${isLast ? camColor + '33' : 'transparent'}`,
                                                        }}>
                                                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                                <CameraIcon size={12} />
                                                                <span style={{
                                                                    fontSize: '0.75rem',
                                                                    fontWeight: 600,
                                                                    color: camColor,
                                                                }}>
                                                                    {sighting.camera_id.length > 20
                                                                        ? sighting.camera_id.slice(0, 20) + '…'
                                                                        : sighting.camera_id
                                                                    }
                                                                </span>
                                                                {isFirst && (
                                                                    <span style={{
                                                                        fontSize: '0.625rem',
                                                                        padding: '1px 5px',
                                                                        borderRadius: '3px',
                                                                        background: 'rgba(16, 185, 129, 0.15)',
                                                                        color: 'var(--success)',
                                                                    }}>FIRST</span>
                                                                )}
                                                                {isLast && !isFirst && (
                                                                    <span style={{
                                                                        fontSize: '0.625rem',
                                                                        padding: '1px 5px',
                                                                        borderRadius: '3px',
                                                                        background: 'rgba(99, 102, 241, 0.15)',
                                                                        color: '#6366f1',
                                                                    }}>LATEST</span>
                                                                )}
                                                            </div>
                                                            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.6875rem', color: 'var(--text-muted)' }}>
                                                                <ClockIcon size={11} />
                                                                {formatTime(sighting.timestamp)}
                                                            </div>
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                        </div>

                                        {/* Summary */}
                                        <div style={{
                                            marginTop: '12px',
                                            padding: '8px 12px',
                                            borderRadius: 'var(--radius-sm)',
                                            background: 'var(--bg-tertiary)',
                                            fontSize: '0.6875rem',
                                            color: 'var(--text-muted)',
                                            display: 'flex',
                                            gap: '16px',
                                        }}>
                                            <span>ID: {person.reid_id.slice(0, 12)}…</span>
                                            <span>First: {formatTime(person.first_seen)}</span>
                                            <span>Last: {formatTime(person.last_seen)}</span>
                                        </div>
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>
            )}
        </AppShell>
    );
}
