'use client';

/**
 * Teachers Currently on Campus — near-real-time faculty presence.
 *
 * Two update paths, matching how the dashboard and security-alerts pages
 * already work:
 *   1. a 30s poll, which is the source of truth
 *   2. the existing live-events WebSocket, which triggers a debounced
 *      refetch when a known person is detected, so a teacher walking past a
 *      camera appears within seconds instead of up to 30
 *
 * The headline is deliberately split (SPEC §0.2): "present now" counts only
 * people seen inside the confidence window, and anyone detected longer ago
 * is reported separately. A single merged number would either overstate
 * presence or hide people who are genuinely here.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import AppShell from '@/components/AppShell';
import FacultyPresenceCard from '@/components/FacultyPresenceCard';
import api from '@/lib/api';
import {
    UsersIcon, AlertIcon, RefreshIcon, SearchIcon, CampusIcon,
} from '@/components/Icons';

const POLL_MS = 30000;
const WS_DEBOUNCE_MS = 1500;

export default function OnCampusPage() {
    const [data, setData] = useState(null);
    const [department, setDepartment] = useState('');
    const [search, setSearch] = useState('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [wsLive, setWsLive] = useState(false);

    const wsRef = useRef(null);
    const debounceRef = useRef(null);
    // Read by the WS handler, which is created once and must not close over
    // a stale `department`.
    const deptRef = useRef(department);
    deptRef.current = department;

    const load = useCallback(async (dept) => {
        try {
            const result = await api.getOnCampus({ department: dept || undefined });
            setData(result);
            setError(null);
        } catch (e) {
            setError(e.message || 'Could not load presence. Make sure the backend is running.');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        setLoading(true);
        load(department);
        const poll = setInterval(() => load(department), POLL_MS);
        return () => clearInterval(poll);
    }, [department, load]);

    useEffect(() => {
        // Any entry/exit for an identified person can change the board, so
        // refetch rather than trying to patch a card from the event payload —
        // the state machine lives on the server and must stay the authority.
        try {
            wsRef.current = api.connectLiveEvents((event) => {
                setWsLive(true);
                const relevant =
                    (event.event_type === 'entry' || event.event_type === 'exit')
                    && event.person_id
                    && !String(event.person_id).startsWith('anon:');
                if (!relevant) return;
                if (debounceRef.current) clearTimeout(debounceRef.current);
                debounceRef.current = setTimeout(() => load(deptRef.current), WS_DEBOUNCE_MS);
            });
        } catch (e) {
            console.warn('WebSocket not available:', e.message);
        }
        return () => {
            if (debounceRef.current) clearTimeout(debounceRef.current);
            if (wsRef.current) wsRef.current.close();
        };
    }, [load]);

    const faculty = data?.faculty || [];
    const departments = [...new Set(faculty.map((f) => f.department))].sort();

    const needle = search.trim().toLowerCase();
    const visible = needle
        ? faculty.filter((f) => f.name.toLowerCase().includes(needle))
        : faculty;

    const present = visible.filter((f) => f.state === 'LIVE' || f.state === 'ON_CAMPUS');
    const earlier = visible.filter((f) => f.state === 'SEEN_EARLIER');
    const departed = visible.filter((f) => f.state === 'DEPARTED');

    return (
        <AppShell>
            <div className="page-header">
                <div className="toolbar-responsive" style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px',
                }}>
                    <div>
                        <h1>Currently on Campus</h1>
                        <p>
                            Faculty presence from camera detections
                            <span style={{
                                display: 'inline-flex', alignItems: 'center', gap: '6px', marginLeft: '12px',
                            }}>
                                <span style={{
                                    width: '8px', height: '8px', borderRadius: '50%',
                                    background: wsLive ? 'var(--success)' : 'var(--text-muted)',
                                    animation: wsLive ? 'pulse-dot 2s infinite' : 'none',
                                }} />
                                <span style={{
                                    fontSize: '0.75rem', fontWeight: 600,
                                    color: wsLive ? 'var(--success)' : 'var(--text-muted)',
                                }}>
                                    {wsLive ? 'LIVE' : 'POLLING'}
                                </span>
                            </span>
                        </p>
                    </div>
                    <button
                        className="btn btn-secondary btn-sm"
                        onClick={() => load(department)}
                        id="on-campus-refresh"
                    >
                        <RefreshIcon size={14} /> Refresh
                    </button>
                </div>
            </div>

            {loading && !data ? (
                <div className="card" style={{ padding: '80px 20px', textAlign: 'center' }}>
                    <div style={{
                        width: '40px', height: '40px', margin: '0 auto 16px',
                        border: '3px solid var(--border-color)', borderTopColor: 'var(--primary)',
                        borderRadius: '50%', animation: 'spin 1s linear infinite',
                    }} />
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                        Loading presence…
                    </div>
                </div>
            ) : error ? (
                <div className="card" style={{ padding: '60px 20px', textAlign: 'center' }}>
                    <div style={{ opacity: 0.3, marginBottom: '12px' }}><AlertIcon size={48} /></div>
                    <div style={{ fontWeight: 600, fontSize: '1.125rem', marginBottom: '8px' }}>
                        Presence Unavailable
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: '20px' }}>
                        {error}
                    </div>
                    <button className="btn btn-primary" onClick={() => load(department)}>Retry</button>
                </div>
            ) : (
                <>
                    {/* Split headline — neither number has to be defended */}
                    <div className="card" style={{ marginBottom: '20px' }}>
                        <div style={{ display: 'flex', alignItems: 'baseline', gap: '14px', flexWrap: 'wrap' }}>
                            <span style={{
                                fontSize: '2.5rem', fontWeight: 800, lineHeight: 1,
                                color: present.length > 0 ? 'var(--success)' : 'var(--text-muted)',
                                fontFamily: "'JetBrains Mono', monospace",
                            }}>
                                {present.length}
                            </span>
                            <div>
                                <div style={{ fontWeight: 600, fontSize: '0.9375rem' }}>
                                    present now
                                </div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                    seen within the last {data?.thresholds?.confident_minutes ?? 20} minutes
                                    {earlier.length > 0 && (
                                        <> · <strong style={{ color: 'var(--text-secondary)' }}>
                                            +{earlier.length} seen earlier today
                                        </strong></>
                                    )}
                                </div>
                            </div>
                            <div style={{
                                marginLeft: 'auto', fontSize: '0.75rem',
                                color: 'var(--text-muted)', textAlign: 'right',
                            }}>
                                {data?.roster_total ?? 0} faculty on roster
                                {!data?.live_tracking_active && (
                                    <div style={{ color: 'var(--warning-light)', marginTop: '2px' }}>
                                        No cameras running — showing recorded detections only
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Filters */}
                    <div className="card" style={{ marginBottom: '20px' }}>
                        <div className="toolbar-responsive" style={{
                            display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap',
                        }}>
                            <div style={{ position: 'relative', flex: '1 1 260px', maxWidth: '320px' }}>
                                <span style={{
                                    position: 'absolute', left: '10px', top: '50%',
                                    transform: 'translateY(-50%)', color: 'var(--text-muted)',
                                    display: 'flex', pointerEvents: 'none',
                                }}>
                                    <SearchIcon size={15} />
                                </span>
                                <input
                                    type="text"
                                    className="input"
                                    placeholder="Search faculty…"
                                    value={search}
                                    onChange={(e) => setSearch(e.target.value)}
                                    style={{ marginBottom: 0, paddingLeft: '32px' }}
                                    id="on-campus-search"
                                />
                            </div>
                            <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                                <button
                                    className={`btn btn-sm ${!department ? 'btn-primary' : 'btn-secondary'}`}
                                    onClick={() => setDepartment('')}
                                >
                                    All
                                </button>
                                {departments.map((d) => (
                                    <button
                                        key={d}
                                        className={`btn btn-sm ${department === d ? 'btn-primary' : 'btn-secondary'}`}
                                        onClick={() => setDepartment(d)}
                                    >
                                        {d}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>

                    {visible.length === 0 ? (
                        <div className="card">
                            <div className="empty-state">
                                <div className="empty-state-icon"><CampusIcon size={32} /></div>
                                <div className="empty-state-title">
                                    {search || department ? 'No matching faculty' : 'Nobody detected yet today'}
                                </div>
                                <div className="empty-state-text">
                                    {search || department
                                        ? 'This board only lists faculty detected today. Someone not detected yet will not appear here.'
                                        : 'Faculty appear here as soon as a camera recognises them. Start a camera from the Cameras page to begin.'}
                                </div>
                                {search && (
                                    <button
                                        className="btn btn-primary"
                                        style={{ marginTop: '16px' }}
                                        onClick={() => window.dispatchEvent(
                                            new CustomEvent('sentinel:open-search')
                                        )}
                                        id="on-campus-search-all"
                                    >
                                        <SearchIcon size={15} /> Search all faculty
                                    </button>
                                )}
                            </div>
                        </div>
                    ) : (
                        <>
                            <PresenceGroup items={present} />
                            {earlier.length > 0 && (
                                <PresenceGroup
                                    items={earlier}
                                    title="Seen earlier today"
                                    caption={`Detected more than ${data?.thresholds?.confident_minutes ?? 20} minutes ago — likely still on campus, but not recently confirmed`}
                                />
                            )}
                            {departed.length > 0 && (
                                <PresenceGroup
                                    items={departed}
                                    title="Departed"
                                    caption="Last detected leaving through a gate camera, or after the campus day ended"
                                />
                            )}
                        </>
                    )}

                    <div style={{
                        marginTop: '20px', fontSize: '0.75rem',
                        color: 'var(--text-muted)', lineHeight: 1.6,
                    }}>
                        Presence is inferred from camera detections. Someone out of camera view
                        is not detected, so an absence here is not proof they have left.
                    </div>
                </>
            )}
        </AppShell>
    );
}

function PresenceGroup({ items, title, caption }) {
    if (!items.length) return null;
    return (
        <div style={{ marginBottom: '24px' }}>
            {title && (
                <div style={{ marginBottom: '10px' }}>
                    <h3 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        {title}
                        <span style={{
                            fontSize: '0.6875rem', fontWeight: 600, color: 'var(--text-muted)',
                            background: 'var(--border-color)', padding: '1px 7px', borderRadius: '10px',
                        }}>
                            {items.length}
                        </span>
                    </h3>
                    {caption && (
                        <div className="card-subtitle" style={{ marginTop: '2px' }}>{caption}</div>
                    )}
                </div>
            )}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
                gap: '12px',
            }}>
                {items.map((f) => (
                    <FacultyPresenceCard key={f.person_id} person={f} />
                ))}
            </div>
        </div>
    );
}
