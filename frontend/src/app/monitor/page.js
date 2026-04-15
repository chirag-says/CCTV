'use client';

/**
 * Live Monitor Page — Real-time camera feeds with detection overlays.
 * Shows detected person names and active persons panel.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import AppShell from '@/components/AppShell';
import api, { API_BASE } from '@/lib/api';
import {
    GridIcon,
    SquareIcon,
    VideoCameraIcon,
    VideoCameraOffIcon,
    UsersIcon,
    SignalIcon,
    EntryIcon,
    ExitIcon,
    DetectionIcon,
    UnknownFaceIcon,
    LiveIcon,
} from '@/components/Icons';



export default function MonitorPage() {
    const [cameras, setCameras] = useState([]);
    const [selectedCamera, setSelectedCamera] = useState(null);
    const [layout, setLayout] = useState(() => {
        if (typeof window !== 'undefined') {
            return localStorage.getItem('sentinel-monitor-layout') || 'single';
        }
        return 'single';
    });
    /**
     * presenceLog: Array of presence entries, each representing a person's
     * visit to the camera view. One entry per person per visit.
     * 
     * Shape: {
     *   id: string,           // unique key (person_id + entry timestamp)
     *   personId: string,
     *   name: string,
     *   status: 'present' | 'left',
     *   confidence: number,
     *   camera: string,
     *   entryTime: number,    // Date.now() when they entered
     *   exitTime: number|null,// Date.now() when they left
     *   duration: number|null,// seconds (filled on exit)
     *   entryTimeStr: string, // formatted time string
     * }
     */
    const [presenceLog, setPresenceLog] = useState([]);
    const [activePersons, setActivePersons] = useState([]);
    const [loading, setLoading] = useState(true);
    const wsRef = useRef(null);
    const pollRef = useRef(null);

    useEffect(() => {
        loadCameras();
        connectWebSocket();
        startActivePersonsPoll();

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
            if (pollRef.current) {
                clearInterval(pollRef.current);
                pollRef.current = null;
            }
        };
    }, []);

    async function loadCameras() {
        setLoading(true);
        try {
            const result = await api.getCameras();
            if (result && result.length > 0) {
                setCameras(result);
                setSelectedCamera(result[0]);
            }
        } catch (err) {
            console.warn('Could not load cameras from API:', err.message);
            setCameras([]);
            setSelectedCamera(null);
        }
        setLoading(false);
    }

    function formatTime(date) {
        return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false,
        });
    }

    function connectWebSocket() {
        try {
            const ws = api.connectLiveEvents((event) => {
                const now = new Date();
                const eventType = event.event_type;

                // Only process meaningful state-change events
                if (eventType === 'entry') {
                    const personId = event.person_id || '';
                    const entry = {
                        id: `${personId}-${now.getTime()}`,
                        personId,
                        name: event.person_name || 'Unknown',
                        status: 'present',
                        confidence: event.confidence || 0,
                        camera: event.camera_id || '',
                        entryTime: now.getTime(),
                        exitTime: null,
                        duration: null,
                        entryTimeStr: formatTime(now),
                    };

                    setPresenceLog((prev) => {
                        // If this person already has a "present" entry, don't duplicate
                        const existingIdx = prev.findIndex(
                            (p) => p.personId === personId && p.status === 'present'
                        );
                        if (existingIdx !== -1) {
                            // Update confidence/camera of existing entry
                            const updated = [...prev];
                            updated[existingIdx] = {
                                ...updated[existingIdx],
                                confidence: event.confidence || updated[existingIdx].confidence,
                                camera: event.camera_id || updated[existingIdx].camera,
                            };
                            return updated;
                        }
                        // Add new entry at top, keep max 50
                        return [entry, ...prev].slice(0, 50);
                    });

                    fetchActivePersons();

                } else if (eventType === 'exit') {
                    const personId = event.person_id || '';
                    const durationSec = event.duration_sec || 0;

                    setPresenceLog((prev) => {
                        const updated = prev.map((p) => {
                            if (p.personId === personId && p.status === 'present') {
                                return {
                                    ...p,
                                    status: 'left',
                                    exitTime: now.getTime(),
                                    duration: durationSec,
                                };
                            }
                            return p;
                        });
                        return updated;
                    });

                    fetchActivePersons();

                } else if (eventType === 'unknown') {
                    const entry = {
                        id: `unknown-${now.getTime()}`,
                        personId: `unknown-${now.getTime()}`,
                        name: 'Unknown Person',
                        status: 'unknown',
                        confidence: 0,
                        camera: event.camera_id || '',
                        entryTime: now.getTime(),
                        exitTime: null,
                        duration: null,
                        entryTimeStr: formatTime(now),
                    };

                    setPresenceLog((prev) => [entry, ...prev].slice(0, 50));
                }
                // Ignore 'detection' type events (no longer emitted, but just in case)
            });

            wsRef.current = ws;
        } catch (err) {
            console.warn('WebSocket connection failed:', err);
        }
    }

    function startActivePersonsPoll() {
        fetchActivePersons();
        pollRef.current = setInterval(fetchActivePersons, 3000);
    }

    async function fetchActivePersons() {
        try {
            const result = await api.getActivePersons();
            if (result && result.persons) {
                setActivePersons(result.persons);
            }
        } catch {
            // Silently fail
        }
    }

    const getPresenceIcon = (status) => {
        switch (status) {
            case 'present': return <EntryIcon size={14} />;
            case 'left': return <ExitIcon size={14} />;
            case 'unknown': return <UnknownFaceIcon size={14} />;
            default: return <DetectionIcon size={14} />;
        }
    };

    function formatDuration(seconds) {
        if (seconds == null) return '';
        if (seconds < 60) return `${seconds}s`;
        const m = Math.floor(seconds / 60);
        const s = seconds % 60;
        return `${m}m ${s}s`;
    }

    const onlineCameras = cameras.filter(c => c.status === 'online');

    return (
        <AppShell>
            <div className="page-header">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div>
                        <h1>Live Monitor</h1>
                        <p>{onlineCameras.length} of {cameras.length} cameras online</p>
                    </div>
                    <div style={{ display: 'flex', gap: '6px' }}>
                        {[
                            { key: 'single', label: '1×1', icon: <SquareIcon size={14} /> },
                            { key: '2x2', label: '2×2', icon: <GridIcon size={14} /> },
                            { key: '3x3', label: '3×3', icon: <GridIcon size={12} /> },
                        ].map(opt => (
                            <button
                                key={opt.key}
                                className={`btn btn-sm ${layout === opt.key ? 'btn-primary' : 'btn-secondary'}`}
                                onClick={() => {
                                    setLayout(opt.key);
                                    localStorage.setItem('sentinel-monitor-layout', opt.key);
                                }}
                                id={`layout-${opt.key}`}
                                style={{ display: 'flex', alignItems: 'center', gap: '5px', minWidth: 'auto' }}
                            >
                                {opt.icon} {opt.label}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {loading ? (
                <div style={{ display: 'flex', justifyContent: 'center', padding: '60px' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Loading cameras...</div>
                </div>
            ) : cameras.length === 0 ? (
                <div className="card">
                    <div className="empty-state">
                        <div className="empty-state-icon"><VideoCameraOffIcon size={32} /></div>
                        <div className="empty-state-title">No Cameras Configured</div>
                        <div className="empty-state-text">Add a camera from the Cameras page to start monitoring.</div>
                    </div>
                </div>
            ) : layout === '2x2' ? (
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(2, 1fr)',
                    gap: '14px',
                }}>
                    {cameras.slice(0, 4).map((camera) => (
                        <CameraFeedCard
                            key={camera.id}
                            camera={camera}
                            onClick={() => {
                                setSelectedCamera(camera);
                                setLayout('single');
                                localStorage.setItem('sentinel-monitor-layout', 'single');
                            }}
                        />
                    ))}
                </div>
            ) : layout === '3x3' ? (
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(3, 1fr)',
                    gap: '12px',
                }}>
                    {cameras.slice(0, 9).map((camera) => (
                        <CameraFeedCard
                            key={camera.id}
                            camera={camera}
                            compact
                            onClick={() => {
                                setSelectedCamera(camera);
                                setLayout('single');
                                localStorage.setItem('sentinel-monitor-layout', 'single');
                            }}
                        />
                    ))}
                </div>
            ) : (
                <div className="dashboard-grid-3">
                    {/* Main Feed */}
                    <div>
                        <CameraFeedCard
                            camera={selectedCamera || cameras[0]}
                            large
                        />
                    </div>

                    {/* Right Sidebar Panel */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                        {/* Camera List */}
                        <div className="card">
                            <div className="card-header">
                                <h3 className="card-title">Cameras</h3>
                            </div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                {cameras.map((camera) => (
                                    <button
                                        key={camera.id}
                                        className={`nav-item ${selectedCamera?.id === camera.id ? 'active' : ''}`}
                                        onClick={() => setSelectedCamera(camera)}
                                    >
                                        <span className="nav-item-icon">
                                            {camera.status === 'online'
                                                ? <VideoCameraIcon size={16} />
                                                : <VideoCameraOffIcon size={16} />
                                            }
                                        </span>
                                        <div style={{ flex: 1, textAlign: 'left' }}>
                                            <div style={{ fontSize: '0.813rem', fontWeight: 500 }}>{camera.name}</div>
                                            <div style={{ fontSize: '0.688rem', color: 'var(--text-muted)' }}>{camera.location}</div>
                                        </div>
                                        <span className={`badge ${camera.status}`} style={{ fontSize: '0.625rem' }}>
                                            {camera.status === 'online' && <span className="badge-dot"></span>}
                                            {camera.status}
                                        </span>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* People Present Panel */}
                        <div className="card">
                            <div className="card-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                <h3 className="card-title" style={{ fontSize: '0.875rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <UsersIcon size={16} /> People Present
                                </h3>
                                <span className="badge online" style={{ fontSize: '0.688rem' }}>
                                    <span className="badge-dot"></span>
                                    {activePersons.length}
                                </span>
                            </div>
                            <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
                                {activePersons.length > 0 ? (
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                                        {activePersons.map((person) => (
                                            <div
                                                key={person.person_id}
                                                style={{
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    gap: '10px',
                                                    padding: '8px 10px',
                                                    borderRadius: 'var(--radius-sm)',
                                                    background: 'var(--bg-tertiary)',
                                                    transition: 'background 0.2s ease',
                                                }}
                                            >
                                                <div style={{
                                                    width: '32px',
                                                    height: '32px',
                                                    borderRadius: '50%',
                                                    background: 'var(--primary-dark)',
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center',
                                                    fontSize: '0.75rem',
                                                    fontWeight: 700,
                                                    color: '#fff',
                                                    flexShrink: 0,
                                                }}>
                                                    {person.person_name?.charAt(0)?.toUpperCase() || '?'}
                                                </div>
                                                <div style={{ flex: 1, minWidth: 0 }}>
                                                    <div style={{
                                                        fontSize: '0.813rem',
                                                        fontWeight: 600,
                                                        whiteSpace: 'nowrap',
                                                        overflow: 'hidden',
                                                        textOverflow: 'ellipsis',
                                                    }}>
                                                        {person.person_name}
                                                    </div>
                                                    <div style={{
                                                        fontSize: '0.625rem',
                                                        color: 'var(--text-muted)',
                                                    }}>
                                                        {person.camera_id} &bull; {person.duration_sec}s
                                                    </div>
                                                </div>
                                                <div style={{
                                                    width: '8px',
                                                    height: '8px',
                                                    borderRadius: '50%',
                                                    background: '#22c55e',
                                                    flexShrink: 0,
                                                    animation: 'pulse 2s ease-in-out infinite',
                                                }} />
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <div style={{
                                        padding: '16px',
                                        textAlign: 'center',
                                        color: 'var(--text-muted)',
                                        fontSize: '0.75rem',
                                    }}>
                                        No registered persons detected yet.
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Activity Log — Presence-based tracking */}
                        <div className="card">
                            <div className="card-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                <h4 className="card-title" style={{ fontSize: '0.875rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <SignalIcon size={16} /> Activity Log
                                </h4>
                                {presenceLog.length > 0 && (
                                    <span style={{
                                        fontSize: '0.625rem',
                                        color: 'var(--text-muted)',
                                        cursor: 'pointer',
                                    }}
                                        onClick={() => setPresenceLog([])}
                                    >
                                        Clear
                                    </span>
                                )}
                            </div>
                            <div className="activity-feed" style={{ maxHeight: '300px' }}>
                                {presenceLog.length > 0 ? presenceLog.map((p) => (
                                    <div key={p.id} className="activity-item" style={{
                                        padding: '10px 12px',
                                        borderLeft: `3px solid ${p.status === 'present' ? 'var(--success)'
                                                : p.status === 'left' ? 'var(--text-muted)'
                                                    : 'var(--warning)'
                                            }`,
                                        opacity: p.status === 'left' ? 0.7 : 1,
                                        transition: 'opacity 0.3s ease',
                                    }}>
                                        <div className={`activity-icon ${p.status === 'present' ? 'entry' : p.status === 'left' ? 'exit' : 'unknown'}`}
                                            style={{ width: '28px', height: '28px', fontSize: '0.75rem' }}>
                                            {getPresenceIcon(p.status)}
                                        </div>
                                        <div className="activity-content" style={{ flex: 1, minWidth: 0 }}>
                                            <div style={{ fontSize: '0.75rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                <span style={{
                                                    whiteSpace: 'nowrap',
                                                    overflow: 'hidden',
                                                    textOverflow: 'ellipsis',
                                                }}>{p.name}</span>
                                                {p.status === 'present' && (
                                                    <span style={{
                                                        display: 'inline-flex',
                                                        alignItems: 'center',
                                                        gap: '4px',
                                                        color: 'var(--success)',
                                                        fontSize: '0.625rem',
                                                        fontWeight: 700,
                                                        letterSpacing: '0.5px',
                                                    }}>
                                                        <span style={{
                                                            width: '6px',
                                                            height: '6px',
                                                            borderRadius: '50%',
                                                            background: 'var(--success)',
                                                            animation: 'pulse 2s ease-in-out infinite',
                                                            display: 'inline-block',
                                                        }} />
                                                        PRESENT
                                                    </span>
                                                )}
                                                {p.status === 'left' && (
                                                    <span style={{
                                                        color: 'var(--text-muted)',
                                                        fontSize: '0.625rem',
                                                        fontWeight: 500,
                                                    }}>
                                                        LEFT {p.duration != null ? `• ${formatDuration(p.duration)}` : ''}
                                                    </span>
                                                )}
                                                {p.status === 'unknown' && (
                                                    <span style={{
                                                        color: 'var(--warning)',
                                                        fontSize: '0.625rem',
                                                        fontWeight: 700,
                                                        letterSpacing: '0.5px',
                                                    }}>
                                                        UNKNOWN
                                                    </span>
                                                )}
                                            </div>
                                            <div style={{ fontSize: '0.625rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                                                {p.confidence > 0 && `${(p.confidence * 100).toFixed(0)}% match \u2022 `}
                                                {p.camera}
                                            </div>
                                        </div>
                                        <span className="activity-time" style={{ fontSize: '0.625rem', flexShrink: 0 }}>
                                            {p.entryTimeStr}
                                        </span>
                                    </div>
                                )) : (
                                    <div style={{ padding: '16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                                        No activity yet. Detections will appear when persons enter or leave.
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </AppShell>
    );
}

function CameraFeedCard({ camera, large = false, compact = false, onClick }) {
    const imgRef = useRef(null);
    const [streamStatus, setStreamStatus] = useState('connecting');

    useEffect(() => {
        if (camera.status !== 'online') {
            setStreamStatus('offline');
            return;
        }

        let mounted = true;
        let ws = null;
        let previousUrl = null;
        let reconnectTimer = null;
        let pendingUrl = null;
        let rafId = null;

        // Use requestAnimationFrame for smooth frame rendering
        const renderFrame = () => {
            if (!mounted) return;
            if (pendingUrl && imgRef.current) {
                const oldSrc = imgRef.current.src;
                imgRef.current.src = pendingUrl;
                // Revoke the previous blob URL to free memory
                if (previousUrl && previousUrl.startsWith('blob:')) {
                    URL.revokeObjectURL(previousUrl);
                }
                previousUrl = pendingUrl;
                pendingUrl = null;
                setStreamStatus('streaming');
            }
            rafId = requestAnimationFrame(renderFrame);
        };

        const connectStream = () => {
            if (!mounted) return;

            try {
                const wsUrl = API_BASE.replace(/^http/, 'ws');
                ws = new WebSocket(`${wsUrl}/api/cameras/${camera.id}/stream`);
                ws.binaryType = 'arraybuffer';

                ws.onopen = () => {
                    if (!mounted) return;
                    console.log(`[Stream] WebSocket connected for ${camera.name}`);
                    setStreamStatus('connecting');
                    // Keep-alive ping every 25 seconds
                    ws._keepAlive = setInterval(() => {
                        if (ws.readyState === WebSocket.OPEN) {
                            ws.send('ping');
                        }
                    }, 25000);
                };

                ws.onmessage = (event) => {
                    if (!mounted) return;
                    // Create blob URL from binary JPEG data
                    const blob = new Blob([event.data], { type: 'image/jpeg' });
                    // If there's already a pending frame we haven't rendered, revoke it
                    if (pendingUrl) {
                        URL.revokeObjectURL(pendingUrl);
                    }
                    pendingUrl = URL.createObjectURL(blob);
                };

                ws.onerror = (err) => {
                    console.warn(`[Stream] WebSocket error for ${camera.name}:`, err);
                };

                ws.onclose = () => {
                    if (ws._keepAlive) clearInterval(ws._keepAlive);
                    if (!mounted) return;
                    console.log(`[Stream] WebSocket closed for ${camera.name}, reconnecting in 2s...`);
                    setStreamStatus('reconnecting');
                    // Auto-reconnect after 2 seconds
                    reconnectTimer = setTimeout(connectStream, 2000);
                };
            } catch (err) {
                console.error(`[Stream] Failed to connect for ${camera.name}:`, err);
                if (mounted) {
                    setStreamStatus('reconnecting');
                    reconnectTimer = setTimeout(connectStream, 2000);
                }
            }
        };

        setStreamStatus('connecting');
        connectStream();
        rafId = requestAnimationFrame(renderFrame);

        return () => {
            mounted = false;
            if (rafId) cancelAnimationFrame(rafId);
            if (reconnectTimer) clearTimeout(reconnectTimer);
            if (ws) {
                if (ws._keepAlive) clearInterval(ws._keepAlive);
                ws.onclose = null; // Prevent reconnect on intentional close
                ws.close();
            }
            if (previousUrl && previousUrl.startsWith('blob:')) {
                URL.revokeObjectURL(previousUrl);
            }
            if (pendingUrl) {
                URL.revokeObjectURL(pendingUrl);
            }
        };
    }, [camera]);

    return (
        <div className="card" style={{ padding: 0, overflow: 'hidden', cursor: onClick ? 'pointer' : 'default' }} onClick={onClick}>
            <div className="live-feed-container" style={{ aspectRatio: '16/9', position: 'relative', background: '#000000' }}>
                {camera.status === 'online' ? (
                    <>
                        <div className="live-indicator">
                            <span className="live-indicator-dot"></span>
                            {streamStatus === 'streaming' ? 'LIVE' : streamStatus === 'connecting' ? 'CONNECTING' : 'RECONNECTING'}
                        </div>

                        <img
                            ref={imgRef}
                            alt={`${camera.name} live feed`}
                            style={{
                                width: '100%',
                                height: '100%',
                                objectFit: 'contain',
                                display: streamStatus === 'streaming' ? 'block' : 'none',
                            }}
                        />

                        {streamStatus !== 'streaming' && (
                            <div className="live-feed-placeholder">
                                <div style={{ animation: 'pulse 2s ease-in-out infinite', display: 'flex' }}>
                                    <VideoCameraIcon size={32} />
                                </div>
                                <div style={{ fontWeight: 500 }}>{camera.name}</div>
                                <div style={{ fontSize: '0.75rem' }}>
                                    {streamStatus === 'connecting' ? 'Connecting to video stream...' : 'Reconnecting...'}
                                </div>
                                <div style={{ fontSize: '0.688rem', opacity: 0.5 }}>
                                    Waiting for frames from backend pipeline
                                </div>
                            </div>
                        )}
                    </>
                ) : (
                    <div className="live-feed-placeholder">
                        <div style={{ opacity: 0.2, display: 'flex' }}>
                            <VideoCameraOffIcon size={32} />
                        </div>
                        <div style={{ fontWeight: 500 }}>{camera.name}</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--danger)' }}>Camera Offline</div>
                    </div>
                )}
            </div>
            {compact ? (
                <div style={{ padding: '8px 12px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ fontWeight: 600, fontSize: '0.75rem' }}>{camera.name}</div>
                    <span className={`badge ${camera.status}`} style={{ fontSize: '0.625rem', padding: '2px 6px' }}>
                        {camera.status}
                    </span>
                </div>
            ) : (
                <div style={{ padding: '12px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div>
                        <div style={{ fontWeight: 600, fontSize: '0.875rem' }}>{camera.name}</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{camera.location}</div>
                    </div>
                    <span className={`badge ${camera.status}`}>
                        {camera.status === 'online' && <span className="badge-dot"></span>}
                        {camera.status}
                    </span>
                </div>
            )}
        </div>
    );
}
