'use client';

/**
 * Live Monitor Page — Real-time camera feeds with detection overlays.
 * Shows detected person names and active persons panel.
 * Supports browser-side RF-DETR WebGPU object detection overlay.
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import AppShell from '@/components/AppShell';
import api, { API_BASE } from '@/lib/api';
import DetectionOverlay from '@/components/DetectionOverlay';
import Go2RtcPlayer from '@/components/Go2RtcPlayer';
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
    const [aiDetectEnabled, setAiDetectEnabled] = useState(false);
    const [aiStats, setAiStats] = useState(null);
    const [layout, setLayout] = useState(() => {
        if (typeof window !== 'undefined') {
            return localStorage.getItem('sentinel-monitor-layout') || 'single';
        }
        return 'single';
    });

    // Grid slot assignments: { slotIndex: cameraId } persisted in localStorage
    const [gridSlots, setGridSlots] = useState(() => {
        if (typeof window !== 'undefined') {
            try {
                return JSON.parse(localStorage.getItem('sentinel-grid-slots') || '{}');
            } catch { return {}; }
        }
        return {};
    });

    const updateGridSlot = useCallback((slotIndex, cameraId) => {
        setGridSlots(prev => {
            const next = { ...prev, [slotIndex]: cameraId };
            localStorage.setItem('sentinel-grid-slots', JSON.stringify(next));
            return next;
        });
    }, []);

    // Resolve which camera goes in each slot, falling back to array order
    const resolveGridCamera = useCallback((slotIndex, fallbackCameras) => {
        const assignedId = gridSlots[slotIndex];
        if (assignedId) {
            const found = cameras.find(c => c.id === assignedId);
            if (found) return found;
        }
        return fallbackCameras[slotIndex] || null;
    }, [gridSlots, cameras]);

    // Sort cameras with online first for the sidebar
    const sortedCameras = useMemo(() => {
        return [...cameras].sort((a, b) => {
            if (a.status === 'online' && b.status !== 'online') return -1;
            if (a.status !== 'online' && b.status === 'online') return 1;
            return 0;
        });
    }, [cameras]);
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

    // Zone bunches for the 'bunch' layout. The backend computes `group` from
    // location (falling back to the first name token); the same fallback here
    // covers cached or older API responses without the field. Largest bunches
    // first, so COE and Front Gate lead the page.
    const cameraBunches = (() => {
        const byGroup = new Map();
        for (const cam of cameras) {
            const group = cam.group || cam.location || (cam.name || '').split(' ')[0] || 'Ungrouped';
            if (!byGroup.has(group)) byGroup.set(group, []);
            byGroup.get(group).push(cam);
        }
        return [...byGroup.entries()]
            .map(([group, cams]) => ({ group, cams }))
            .sort((a, b) => b.cams.length - a.cams.length || a.group.localeCompare(b.group));
    })();

    const openSingle = (camera) => {
        setSelectedCamera(camera);
        setLayout('single');
        localStorage.setItem('sentinel-monitor-layout', 'single');
    };

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
                    <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                        <button
                            className={`btn btn-sm ${aiDetectEnabled ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={() => setAiDetectEnabled(!aiDetectEnabled)}
                            id="ai-detect-toggle"
                            title="Toggle browser-side RF-DETR object detection"
                            style={{
                                display: 'flex', alignItems: 'center', gap: '5px', minWidth: 'auto',
                                ...(aiDetectEnabled ? { boxShadow: '0 0 12px rgba(99, 102, 241, 0.4)' } : {}),
                            }}
                        >
                            <DetectionIcon size={14} />
                            {aiDetectEnabled ? 'AI On' : 'AI Detect'}
                        </button>
                        {aiDetectEnabled && aiStats && (
                            <span style={{ fontSize: '0.625rem', color: 'var(--accent-light)', fontWeight: 600 }}>
                                {aiStats.inferenceMs}ms • {aiStats.detectionCount} obj
                            </span>
                        )}
                        <div style={{ width: '1px', height: '20px', background: 'var(--border-color)' }} />
                        {[
                            { key: 'single', label: '1×1', icon: <SquareIcon size={14} /> },
                            { key: '2x2', label: '2×2', icon: <GridIcon size={14} /> },
                            { key: '3x3', label: '3×3', icon: <GridIcon size={12} /> },
                            { key: 'bunch', label: 'Bunch', icon: <UsersIcon size={12} /> },
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
                    {[0, 1, 2, 3].map((slotIdx) => {
                        const cam = resolveGridCamera(slotIdx, cameras);
                        if (!cam) return null;
                        return (
                            <CameraFeedCard
                                key={`slot-${slotIdx}-${cam.id}`}
                                camera={cam}
                                aiDetectEnabled={aiDetectEnabled}
                                onAiStats={setAiStats}
                                slotIndex={slotIdx}
                                allCameras={cameras}
                                onSlotChange={updateGridSlot}
                                onClick={() => {
                                    setSelectedCamera(cam);
                                    setLayout('single');
                                    localStorage.setItem('sentinel-monitor-layout', 'single');
                                }}
                            />
                        );
                    })}
                </div>
            ) : layout === 'bunch' ? (
                /* One section per zone: every COE camera side by side, every
                   Front Gate camera side by side. A person crossing the zone
                   carries the same identity label on each feed, so
                   cross-camera re-ID can be watched happening. */
                <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                    {cameraBunches.map(({ group, cams }) => {
                        const online = cams.filter(c => c.status === 'online').length;
                        return (
                            <div key={group}>
                                <div style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '10px',
                                    marginBottom: '10px',
                                }}>
                                    <h3 style={{ fontSize: '0.938rem', fontWeight: 600, margin: 0 }}>
                                        {group}
                                    </h3>
                                    <span className={`badge ${online > 0 ? 'online' : 'offline'}`} style={{ fontSize: '0.625rem' }}>
                                        {online > 0 && <span className="badge-dot"></span>}
                                        {online}/{cams.length} online
                                    </span>
                                    <div style={{ flex: 1, height: '1px', background: 'var(--border-color)' }} />
                                </div>
                                <div style={{
                                    display: 'grid',
                                    gridTemplateColumns: `repeat(${cams.length >= 5 ? 3 : 2}, 1fr)`,
                                    gap: '12px',
                                }}>
                                    {cams.map((camera) => (
                                        <CameraFeedCard
                                            key={camera.id}
                                            camera={camera}
                                            compact
                                            aiDetectEnabled={aiDetectEnabled}
                                            onAiStats={setAiStats}
                                            onClick={() => openSingle(camera)}
                                        />
                                    ))}
                                </div>
                            </div>
                        );
                    })}
                </div>
            ) : layout === '3x3' ? (
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(3, 1fr)',
                    gap: '12px',
                }}>
                    {[0, 1, 2, 3, 4, 5, 6, 7, 8].map((slotIdx) => {
                        const cam = resolveGridCamera(slotIdx, cameras);
                        if (!cam) return null;
                        return (
                            <CameraFeedCard
                                key={`slot-${slotIdx}-${cam.id}`}
                                camera={cam}
                                compact
                                aiDetectEnabled={aiDetectEnabled}
                                onAiStats={setAiStats}
                                slotIndex={slotIdx}
                                allCameras={cameras}
                                onSlotChange={updateGridSlot}
                                onClick={() => {
                                    setSelectedCamera(cam);
                                    setLayout('single');
                                    localStorage.setItem('sentinel-monitor-layout', 'single');
                                }}
                            />
                        );
                    })}
                </div>
            ) : (
                <div className="dashboard-grid-3">
                    {/* Main Feed */}
                    <div>
                        <CameraFeedCard
                            camera={selectedCamera || cameras[0]}
                            large
                            aiDetectEnabled={aiDetectEnabled}
                            onAiStats={setAiStats}
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
                                {sortedCameras.map((camera) => (
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

/**
 * Is MJPEG being forced for this session?
 *
 * WebRTC is the default path; this flag exists because a relay that will not
 * start, a browser with WebRTC blocked, or a camera go2rtc cannot serve are
 * all real situations, and "the live view is broken" is a worse answer than
 * "the live view is the old one". Reachable at /monitor?mjpeg=1.
 */
function mjpegForced() {
    if (typeof window === 'undefined') return false;
    return new URLSearchParams(window.location.search).get('mjpeg') === '1';
}

/**
 * MJPEG over WebSocket — the fallback transport.
 *
 * Frames are annotated server-side, so this path draws no overlay of its own.
 * Connecting starts the camera's JPEG encoder thread; nothing else does.
 */
function MjpegFeed({ camera, onStatus, onStale, aiDetectEnabled, onAiStats }) {
    const imgRef = useRef(null);
    const [streamStatus, setStreamStatus] = useState('connecting');

    useEffect(() => {
        if (onStatus) onStatus(streamStatus);
    }, [streamStatus, onStatus]);

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
                // Use the shared helper so the JWT is attached -- the server
                // now rejects this socket with 4001 if ?token= is missing.
                // fallback=1 says this viewer is deliberately on the old
                // transport, and is what starts the JPEG encoder thread.
                ws = new WebSocket(
                    `${api.wsUrl(`/api/cameras/${camera.id}/stream`)}&fallback=1`
                );
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

                    // Handle JSON text messages (stream_stale notifications)
                    if (typeof event.data === 'string') {
                        try {
                            const msg = JSON.parse(event.data);
                            if (msg.type === 'stream_stale') {
                                if (onStale) onStale(msg);
                                setStreamStatus('stream_frozen');
                                return;
                            }
                        } catch (e) {
                            // Not JSON, ignore
                        }
                        return;
                    }

                    // Binary JPEG frame — stream is healthy again
                    if (onStale) onStale(null);
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
    }, [camera, onStale]);

    return (
        <>
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

            {/* RF-DETR WebGPU Detection Overlay (browser-side, MJPEG only) */}
            {aiDetectEnabled && streamStatus === 'streaming' && (
                <DetectionOverlay
                    videoRef={imgRef}
                    enabled={aiDetectEnabled}
                    onStats={onAiStats}
                />
            )}
        </>
    );
}

/**
 * One camera tile.
 *
 * Chooses the transport, then renders the chrome around whichever one it got.
 * WebRTC (video from go2rtc plus a metadata canvas) is the default; MJPEG is
 * used when go2rtc cannot relay this camera -- a webcam index it does not own,
 * a browser-pushed session -- or when ?mjpeg=1 forces it.
 */
function CameraFeedCard({ camera, large = false, compact = false, onClick, aiDetectEnabled = false, onAiStats, slotIndex, allCameras, onSlotChange }) {
    const [streamStatus, setStreamStatus] = useState('connecting');
    const [staleInfo, setStaleInfo] = useState(null);
    const [webrtc, setWebrtc] = useState(null);
    const [everStreamed, setEverStreamed] = useState(false);
    const forceMjpeg = mjpegForced();

    // ── Zoom & Pan state ──────────────────────────────────────────
    const [zoom, setZoom] = useState(1);
    const [pan, setPan] = useState({ x: 0, y: 0 });
    const [dragging, setDragging] = useState(false);
    const dragStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 });
    const containerRef = useRef(null);

    // ── Slot picker state ─────────────────────────────────────────
    const [pickerOpen, setPickerOpen] = useState(false);
    const pickerRef = useRef(null);

    // Close picker on click outside
    useEffect(() => {
        if (!pickerOpen) return;
        const handler = (e) => {
            if (pickerRef.current && !pickerRef.current.contains(e.target)) {
                setPickerOpen(false);
            }
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, [pickerOpen]);

    useEffect(() => {
        if (camera.status !== 'online' || forceMjpeg) return undefined;
        let mounted = true;
        api.getCameraWebrtc(camera.id)
            .then((info) => { if (mounted) setWebrtc(info); })
            .catch(() => {
                if (mounted) setWebrtc({ available: false });
            });
        return () => { mounted = false; };
    }, [camera.id, camera.status, forceMjpeg]);

    const handleStatus = useCallback((next) => {
        setStreamStatus(next);
        if (next === 'streaming') setEverStreamed(true);
    }, []);

    // ── Zoom helpers ──────────────────────────────────────────────
    const MIN_ZOOM = 1;
    const MAX_ZOOM = 5;

    const clampPan = useCallback((x, y, z) => {
        if (z <= 1) return { x: 0, y: 0 };
        const el = containerRef.current;
        if (!el) return { x: 0, y: 0 };
        const w = el.offsetWidth;
        const h = el.offsetHeight;
        const maxX = w * (z - 1);
        const maxY = h * (z - 1);
        return {
            x: Math.max(-maxX, Math.min(0, x)),
            y: Math.max(-maxY, Math.min(0, y)),
        };
    }, []);

    const handleWheel = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        const delta = e.deltaY < 0 ? 0.3 : -0.3;
        setZoom(prev => {
            const next = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, prev + delta));
            if (next <= 1) {
                setPan({ x: 0, y: 0 });
            } else {
                // Re-clamp pan for new zoom
                setPan(p => clampPan(p.x, p.y, next));
            }
            return next;
        });
    }, [clampPan]);

    const handleMouseDown = useCallback((e) => {
        if (zoom <= 1) return;
        e.preventDefault();
        e.stopPropagation();
        setDragging(true);
        dragStart.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y };
    }, [zoom, pan]);

    const handleMouseMove = useCallback((e) => {
        if (!dragging) return;
        e.preventDefault();
        const dx = e.clientX - dragStart.current.x;
        const dy = e.clientY - dragStart.current.y;
        setPan(clampPan(dragStart.current.panX + dx, dragStart.current.panY + dy, zoom));
    }, [dragging, zoom, clampPan]);

    const handleMouseUp = useCallback(() => {
        setDragging(false);
    }, []);

    useEffect(() => {
        if (!dragging) return;
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, [dragging, handleMouseMove, handleMouseUp]);

    const zoomIn = (e) => { e.stopPropagation(); setZoom(z => Math.min(MAX_ZOOM, z + 0.5)); };
    const zoomOut = (e) => {
        e.stopPropagation();
        setZoom(z => {
            const next = Math.max(MIN_ZOOM, z - 0.5);
            if (next <= 1) setPan({ x: 0, y: 0 });
            else setPan(p => clampPan(p.x, p.y, next));
            return next;
        });
    };
    const zoomReset = (e) => { e.stopPropagation(); setZoom(1); setPan({ x: 0, y: 0 }); };

    const decided = forceMjpeg || webrtc !== null;
    const useWebrtc = !forceMjpeg && Boolean(webrtc?.available && webrtc?.signalling_url);
    const transportLabel = useWebrtc ? 'WEBRTC' : 'MJPEG';
    const status = camera.status === 'online' ? streamStatus : 'offline';
    const isZoomed = zoom > 1;

    // Sorted cameras for slot picker: online first
    const pickerCameras = useMemo(() => {
        if (!allCameras) return [];
        const online = allCameras.filter(c => c.status === 'online');
        const offline = allCameras.filter(c => c.status !== 'online');
        return { online, offline };
    }, [allCameras]);

    const hasSlotPicker = slotIndex !== undefined && allCameras && onSlotChange;

    return (
        <div className="card" style={{ padding: 0, overflow: 'hidden', cursor: onClick && !isZoomed ? 'pointer' : 'default' }} onClick={!isZoomed ? onClick : undefined}>
            <div
                ref={containerRef}
                className="live-feed-container"
                style={{ aspectRatio: '16/9', position: 'relative', background: '#000000', overflow: 'hidden' }}
                onWheel={handleWheel}
            >
                {/* Zoom Controls */}
                <div className={`zoom-controls${isZoomed ? ' zoom-active' : ''}`}>
                    <button className="zoom-btn" onClick={zoomIn} title="Zoom in">+</button>
                    <button className="zoom-btn" onClick={zoomOut} title="Zoom out">−</button>
                    {isZoomed && (
                        <>
                            <div className="zoom-level-badge">{zoom.toFixed(1)}x</div>
                            <button className="zoom-btn reset" onClick={zoomReset} title="Reset zoom">RST</button>
                        </>
                    )}
                </div>

                <div
                    className={`live-feed-content${isZoomed ? ' zoomed' : ''}${dragging ? ' grabbing' : ''}`}
                    style={{
                        transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                    }}
                    onMouseDown={handleMouseDown}
                >
                    {camera.status === 'online' ? (
                        <>
                            <div className="live-indicator">
                                <span className={`live-indicator-dot${status === 'stream_frozen' ? ' frozen' : ''}`}></span>
                                {status === 'streaming' ? 'LIVE'
                                    : status === 'stream_frozen' ? (staleInfo?.stream_state === 'OFFLINE' ? 'OFFLINE' : 'FROZEN')
                                    : status === 'connecting' ? 'CONNECTING'
                                    : status === 'unavailable' ? 'NO RELAY'
                                    : 'RECONNECTING'}
                                <span style={{
                                    marginLeft: '6px',
                                    padding: '1px 5px',
                                    borderRadius: '4px',
                                    background: 'rgba(148, 163, 184, 0.2)',
                                    border: '1px solid rgba(148, 163, 184, 0.35)',
                                    fontSize: '0.563rem',
                                    fontWeight: 700,
                                    color: '#cbd5e1',
                                    letterSpacing: '0.5px',
                                }}>{transportLabel}</span>
                                {aiDetectEnabled && !useWebrtc && status === 'streaming' && (
                                    <span style={{
                                        marginLeft: '6px',
                                        padding: '1px 5px',
                                        borderRadius: '4px',
                                        background: 'rgba(99, 102, 241, 0.3)',
                                        border: '1px solid rgba(99, 102, 241, 0.5)',
                                        fontSize: '0.563rem',
                                        fontWeight: 700,
                                        color: '#a5b4fc',
                                        letterSpacing: '0.5px',
                                    }}>RF-DETR</span>
                                )}
                            </div>

                            {!decided ? null : useWebrtc ? (
                                <Go2RtcPlayer
                                    camera={camera}
                                    signallingUrl={webrtc.signalling_url}
                                    onStatus={handleStatus}
                                />
                            ) : (
                                <MjpegFeed
                                    camera={camera}
                                    onStatus={handleStatus}
                                    onStale={setStaleInfo}
                                    aiDetectEnabled={aiDetectEnabled}
                                    onAiStats={onAiStats}
                                />
                            )}

                            {status === 'stream_frozen' && (
                                <div className="stream-frozen-overlay">
                                    <div className="stream-frozen-icon">
                                        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                                            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                                            <line x1="12" y1="9" x2="12" y2="13" />
                                            <line x1="12" y1="17" x2="12.01" y2="17" />
                                        </svg>
                                    </div>
                                    <div className="stream-frozen-title">
                                        {staleInfo?.stream_state === 'OFFLINE' ? 'STREAM OFFLINE' : 'STREAM FROZEN'}
                                    </div>
                                    <div className="stream-frozen-detail">
                                        {staleInfo?.stream_state === 'RECONNECTING'
                                            ? `Attempting reconnection... #${staleInfo.reconnect_attempts || 0}`
                                            : staleInfo?.stream_state === 'OFFLINE'
                                            ? 'Max reconnect attempts exceeded'
                                            : 'No new frames received'}
                                    </div>
                                    {staleInfo?.last_frame_age && (
                                        <div className="stream-frozen-age">
                                            Last frame: {staleInfo.last_frame_age >= 60
                                                ? `${Math.floor(staleInfo.last_frame_age / 60)}m ${Math.round(staleInfo.last_frame_age % 60)}s ago`
                                                : `${Math.round(staleInfo.last_frame_age)}s ago`}
                                        </div>
                                    )}
                                </div>
                            )}

                            {status !== 'streaming' && status !== 'stream_frozen' && !everStreamed && (
                                <div className="live-feed-placeholder">
                                    <div style={{ animation: 'pulse 2s ease-in-out infinite', display: 'flex' }}>
                                        <VideoCameraIcon size={32} />
                                    </div>
                                    <div style={{ fontWeight: 500 }}>{camera.name}</div>
                                    <div style={{ fontSize: '0.75rem' }}>
                                        {status === 'unavailable'
                                            ? 'Relay has no stream for this camera'
                                            : status === 'connecting'
                                            ? 'Connecting to video stream...'
                                            : 'Reconnecting...'}
                                    </div>
                                    <div style={{ fontSize: '0.688rem', opacity: 0.5 }}>
                                        {useWebrtc
                                            ? 'Negotiating WebRTC with the relay'
                                            : 'Waiting for frames from backend pipeline'}
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
            </div>
            {compact ? (
                <div style={{ padding: '8px 12px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', position: 'relative' }}>
                    <div style={{ fontWeight: 600, fontSize: '0.75rem' }}>{camera.name}</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span className={`badge ${camera.status}`} style={{ fontSize: '0.625rem', padding: '2px 6px' }}>
                            {camera.status}
                        </span>
                        {hasSlotPicker && (
                            <div ref={pickerRef} style={{ position: 'relative' }}>
                                <button
                                    className="slot-swap-btn"
                                    onClick={(e) => { e.stopPropagation(); setPickerOpen(!pickerOpen); }}
                                    title="Change camera for this slot"
                                >
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <polyline points="16 3 21 3 21 8" />
                                        <line x1="4" y1="20" x2="21" y2="3" />
                                        <polyline points="21 16 21 21 16 21" />
                                        <line x1="15" y1="15" x2="21" y2="21" />
                                        <line x1="4" y1="4" x2="9" y2="9" />
                                    </svg>
                                </button>
                                {pickerOpen && (
                                    <div className="slot-picker-dropdown">
                                        {pickerCameras.online?.length > 0 && (
                                            <>
                                                <div className="slot-picker-divider">Online</div>
                                                {pickerCameras.online.map(c => (
                                                    <button
                                                        key={c.id}
                                                        className={`slot-picker-item${c.id === camera.id ? ' active' : ''}`}
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            onSlotChange(slotIndex, c.id);
                                                            setPickerOpen(false);
                                                        }}
                                                    >
                                                        <VideoCameraIcon size={14} />
                                                        <div style={{ flex: 1, minWidth: 0 }}>
                                                            <div className="picker-cam-name">{c.name}</div>
                                                            <div className="picker-cam-loc">{c.location}</div>
                                                        </div>
                                                        <span className="badge online" style={{ fontSize: '0.5rem', padding: '1px 4px' }}>
                                                            <span className="badge-dot"></span>
                                                        </span>
                                                    </button>
                                                ))}
                                            </>
                                        )}
                                        {pickerCameras.offline?.length > 0 && (
                                            <>
                                                <div className="slot-picker-divider">Offline</div>
                                                {pickerCameras.offline.map(c => (
                                                    <button
                                                        key={c.id}
                                                        className={`slot-picker-item${c.id === camera.id ? ' active' : ''}`}
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            onSlotChange(slotIndex, c.id);
                                                            setPickerOpen(false);
                                                        }}
                                                    >
                                                        <VideoCameraOffIcon size={14} />
                                                        <div style={{ flex: 1, minWidth: 0 }}>
                                                            <div className="picker-cam-name">{c.name}</div>
                                                            <div className="picker-cam-loc">{c.location}</div>
                                                        </div>
                                                    </button>
                                                ))}
                                            </>
                                        )}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            ) : (
                <div style={{ padding: '12px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', position: 'relative' }}>
                    <div>
                        <div style={{ fontWeight: 600, fontSize: '0.875rem' }}>{camera.name}</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{camera.location}</div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span className={`badge ${camera.status}`}>
                            {camera.status === 'online' && <span className="badge-dot"></span>}
                            {camera.status}
                        </span>
                        {hasSlotPicker && (
                            <div ref={pickerRef} style={{ position: 'relative' }}>
                                <button
                                    className="slot-swap-btn"
                                    onClick={(e) => { e.stopPropagation(); setPickerOpen(!pickerOpen); }}
                                    title="Change camera for this slot"
                                >
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <polyline points="16 3 21 3 21 8" />
                                        <line x1="4" y1="20" x2="21" y2="3" />
                                        <polyline points="21 16 21 21 16 21" />
                                        <line x1="15" y1="15" x2="21" y2="21" />
                                        <line x1="4" y1="4" x2="9" y2="9" />
                                    </svg>
                                </button>
                                {pickerOpen && (
                                    <div className="slot-picker-dropdown">
                                        {pickerCameras.online?.length > 0 && (
                                            <>
                                                <div className="slot-picker-divider">Online</div>
                                                {pickerCameras.online.map(c => (
                                                    <button
                                                        key={c.id}
                                                        className={`slot-picker-item${c.id === camera.id ? ' active' : ''}`}
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            onSlotChange(slotIndex, c.id);
                                                            setPickerOpen(false);
                                                        }}
                                                    >
                                                        <VideoCameraIcon size={14} />
                                                        <div style={{ flex: 1, minWidth: 0 }}>
                                                            <div className="picker-cam-name">{c.name}</div>
                                                            <div className="picker-cam-loc">{c.location}</div>
                                                        </div>
                                                        <span className="badge online" style={{ fontSize: '0.5rem', padding: '1px 4px' }}>
                                                            <span className="badge-dot"></span>
                                                        </span>
                                                    </button>
                                                ))}
                                            </>
                                        )}
                                        {pickerCameras.offline?.length > 0 && (
                                            <>
                                                <div className="slot-picker-divider">Offline</div>
                                                {pickerCameras.offline.map(c => (
                                                    <button
                                                        key={c.id}
                                                        className={`slot-picker-item${c.id === camera.id ? ' active' : ''}`}
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            onSlotChange(slotIndex, c.id);
                                                            setPickerOpen(false);
                                                        }}
                                                    >
                                                        <VideoCameraOffIcon size={14} />
                                                        <div style={{ flex: 1, minWidth: 0 }}>
                                                            <div className="picker-cam-name">{c.name}</div>
                                                            <div className="picker-cam-loc">{c.location}</div>
                                                        </div>
                                                    </button>
                                                ))}
                                            </>
                                        )}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
