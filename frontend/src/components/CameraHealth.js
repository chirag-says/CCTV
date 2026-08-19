'use client';

/**
 * CameraHealth — Dashboard widget showing camera status overview.
 * Displays 4-state stream health (LIVE/STALE/RECONNECTING/OFFLINE)
 * with color-coded indicators and quick status summaries.
 */

import { useState, useEffect } from 'react';
import api from '@/lib/api';
import { VideoCameraIcon, VideoCameraOffIcon } from './Icons';

function getStreamHealthInfo(cam) {
    // Use stream_health from pipeline status if available
    const sh = cam.stream_health;
    if (sh && sh.stream_state) {
        switch (sh.stream_state) {
            case 'LIVE':
                return { color: 'var(--success)', bg: 'rgba(16, 185, 129, 0.12)', label: 'LIVE', glow: true };
            case 'STALE':
                return { color: 'var(--warning)', bg: 'rgba(245, 158, 11, 0.12)', label: 'STALE', glow: false };
            case 'RECONNECTING':
                return { color: 'var(--warning)', bg: 'rgba(245, 158, 11, 0.12)', label: 'RECONNECTING', glow: false, pulse: true };
            case 'OFFLINE':
                return { color: 'var(--danger)', bg: 'rgba(239, 68, 68, 0.1)', label: 'OFFLINE', glow: false };
            default:
                break;
        }
    }
    // Fallback: simple online/offline
    if (cam.status === 'online') {
        return { color: 'var(--success)', bg: 'rgba(16, 185, 129, 0.12)', label: 'LIVE', glow: true };
    }
    return { color: 'var(--danger)', bg: 'rgba(239, 68, 68, 0.1)', label: 'OFFLINE', glow: false };
}

export default function CameraHealth() {
    const [cameras, setCameras] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadCameras();
        // Refresh every 15s (faster for health monitoring)
        const interval = setInterval(loadCameras, 15000);
        return () => clearInterval(interval);
    }, []);

    async function loadCameras() {
        try {
            const result = await api.getCameras();
            setCameras(result || []);
        } catch {
            // Silent fail — widget is non-critical
        } finally {
            setLoading(false);
        }
    }

    const online = cameras.filter(c => c.status === 'online');
    const stale = cameras.filter(c => c.stream_health?.stream_state === 'STALE' || c.stream_health?.stream_state === 'RECONNECTING');
    const healthPct = cameras.length > 0 ? Math.round((online.length / cameras.length) * 100) : 0;

    return (
        <div className="card">
            <div className="card-header">
                <div>
                    <h3 className="card-title">Camera Health</h3>
                    <div className="card-subtitle">{cameras.length} cameras configured</div>
                </div>
                <div style={{
                    padding: '4px 10px',
                    borderRadius: '8px',
                    background: healthPct === 100 && stale.length === 0 ? 'rgba(16, 185, 129, 0.1)' : healthPct > 50 ? 'rgba(245, 158, 11, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                    color: healthPct === 100 && stale.length === 0 ? 'var(--success)' : healthPct > 50 ? 'var(--warning)' : 'var(--danger)',
                    fontSize: '0.75rem',
                    fontWeight: 700,
                }}>
                    {stale.length > 0 ? `${stale.length} Stale` : `${healthPct}% Online`}
                </div>
            </div>

            {loading ? (
                <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
                    Loading...
                </div>
            ) : cameras.length === 0 ? (
                <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
                    No cameras configured
                </div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {cameras.map(cam => {
                        const health = getStreamHealthInfo(cam);
                        return (
                            <div
                                key={cam.id}
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '10px',
                                    padding: '8px 12px',
                                    borderRadius: '8px',
                                    background: 'var(--bg-tertiary)',
                                    transition: 'background 0.15s',
                                }}
                            >
                                <div style={{
                                    width: '30px', height: '30px', borderRadius: '8px',
                                    background: health.bg,
                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                    color: health.color,
                                    flexShrink: 0,
                                }}>
                                    {cam.status === 'online' ? <VideoCameraIcon size={14} /> : <VideoCameraOffIcon size={14} />}
                                </div>
                                <div style={{ flex: 1, minWidth: 0 }}>
                                    <div style={{
                                        fontSize: '0.8125rem', fontWeight: 600,
                                        color: 'var(--text-primary)',
                                        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                                    }}>
                                        {cam.name}
                                    </div>
                                    <div style={{ fontSize: '0.625rem', color: 'var(--text-muted)' }}>
                                        {cam.location || cam.camera_type}
                                    </div>
                                </div>
                                <div style={{
                                    display: 'flex', alignItems: 'center', gap: '6px',
                                }}>
                                    <span style={{
                                        fontSize: '0.563rem',
                                        fontWeight: 700,
                                        color: health.color,
                                        letterSpacing: '0.5px',
                                    }}>
                                        {health.label}
                                    </span>
                                    <div style={{
                                        width: '8px', height: '8px', borderRadius: '50%',
                                        background: health.color,
                                        boxShadow: health.glow ? `0 0 8px ${health.color}40` : 'none',
                                        animation: health.pulse ? 'pulse 2s ease-in-out infinite' : 'none',
                                        flexShrink: 0,
                                    }} />
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}

