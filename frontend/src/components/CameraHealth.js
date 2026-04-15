'use client';

/**
 * CameraHealth — Dashboard widget showing camera status overview.
 * Displays online/offline status, uptime indicators, and quick actions.
 */

import { useState, useEffect } from 'react';
import api from '@/lib/api';
import { VideoCameraIcon, VideoCameraOffIcon } from './Icons';

export default function CameraHealth() {
    const [cameras, setCameras] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadCameras();
        // Refresh every 30s
        const interval = setInterval(loadCameras, 30000);
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
    const offline = cameras.filter(c => c.status !== 'online');
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
                    background: healthPct === 100 ? 'rgba(16, 185, 129, 0.1)' : healthPct > 50 ? 'rgba(245, 158, 11, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                    color: healthPct === 100 ? 'var(--success)' : healthPct > 50 ? 'var(--warning)' : 'var(--danger)',
                    fontSize: '0.75rem',
                    fontWeight: 700,
                }}>
                    {healthPct}% Online
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
                    {cameras.map(cam => (
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
                                background: cam.status === 'online' ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.1)',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                color: cam.status === 'online' ? 'var(--success)' : 'var(--danger)',
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
                                width: '8px', height: '8px', borderRadius: '50%',
                                background: cam.status === 'online' ? 'var(--success)' : 'var(--danger)',
                                boxShadow: cam.status === 'online' ? '0 0 8px rgba(16, 185, 129, 0.4)' : 'none',
                                flexShrink: 0,
                            }} />
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
