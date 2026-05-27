'use client';

/**
 * Share Camera Page — Stream your phone/browser camera to SentinelAI.
 * Redesigned to match SentinelAI's pitch-black glassmorphism theme.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { API_BASE } from '@/lib/api';

function generateSessionId() {
    return 'bc_' + Math.random().toString(36).substring(2, 10) + '_' + Date.now().toString(36);
}

/* ── Inline SVG Icons ─────────────────────────────────────────── */
const Icons = {
    signal: (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M2 20h.01"/><path d="M7 20v-4"/><path d="M12 20v-8"/><path d="M17 20V8"/><path d="M22 4v16"/>
        </svg>
    ),
    camera: (
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/>
            <circle cx="12" cy="13" r="3"/>
        </svg>
    ),
    play: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
    ),
    stop: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="6" y="6" width="12" height="12" rx="2"/>
        </svg>
    ),
    switchCam: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M11 19H4a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h5"/><path d="M13 5h7a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-5"/>
            <polyline points="16 3 18 5 16 7"/><polyline points="8 17 6 19 8 21"/>
        </svg>
    ),
    shield: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        </svg>
    ),
    settings: (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
        </svg>
    ),
    info: (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>
        </svg>
    ),
};

export default function ShareCameraPage() {
    const [status, setStatus] = useState('idle');
    const [errorMsg, setErrorMsg] = useState('');
    const [sessionId, setSessionId] = useState('');
    const [cameraLabel, setCameraLabel] = useState('Remote Camera');
    const [facingMode, setFacingMode] = useState('environment');
    const [frameCount, setFrameCount] = useState(0);
    const [fps, setFps] = useState(0);
    const [quality, setQuality] = useState(0.6);
    const [targetFps, setTargetFps] = useState(10);

    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const wsRef = useRef(null);
    const streamRef = useRef(null);
    const intervalRef = useRef(null);
    const fpsCounterRef = useRef({ count: 0, lastTime: Date.now() });

    const stopStreaming = useCallback(() => {
        if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
        if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
        if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
        if (videoRef.current) { videoRef.current.srcObject = null; }
        setStatus('stopped');
    }, []);

    const startStreaming = useCallback(async () => {
        setStatus('connecting');
        setErrorMsg('');
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode, width: { ideal: 640 }, height: { ideal: 480 } },
                audio: false,
            });
            streamRef.current = stream;
            if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); }

            const wsUrl = API_BASE.replace(/^http/, 'ws');
            const ws = new WebSocket(`${wsUrl}/api/browser-cam/stream/${sessionId}?label=${encodeURIComponent(cameraLabel)}`);
            ws.binaryType = 'arraybuffer';

            ws.onopen = () => {
                setStatus('streaming');
                const canvas = canvasRef.current;
                const video = videoRef.current;
                const ctx = canvas.getContext('2d', { willReadFrequently: true });
                intervalRef.current = setInterval(() => {
                    if (!video || video.readyState < 2) return;
                    canvas.width = video.videoWidth || 640;
                    canvas.height = video.videoHeight || 480;
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    canvas.toBlob((blob) => {
                        if (blob && ws.readyState === WebSocket.OPEN) {
                            blob.arrayBuffer().then(buf => {
                                ws.send(buf);
                                setFrameCount(prev => prev + 1);
                                fpsCounterRef.current.count++;
                                const now = Date.now();
                                if (now - fpsCounterRef.current.lastTime >= 1000) {
                                    setFps(fpsCounterRef.current.count);
                                    fpsCounterRef.current.count = 0;
                                    fpsCounterRef.current.lastTime = now;
                                }
                            });
                        }
                    }, 'image/jpeg', quality);
                }, 1000 / targetFps);
            };
            ws.onerror = () => { setErrorMsg('WebSocket connection failed. Is the backend running?'); setStatus('error'); };
            ws.onclose = () => { if (status === 'streaming') setStatus('stopped'); };
            wsRef.current = ws;
        } catch (err) {
            setErrorMsg(err.message || 'Camera access denied');
            setStatus('error');
        }
    }, [facingMode, sessionId, cameraLabel, quality, targetFps]);

    useEffect(() => { setSessionId(generateSessionId()); }, []);
    useEffect(() => {
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
            if (wsRef.current) wsRef.current.close();
            if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
        };
    }, []);

    const toggleCamera = useCallback(() => {
        const newMode = facingMode === 'environment' ? 'user' : 'environment';
        setFacingMode(newMode);
        if (status === 'streaming') { stopStreaming(); setTimeout(() => startStreaming(), 500); }
    }, [facingMode, status, stopStreaming, startStreaming]);

    const isActive = status === 'streaming';
    const statusColor = { idle: 'var(--text-muted)', connecting: 'var(--warning)', streaming: 'var(--success)', error: 'var(--danger)', stopped: 'var(--text-muted)' }[status];
    const statusText = { idle: 'Ready to stream', connecting: 'Connecting...', streaming: `Streaming · ${fps} FPS`, error: 'Connection Error', stopped: 'Stopped' }[status];

    return (
        <div className="share-camera-page">
            <style>{`
                .share-camera-page {
                    min-height: 100vh;
                    background: var(--bg-primary);
                    color: var(--text-primary);
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    display: flex; flex-direction: column; align-items: center;
                    padding: 24px 16px; box-sizing: border-box;
                    position: relative; z-index: 1;
                }
                .sc-header { text-align: center; margin-bottom: 20px; }
                .sc-brand { display: inline-flex; align-items: center; gap: 10px; margin-bottom: 6px; }
                .sc-brand-icon {
                    width: 36px; height: 36px; border-radius: var(--radius-md);
                    background: var(--primary); display: flex; align-items: center;
                    justify-content: center; color: white;
                }
                .sc-title {
                    font-size: 1.375rem; font-weight: 700; letter-spacing: -0.02em;
                    color: var(--text-primary);
                }
                .sc-subtitle { font-size: 0.813rem; color: var(--text-secondary); margin: 0; }
                .sc-status-badge {
                    display: inline-flex; align-items: center; gap: 8px;
                    padding: 5px 14px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;
                    margin-bottom: 16px; letter-spacing: 0.02em;
                }
                .sc-dot {
                    width: 7px; height: 7px; border-radius: 50%;
                }
                .sc-dot.active { animation: sc-pulse 2s ease-in-out infinite; }
                @keyframes sc-pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

                .sc-video-card {
                    width: 100%; max-width: 500px; border-radius: var(--radius-lg);
                    overflow: hidden; position: relative; aspect-ratio: 4/3;
                    background: var(--bg-secondary); border: 1px solid var(--border-color);
                    margin-bottom: 16px; transition: border-color var(--transition-fast);
                }
                .sc-video-card.live { border-color: rgba(16,185,129,0.35); }
                .sc-video-card video {
                    width: 100%; height: 100%; object-fit: cover; display: block;
                }
                .sc-overlay {
                    position: absolute; inset: 0; display: flex; flex-direction: column;
                    align-items: center; justify-content: center;
                    background: rgba(0,0,0,0.65); backdrop-filter: blur(4px);
                    color: var(--text-secondary);
                }
                .sc-overlay-icon { color: var(--text-muted); margin-bottom: 10px; opacity: 0.6; }
                .sc-overlay-text { font-size: 0.875rem; }
                .sc-live-badge {
                    position: absolute; top: 10px; left: 10px;
                    padding: 4px 10px; border-radius: var(--radius-sm);
                    background: rgba(0,0,0,0.65); backdrop-filter: blur(6px);
                    border: 1px solid rgba(255,255,255,0.06);
                    font-size: 0.688rem; font-weight: 600; color: var(--success);
                    display: flex; align-items: center; gap: 10px;
                }
                .sc-live-frames { color: var(--text-muted); font-weight: 500; }

                .sc-input {
                    width: 100%; max-width: 500px; margin-bottom: 12px;
                }
                .sc-input input {
                    width: 100%; padding: 11px 16px; border-radius: var(--radius-md);
                    border: 1px solid var(--border-color); background: var(--bg-secondary);
                    color: var(--text-primary); font-size: 0.875rem; font-family: inherit;
                    outline: none; box-sizing: border-box; transition: border-color var(--transition-fast);
                }
                .sc-input input:focus { border-color: var(--primary); }
                .sc-input label {
                    display: block; font-size: 0.688rem; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.08em;
                    color: var(--text-muted); margin-bottom: 6px;
                }

                .sc-controls {
                    display: flex; gap: 8px; width: 100%; max-width: 500px; margin-bottom: 16px;
                }
                .sc-btn-start {
                    flex: 1; padding: 12px; border-radius: var(--radius-md); border: none;
                    cursor: pointer; font-weight: 600; font-size: 0.875rem; font-family: inherit;
                    color: white; background: var(--primary);
                    display: flex; align-items: center; justify-content: center; gap: 8px;
                    transition: all var(--transition-fast);
                }
                .sc-btn-start:hover { background: var(--primary-light); }
                .sc-btn-stop {
                    flex: 1; padding: 12px; border-radius: var(--radius-md); border: none;
                    cursor: pointer; font-weight: 600; font-size: 0.875rem; font-family: inherit;
                    color: var(--danger-light); background: rgba(239,68,68,0.1);
                    border: 1px solid rgba(239,68,68,0.15);
                    display: flex; align-items: center; justify-content: center; gap: 8px;
                    transition: all var(--transition-fast);
                }
                .sc-btn-stop:hover { background: rgba(239,68,68,0.18); }
                .sc-btn-switch {
                    padding: 12px 14px; border-radius: var(--radius-md);
                    border: 1px solid var(--border-color); cursor: pointer;
                    background: var(--bg-secondary); color: var(--text-secondary);
                    display: flex; align-items: center; justify-content: center;
                    transition: all var(--transition-fast);
                }
                .sc-btn-switch:hover { border-color: var(--border-hover); color: var(--text-primary); }

                .sc-settings-card {
                    width: 100%; max-width: 500px; padding: 16px;
                    border-radius: var(--radius-lg); background: var(--bg-card);
                    border: 1px solid var(--border-color); margin-bottom: 12px;
                }
                .sc-settings-title {
                    font-size: 0.688rem; font-weight: 600; text-transform: uppercase;
                    letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 14px;
                    display: flex; align-items: center; gap: 6px;
                }
                .sc-setting-row {
                    display: flex; justify-content: space-between; align-items: center;
                    padding: 6px 0;
                }
                .sc-setting-row + .sc-setting-row { border-top: 1px solid var(--border-color); padding-top: 10px; margin-top: 4px; }
                .sc-setting-label { font-size: 0.813rem; color: var(--text-secondary); }
                .sc-select {
                    padding: 5px 10px; border-radius: var(--radius-sm);
                    background: var(--bg-primary); border: 1px solid var(--border-color);
                    color: var(--text-primary); font-size: 0.75rem; font-family: inherit;
                    outline: none; cursor: pointer;
                }
                .sc-select:focus { border-color: var(--primary); }

                .sc-info-card {
                    width: 100%; max-width: 500px; padding: 14px 16px;
                    border-radius: var(--radius-lg); background: var(--bg-card);
                    border: 1px solid var(--border-color);
                    font-size: 0.688rem; color: var(--text-muted);
                    font-family: 'JetBrains Mono', monospace;
                }
                .sc-info-title {
                    font-size: 0.688rem; font-weight: 600; text-transform: uppercase;
                    letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 10px;
                    display: flex; align-items: center; gap: 6px;
                    font-family: 'Inter', sans-serif;
                }
                .sc-info-row { display: flex; gap: 8px; padding: 2px 0; }
                .sc-info-label { color: var(--text-secondary); font-weight: 600; min-width: 60px; font-family: 'Inter', sans-serif; }
            `}</style>

            {/* Header */}
            <div className="sc-header">
                <div className="sc-brand">
                    <div className="sc-brand-icon">{Icons.shield}</div>
                    <span className="sc-title">SentinelAI Remote Camera</span>
                </div>
                <p className="sc-subtitle">Stream your camera to SentinelAI for AI-powered monitoring</p>
            </div>

            {/* Status Badge */}
            <div className="sc-status-badge" style={{
                background: `color-mix(in srgb, ${statusColor} 12%, transparent)`,
                border: `1px solid color-mix(in srgb, ${statusColor} 25%, transparent)`,
                color: statusColor,
            }}>
                <span className={`sc-dot ${isActive ? 'active' : ''}`} style={{ background: statusColor }} />
                {statusText}
            </div>

            {/* Video Preview */}
            <div className={`sc-video-card ${isActive ? 'live' : ''}`}>
                <video
                    ref={videoRef} autoPlay playsInline muted
                    style={{ transform: facingMode === 'user' ? 'scaleX(-1)' : 'none' }}
                />
                <canvas ref={canvasRef} style={{ display: 'none' }} />

                {isActive && (
                    <div className="sc-live-badge">
                        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <span className="sc-dot active" style={{ background: 'var(--success)', width: 6, height: 6 }} />
                            LIVE
                        </span>
                        <span className="sc-live-frames">{frameCount.toLocaleString()} frames</span>
                    </div>
                )}

                {!isActive && status !== 'connecting' && (
                    <div className="sc-overlay">
                        <div className="sc-overlay-icon">{Icons.camera}</div>
                        <div className="sc-overlay-text">
                            {status === 'error' ? errorMsg : 'Tap "Start Streaming" to begin'}
                        </div>
                    </div>
                )}

                {status === 'connecting' && (
                    <div className="sc-overlay">
                        <div className="sc-overlay-icon" style={{ animation: 'sc-pulse 1s ease-in-out infinite' }}>{Icons.signal}</div>
                        <div className="sc-overlay-text">Connecting to server...</div>
                    </div>
                )}
            </div>

            {/* Camera Label */}
            {(status === 'idle' || status === 'stopped' || status === 'error') && (
                <div className="sc-input">
                    <label>Camera Name</label>
                    <input
                        type="text" value={cameraLabel}
                        onChange={e => setCameraLabel(e.target.value)}
                        placeholder="e.g. Front Gate, Parking Lot"
                    />
                </div>
            )}

            {/* Controls */}
            <div className="sc-controls">
                {status === 'idle' || status === 'stopped' || status === 'error' ? (
                    <button className="sc-btn-start" onClick={startStreaming}>
                        {Icons.play} Start Streaming
                    </button>
                ) : (
                    <button className="sc-btn-stop" onClick={stopStreaming}>
                        {Icons.stop} Stop Streaming
                    </button>
                )}
                <button className="sc-btn-switch" onClick={toggleCamera} title="Switch camera">
                    {Icons.switchCam}
                </button>
            </div>

            {/* Stream Settings */}
            {isActive && (
                <div className="sc-settings-card">
                    <div className="sc-settings-title">{Icons.settings} Stream Settings</div>
                    <div className="sc-setting-row">
                        <span className="sc-setting-label">Quality</span>
                        <select className="sc-select" value={quality} onChange={e => setQuality(Number(e.target.value))}>
                            <option value={0.3}>Low (saves data)</option>
                            <option value={0.5}>Medium</option>
                            <option value={0.6}>Good</option>
                            <option value={0.8}>High</option>
                        </select>
                    </div>
                    <div className="sc-setting-row">
                        <span className="sc-setting-label">Frame Rate</span>
                        <select className="sc-select" value={targetFps} onChange={e => setTargetFps(Number(e.target.value))}>
                            <option value={5}>5 FPS (saves data)</option>
                            <option value={10}>10 FPS</option>
                            <option value={15}>15 FPS</option>
                        </select>
                    </div>
                </div>
            )}

            {/* Connection Info */}
            <div className="sc-info-card">
                <div className="sc-info-title">{Icons.info} Connection Details</div>
                <div className="sc-info-row"><span className="sc-info-label">Session</span><span>{sessionId}</span></div>
                <div className="sc-info-row"><span className="sc-info-label">Server</span><span>{API_BASE}</span></div>
                <div className="sc-info-row"><span className="sc-info-label">Camera</span><span>{facingMode === 'environment' ? 'Rear Camera' : 'Front Camera'}</span></div>
            </div>
        </div>
    );
}
