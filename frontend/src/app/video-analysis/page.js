'use client';

/**
 * Video Analysis Page — Upload and analyze videos using AI.
 *
 * Processes uploaded video files through the same vision pipeline
 * used for live camera feeds (face recognition, ANPR, hazard detection, etc.)
 * and presents a comprehensive analysis report.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import AppShell from '@/components/AppShell';
import api from '@/lib/api';
import {
    VideoUploadIcon,
    UsersIcon,
    AlertIcon,
    ClockIcon,
    CheckCircleIcon,
    TrashIcon,
    RefreshIcon,
    CameraIcon,
    SecurityIcon,
    VehicleIcon,
    PlateIcon,
    AnalyticsIcon,
    UserIcon,
    EventLogIcon,
    ImageIcon,
    HazardIcon,
    SearchIcon,
    EntryIcon,
    ExitIcon,
    DetectionIcon,
} from '@/components/Icons';

// ── Tab Configuration ───────────────────────────────────────────
const TABS = [
    { id: 'summary', label: 'Summary', Icon: AnalyticsIcon },
    { id: 'persons', label: 'Persons', Icon: UserIcon },
    { id: 'vehicles', label: 'Vehicles', Icon: VehicleIcon },
    { id: 'alerts', label: 'Alerts', Icon: AlertIcon },
    { id: 'timeline', label: 'Timeline', Icon: EventLogIcon },
    { id: 'keyframes', label: 'Key Frames', Icon: ImageIcon },
];

// ── Utility Functions ───────────────────────────────────────────
function formatDuration(seconds) {
    if (!seconds || seconds <= 0) return '0s';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.round(seconds % 60);
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}

function formatVideoTime(seconds) {
    if (!seconds || seconds <= 0) return '0:00';
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function formatFileSize(bytes) {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ═══════════════════════════════════════════════════════════════
//  Main Page Component
// ═══════════════════════════════════════════════════════════════

export default function VideoAnalysisPage() {
    // ── State ────────────────────────────────────────────────
    const [mode, setMode] = useState('upload'); // 'upload' | 'processing' | 'results' | 'history'
    const [uploadProgress, setUploadProgress] = useState(0);
    const [analysisProgress, setAnalysisProgress] = useState(0);
    const [analysisStatus, setAnalysisStatus] = useState('');
    const [currentJobId, setCurrentJobId] = useState(null);
    const [results, setResults] = useState(null);
    const [history, setHistory] = useState([]);
    const [activeTab, setActiveTab] = useState('summary');
    const [error, setError] = useState(null);
    const [dragOver, setDragOver] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);

    const fileInputRef = useRef(null);
    const wsRef = useRef(null);
    const pollRef = useRef(null);

    // ── Load History on Mount ────────────────────────────────
    useEffect(() => {
        loadHistory();
        return () => {
            if (wsRef.current) wsRef.current.close();
            if (pollRef.current) clearInterval(pollRef.current);
        };
    }, []);

    async function loadHistory() {
        try {
            const data = await api.getAnalysisHistory();
            setHistory(data.analyses || []);
        } catch (e) {
            console.warn('Could not load history:', e.message);
        }
    }

    // ── File Handling ────────────────────────────────────────
    const ALLOWED_EXTENSIONS = ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm'];

    function validateFile(file) {
        const ext = file.name.split('.').pop()?.toLowerCase();
        if (!ALLOWED_EXTENSIONS.includes(ext)) {
            return `Invalid file type ".${ext}". Allowed: ${ALLOWED_EXTENSIONS.join(', ')}`;
        }
        if (file.size > 500 * 1024 * 1024) {
            return 'File too large. Maximum size: 500 MB';
        }
        return null;
    }

    const handleFileSelect = useCallback((file) => {
        const err = validateFile(file);
        if (err) {
            setError(err);
            return;
        }
        setError(null);
        setSelectedFile(file);
    }, []);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files[0];
        if (file) handleFileSelect(file);
    }, [handleFileSelect]);

    const handleDragOver = useCallback((e) => {
        e.preventDefault();
        setDragOver(true);
    }, []);

    const handleDragLeave = useCallback(() => {
        setDragOver(false);
    }, []);

    // ── Upload & Analyze ─────────────────────────────────────
    async function startAnalysis() {
        if (!selectedFile) return;
        setError(null);
        setMode('processing');
        setUploadProgress(0);
        setAnalysisProgress(0);
        setAnalysisStatus('uploading');

        try {
            // Upload file
            const uploadResult = await api.uploadVideo(selectedFile, (progress) => {
                setUploadProgress(progress);
            });

            const jobId = uploadResult.job_id;
            setCurrentJobId(jobId);
            setAnalysisStatus('processing');

            // Try WebSocket for real-time progress
            try {
                const wsUrl = API_BASE.replace(/^http/, 'ws');
                const ws = new WebSocket(`${wsUrl}/api/video-analysis/${jobId}/stream`);
                wsRef.current = ws;

                ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        setAnalysisProgress(data.progress || 0);
                        setAnalysisStatus(data.status || 'processing');

                        if (data.completed || data.status === 'completed') {
                            ws.close();
                            loadResults(jobId);
                        } else if (data.status === 'failed') {
                            ws.close();
                            setError(data.error || 'Analysis failed');
                            setMode('upload');
                        }
                    } catch (e) {
                        console.warn('WS parse error:', e);
                    }
                };

                ws.onerror = () => {
                    // Fallback to polling
                    startPolling(jobId);
                };

                ws.onclose = () => {
                    wsRef.current = null;
                };
            } catch {
                // Fallback to polling
                startPolling(jobId);
            }

        } catch (e) {
            setError(e.message || 'Upload failed');
            setMode('upload');
        }
    }

    function startPolling(jobId) {
        if (pollRef.current) clearInterval(pollRef.current);

        pollRef.current = setInterval(async () => {
            try {
                const status = await api.getAnalysisStatus(jobId);
                setAnalysisProgress(status.progress || 0);
                setAnalysisStatus(status.status || 'processing');

                if (status.status === 'completed') {
                    clearInterval(pollRef.current);
                    pollRef.current = null;
                    loadResults(jobId);
                } else if (status.status === 'failed') {
                    clearInterval(pollRef.current);
                    pollRef.current = null;
                    setError(status.error || 'Analysis failed');
                    setMode('upload');
                }
            } catch (e) {
                console.warn('Poll error:', e.message);
            }
        }, 1000);
    }

    async function loadResults(jobId) {
        try {
            const data = await api.getAnalysisResults(jobId);
            setResults(data);
            setMode('results');
            loadHistory();
        } catch (e) {
            setError('Failed to load results: ' + e.message);
            setMode('upload');
        }
    }

    async function viewHistoryResult(jobId) {
        setError(null);
        try {
            const data = await api.getAnalysisResults(jobId);
            if (data.status === 'processing') {
                setCurrentJobId(jobId);
                setAnalysisProgress(data.progress || 0);
                setAnalysisStatus('processing');
                setMode('processing');
                startPolling(jobId);
            } else {
                setResults(data);
                setCurrentJobId(jobId);
                setMode('results');
            }
        } catch (e) {
            setError('Failed to load results: ' + e.message);
        }
    }

    async function deleteJob(jobId) {
        try {
            await api.deleteAnalysis(jobId);
            loadHistory();
            if (currentJobId === jobId) {
                setMode('upload');
                setResults(null);
                setCurrentJobId(null);
            }
        } catch (e) {
            setError('Failed to delete: ' + e.message);
        }
    }

    function resetToUpload() {
        setMode('upload');
        setSelectedFile(null);
        setResults(null);
        setCurrentJobId(null);
        setError(null);
        setUploadProgress(0);
        setAnalysisProgress(0);
    }

    // ═══════════════════════════════════════════════════════════
    //  RENDER
    // ═══════════════════════════════════════════════════════════

    return (
        <AppShell>
            <div className="page-header">
                <h1>Video Analysis</h1>
                <p>
                    Upload a video and let AI analyze it — detect faces, read license plates, identify threats
                    <span style={{
                        display: 'inline-flex', alignItems: 'center', gap: '6px', marginLeft: '12px',
                    }}>
                        <span style={{
                            width: '8px', height: '8px', borderRadius: '50%',
                            background: 'var(--accent)', animation: 'pulse-dot 2s infinite',
                        }}></span>
                        <span style={{
                            fontSize: '0.75rem', color: 'var(--accent-light)', fontWeight: 600,
                        }}>AI POWERED</span>
                    </span>
                </p>
            </div>

            {/* Error Banner */}
            {error && (
                <div style={{
                    background: 'rgba(239, 68, 68, 0.1)',
                    border: '1px solid rgba(239, 68, 68, 0.3)',
                    borderRadius: '12px',
                    padding: '14px 18px',
                    marginBottom: '20px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    color: 'var(--danger-light)',
                    fontSize: '0.875rem',
                }}>
                    <AlertIcon size={16} />
                    {error}
                    <button
                        onClick={() => setError(null)}
                        style={{
                            marginLeft: 'auto', background: 'none', border: 'none',
                            color: 'var(--danger-light)', cursor: 'pointer', fontSize: '1.2rem',
                        }}
                    >×</button>
                </div>
            )}

            {/* Mode: Upload */}
            {mode === 'upload' && (
                <>
                    {/* Upload Zone */}
                    <div
                        className="card"
                        style={{
                            marginBottom: '24px',
                            padding: '0',
                            overflow: 'hidden',
                        }}
                    >
                        <div
                            onDrop={handleDrop}
                            onDragOver={handleDragOver}
                            onDragLeave={handleDragLeave}
                            onClick={() => fileInputRef.current?.click()}
                            style={{
                                padding: selectedFile ? '40px 30px' : '80px 30px',
                                textAlign: 'center',
                                cursor: 'pointer',
                                border: `2px dashed ${dragOver ? 'var(--accent)' : 'var(--border-color)'}`,
                                borderRadius: '12px',
                                margin: '20px',
                                background: dragOver
                                    ? 'rgba(99, 102, 241, 0.05)'
                                    : 'var(--hover-surface)',
                                transition: 'all 0.3s ease',
                            }}
                        >
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept=".mp4,.avi,.mkv,.mov,.wmv,.flv,.webm"
                                style={{ display: 'none' }}
                                onChange={(e) => {
                                    const file = e.target.files?.[0];
                                    if (file) handleFileSelect(file);
                                }}
                            />

                            {selectedFile ? (
                                <>
                                    <div style={{
                                        width: '64px', height: '64px', borderRadius: '50%',
                                        background: 'linear-gradient(135deg, var(--accent), var(--primary))',
                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                        margin: '0 auto 20px', boxShadow: '0 8px 30px rgba(99, 102, 241, 0.3)',
                                    }}>
                                        <VideoUploadIcon size={28} style={{ color: 'var(--text-inverse)' }} />
                                    </div>
                                    <div style={{
                                        fontWeight: 600, fontSize: '1.125rem',
                                        color: 'var(--text-primary)', marginBottom: '6px',
                                    }}>
                                        {selectedFile.name}
                                    </div>
                                    <div style={{
                                        fontSize: '0.875rem', color: 'var(--text-muted)', marginBottom: '20px',
                                    }}>
                                        {formatFileSize(selectedFile.size)} • Click to change file
                                    </div>
                                    <button
                                        className="btn btn-primary"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            startAnalysis();
                                        }}
                                        style={{
                                            padding: '12px 40px',
                                            fontSize: '1rem',
                                            fontWeight: 600,
                                            borderRadius: '12px',
                                            background: 'linear-gradient(135deg, var(--accent), var(--primary))',
                                            boxShadow: '0 4px 20px rgba(99, 102, 241, 0.3)',
                                        }}
                                    >
                                        <VideoUploadIcon size={16} style={{ marginRight: '4px' }} /> Start AI Analysis
                                    </button>
                                </>
                            ) : (
                                <>
                                    <div style={{
                                        width: '80px', height: '80px', borderRadius: '50%',
                                        background: 'rgba(99, 102, 241, 0.1)',
                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                        margin: '0 auto 24px',
                                    }}>
                                        <VideoUploadIcon size={36} style={{ color: 'var(--accent)' }} />
                                    </div>
                                    <div style={{
                                        fontWeight: 600, fontSize: '1.25rem',
                                        color: 'var(--text-primary)', marginBottom: '8px',
                                    }}>
                                        Drop your video here or click to browse
                                    </div>
                                    <div style={{
                                        fontSize: '0.875rem', color: 'var(--text-muted)',
                                        marginBottom: '24px', lineHeight: 1.6,
                                    }}>
                                        Supports MP4, AVI, MKV, MOV, WMV, FLV, WebM • Max 500 MB
                                    </div>
                                    <div style={{
                                        display: 'flex', gap: '20px', justifyContent: 'center',
                                        flexWrap: 'wrap', fontSize: '0.75rem', color: 'var(--text-muted)',
                                    }}>
                                        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                            <UsersIcon size={14} /> Face Recognition
                                        </span>
                                        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                            <PlateIcon size={14} /> License Plates
                                        </span>
                                        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                            <SecurityIcon size={14} /> Threat Detection
                                        </span>
                                        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                            <VehicleIcon size={14} /> Vehicle Tracking
                                        </span>
                                    </div>
                                </>
                            )}
                        </div>
                    </div>

                    {/* History Section */}
                    {history.length > 0 && (
                        <div className="card">
                            <div style={{
                                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                marginBottom: '16px',
                            }}>
                                <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 600 }}>
                                    Past Analyses
                                </h3>
                                <button className="btn btn-secondary btn-sm" onClick={loadHistory}>
                                    <RefreshIcon size={14} /> Refresh
                                </button>
                            </div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                                {history.map((job) => (
                                    <div
                                        key={job.job_id}
                                        className="animate-in"
                                        style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '14px',
                                            padding: '14px 16px',
                                            background: 'var(--hover-surface)',
                                            borderRadius: '10px',
                                            border: '1px solid var(--border-color)',
                                            cursor: job.status === 'completed' || job.status === 'processing' ? 'pointer' : 'default',
                                            transition: 'all 0.2s ease',
                                        }}
                                        onClick={() => {
                                            if (job.status === 'completed' || job.status === 'processing') {
                                                viewHistoryResult(job.job_id);
                                            }
                                        }}
                                        onMouseEnter={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
                                        onMouseLeave={(e) => e.currentTarget.style.borderColor = 'var(--border-color)'}
                                    >
                                        <div style={{
                                            width: '40px', height: '40px', borderRadius: '10px',
                                            background: job.status === 'completed'
                                                ? 'rgba(34, 197, 94, 0.1)'
                                                : job.status === 'processing'
                                                    ? 'rgba(99, 102, 241, 0.1)'
                                                    : 'rgba(239, 68, 68, 0.1)',
                                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                                            flexShrink: 0,
                                        }}>
                                            {job.status === 'completed' && <CheckCircleIcon size={18} style={{ color: 'var(--success)' }} />}
                                            {job.status === 'processing' && (
                                                <div style={{
                                                    width: '18px', height: '18px',
                                                    border: '2px solid var(--border-color)',
                                                    borderTopColor: 'var(--accent)',
                                                    borderRadius: '50%',
                                                    animation: 'spin 1s linear infinite',
                                                }} />
                                            )}
                                            {job.status === 'failed' && <AlertIcon size={18} style={{ color: 'var(--danger)' }} />}
                                        </div>

                                        <div style={{ flex: 1, minWidth: 0 }}>
                                            <div style={{
                                                fontWeight: 500, fontSize: '0.875rem',
                                                color: 'var(--text-primary)',
                                                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                                            }}>
                                                {job.filename}
                                            </div>
                                            <div style={{
                                                fontSize: '0.75rem', color: 'var(--text-muted)',
                                                display: 'flex', gap: '8px', marginTop: '2px',
                                            }}>
                                                <span>{job.video_metadata?.duration_seconds
                                                    ? formatDuration(job.video_metadata.duration_seconds)
                                                    : '—'}</span>
                                                <span>•</span>
                                                <span>{job.status === 'processing'
                                                    ? `${Math.round(job.progress)}%`
                                                    : job.status}</span>
                                                {job.summary?.unique_persons_recognized > 0 && (
                                                    <>
                                                        <span>•</span>
                                                        <span>{job.summary.unique_persons_recognized} persons</span>
                                                    </>
                                                )}
                                            </div>
                                        </div>

                                        <button
                                            className="btn btn-secondary btn-sm"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                deleteJob(job.job_id);
                                            }}
                                            style={{ padding: '6px 8px' }}
                                        >
                                            <TrashIcon size={14} />
                                        </button>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </>
            )}

            {/* Mode: Processing */}
            {mode === 'processing' && (
                <div className="card" style={{ padding: '60px 30px', textAlign: 'center' }}>
                    <div style={{
                        width: '100px', height: '100px', borderRadius: '50%',
                        background: 'linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.15))',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        margin: '0 auto 28px',
                        position: 'relative',
                    }}>
                        <div style={{
                            position: 'absolute', inset: '-4px',
                            border: '3px solid transparent',
                            borderTopColor: 'var(--accent)',
                            borderRadius: '50%',
                            animation: 'spin 1.5s linear infinite',
                        }} />
                        <VideoUploadIcon size={42} style={{ color: 'var(--accent)' }} />
                    </div>

                    <div style={{
                        fontWeight: 700, fontSize: '1.5rem',
                        color: 'var(--text-primary)', marginBottom: '8px',
                    }}>
                        {analysisStatus === 'uploading' ? 'Uploading Video...' : 'Analyzing Video...'}
                    </div>

                    <div style={{
                        fontSize: '0.875rem', color: 'var(--text-muted)',
                        marginBottom: '32px', maxWidth: '400px', margin: '0 auto 32px',
                    }}>
                        {analysisStatus === 'uploading'
                            ? 'Uploading your video to the server...'
                            : 'AI is processing each frame — detecting faces, reading plates, identifying threats...'}
                    </div>

                    {/* Progress Bar */}
                    <div style={{
                        maxWidth: '500px', margin: '0 auto 16px',
                    }}>
                        <div style={{
                            display: 'flex', justifyContent: 'space-between',
                            fontSize: '0.813rem', color: 'var(--text-muted)', marginBottom: '8px',
                        }}>
                            <span>{analysisStatus === 'uploading' ? 'Upload Progress' : 'Analysis Progress'}</span>
                            <span style={{ fontWeight: 600, color: 'var(--accent-light)' }}>
                                {analysisStatus === 'uploading'
                                    ? `${uploadProgress}%`
                                    : `${Math.round(analysisProgress)}%`}
                            </span>
                        </div>
                        <div style={{
                            height: '10px',
                            background: 'var(--hover-surface)',
                            borderRadius: '10px',
                            overflow: 'hidden',
                        }}>
                            <div style={{
                                height: '100%',
                                width: `${analysisStatus === 'uploading' ? uploadProgress : analysisProgress}%`,
                                background: 'linear-gradient(90deg, var(--accent), var(--primary))',
                                borderRadius: '10px',
                                transition: 'width 0.5s ease',
                                boxShadow: '0 0 12px rgba(99, 102, 241, 0.4)',
                            }} />
                        </div>
                    </div>

                    <div style={{
                        display: 'flex', gap: '24px', justifyContent: 'center',
                        fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '24px',
                    }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <SearchIcon size={13} /> Face Detection
                        </span>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <PlateIcon size={13} /> ANPR
                        </span>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <HazardIcon size={13} /> Hazard Scan
                        </span>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <AnalyticsIcon size={13} /> Analytics
                        </span>
                    </div>
                </div>
            )}

            {/* Mode: Results */}
            {mode === 'results' && results && (
                <>
                    {/* Header Bar */}
                    <div style={{
                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                        marginBottom: '20px', flexWrap: 'wrap', gap: '12px',
                    }}>
                        <div>
                            <div style={{
                                fontWeight: 600, fontSize: '1rem', color: 'var(--text-primary)',
                                display: 'flex', alignItems: 'center', gap: '8px',
                            }}>
                                <CheckCircleIcon size={18} style={{ color: 'var(--success)' }} />
                                {results.filename}
                            </div>
                            <div style={{ fontSize: '0.813rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                                {formatDuration(results.video_metadata?.duration_seconds)} •{' '}
                                {results.video_metadata?.resolution?.width}×{results.video_metadata?.resolution?.height} •{' '}
                                {results.video_metadata?.fps} fps
                            </div>
                        </div>
                        <div style={{ display: 'flex', gap: '8px' }}>
                            <button className="btn btn-secondary btn-sm" onClick={resetToUpload}>
                                <VideoUploadIcon size={14} /> New Analysis
                            </button>
                        </div>
                    </div>

                    {/* Summary Stats */}
                    <div className="security-stats" style={{ marginBottom: '20px' }}>
                        <div className="security-stat info">
                            <div className="stat-number">{results.summary?.unique_persons_recognized || 0}</div>
                            <div className="stat-label">Persons Found</div>
                        </div>
                        <div className="security-stat total">
                            <div className="stat-number">{results.summary?.unknown_faces_detected || 0}</div>
                            <div className="stat-label">Unknown Faces</div>
                        </div>
                        <div className="security-stat warning">
                            <div className="stat-number">{results.summary?.vehicles_detected || 0}</div>
                            <div className="stat-label">Vehicles</div>
                        </div>
                        <div className="security-stat danger">
                            <div className="stat-number">{results.summary?.security_alerts || 0}</div>
                            <div className="stat-label">Alerts</div>
                        </div>
                    </div>

                    {/* Tab Bar */}
                    <div className="card" style={{ marginBottom: '20px', padding: '0' }}>
                        <div style={{
                            display: 'flex', gap: '0', overflowX: 'auto',
                            borderBottom: '1px solid var(--border-color)',
                        }}>
                            {TABS.map((tab) => (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id)}
                                    style={{
                                        padding: '14px 20px',
                                        background: 'none',
                                        border: 'none',
                                        borderBottom: activeTab === tab.id
                                            ? '2px solid var(--accent)'
                                            : '2px solid transparent',
                                        color: activeTab === tab.id
                                            ? 'var(--accent-light)'
                                            : 'var(--text-muted)',
                                        fontSize: '0.875rem',
                                        fontWeight: activeTab === tab.id ? 600 : 400,
                                        cursor: 'pointer',
                                        whiteSpace: 'nowrap',
                                        transition: 'all 0.2s ease',
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '6px',
                                    }}
                                >
                                    <tab.Icon size={15} />
                                    {tab.label}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Tab Content */}
                    <div className="card animate-in">
                        {/* SUMMARY TAB */}
                        {activeTab === 'summary' && (
                            <div>
                                <h3 style={{ marginTop: 0, marginBottom: '20px', fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <AnalyticsIcon size={18} /> Analysis Summary
                                </h3>
                                <div style={{
                                    display: 'grid',
                                    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                                    gap: '16px',
                                }}>
                                    {[
                                        { label: 'Total Frames', value: results.summary?.total_frames?.toLocaleString() || '0' },
                                        { label: 'Processed Frames', value: results.summary?.processed_frames?.toLocaleString() || '0' },
                                        { label: 'Duration', value: formatDuration(results.summary?.duration_seconds) },
                                        { label: 'Known Persons', value: results.summary?.unique_persons_recognized || 0 },
                                        { label: 'Unknown Faces', value: results.summary?.unknown_faces_detected || 0 },
                                        { label: 'Vehicles Detected', value: results.summary?.vehicles_detected || 0 },
                                        { label: 'Security Alerts', value: results.summary?.security_alerts || 0 },
                                        { label: 'Traffic Alerts', value: results.summary?.traffic_alerts || 0 },
                                        { label: 'Total Events', value: results.summary?.total_events || 0 },
                                        { label: 'Key Frames', value: results.summary?.key_frames_saved || 0 },
                                    ].map((item) => (
                                        <div key={item.label} style={{
                                            padding: '16px',
                                            background: 'var(--hover-surface)',
                                            borderRadius: '10px',
                                            border: '1px solid var(--border-color)',
                                        }}>
                                            <div style={{
                                                fontSize: '1.5rem', fontWeight: 700,
                                                color: 'var(--text-primary)',
                                            }}>{item.value}</div>
                                            <div style={{
                                                fontSize: '0.75rem', color: 'var(--text-muted)',
                                                marginTop: '4px',
                                            }}>{item.label}</div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* PERSONS TAB */}
                        {activeTab === 'persons' && (
                            <div>
                                <h3 style={{ marginTop: 0, marginBottom: '20px', fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <UserIcon size={18} /> Detected Persons ({(results.detected_persons?.length || 0) + (results.unknown_persons?.length || 0)})
                                </h3>

                                {/* Known Persons */}
                                {results.detected_persons?.length > 0 && (
                                    <>
                                        <div style={{
                                            fontSize: '0.813rem', fontWeight: 600,
                                            color: 'var(--accent-light)', marginBottom: '12px',
                                            textTransform: 'uppercase', letterSpacing: '0.5px',
                                        }}>Recognized Persons</div>
                                        <div style={{
                                            display: 'grid',
                                            gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
                                            gap: '12px',
                                            marginBottom: '24px',
                                        }}>
                                            {results.detected_persons.map((person, idx) => (
                                                <div key={idx} className="animate-in" style={{
                                                    display: 'flex', gap: '14px', alignItems: 'center',
                                                    padding: '14px 16px',
                                                    background: 'var(--hover-surface)',
                                                    borderRadius: '10px',
                                                    border: '1px solid var(--border-color)',
                                                    animationDelay: `${idx * 30}ms`,
                                                }}>
                                                    {person.snapshot_url ? (
                                                        <img
                                                            src={`${API_BASE}${person.snapshot_url}`}
                                                            alt={person.person_name}
                                                            style={{
                                                                width: '50px', height: '50px',
                                                                borderRadius: '10px', objectFit: 'cover',
                                                                border: '2px solid var(--accent)',
                                                            }}
                                                        />
                                                    ) : (
                                                        <div style={{
                                                            width: '50px', height: '50px',
                                                            borderRadius: '10px',
                                                            background: 'linear-gradient(135deg, var(--accent), var(--primary))',
                                                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                            color: '#fff', fontWeight: 700, fontSize: '1.1rem',
                                                        }}>
                                                            {(person.person_name || '?')[0]?.toUpperCase()}
                                                        </div>
                                                    )}
                                                    <div style={{ flex: 1, minWidth: 0 }}>
                                                        <div style={{
                                                            fontWeight: 600, fontSize: '0.875rem',
                                                            color: 'var(--text-primary)',
                                                        }}>{person.person_name}</div>
                                                        <div style={{
                                                            fontSize: '0.75rem', color: 'var(--text-muted)',
                                                            display: 'flex', gap: '8px', marginTop: '2px',
                                                        }}>
                                                            <span>First: {formatVideoTime(person.first_seen_time)}</span>
                                                            <span>•</span>
                                                            <span>Last: {formatVideoTime(person.last_seen_time)}</span>
                                                        </div>
                                                        <div style={{
                                                            fontSize: '0.688rem', color: 'var(--text-muted)',
                                                            marginTop: '2px',
                                                        }}>
                                                            {person.appearances} appearances •
                                                            Confidence: {((person.max_confidence || 0) * 100).toFixed(0)}%
                                                            {person.attributes?.gender && ` • ${person.attributes.gender}`}
                                                        </div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </>
                                )}

                                {/* Unknown Persons */}
                                {results.unknown_persons?.length > 0 && (
                                    <>
                                        <div style={{
                                            fontSize: '0.813rem', fontWeight: 600,
                                            color: 'var(--danger-light)', marginBottom: '12px',
                                            textTransform: 'uppercase', letterSpacing: '0.5px',
                                        }}>Unknown Faces</div>
                                        <div style={{
                                            display: 'grid',
                                            gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
                                            gap: '12px',
                                        }}>
                                            {results.unknown_persons.map((person, idx) => (
                                                <div key={idx} className="animate-in" style={{
                                                    padding: '12px',
                                                    background: 'rgba(239, 68, 68, 0.05)',
                                                    borderRadius: '10px',
                                                    border: '1px solid rgba(239, 68, 68, 0.2)',
                                                    textAlign: 'center',
                                                    animationDelay: `${idx * 20}ms`,
                                                }}>
                                                    {person.snapshot_url ? (
                                                        <img
                                                            src={`${API_BASE}${person.snapshot_url}`}
                                                            alt="Unknown"
                                                            style={{
                                                                width: '80px', height: '80px',
                                                                borderRadius: '10px', objectFit: 'cover',
                                                                margin: '0 auto 8px', display: 'block',
                                                                border: '2px solid rgba(239, 68, 68, 0.3)',
                                                            }}
                                                        />
                                                    ) : (
                                                        <div style={{
                                                            width: '80px', height: '80px',
                                                            borderRadius: '10px',
                                                            background: 'rgba(239, 68, 68, 0.1)',
                                                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                            margin: '0 auto 8px', color: 'var(--danger)', fontSize: '1.5rem',
                                                        }}>?</div>
                                                    )}
                                                    <div style={{
                                                        fontSize: '0.75rem', color: 'var(--text-muted)',
                                                    }}>
                                                        at {formatVideoTime(person.video_time)}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </>
                                )}

                                {(!results.detected_persons?.length && !results.unknown_persons?.length) && (
                                    <EmptyState icon={<UserIcon size={40} />} title="No Persons Detected" description="No faces were found in this video." />
                                )}
                            </div>
                        )}

                        {/* VEHICLES TAB */}
                        {activeTab === 'vehicles' && (
                            <div>
                                <h3 style={{ marginTop: 0, marginBottom: '20px', fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <VehicleIcon size={18} /> Detected Vehicles ({results.detected_vehicles?.length || 0})
                                </h3>
                                {results.detected_vehicles?.length > 0 ? (
                                    <div style={{
                                        display: 'grid',
                                        gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
                                        gap: '12px',
                                    }}>
                                        {results.detected_vehicles.map((vehicle, idx) => (
                                            <div key={idx} className="animate-in" style={{
                                                padding: '18px',
                                                background: 'var(--hover-surface)',
                                                borderRadius: '10px',
                                                border: '1px solid var(--border-color)',
                                                animationDelay: `${idx * 30}ms`,
                                            }}>
                                                <div style={{
                                                    display: 'flex', alignItems: 'center', gap: '12px',
                                                    marginBottom: '10px',
                                                }}>
                                                    <div style={{
                                                        width: '42px', height: '42px', borderRadius: '10px',
                                                        background: 'rgba(234, 179, 8, 0.1)',
                                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                    }}>
                                                        <PlateIcon size={20} style={{ color: 'var(--warning-light)' }} />
                                                    </div>
                                                    <div>
                                                        <div style={{
                                                            fontWeight: 700, fontSize: '1.1rem',
                                                            color: 'var(--warning-light)',
                                                            letterSpacing: '1px', fontFamily: 'monospace',
                                                        }}>{vehicle.plate}</div>
                                                        <div style={{
                                                            fontSize: '0.75rem', color: 'var(--text-muted)',
                                                        }}>{vehicle.vehicle_type}</div>
                                                    </div>
                                                </div>
                                                <div style={{
                                                    fontSize: '0.75rem', color: 'var(--text-muted)',
                                                }}>
                                                    Confidence: {((vehicle.confidence || 0) * 100).toFixed(0)}%
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <EmptyState icon={<VehicleIcon size={40} />} title="No Vehicles Detected" description="No license plates were detected in this video. Make sure Traffic/ANPR mode is enabled." />
                                )}
                            </div>
                        )}

                        {/* ALERTS TAB */}
                        {activeTab === 'alerts' && (
                            <div>
                                <h3 style={{ marginTop: 0, marginBottom: '20px', fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <AlertIcon size={18} /> Security & Traffic Alerts ({(results.security_alerts?.length || 0) + (results.traffic_alerts?.length || 0)})
                                </h3>
                                {(results.security_alerts?.length > 0 || results.traffic_alerts?.length > 0) ? (
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                                        {[...(results.security_alerts || []), ...(results.traffic_alerts || [])].map((alert, idx) => (
                                            <div key={idx} className="animate-in" style={{
                                                padding: '14px 16px',
                                                background: 'rgba(239, 68, 68, 0.05)',
                                                borderRadius: '10px',
                                                border: '1px solid rgba(239, 68, 68, 0.2)',
                                                display: 'flex', gap: '12px', alignItems: 'center',
                                                animationDelay: `${idx * 30}ms`,
                                            }}>
                                                <div style={{
                                                    width: '36px', height: '36px', borderRadius: '8px',
                                                    background: 'rgba(239, 68, 68, 0.15)',
                                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                    flexShrink: 0,
                                                }}>
                                                    <AlertIcon size={16} style={{ color: 'var(--danger-light)' }} />
                                                </div>
                                                <div style={{ flex: 1 }}>
                                                    <div style={{
                                                        fontWeight: 500, fontSize: '0.875rem',
                                                        color: 'var(--text-primary)',
                                                    }}>
                                                        {alert.subtype || alert.metadata?.subtype || alert.event_type || 'Alert'}
                                                    </div>
                                                    <div style={{
                                                        fontSize: '0.75rem', color: 'var(--text-muted)',
                                                    }}>
                                                        {alert.alert_source || 'security'} •
                                                        Camera: {alert.camera_id || '—'}
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <EmptyState icon={<CheckCircleIcon size={40} style={{ color: 'var(--success)' }} />} title="No Alerts" description="No security or traffic alerts were triggered during this video." />
                                )}
                            </div>
                        )}

                        {/* TIMELINE TAB */}
                        {activeTab === 'timeline' && (
                            <div>
                                <h3 style={{ marginTop: 0, marginBottom: '20px', fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <EventLogIcon size={18} /> Event Timeline ({results.events_timeline?.length || 0})
                                </h3>
                                {results.events_timeline?.length > 0 ? (
                                    <div style={{
                                        position: 'relative',
                                        paddingLeft: '24px',
                                        borderLeft: '2px solid var(--border-color)',
                                        marginLeft: '12px',
                                    }}>
                                        {results.events_timeline.map((event, idx) => (
                                            <div key={idx} className="animate-in" style={{
                                                padding: '14px 18px',
                                                marginBottom: '12px',
                                                background: 'var(--hover-surface)',
                                                borderRadius: '10px',
                                                border: '1px solid var(--border-color)',
                                                position: 'relative',
                                                animationDelay: `${idx * 30}ms`,
                                            }}>
                                                {/* Timeline dot */}
                                                <div style={{
                                                    position: 'absolute', left: '-33px', top: '20px',
                                                    width: '10px', height: '10px', borderRadius: '50%',
                                                    background: (event.event_type === 'vehicle_entered' || event.event_type === 'person_entered' || event.event_type === 'entry') ? 'var(--success)'
                                                        : (event.event_type === 'vehicle_left' || event.event_type === 'person_left' || event.event_type === 'exit') ? 'var(--warning)'
                                                            : event.event_type === 'security_alert' ? 'var(--danger)'
                                                                : 'var(--accent)',
                                                    border: '2px solid var(--bg-elevated)',
                                                }} />

                                                <div style={{
                                                    display: 'flex', justifyContent: 'space-between',
                                                    alignItems: 'flex-start', gap: '12px',
                                                }}>
                                                    <div style={{ flex: 1 }}>
                                                        {/* Description or event type */}
                                                        <div style={{
                                                            fontWeight: 500, fontSize: '0.875rem',
                                                            color: 'var(--text-primary)',
                                                            lineHeight: 1.5,
                                                        }}>
                                                            {event.description || event.event_type || 'Event'}
                                                        </div>

                                                        {/* Video timestamp */}
                                                        {event.video_time && (
                                                            <div style={{
                                                                fontSize: '0.75rem', color: 'var(--text-muted)',
                                                                marginTop: '4px',
                                                                display: 'flex', alignItems: 'center', gap: '4px',
                                                            }}>
                                                                <ClockIcon size={10} />
                                                                Video time: {event.video_time}
                                                            </div>
                                                        )}

                                                        {/* Person name (for person events) */}
                                                        {event.person_name && (
                                                            <div style={{
                                                                fontSize: '0.75rem', color: 'var(--text-muted)',
                                                                marginTop: '2px',
                                                            }}>
                                                                Person: {event.person_name}
                                                            </div>
                                                        )}
                                                    </div>

                                                    {/* Badge */}
                                                    <span className={`badge ${(event.event_type === 'vehicle_entered' || event.event_type === 'person_entered' || event.event_type === 'entry') ? 'entry'
                                                        : (event.event_type === 'vehicle_left' || event.event_type === 'person_left' || event.event_type === 'exit') ? 'exit'
                                                            : event.event_type === 'security_alert' ? 'security-alert hazard'
                                                                : 'vehicle'
                                                        }`} style={{
                                                            fontSize: '0.625rem',
                                                            whiteSpace: 'nowrap',
                                                            flexShrink: 0,
                                                        }}>
                                                        {(event.event_type === 'vehicle_entered' || event.event_type === 'person_entered' || event.event_type === 'entry') ? 'entered'
                                                            : (event.event_type === 'vehicle_left' || event.event_type === 'person_left' || event.event_type === 'exit') ? 'left'
                                                                : event.event_type}
                                                    </span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <EmptyState icon={<EventLogIcon size={40} />} title="No Events" description="No entry/exit or detection events were generated." />
                                )}
                            </div>
                        )}

                        {/* KEY FRAMES TAB */}
                        {activeTab === 'keyframes' && (
                            <div>
                                <h3 style={{ marginTop: 0, marginBottom: '20px', fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <ImageIcon size={18} /> Key Frames ({results.key_frames?.length || 0})
                                </h3>
                                {results.key_frames?.length > 0 ? (
                                    <div style={{
                                        display: 'grid',
                                        gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
                                        gap: '14px',
                                    }}>
                                        {results.key_frames.map((frame, idx) => (
                                            <div key={idx} className="animate-in" style={{
                                                borderRadius: '10px',
                                                overflow: 'hidden',
                                                border: '1px solid var(--border-color)',
                                                animationDelay: `${idx * 30}ms`,
                                            }}>
                                                <img
                                                    src={`${API_BASE}${frame.snapshot_url}`}
                                                    alt={`Frame ${frame.frame}`}
                                                    style={{
                                                        width: '100%', height: '150px',
                                                        objectFit: 'cover', display: 'block',
                                                    }}
                                                />
                                                <div style={{
                                                    padding: '10px 12px',
                                                    background: 'var(--hover-surface)',
                                                    fontSize: '0.75rem',
                                                    color: 'var(--text-muted)',
                                                    display: 'flex',
                                                    justifyContent: 'space-between',
                                                }}>
                                                    <span>Frame #{frame.frame}</span>
                                                    <span>{formatVideoTime(frame.video_time)}</span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <EmptyState icon={<ImageIcon size={40} />} title="No Key Frames" description="No key frames were captured during analysis." />
                                )}
                            </div>
                        )}
                    </div>
                </>
            )}
        </AppShell>
    );
}


// ── Empty State Component ────────────────────────────────────────
function EmptyState({ icon, title, description }) {
    return (
        <div style={{
            padding: '60px 20px',
            textAlign: 'center',
        }}>
            <div style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                marginBottom: '16px', opacity: 0.3, color: 'var(--text-muted)',
            }}>{icon}</div>
            <div style={{
                fontWeight: 600, fontSize: '1.125rem',
                color: 'var(--text-secondary)', marginBottom: '8px',
            }}>{title}</div>
            <div style={{
                fontSize: '0.875rem', color: 'var(--text-muted)',
                maxWidth: '400px', margin: '0 auto', lineHeight: 1.6,
            }}>{description}</div>
        </div>
    );
}
