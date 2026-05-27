'use client';

/**
 * Camera Management Page
 * All data fetched from backend API — no mock/dummy data.
 */

import { useState, useEffect } from 'react';
import AppShell from '@/components/AppShell';
import api from '@/lib/api';
import { PlusIcon, SettingsIcon, XIcon, VideoCameraIcon, VideoCameraOffIcon, TrashIcon } from '@/components/Icons';

export default function CamerasPage() {
    const [cameras, setCameras] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [showModal, setShowModal] = useState(false);
    const [formData, setFormData] = useState({ name: '', location: '', stream_url: '', camera_type: 'webcam' });
    const [startingWebcam, setStartingWebcam] = useState(false);
    const [deleteConfirm, setDeleteConfirm] = useState(null); // camera object to delete
    const [deleting, setDeleting] = useState(false);

    useEffect(() => { loadCameras(); }, []);

    async function loadCameras() {
        setLoading(true);
        setError(null);
        try {
            const result = await api.getCameras();
            setCameras(result || []);
        } catch (e) {
            setError('Could not load cameras. Make sure the backend is running.');
            setCameras([]);
        } finally {
            setLoading(false);
        }
    }

    async function handleAdd() {
        try {
            await api.createCamera(formData);
            setShowModal(false);
            setFormData({ name: '', location: '', stream_url: '', camera_type: 'webcam' });
            loadCameras();
        } catch (e) { alert(e.message); }
    }

    async function handleToggle(cam) {
        try {
            if (cam.status === 'online') await api.stopCamera(cam.id);
            else await api.startCamera(cam.id);
            loadCameras();
        } catch (e) { alert(e.message); }
    }

    async function handleQuickWebcam() {
        setStartingWebcam(true);
        try {
            const cam = await api.createCamera({
                name: 'My Webcam',
                location: 'Local Machine',
                stream_url: '0',
                camera_type: 'webcam',
            });
            if (cam?.id) {
                try { await api.startCamera(cam.id); } catch (_) { /* camera added but start may fail */ }
            }
            loadCameras();
        } catch (e) {
            alert('Could not start webcam: ' + e.message);
        } finally {
            setStartingWebcam(false);
        }
    }

    async function handleDelete(cam) {
        setDeleting(true);
        try {
            // Stop first if running
            if (cam.status === 'online') {
                try { await api.stopCamera(cam.id); } catch (_) {}
            }
            await api.deleteCamera(cam.id);
            setDeleteConfirm(null);
            loadCameras();
        } catch (e) {
            alert('Failed to delete camera: ' + e.message);
        } finally {
            setDeleting(false);
        }
    }

    return (
        <AppShell>
            <div className="page-header">
                <h1>Camera Management</h1>
                <p>Configure and control surveillance cameras</p>
            </div>
            <div className="card toolbar-responsive" style={{ marginBottom: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', gap: '16px' }}>
                    <div><span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Total</span><div style={{ fontSize: '1.25rem', fontWeight: 700 }}>{cameras.length}</div></div>
                    <div><span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Online</span><div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--success)' }}>{cameras.filter(c => c.status === 'online').length}</div></div>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                    {!cameras.some(c => c.camera_type === 'webcam') && (
                        <button
                            className="btn btn-success"
                            onClick={handleQuickWebcam}
                            disabled={startingWebcam}
                            id="quick-webcam-btn"
                            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
                        >
                            <VideoCameraIcon size={16} />
                            {startingWebcam ? 'Starting Webcam...' : 'Start Webcam'}
                        </button>
                    )}
                    <button className="btn btn-primary" onClick={() => setShowModal(true)} id="add-camera-btn"><PlusIcon size={16} /> Add Camera</button>
                    <a
                        href="/share-camera"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="btn btn-secondary"
                        style={{ display: 'flex', alignItems: 'center', gap: '6px', textDecoration: 'none' }}
                        title="Open on a remote phone to use its camera"
                    >
                        📱 Remote Camera
                    </a>
                </div>
            </div>

            {loading ? (
                <div className="card" style={{ padding: '60px 20px', textAlign: 'center' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Loading cameras...</div>
                </div>
            ) : error ? (
                <div className="card" style={{ padding: '60px 20px', textAlign: 'center' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: '16px' }}>{error}</div>
                    <button className="btn btn-primary btn-sm" onClick={loadCameras}>Retry</button>
                </div>
            ) : cameras.length === 0 ? (
                <div className="card">
                    <div className="empty-state">
                        <div className="empty-state-icon"><VideoCameraIcon size={32} /></div>
                        <div className="empty-state-title">No cameras configured</div>
                        <div className="empty-state-text">Get started by launching your webcam or adding an RTSP camera.</div>
                        <div style={{ marginTop: '20px', display: 'flex', gap: '12px', justifyContent: 'center', flexWrap: 'wrap' }}>
                            <button
                                className="btn btn-success"
                                onClick={handleQuickWebcam}
                                disabled={startingWebcam}
                                style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
                            >
                                <VideoCameraIcon size={16} />
                                {startingWebcam ? 'Starting Webcam...' : 'Quick Start Webcam'}
                            </button>
                            <button className="btn btn-primary" onClick={() => setShowModal(true)}>
                                <PlusIcon size={16} /> Add Camera Manually
                            </button>
                        </div>
                    </div>
                </div>
            ) : (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(280px,1fr))', gap: '16px' }}>
                    {cameras.map(cam => (
                        <div key={cam.id} className="card">
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                                <div>
                                    <div style={{ fontWeight: 600, fontSize: '1rem' }}>{cam.name}</div>
                                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{cam.location}</div>
                                </div>
                                <span className={`badge ${cam.status}`}>{cam.status === 'online' && <span className="badge-dot" />}{cam.status}</span>
                            </div>
                            <div style={{ fontSize: '0.813rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                                <strong>Type:</strong> {cam.camera_type}
                            </div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: "'JetBrains Mono',monospace", marginBottom: '16px', wordBreak: 'break-all' }}>
                                {cam.stream_url}
                            </div>
                            <div style={{ display: 'flex', gap: '8px' }}>
                                <button className={`btn btn-sm ${cam.status === 'online' ? 'btn-danger' : 'btn-success'}`} onClick={() => handleToggle(cam)} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                    {cam.status === 'online' ? <><VideoCameraOffIcon size={14} /> Stop</> : <><VideoCameraIcon size={14} /> Start</>}
                                </button>
                                <button className="btn btn-secondary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><SettingsIcon size={14} /> Config</button>
                                <button
                                    className="btn btn-sm"
                                    onClick={() => setDeleteConfirm(cam)}
                                    title="Remove camera"
                                    style={{
                                        display: 'flex', alignItems: 'center', gap: '4px',
                                        background: 'rgba(239,68,68,0.1)', color: '#ef4444',
                                        border: '1px solid rgba(239,68,68,0.2)',
                                    }}
                                >
                                    <TrashIcon size={14} />
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {showModal && (
                <div className="modal-overlay" onClick={() => setShowModal(false)}>
                    <div className="modal" onClick={e => e.stopPropagation()}>
                        <div className="modal-header">
                            <h3 className="modal-title">Add Camera</h3>
                            <button className="modal-close" onClick={() => setShowModal(false)}><XIcon size={18} /></button>
                        </div>
                        <div className="input-group"><label className="input-label">Name *</label><input className="input" value={formData.name} onChange={e => setFormData({ ...formData, name: e.target.value })} placeholder="e.g., Main Entrance" /></div>
                        <div className="input-group"><label className="input-label">Location</label><input className="input" value={formData.location} onChange={e => setFormData({ ...formData, location: e.target.value })} placeholder="Building A, Floor 1" /></div>
                        <div className="input-group"><label className="input-label">Stream URL</label><input className="input" value={formData.stream_url} onChange={e => setFormData({ ...formData, stream_url: e.target.value })} placeholder="rtsp://... or 0 for webcam" /></div>
                        <div className="input-group"><label className="input-label">Type</label>
                            <select className="input" value={formData.camera_type} onChange={e => setFormData({ ...formData, camera_type: e.target.value })}>
                                <option value="webcam">Webcam</option><option value="rtsp">RTSP</option><option value="ip">IP Camera</option>
                            </select>
                        </div>
                        <div className="modal-actions">
                            <button className="btn btn-secondary" onClick={() => setShowModal(false)}>Cancel</button>
                            <button className="btn btn-primary" onClick={handleAdd} disabled={!formData.name}>Add Camera</button>
                        </div>
                    </div>
                </div>
            )}

            {/* Delete Confirmation Modal */}
            {deleteConfirm && (
                <div className="modal-overlay" onClick={() => !deleting && setDeleteConfirm(null)}>
                    <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: '420px' }}>
                        <div className="modal-header">
                            <h3 className="modal-title" style={{ color: '#ef4444', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <TrashIcon size={18} /> Remove Camera
                            </h3>
                            <button className="modal-close" onClick={() => !deleting && setDeleteConfirm(null)}><XIcon size={18} /></button>
                        </div>
                        <div style={{ padding: '0 24px 8px', fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                            Are you sure you want to remove <strong style={{ color: 'var(--text-primary)' }}>{deleteConfirm.name}</strong>?
                            {deleteConfirm.status === 'online' && (
                                <div style={{
                                    marginTop: '8px', padding: '8px 12px', borderRadius: '8px',
                                    background: 'rgba(245,158,11,0.1)', border: '1px solid rgba(245,158,11,0.2)',
                                    fontSize: '0.75rem', color: '#f59e0b',
                                }}>
                                    This camera is currently online. It will be stopped before removal.
                                </div>
                            )}
                        </div>
                        <div className="modal-actions">
                            <button className="btn btn-secondary" onClick={() => setDeleteConfirm(null)} disabled={deleting}>Cancel</button>
                            <button
                                className="btn btn-danger"
                                onClick={() => handleDelete(deleteConfirm)}
                                disabled={deleting}
                                style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
                            >
                                <TrashIcon size={14} />
                                {deleting ? 'Removing...' : 'Remove Camera'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </AppShell>
    );
}
