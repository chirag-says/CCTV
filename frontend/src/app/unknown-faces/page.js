'use client';

/**
 * Unknown Faces Page — Admin review and enrollment queue.
 * All data fetched from backend API — no mock/dummy data.
 */

import { useState, useEffect } from 'react';
import AppShell from '@/components/AppShell';
import api from '@/lib/api';
import { UserIcon, CameraIcon, CheckIcon, XIcon, CheckCircleIcon, UnknownFaceIcon } from '@/components/Icons';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function UnknownFacesPage() {
    const [unknowns, setUnknowns] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [showEnrollModal, setShowEnrollModal] = useState(false);
    const [selectedUnknown, setSelectedUnknown] = useState(null);
    const [enrollData, setEnrollData] = useState({ name: '', role: 'visitor', department: '', phone: '', email: '' });

    useEffect(() => {
        loadUnknowns();
    }, []);

    async function loadUnknowns() {
        setLoading(true);
        setError(null);
        try {
            const result = await api.getUnknownFaces({ status: 'pending' });
            setUnknowns(result.data || []);
        } catch (e) {
            setError('Could not load unknown faces. Make sure the backend is running.');
            setUnknowns([]);
        } finally {
            setLoading(false);
        }
    }

    function openEnrollModal(unknown) {
        setSelectedUnknown(unknown);
        setEnrollData({ name: '', role: 'visitor', department: '', phone: '', email: '' });
        setShowEnrollModal(true);
    }

    async function handleEnroll() {
        try {
            await api.enrollUnknownFace(selectedUnknown.id, enrollData);
            setShowEnrollModal(false);
            setUnknowns(unknowns.filter(u => u.id !== selectedUnknown.id));
        } catch (e) {
            alert(e.message);
        }
    }

    async function handleDismiss(unknownId) {
        if (!confirm('Dismiss this unknown face? It will be removed from the queue.')) return;
        try {
            await api.dismissUnknownFace(unknownId);
            setUnknowns(unknowns.filter(u => u.id !== unknownId));
        } catch (e) {
            alert(e.message);
        }
    }

    const formatDate = (dateStr) => {
        if (!dateStr) return '—';
        try {
            return new Date(dateStr).toLocaleString('en-US', {
                month: 'short', day: 'numeric',
                hour: '2-digit', minute: '2-digit',
                hour12: false,
            });
        } catch { return '—'; }
    };

    return (
        <AppShell unknownCount={unknowns.length}>
            <div className="page-header">
                <h1>Unknown Faces</h1>
                <p>Review unrecognized persons and enroll them into the system</p>
            </div>

            {loading ? (
                <div className="card" style={{ padding: '60px 20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                    Loading unknown faces...
                </div>
            ) : error ? (
                <div className="card" style={{ padding: '60px 20px', textAlign: 'center' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: '16px' }}>{error}</div>
                    <button className="btn btn-primary btn-sm" onClick={loadUnknowns}>Retry</button>
                </div>
            ) : (
                <>
                    {/* Summary Bar */}
                    <div className="card" style={{
                        marginBottom: '20px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        flexWrap: 'wrap',
                        gap: '12px',
                    }}>
                        <div style={{ display: 'flex', gap: '24px' }}>
                            <div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Pending Review</div>
                                <div style={{ fontSize: '1.5rem', fontWeight: 800, color: unknowns.length > 0 ? 'var(--warning)' : 'var(--success)' }}>{unknowns.length}</div>
                            </div>
                            <div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Total Sightings</div>
                                <div style={{ fontSize: '1.5rem', fontWeight: 800 }}>{unknowns.reduce((acc, u) => acc + (u.occurrence || 0), 0)}</div>
                            </div>
                        </div>
                    </div>

                    {/* Unknown Faces Grid */}
                    {unknowns.length > 0 ? (
                        <div className="unknown-faces-grid">
                            {unknowns.map((unknown) => {
                                const imageUrl = unknown.context_url || unknown.snapshot_url;
                                return (
                                    <div key={unknown.id} className="unknown-face-card">
                                        {/* Face Image */}
                                        {imageUrl ? (
                                            <div className="unknown-face-image" style={{ padding: 0, overflow: 'hidden' }}>
                                                <img
                                                    src={`${API_BASE}${imageUrl}`}
                                                    alt="Face Snapshot"
                                                    style={{
                                                        width: '100%',
                                                        height: '100%',
                                                        objectFit: 'cover',
                                                        display: 'block',
                                                    }}
                                                />
                                            </div>
                                        ) : (
                                            <div className="unknown-face-image" style={{
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                background: 'linear-gradient(135deg, var(--bg-tertiary), var(--bg-secondary))',
                                            }}>
                                                <div style={{ textAlign: 'center' }}>
                                                    <div style={{ fontSize: '3rem', opacity: 0.3, display: 'flex', justifyContent: 'center' }}><UnknownFaceIcon size={48} /></div>
                                                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px' }}>No Image</div>
                                                </div>
                                            </div>
                                        )}

                                        <div className="unknown-face-info">
                                            <div className="unknown-face-meta">
                                                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><CameraIcon size={14} /> {unknown.camera_name || unknown.camera_id}</span>
                                                <span className="badge pending">
                                                    {unknown.occurrence}× seen
                                                </span>
                                            </div>
                                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '12px' }}>
                                                <div>First: {formatDate(unknown.first_seen)}</div>
                                                <div>Last: {formatDate(unknown.last_seen)}</div>
                                            </div>
                                            <div className="unknown-face-actions">
                                                <button
                                                    className="btn btn-success btn-sm"
                                                    onClick={() => openEnrollModal(unknown)}
                                                >
                                                    <CheckIcon size={14} /> Enroll
                                                </button>
                                                <button
                                                    className="btn btn-danger btn-sm"
                                                    onClick={() => handleDismiss(unknown.id)}
                                                >
                                                    <XIcon size={14} /> Dismiss
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                )
                            })}
                        </div>
                    ) : (
                        <div className="card">
                            <div className="empty-state">
                                <div className="empty-state-icon"><CheckCircleIcon size={32} /></div>
                                <div className="empty-state-title">All Clear!</div>
                                <div className="empty-state-text">No unknown faces pending review. All detected persons have been processed.</div>
                            </div>
                        </div>
                    )}
                </>
            )}

            {/* Enroll Modal */}
            {showEnrollModal && (
                <div className="modal-overlay" onClick={() => setShowEnrollModal(false)}>
                    <div className="modal" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <h3 className="modal-title">Enroll Unknown Person</h3>
                            <button className="modal-close" onClick={() => setShowEnrollModal(false)}><XIcon size={18} /></button>
                        </div>

                        <div style={{
                            background: 'var(--bg-tertiary)',
                            borderRadius: 'var(--radius-md)',
                            padding: '16px',
                            marginBottom: '20px',
                            fontSize: '0.813rem',
                            color: 'var(--text-secondary)',
                        }}>
                            <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
                                <div style={{ display: 'flex' }}><UserIcon size={32} /></div>
                                <div>
                                    <div>Camera: {selectedUnknown?.camera_name || selectedUnknown?.camera_id}</div>
                                    <div>Seen {selectedUnknown?.occurrence} times</div>
                                    <div>First seen: {formatDate(selectedUnknown?.first_seen)}</div>
                                </div>
                            </div>
                        </div>

                        <div className="input-group">
                            <label className="input-label">Full Name *</label>
                            <input
                                className="input"
                                value={enrollData.name}
                                onChange={(e) => setEnrollData({ ...enrollData, name: e.target.value })}
                                placeholder="Enter person's name"
                                autoFocus
                                id="enroll-name"
                            />
                        </div>

                        <div className="input-group">
                            <label className="input-label">Role</label>
                            <select
                                className="input"
                                value={enrollData.role}
                                onChange={(e) => setEnrollData({ ...enrollData, role: e.target.value })}
                                id="enroll-role"
                            >
                                <option value="visitor">Visitor</option>
                                <option value="employee">Employee</option>
                                <option value="vip">VIP</option>
                            </select>
                        </div>

                        <div className="input-group">
                            <label className="input-label">Department</label>
                            <input
                                className="input"
                                value={enrollData.department}
                                onChange={(e) => setEnrollData({ ...enrollData, department: e.target.value })}
                                placeholder="e.g., Engineering"
                                id="enroll-department"
                            />
                        </div>

                        <div className="modal-actions">
                            <button className="btn btn-secondary" onClick={() => setShowEnrollModal(false)}>Cancel</button>
                            <button
                                className="btn btn-primary"
                                onClick={handleEnroll}
                                disabled={!enrollData.name.trim()}
                                id="enroll-submit"
                            >
                                Enroll Person
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </AppShell>
    );
}
