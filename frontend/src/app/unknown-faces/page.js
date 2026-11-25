'use client';

/**
 * Unknown Faces Page — Admin review and enrollment queue.
 * All data fetched from backend API — no mock/dummy data.
 *
 * Features:
 * - View pending unknown faces
 * - Enroll as new person OR merge into existing person
 * - Dismiss unwanted entries
 */

import { useState, useEffect, useCallback } from 'react';
import AppShell from '@/components/AppShell';
import api, { API_BASE } from '@/lib/api';
import { useToast } from '@/lib/ToastContext';
import { UserIcon, CameraIcon, CheckIcon, XIcon, CheckCircleIcon, UnknownFaceIcon } from '@/components/Icons';

/** First / previous / page numbers / next / last, with a line saying where you are. */
function Pager({ page, pageCount, onPage, summary, disabled }) {
    // Up to five page numbers centred on the current one.
    const from = Math.max(0, Math.min(page - 2, pageCount - 5));
    const numbers = Array.from({ length: Math.min(5, pageCount) }, (_, i) => from + i);
    const go = (target) => onPage(Math.min(pageCount - 1, Math.max(0, target)));
    return (
        <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            flexWrap: 'wrap', gap: '10px', margin: '12px 0',
        }}>
            <span style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>{summary}</span>
            {pageCount > 1 && (
                <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                    <button className="btn btn-secondary btn-sm" onClick={() => go(0)} disabled={disabled || page === 0} aria-label="First page">«</button>
                    <button className="btn btn-secondary btn-sm" onClick={() => go(page - 1)} disabled={disabled || page === 0}>Previous</button>
                    {numbers.map((n) => (
                        <button
                            key={n}
                            className={`btn btn-sm ${n === page ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={() => go(n)}
                            disabled={disabled}
                            aria-current={n === page ? 'page' : undefined}
                            style={{ minWidth: '36px' }}
                        >
                            {n + 1}
                        </button>
                    ))}
                    <button className="btn btn-secondary btn-sm" onClick={() => go(page + 1)} disabled={disabled || page >= pageCount - 1}>Next</button>
                    <button className="btn btn-secondary btn-sm" onClick={() => go(pageCount - 1)} disabled={disabled || page >= pageCount - 1} aria-label="Last page">»</button>
                </div>
            )}
        </div>
    );
}

export default function UnknownFacesPage() {
    const [unknowns, setUnknowns] = useState([]);
    const [suggestions, setSuggestions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [showEnrollModal, setShowEnrollModal] = useState(false);
    const [selectedUnknown, setSelectedUnknown] = useState(null);
    const [enrollData, setEnrollData] = useState({ name: '', role: 'visitor', department: '', phone: '', email: '' });
    const { showToast } = useToast();

    // ── Merge mode state ──────────────────────────────────────
    const [enrollMode, setEnrollMode] = useState('new'); // 'new' | 'merge'
    const [persons, setPersons] = useState([]);
    const [personsLoading, setPersonsLoading] = useState(false);
    const [personSearch, setPersonSearch] = useState('');
    const [selectedPerson, setSelectedPerson] = useState(null);
    const [merging, setMerging] = useState(false);

    // ── Sorting and paging ────────────────────────────────────
    // The server does both. The queue runs to thousands of faces, so the page
    // only ever holds the slice it is showing, and the totals come from the
    // server rather than from counting that slice.
    const [sortKey, setSortKey] = useState('last_seen:desc');
    const [pageSize, setPageSize] = useState(24);
    const [page, setPage] = useState(0);
    const [total, setTotal] = useState(0);
    const [totalSightings, setTotalSightings] = useState(0);
    const [retentionDays, setRetentionDays] = useState(0);
    const [refreshing, setRefreshing] = useState(false);

    useEffect(() => {
        loadSuggestions();
    }, []);

    useEffect(() => {
        // Only the very first load swaps in the "Loading" card; turning a
        // page or re-sorting keeps the grid on screen while it updates.
        loadUnknowns({ quiet: unknowns.length > 0 });
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [sortKey, pageSize, page]);

    // Load persons when switching to merge mode
    useEffect(() => {
        if (enrollMode === 'merge' && showEnrollModal) {
            loadPersons();
        }
    }, [enrollMode, showEnrollModal]);

    async function loadSuggestions() {
        try {
            const result = await api.request('/api/unknown-faces/suggestions?min_occurrences=3');
            setSuggestions(result.suggestions || []);
        } catch {
            // Silently fail — suggestions are a bonus feature
        }
    }

    // `quiet` reloads in place (after an enrol or dismiss, or the Refresh
    // button) without swapping the whole grid for a "Loading" card.
    async function loadUnknowns({ quiet = false, atPage = page } = {}) {
        if (quiet) setRefreshing(true); else setLoading(true);
        setError(null);
        try {
            const [sort, order] = sortKey.split(':');
            const result = await api.getUnknownFaces({
                status: 'pending', sort, order, limit: pageSize, offset: atPage * pageSize,
            });
            const rows = result.data || [];
            const count = result.total ?? rows.length;
            // The last face on the last page was just dealt with: step back
            // rather than showing an empty page with faces still waiting.
            if (rows.length === 0 && count > 0 && atPage > 0) {
                setPage(Math.max(0, Math.ceil(count / pageSize) - 1));
                return;
            }
            setUnknowns(rows);
            setTotal(count);
            setTotalSightings(result.total_sightings ?? rows.reduce((acc, u) => acc + (u.occurrence || 0), 0));
            setRetentionDays(result.retention_days || 0);
        } catch (e) {
            setError('Could not load unknown faces. Make sure the backend is running.');
            setUnknowns([]);
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    }

    function changeSort(value) {
        setSortKey(value);
        setPage(0);
    }

    function changePageSize(value) {
        setPageSize(Number(value));
        setPage(0);
    }

    const pageCount = Math.max(1, Math.ceil(total / pageSize));
    const firstShown = total === 0 ? 0 : page * pageSize + 1;
    const lastShown = Math.min(total, page * pageSize + unknowns.length);

    async function loadPersons(search = '') {
        setPersonsLoading(true);
        try {
            const params = { is_active: true };
            if (search) params.search = search;
            const result = await api.getPersons(params);
            setPersons(result.data || []);
        } catch {
            setPersons([]);
        } finally {
            setPersonsLoading(false);
        }
    }

    // Debounced person search
    const handlePersonSearch = useCallback((value) => {
        setPersonSearch(value);
        // Simple debounce via timeout
        clearTimeout(window._personSearchTimeout);
        window._personSearchTimeout = setTimeout(() => loadPersons(value), 300);
    }, []);

    function openEnrollModal(unknown) {
        setSelectedUnknown(unknown);
        setEnrollData({ name: '', role: 'visitor', department: '', phone: '', email: '' });
        setEnrollMode('new');
        setSelectedPerson(null);
        setPersonSearch('');
        setShowEnrollModal(true);
    }

    function closeEnrollModal() {
        setShowEnrollModal(false);
        setSelectedPerson(null);
        setPersonSearch('');
        setMerging(false);
    }

    async function handleEnroll() {
        try {
            await api.enrollUnknownFace(selectedUnknown.id, enrollData);
            closeEnrollModal();
            setUnknowns(unknowns.filter(u => u.id !== selectedUnknown.id));
            loadUnknowns({ quiet: true });
            showToast({ type: 'success', message: `Enrolled ${enrollData.name} successfully` });
        } catch (e) {
            showToast({ type: 'error', message: e.message });
        }
    }

    async function handleMerge() {
        if (!selectedPerson) return;
        setMerging(true);
        try {
            const result = await api.mergeUnknownFace(selectedUnknown.id, selectedPerson.id);
            closeEnrollModal();
            setUnknowns(unknowns.filter(u => u.id !== selectedUnknown.id));
            loadUnknowns({ quiet: true });
            showToast({
                type: 'success',
                message: result.message || `Merged face into ${selectedPerson.name}`,
            });
        } catch (e) {
            showToast({ type: 'error', message: e.message });
        } finally {
            setMerging(false);
        }
    }

    async function handleDismiss(unknownId) {
        if (!confirm('Dismiss this unknown face? It will be removed from the queue.')) return;
        try {
            await api.dismissUnknownFace(unknownId);
            setUnknowns(unknowns.filter(u => u.id !== unknownId));
            loadUnknowns({ quiet: true });
        } catch (e) {
            showToast({ type: 'error', message: e.message });
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

    const getRoleBadgeClass = (role) => {
        switch (role) {
            case 'employee': return 'active';
            case 'vip': return 'detection';
            case 'visitor': return 'pending';
            case 'banned': return 'exit';
            default: return 'pending';
        }
    };

    // ── Segmented Control Style ──────────────────────────────
    const segmentStyle = (active) => ({
        flex: 1,
        padding: '8px 16px',
        fontSize: '0.8125rem',
        fontWeight: 600,
        border: 'none',
        cursor: 'pointer',
        borderRadius: '8px',
        transition: 'all 0.25s ease',
        background: active ? 'var(--primary)' : 'transparent',
        color: active ? '#fff' : 'var(--text-secondary)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '6px',
    });

    return (
        <AppShell unknownCount={total}>
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
                                <div style={{ fontSize: '1.5rem', fontWeight: 800, color: total > 0 ? 'var(--warning)' : 'var(--success)' }}>{total.toLocaleString()}</div>
                            </div>
                            <div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Total Sightings</div>
                                <div style={{ fontSize: '1.5rem', fontWeight: 800 }}>{totalSightings.toLocaleString()}</div>
                            </div>
                        </div>

                        <div style={{ display: 'flex', gap: '10px', alignItems: 'flex-end', flexWrap: 'wrap' }}>
                            <label style={{ fontSize: '0.688rem', color: 'var(--text-muted)' }}>
                                Sort by
                                <select
                                    className="input"
                                    id="unknown-sort"
                                    value={sortKey}
                                    onChange={(e) => changeSort(e.target.value)}
                                    style={{ display: 'block', marginTop: '4px', fontSize: '0.8125rem', minWidth: '210px' }}
                                >
                                    <option value="last_seen:desc">Last seen — newest first</option>
                                    <option value="last_seen:asc">Last seen — oldest first</option>
                                    <option value="first_seen:desc">First seen — newest first</option>
                                    <option value="first_seen:asc">First seen — oldest first</option>
                                    <option value="occurrence:desc">Seen most often</option>
                                    <option value="occurrence:asc">Seen least often</option>
                                </select>
                            </label>
                            <label style={{ fontSize: '0.688rem', color: 'var(--text-muted)' }}>
                                Per page
                                <select
                                    className="input"
                                    id="unknown-page-size"
                                    value={pageSize}
                                    onChange={(e) => changePageSize(e.target.value)}
                                    style={{ display: 'block', marginTop: '4px', fontSize: '0.8125rem' }}
                                >
                                    {[12, 24, 48, 96].map((n) => <option key={n} value={n}>{n}</option>)}
                                </select>
                            </label>
                            <button
                                className="btn btn-secondary btn-sm"
                                id="unknown-refresh"
                                onClick={() => { loadUnknowns({ quiet: true }); loadSuggestions(); }}
                                disabled={refreshing}
                            >
                                {refreshing ? 'Refreshing…' : 'Refresh'}
                            </button>
                        </div>
                    </div>

                    {retentionDays > 0 && (
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '-8px 0 16px' }}>
                            Unknown faces not seen for {retentionDays} days are removed automatically. Enrolled faces are kept.
                        </div>
                    )}

                    {/* Enrollment Suggestions */}
                    {suggestions.length > 0 && (
                        <div className="card" style={{
                            marginBottom: '20px',
                            borderLeft: '4px solid var(--warning)',
                            background: 'var(--bg-card)',
                        }}>
                            <div className="card-header" style={{ paddingBottom: '8px' }}>
                                <div>
                                    <h3 className="card-title" style={{ color: 'var(--warning)' }}>
                                        ⭐ Frequently Seen — Consider Enrolling
                                    </h3>
                                    <div className="card-subtitle">
                                        These faces have been seen {'>'} 3 times. They may be regulars.
                                    </div>
                                </div>
                            </div>
                            <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', padding: '0 0 4px' }}>
                                {suggestions.map(s => {
                                    const imgUrl = s.context_url || s.snapshot_url;
                                    return (
                                        <div key={s.id} style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '10px',
                                            padding: '8px 12px',
                                            background: 'var(--bg-tertiary)',
                                            borderRadius: '10px',
                                            border: '1px solid rgba(245, 158, 11, 0.15)',
                                        }}>
                                            {imgUrl ? (
                                                <img src={`${API_BASE}${imgUrl}`} alt="Unknown" style={{
                                                    width: 36, height: 36, borderRadius: '8px', objectFit: 'cover',
                                                }} />
                                            ) : (
                                                <div style={{
                                                    width: 36, height: 36, borderRadius: '8px',
                                                    background: 'var(--bg-secondary)',
                                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                }}>
                                                    <UnknownFaceIcon size={16} />
                                                </div>
                                            )}
                                            <div>
                                                <div style={{ fontSize: '0.75rem', fontWeight: 600 }}>Seen {s.occurrence}x</div>
                                                <div style={{ fontSize: '0.625rem', color: 'var(--text-muted)' }}>ID: {s.id.slice(0, 8)}</div>
                                            </div>
                                            <button
                                                className="btn btn-sm btn-primary"
                                                style={{ marginLeft: '4px', minWidth: 'auto', padding: '4px 10px', fontSize: '0.6875rem' }}
                                                onClick={() => openEnrollModal(s)}
                                            >
                                                Enroll
                                            </button>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {/* Unknown Faces Grid */}
                    {unknowns.length > 0 ? (
                        <>
                        <Pager
                            page={page} pageCount={pageCount} onPage={setPage} disabled={refreshing}
                            summary={`Showing ${firstShown.toLocaleString()}–${lastShown.toLocaleString()} of ${total.toLocaleString()}`}
                        />
                        <div className="unknown-faces-grid" style={{ opacity: refreshing ? 0.6 : 1, transition: 'opacity 0.15s' }}>
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
                                                <span className="badge pending" style={{ whiteSpace: 'nowrap' }}>
                                                    {unknown.occurrence}x seen
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
                        <Pager
                            page={page} pageCount={pageCount} onPage={setPage} disabled={refreshing}
                            summary={`Page ${page + 1} of ${pageCount.toLocaleString()}`}
                        />
                        </>
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

            {/* ═══ Enroll / Merge Modal ═══════════════════════════════════ */}
            {showEnrollModal && (
                <div className="modal-overlay" onClick={closeEnrollModal}>
                    <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '520px' }}>
                        <div className="modal-header">
                            <h3 className="modal-title">Enroll Unknown Person</h3>
                            <button className="modal-close" onClick={closeEnrollModal}><XIcon size={18} /></button>
                        </div>

                        {/* Unknown face preview */}
                        <div style={{
                            background: 'var(--bg-tertiary)',
                            borderRadius: 'var(--radius-md)',
                            padding: '12px 16px',
                            marginBottom: '16px',
                            fontSize: '0.813rem',
                            color: 'var(--text-secondary)',
                        }}>
                            <div style={{ display: 'flex', gap: '14px', alignItems: 'center' }}>
                                {(() => {
                                    const imgUrl = selectedUnknown?.context_url || selectedUnknown?.snapshot_url;
                                    return imgUrl ? (
                                        <img
                                            src={`${API_BASE}${imgUrl}`}
                                            alt="Unknown face"
                                            style={{
                                                width: 48, height: 48, borderRadius: '10px',
                                                objectFit: 'cover', border: '2px solid var(--border)',
                                            }}
                                        />
                                    ) : (
                                        <div style={{
                                            width: 48, height: 48, borderRadius: '10px',
                                            background: 'var(--bg-secondary)',
                                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                                        }}>
                                            <UserIcon size={24} />
                                        </div>
                                    );
                                })()}
                                <div>
                                    <div>Camera: {selectedUnknown?.camera_name || selectedUnknown?.camera_id}</div>
                                    <div>Seen {selectedUnknown?.occurrence} times</div>
                                    <div>First seen: {formatDate(selectedUnknown?.first_seen)}</div>
                                </div>
                            </div>
                        </div>

                        {/* ── Mode Toggle (Segmented Control) ─────────────── */}
                        <div style={{
                            display: 'flex',
                            background: 'var(--bg-tertiary)',
                            borderRadius: '10px',
                            padding: '3px',
                            marginBottom: '20px',
                            border: '1px solid var(--border)',
                        }}>
                            <button
                                style={segmentStyle(enrollMode === 'new')}
                                onClick={() => { setEnrollMode('new'); setSelectedPerson(null); }}
                                id="mode-new-person"
                            >
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                                    <circle cx="8.5" cy="7" r="4" />
                                    <line x1="20" y1="8" x2="20" y2="14" />
                                    <line x1="23" y1="11" x2="17" y2="11" />
                                </svg>
                                New Person
                            </button>
                            <button
                                style={segmentStyle(enrollMode === 'merge')}
                                onClick={() => setEnrollMode('merge')}
                                id="mode-match-existing"
                            >
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M17 21v-2a4 4 0 0 0-3-3.87" />
                                    <path d="M9 21v-2a4 4 0 0 1 3-3.87" />
                                    <circle cx="13" cy="7" r="4" />
                                    <circle cx="7" cy="7" r="4" />
                                </svg>
                                Match Existing
                            </button>
                        </div>

                        {/* ═══ NEW PERSON MODE ═══════════════════════════ */}
                        {enrollMode === 'new' && (
                            <>
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

                                <div className="input-group">
                                    <label className="input-label">Email</label>
                                    <input
                                        className="input"
                                        type="email"
                                        value={enrollData.email}
                                        onChange={(e) => setEnrollData({ ...enrollData, email: e.target.value })}
                                        placeholder="person@company.com"
                                        id="enroll-email"
                                    />
                                </div>

                                <div className="input-group">
                                    <label className="input-label">Phone</label>
                                    <input
                                        className="input"
                                        value={enrollData.phone}
                                        onChange={(e) => setEnrollData({ ...enrollData, phone: e.target.value })}
                                        placeholder="+91-XXXXXXXXXX"
                                        id="enroll-phone"
                                    />
                                </div>

                                <div className="modal-actions">
                                    <button className="btn btn-secondary" onClick={closeEnrollModal}>Cancel</button>
                                    <button
                                        className="btn btn-primary"
                                        onClick={handleEnroll}
                                        disabled={!enrollData.name.trim()}
                                        id="enroll-submit"
                                    >
                                        Enroll Person
                                    </button>
                                </div>
                            </>
                        )}

                        {/* ═══ MATCH EXISTING MODE ══════════════════════ */}
                        {enrollMode === 'merge' && (
                            <>
                                {/* Search existing persons */}
                                <div className="input-group" style={{ marginBottom: '12px' }}>
                                    <label className="input-label">Search Persons</label>
                                    <input
                                        className="input"
                                        value={personSearch}
                                        onChange={(e) => handlePersonSearch(e.target.value)}
                                        placeholder="Type to search by name..."
                                        autoFocus
                                        id="merge-search"
                                    />
                                </div>

                                {/* Person list */}
                                <div style={{
                                    maxHeight: '220px',
                                    overflowY: 'auto',
                                    border: '1px solid var(--border)',
                                    borderRadius: 'var(--radius-md)',
                                    marginBottom: '16px',
                                }}>
                                    {personsLoading ? (
                                        <div style={{
                                            padding: '30px 16px', textAlign: 'center',
                                            color: 'var(--text-muted)', fontSize: '0.8125rem',
                                        }}>
                                            Loading persons...
                                        </div>
                                    ) : persons.length === 0 ? (
                                        <div style={{
                                            padding: '30px 16px', textAlign: 'center',
                                            color: 'var(--text-muted)', fontSize: '0.8125rem',
                                        }}>
                                            No persons found{personSearch ? ` matching "${personSearch}"` : ''}
                                        </div>
                                    ) : (
                                        persons.map((person) => {
                                            const isSelected = selectedPerson?.id === person.id;
                                            return (
                                                <div
                                                    key={person.id}
                                                    onClick={() => setSelectedPerson(person)}
                                                    id={`merge-person-${person.id}`}
                                                    style={{
                                                        display: 'flex',
                                                        alignItems: 'center',
                                                        gap: '12px',
                                                        padding: '10px 14px',
                                                        cursor: 'pointer',
                                                        transition: 'all 0.15s ease',
                                                        borderBottom: '1px solid var(--border)',
                                                        background: isSelected
                                                            ? 'rgba(99, 102, 241, 0.12)'
                                                            : 'transparent',
                                                        borderLeft: isSelected
                                                            ? '3px solid var(--primary)'
                                                            : '3px solid transparent',
                                                    }}
                                                    onMouseEnter={(e) => {
                                                        if (!isSelected) e.currentTarget.style.background = 'var(--bg-tertiary)';
                                                    }}
                                                    onMouseLeave={(e) => {
                                                        if (!isSelected) e.currentTarget.style.background = 'transparent';
                                                    }}
                                                >
                                                    {/* Avatar */}
                                                    {person.avatar_url ? (
                                                        <img
                                                            src={`${API_BASE}${person.avatar_url}`}
                                                            alt={person.name}
                                                            style={{
                                                                width: 38, height: 38, borderRadius: '10px',
                                                                objectFit: 'cover',
                                                                border: isSelected
                                                                    ? '2px solid var(--primary)'
                                                                    : '2px solid var(--border)',
                                                            }}
                                                        />
                                                    ) : (
                                                        <div style={{
                                                            width: 38, height: 38, borderRadius: '10px',
                                                            background: 'var(--bg-secondary)',
                                                            display: 'flex', alignItems: 'center',
                                                            justifyContent: 'center',
                                                            fontSize: '0.875rem', fontWeight: 700,
                                                            color: 'var(--text-muted)',
                                                            border: isSelected
                                                                ? '2px solid var(--primary)'
                                                                : '2px solid var(--border)',
                                                        }}>
                                                            {person.name.charAt(0).toUpperCase()}
                                                        </div>
                                                    )}

                                                    {/* Info */}
                                                    <div style={{ flex: 1, minWidth: 0 }}>
                                                        <div style={{
                                                            fontWeight: 600, fontSize: '0.8125rem',
                                                            color: isSelected ? 'var(--primary)' : 'var(--text-primary)',
                                                            whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                                                        }}>
                                                            {person.name}
                                                        </div>
                                                        <div style={{
                                                            fontSize: '0.6875rem', color: 'var(--text-muted)',
                                                            display: 'flex', gap: '8px', alignItems: 'center',
                                                        }}>
                                                            <span className={`badge ${getRoleBadgeClass(person.role)}`} style={{
                                                                fontSize: '0.625rem', padding: '1px 6px',
                                                            }}>
                                                                {person.role}
                                                            </span>
                                                            <span>{person.encoding_count} face{person.encoding_count !== 1 ? 's' : ''}</span>
                                                        </div>
                                                    </div>

                                                    {/* Selection indicator */}
                                                    {isSelected && (
                                                        <div style={{
                                                            width: 22, height: 22, borderRadius: '50%',
                                                            background: 'var(--primary)',
                                                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                            flexShrink: 0,
                                                        }}>
                                                            <CheckIcon size={13} style={{ color: '#fff' }} />
                                                        </div>
                                                    )}
                                                </div>
                                            );
                                        })
                                    )}
                                </div>

                                {/* Side-by-side merge preview */}
                                {selectedPerson && (
                                    <div style={{
                                        background: 'var(--bg-tertiary)',
                                        borderRadius: 'var(--radius-md)',
                                        padding: '14px',
                                        marginBottom: '16px',
                                        border: '1px solid rgba(99, 102, 241, 0.25)',
                                    }}>
                                        <div style={{
                                            fontSize: '0.6875rem', color: 'var(--text-muted)',
                                            textTransform: 'uppercase', letterSpacing: '0.08em',
                                            marginBottom: '10px', fontWeight: 600,
                                        }}>
                                            Merge Preview
                                        </div>
                                        <div style={{
                                            display: 'flex', alignItems: 'center',
                                            justifyContent: 'center', gap: '16px',
                                        }}>
                                            {/* Unknown face */}
                                            <div style={{ textAlign: 'center' }}>
                                                {(() => {
                                                    const imgUrl = selectedUnknown?.context_url || selectedUnknown?.snapshot_url;
                                                    return imgUrl ? (
                                                        <img
                                                            src={`${API_BASE}${imgUrl}`}
                                                            alt="Unknown"
                                                            style={{
                                                                width: 56, height: 56, borderRadius: '12px',
                                                                objectFit: 'cover',
                                                                border: '2px solid var(--warning)',
                                                            }}
                                                        />
                                                    ) : (
                                                        <div style={{
                                                            width: 56, height: 56, borderRadius: '12px',
                                                            background: 'var(--bg-secondary)',
                                                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                            border: '2px solid var(--warning)',
                                                        }}>
                                                            <UnknownFaceIcon size={24} />
                                                        </div>
                                                    );
                                                })()}
                                                <div style={{
                                                    fontSize: '0.625rem', color: 'var(--text-muted)',
                                                    marginTop: '4px',
                                                }}>
                                                    Unknown
                                                </div>
                                            </div>

                                            {/* Arrow */}
                                            <div style={{
                                                display: 'flex', flexDirection: 'column',
                                                alignItems: 'center', gap: '2px',
                                            }}>
                                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                                    <line x1="5" y1="12" x2="19" y2="12" />
                                                    <polyline points="12 5 19 12 12 19" />
                                                </svg>
                                                <span style={{
                                                    fontSize: '0.5625rem', color: 'var(--primary)',
                                                    fontWeight: 700, textTransform: 'uppercase',
                                                    letterSpacing: '0.05em',
                                                }}>
                                                    merge
                                                </span>
                                            </div>

                                            {/* Target person */}
                                            <div style={{ textAlign: 'center' }}>
                                                {selectedPerson.avatar_url ? (
                                                    <img
                                                        src={`${API_BASE}${selectedPerson.avatar_url}`}
                                                        alt={selectedPerson.name}
                                                        style={{
                                                            width: 56, height: 56, borderRadius: '12px',
                                                            objectFit: 'cover',
                                                            border: '2px solid var(--success)',
                                                        }}
                                                    />
                                                ) : (
                                                    <div style={{
                                                        width: 56, height: 56, borderRadius: '12px',
                                                        background: 'var(--bg-secondary)',
                                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                        fontSize: '1.25rem', fontWeight: 700,
                                                        color: 'var(--text-muted)',
                                                        border: '2px solid var(--success)',
                                                    }}>
                                                        {selectedPerson.name.charAt(0).toUpperCase()}
                                                    </div>
                                                )}
                                                <div style={{
                                                    fontSize: '0.625rem', color: 'var(--success)',
                                                    marginTop: '4px', fontWeight: 600,
                                                    maxWidth: '80px', whiteSpace: 'nowrap',
                                                    overflow: 'hidden', textOverflow: 'ellipsis',
                                                }}>
                                                    {selectedPerson.name}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                <div className="modal-actions">
                                    <button className="btn btn-secondary" onClick={closeEnrollModal}>Cancel</button>
                                    <button
                                        className="btn btn-primary"
                                        onClick={handleMerge}
                                        disabled={!selectedPerson || merging}
                                        id="merge-submit"
                                        style={{
                                            display: 'flex', alignItems: 'center', gap: '6px',
                                        }}
                                    >
                                        {merging ? (
                                            'Merging...'
                                        ) : (
                                            <>
                                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                                    <path d="M17 21v-2a4 4 0 0 0-3-3.87" />
                                                    <path d="M9 21v-2a4 4 0 0 1 3-3.87" />
                                                    <circle cx="13" cy="7" r="4" />
                                                    <circle cx="7" cy="7" r="4" />
                                                </svg>
                                                Merge Face
                                            </>
                                        )}
                                    </button>
                                </div>
                            </>
                        )}
                    </div>
                </div>
            )}
        </AppShell>
    );
}
