'use client';

/**
 * Person Detail Page — Embedding Management & Suggestions
 *
 * Sections:
 * 1. Person Header — Avatar, name, role, stats
 * 2. Current Embeddings — Gallery of enrolled face encodings with health scores
 * 3. Suggested Embeddings — Admin approval queue with diversity reasons
 * 4. Activity Timeline — Recent detection events
 */

import { useState, useEffect, useCallback } from 'react';
import { useParams, useRouter } from 'next/navigation';
import AppShell from '@/components/AppShell';
import PersonTimeline from '@/components/PersonTimeline';
import JourneyPanel from '@/components/JourneyPanel';
import api, { API_BASE } from '@/lib/api';
import {
    UserIcon, ImageIcon, XIcon, ClockIcon, EditIcon, PlusIcon,
} from '@/components/Icons';

// ── Inline Icons (not in Icons.js) ─────────────────────────────

const CheckIcon = ({ size = 20 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
        <polyline points="20 6 9 17 4 12" />
    </svg>
);

const StarIcon = ({ size = 20 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
    </svg>
);

const TrashIcon = ({ size = 20 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
        <polyline points="3 6 5 6 21 6" />
        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </svg>
);

const ArrowLeftIcon = ({ size = 20 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
        <line x1="19" y1="12" x2="5" y2="12" /><polyline points="12 19 5 12 12 5" />
    </svg>
);

const ShieldIcon = ({ size = 20 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
);


// ── Helper: Pose bucket label ──────────────────────────────────

function poseBucketLabel(bucket) {
    const map = {
        frontal: '正面 Frontal',
        left: '← Left',
        right: 'Right →',
        extreme_left: '⟵ Extreme Left',
        extreme_right: 'Extreme Right ⟶',
    };
    return map[bucket] || bucket || 'Unknown';
}

function poseBucketColor(bucket) {
    const map = {
        frontal: 'var(--primary)',
        left: 'var(--accent)',
        right: 'var(--warning)',
        extreme_left: 'var(--danger)',
        extreme_right: 'var(--danger)',
    };
    return map[bucket] || 'var(--text-muted)';
}

function healthLabel(score) {
    if (score >= 0.7) return { text: 'Excellent', color: 'var(--success)' };
    if (score >= 0.4) return { text: 'Good', color: 'var(--warning)' };
    return { text: 'Low', color: 'var(--danger)' };
}

function qualityBar(score) {
    const pct = Math.round((score || 0) * 100);
    const color = pct >= 80 ? 'var(--success)' : pct >= 50 ? 'var(--warning)' : 'var(--danger)';
    return { pct, color };
}

function formatDate(iso) {
    if (!iso) return '—';
    return new Date(iso).toLocaleDateString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
    });
}


// ── Main Component ─────────────────────────────────────────────

export default function PersonDetailPage() {
    const params = useParams();
    const router = useRouter();
    const personId = params.id;

    const [person, setPerson] = useState(null);
    const [encodings, setEncodings] = useState([]);
    const [suggestions, setSuggestions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    // Journey leads: "where was this person" is the question people open a
    // person page to answer. Embedding management is admin plumbing.
    const [activeTab, setActiveTab] = useState('journey');

    // Replacement modal state
    const [replaceModal, setReplaceModal] = useState(null);

    // Delete person state
    const [showDeleteModal, setShowDeleteModal] = useState(false);
    const [deleteConfirmName, setDeleteConfirmName] = useState('');
    const [deleting, setDeleting] = useState(false);

    const loadData = useCallback(async () => {
        try {
            setLoading(true);
            const [personData, encodingData, suggestionData] = await Promise.all([
                api.getPerson(personId),
                api.getPersonEncodings(personId),
                api.getPersonSuggestions(personId),
            ]);
            setPerson(personData);
            setEncodings(encodingData);
            setSuggestions(suggestionData);
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    }, [personId]);

    useEffect(() => { loadData(); }, [loadData]);

    // ── Actions ────────────────────────────────────────────────

    async function handleDeleteEncoding(encodingId) {
        if (!confirm('Delete this encoding? This cannot be undone.')) return;
        try {
            await api.deletePersonEncoding(personId, encodingId);
            loadData();
        } catch (e) {
            alert(e.message);
        }
    }

    async function handleApprove(suggestionId) {
        try {
            const result = await api.approveSuggestion(personId, suggestionId);
            if (result.status === 'cap_reached') {
                // Show replacement modal
                setReplaceModal({ suggestionId, ...result });
            } else {
                loadData();
            }
        } catch (e) {
            alert(e.message);
        }
    }

    async function handleApproveWithReplace(suggestionId, replaceEncodingId) {
        try {
            await api.approveSuggestion(personId, suggestionId, replaceEncodingId);
            setReplaceModal(null);
            loadData();
        } catch (e) {
            alert(e.message);
        }
    }

    async function handleReject(suggestionId) {
        try {
            await api.rejectSuggestion(personId, suggestionId);
            loadData();
        } catch (e) {
            alert(e.message);
        }
    }

    async function handleDeletePerson(hard = false) {
        setDeleting(true);
        try {
            if (hard) {
                await api.hardDeletePerson(personId);
            } else {
                await api.deletePerson(personId);
            }
            router.push('/persons');
        } catch (e) {
            alert(e.message);
            setDeleting(false);
        }
    }

    // ── Pose bucket summary for current encodings ──────────────

    function getPoseCoverage() {
        const buckets = {};
        encodings.forEach(enc => {
            const b = enc.pose_bucket || 'frontal';
            buckets[b] = (buckets[b] || 0) + 1;
        });
        return buckets;
    }

    // ── Render ─────────────────────────────────────────────────

    if (loading) {
        return (
            <AppShell>
                <div style={{ padding: '60px 20px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    Loading person details...
                </div>
            </AppShell>
        );
    }

    if (error || !person) {
        return (
            <AppShell>
                <div style={{ padding: '60px 20px', textAlign: 'center' }}>
                    <div style={{ color: 'var(--danger)', marginBottom: '16px' }}>
                        {error || 'Person not found'}
                    </div>
                    <button className="btn btn-primary" onClick={() => router.push('/persons')}>
                        <ArrowLeftIcon size={16} /> Back to Persons
                    </button>
                </div>
            </AppShell>
        );
    }

    const poseCoverage = getPoseCoverage();
    const allBuckets = ['frontal', 'left', 'right', 'extreme_left', 'extreme_right'];

    return (
        <AppShell>
            {/* ── Back + Header ──────────────────────────────── */}
            <div style={{ marginBottom: '20px' }}>
                <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => router.push('/persons')}
                    style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', marginBottom: '16px' }}
                >
                    <ArrowLeftIcon size={14} /> Back to Persons
                </button>

                <div className="card" style={{ padding: '24px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '20px', flexWrap: 'wrap' }}>
                        {person.avatar_url ? (
                            <img
                                src={`${API_BASE}${person.avatar_url}`}
                                alt={person.name}
                                style={{
                                    width: '80px', height: '80px', borderRadius: 'var(--radius-lg)',
                                    objectFit: 'cover', border: '3px solid var(--border-color)',
                                }}
                            />
                        ) : (
                            <div style={{
                                width: '80px', height: '80px', borderRadius: 'var(--radius-lg)',
                                background: 'linear-gradient(135deg, var(--primary), var(--accent))',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                fontSize: '2rem', fontWeight: 700, color: 'white',
                            }}>
                                {person.name.charAt(0)}
                            </div>
                        )}

                        <div style={{ flex: 1 }}>
                            <h1 style={{ margin: 0, fontSize: '1.5rem', color: 'var(--text-primary)' }}>
                                {person.name}
                            </h1>
                            <div style={{ display: 'flex', gap: '12px', marginTop: '8px', flexWrap: 'wrap' }}>
                                <span className="badge active">{person.role}</span>
                                {person.department && (
                                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                                        {person.department}
                                    </span>
                                )}
                            </div>
                        </div>

                        {/* Actions */}
                        <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
                            <button
                                className="btn btn-sm"
                                onClick={() => setShowDeleteModal(true)}
                                style={{
                                    background: 'rgba(239, 68, 68, 0.1)',
                                    color: 'var(--danger)',
                                    border: '1px solid rgba(239, 68, 68, 0.3)',
                                    display: 'flex', alignItems: 'center', gap: '6px',
                                    padding: '8px 14px', borderRadius: 'var(--radius-sm)',
                                    cursor: 'pointer', fontSize: '0.8rem', fontWeight: 600,
                                    transition: 'var(--transition-fast)',
                                }}
                                onMouseEnter={e => { e.target.style.background = 'rgba(239, 68, 68, 0.2)'; }}
                                onMouseLeave={e => { e.target.style.background = 'rgba(239, 68, 68, 0.1)'; }}
                            >
                                <TrashIcon size={14} /> Delete Person
                            </button>
                        </div>

                        {/* Stats */}
                        <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
                            <div style={{ textAlign: 'center' }}>
                                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--primary)', fontFamily: "'JetBrains Mono'" }}>
                                    {encodings.length}
                                </div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                    Encodings
                                </div>
                            </div>
                            <div style={{ textAlign: 'center' }}>
                                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: suggestions.length > 0 ? 'var(--warning)' : 'var(--text-muted)', fontFamily: "'JetBrains Mono'" }}>
                                    {suggestions.length}
                                </div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                    Suggestions
                                </div>
                            </div>
                            <div style={{ textAlign: 'center' }}>
                                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--success)', fontFamily: "'JetBrains Mono'" }}>
                                    {Object.keys(poseCoverage).length}/5
                                </div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                    Pose Coverage
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Pose Coverage Bar */}
                    <div style={{ marginTop: '20px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                        {allBuckets.map(bucket => {
                            const count = poseCoverage[bucket] || 0;
                            const label = bucket.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
                            return (
                                <div
                                    key={bucket}
                                    style={{
                                        padding: '6px 12px',
                                        borderRadius: 'var(--radius-sm)',
                                        background: count > 0 ? `${poseBucketColor(bucket)}22` : 'var(--bg-tertiary)',
                                        border: `1px solid ${count > 0 ? poseBucketColor(bucket) : 'var(--border-color)'}`,
                                        color: count > 0 ? poseBucketColor(bucket) : 'var(--text-muted)',
                                        fontSize: '0.75rem',
                                        fontWeight: 600,
                                        transition: 'var(--transition-fast)',
                                    }}
                                >
                                    {count > 0 ? '✓' : '○'} {label} {count > 0 && `(${count})`}
                                </div>
                            );
                        })}
                    </div>
                </div>
            </div>

            {/* ── Delete Confirmation Modal ─────────────────── */}
            {showDeleteModal && (
                <div style={{
                    position: 'fixed', inset: 0, zIndex: 1000,
                    background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                }} onClick={() => !deleting && setShowDeleteModal(false)}>
                    <div style={{
                        background: 'var(--bg-secondary)', borderRadius: 'var(--radius-lg)',
                        padding: '28px', maxWidth: '440px', width: '90%',
                        border: '1px solid var(--border-color)',
                        boxShadow: '0 20px 60px rgba(0,0,0,0.4)',
                    }} onClick={e => e.stopPropagation()}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                            <div style={{
                                width: '40px', height: '40px', borderRadius: '50%',
                                background: 'rgba(239, 68, 68, 0.15)', display: 'flex',
                                alignItems: 'center', justifyContent: 'center', color: 'var(--danger)',
                            }}>
                                <TrashIcon size={20} />
                            </div>
                            <div>
                                <h3 style={{ margin: 0, color: 'var(--text-primary)', fontSize: '1.1rem' }}>
                                    Delete {person.name}?
                                </h3>
                                <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                                    Choose how to remove this person
                                </p>
                            </div>
                        </div>

                        {/* Option 1: Deactivate */}
                        <button
                            disabled={deleting}
                            onClick={() => handleDeletePerson(false)}
                            style={{
                                width: '100%', padding: '14px 16px', marginBottom: '10px',
                                background: 'var(--bg-tertiary)', border: '1px solid var(--border-color)',
                                borderRadius: 'var(--radius-sm)', cursor: deleting ? 'wait' : 'pointer',
                                textAlign: 'left', color: 'var(--text-primary)',
                                transition: 'var(--transition-fast)',
                            }}
                            onMouseEnter={e => { if (!deleting) e.target.style.borderColor = 'var(--warning)'; }}
                            onMouseLeave={e => { e.target.style.borderColor = 'var(--border-color)'; }}
                        >
                            <div style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: '4px' }}>
                                ⏸ Deactivate
                            </div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                Hides from recognition but keeps all data. Can be reactivated later.
                            </div>
                        </button>

                        {/* Option 2: Permanent Delete */}
                        <div style={{
                            padding: '14px 16px', background: 'rgba(239, 68, 68, 0.05)',
                            border: '1px solid rgba(239, 68, 68, 0.2)', borderRadius: 'var(--radius-sm)',
                        }}>
                            <div style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: '4px', color: 'var(--danger)' }}>
                                🗑 Permanently Delete
                            </div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '12px' }}>
                                Removes person, all encodings, events, and tracking history. This cannot be undone.
                            </div>
                            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                                <input
                                    type="text"
                                    placeholder={`Type "${person.name}" to confirm`}
                                    value={deleteConfirmName}
                                    onChange={e => setDeleteConfirmName(e.target.value)}
                                    style={{
                                        flex: 1, padding: '8px 12px', borderRadius: 'var(--radius-sm)',
                                        border: '1px solid var(--border-color)', background: 'var(--bg-primary)',
                                        color: 'var(--text-primary)', fontSize: '0.85rem',
                                        outline: 'none',
                                    }}
                                />
                                <button
                                    disabled={deleteConfirmName !== person.name || deleting}
                                    onClick={() => handleDeletePerson(true)}
                                    style={{
                                        padding: '8px 16px', borderRadius: 'var(--radius-sm)',
                                        background: deleteConfirmName === person.name ? 'var(--danger)' : 'var(--bg-tertiary)',
                                        color: deleteConfirmName === person.name ? 'white' : 'var(--text-muted)',
                                        border: 'none', fontWeight: 600, fontSize: '0.8rem',
                                        cursor: deleteConfirmName === person.name && !deleting ? 'pointer' : 'not-allowed',
                                        transition: 'var(--transition-fast)',
                                        opacity: deleting ? 0.6 : 1,
                                    }}
                                >
                                    {deleting ? 'Deleting...' : 'Delete Forever'}
                                </button>
                            </div>
                        </div>

                        {/* Cancel */}
                        <button
                            onClick={() => { setShowDeleteModal(false); setDeleteConfirmName(''); }}
                            disabled={deleting}
                            style={{
                                width: '100%', padding: '10px', marginTop: '12px',
                                background: 'none', border: '1px solid var(--border-color)',
                                borderRadius: 'var(--radius-sm)', color: 'var(--text-secondary)',
                                cursor: 'pointer', fontSize: '0.85rem',
                            }}
                        >
                            Cancel
                        </button>
                    </div>
                </div>
            )}

            {/* ── Tab Navigation ─────────────────────────────── */}
            <div style={{ display: 'flex', gap: '4px', marginBottom: '20px', borderBottom: '1px solid var(--border-color)', paddingBottom: '0' }}>
                {[
                    { key: 'journey', label: 'Journey', count: null },
                    { key: 'encodings', label: 'Current Embeddings', count: encodings.length },
                    { key: 'suggestions', label: 'Suggested Embeddings', count: suggestions.length },
                    { key: 'timeline', label: 'Activity', count: null },
                ].map(tab => (
                    <button
                        key={tab.key}
                        onClick={() => setActiveTab(tab.key)}
                        style={{
                            padding: '12px 20px',
                            background: 'none',
                            border: 'none',
                            borderBottom: `2px solid ${activeTab === tab.key ? 'var(--primary)' : 'transparent'}`,
                            color: activeTab === tab.key ? 'var(--primary)' : 'var(--text-secondary)',
                            fontWeight: activeTab === tab.key ? 600 : 400,
                            fontSize: '0.875rem',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            transition: 'var(--transition-fast)',
                        }}
                    >
                        {tab.label}
                        {tab.count !== null && tab.count > 0 && (
                            <span style={{
                                background: tab.key === 'suggestions' && tab.count > 0
                                    ? 'var(--warning)' : 'var(--bg-tertiary)',
                                color: tab.key === 'suggestions' && tab.count > 0
                                    ? 'black' : 'var(--text-secondary)',
                                padding: '2px 8px',
                                borderRadius: '10px',
                                fontSize: '0.7rem',
                                fontWeight: 700,
                                fontFamily: "'JetBrains Mono'",
                            }}>
                                {tab.count}
                            </span>
                        )}
                    </button>
                ))}
            </div>

            {/* ── Journey ────────────────────────────────────── */}
            {activeTab === 'journey' && (
                <JourneyPanel personId={personId} />
            )}

            {/* ── Current Embeddings ─────────────────────────── */}
            {activeTab === 'encodings' && (
                <div>
                    {encodings.length === 0 ? (
                        <div className="card" style={{ padding: '60px 20px', textAlign: 'center' }}>
                            <div style={{ color: 'var(--text-muted)', marginBottom: '8px' }}>
                                <ImageIcon size={40} />
                            </div>
                            <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                                No face encodings enrolled yet.
                            </div>
                        </div>
                    ) : (
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
                            gap: '16px',
                        }}>
                            {encodings.map(enc => {
                                const q = qualityBar(enc.quality);
                                const h = healthLabel(enc.health_score);
                                return (
                                    <div key={enc.id} className="card" style={{ padding: '0', overflow: 'hidden', transition: 'var(--transition-normal)' }}>
                                        {/* Face image */}
                                        <div style={{
                                            height: '180px', background: 'var(--bg-tertiary)',
                                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                                            position: 'relative',
                                        }}>
                                            {enc.source_image ? (
                                                <img
                                                    src={`${API_BASE}${enc.source_image}`}
                                                    alt="Face encoding"
                                                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                                                    onError={(e) => { e.target.style.display = 'none'; }}
                                                />
                                            ) : (
                                                <UserIcon size={48} />
                                            )}

                                            {/* Pose bucket badge */}
                                            <span style={{
                                                position: 'absolute', top: '8px', left: '8px',
                                                background: `${poseBucketColor(enc.pose_bucket)}dd`,
                                                color: 'white', padding: '3px 8px', borderRadius: 'var(--radius-sm)',
                                                fontSize: '0.65rem', fontWeight: 700, textTransform: 'uppercase',
                                                backdropFilter: 'blur(8px)',
                                            }}>
                                                {enc.pose_bucket || 'frontal'}
                                            </span>

                                            {/* Health badge */}
                                            <span style={{
                                                position: 'absolute', top: '8px', right: '8px',
                                                background: 'rgba(0,0,0,0.7)', color: h.color,
                                                padding: '3px 8px', borderRadius: 'var(--radius-sm)',
                                                fontSize: '0.65rem', fontWeight: 700,
                                                backdropFilter: 'blur(8px)',
                                            }}>
                                                ♥ {h.text}
                                            </span>
                                        </div>

                                        {/* Metadata */}
                                        <div style={{ padding: '12px 14px' }}>
                                            {/* Quality bar */}
                                            <div style={{ marginBottom: '10px' }}>
                                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
                                                    <span>Quality</span>
                                                    <span style={{ color: q.color, fontFamily: "'JetBrains Mono'", fontWeight: 600 }}>{q.pct}%</span>
                                                </div>
                                                <div style={{ height: '4px', background: 'var(--bg-tertiary)', borderRadius: '2px', overflow: 'hidden' }}>
                                                    <div style={{ height: '100%', width: `${q.pct}%`, background: q.color, borderRadius: '2px', transition: 'width 0.5s ease' }} />
                                                </div>
                                            </div>

                                            {/* Stats grid */}
                                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                                                <div>Yaw: <span style={{ color: 'var(--text-secondary)', fontFamily: "'JetBrains Mono'" }}>{enc.yaw?.toFixed(1) ?? '—'}°</span></div>
                                                <div>Pitch: <span style={{ color: 'var(--text-secondary)', fontFamily: "'JetBrains Mono'" }}>{enc.pitch?.toFixed(1) ?? '—'}°</span></div>
                                                <div>Uses: <span style={{ color: 'var(--text-secondary)', fontFamily: "'JetBrains Mono'" }}>{enc.usage_count}</span></div>
                                                <div>Matches: <span style={{ color: 'var(--success)', fontFamily: "'JetBrains Mono'" }}>{enc.success_count}</span></div>
                                            </div>

                                            {/* Date + Delete */}
                                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '10px', paddingTop: '10px', borderTop: '1px solid var(--border-color)' }}>
                                                <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
                                                    {formatDate(enc.created_at)}
                                                </span>
                                                <button
                                                    className="btn btn-secondary btn-sm"
                                                    onClick={() => handleDeleteEncoding(enc.id)}
                                                    style={{ padding: '4px 8px', color: 'var(--danger)', display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.7rem' }}
                                                    id={`delete-enc-${enc.id}`}
                                                >
                                                    <TrashIcon size={12} /> Delete
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            )}

            {/* ── Suggested Embeddings ───────────────────────── */}
            {activeTab === 'suggestions' && (
                <div>
                    {suggestions.length === 0 ? (
                        <div className="card" style={{ padding: '60px 20px', textAlign: 'center' }}>
                            <div style={{ color: 'var(--text-muted)', marginBottom: '8px' }}>
                                <ShieldIcon size={40} />
                            </div>
                            <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                                No pending suggestions. The system will suggest new embeddings when it detects diverse, high-quality faces.
                            </div>
                        </div>
                    ) : (
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
                            gap: '16px',
                        }}>
                            {suggestions.map(sug => {
                                const q = qualityBar(sug.quality);
                                return (
                                    <div key={sug.id} className="card" style={{ padding: '0', overflow: 'hidden', border: '1px solid var(--warning)33' }}>
                                        {/* Face image */}
                                        <div style={{
                                            height: '200px', background: 'var(--bg-tertiary)',
                                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                                            position: 'relative',
                                        }}>
                                            {sug.snapshot_url ? (
                                                <img
                                                    src={`${API_BASE}${sug.snapshot_url}`}
                                                    alt="Suggested face"
                                                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                                                    onError={(e) => { e.target.style.display = 'none'; }}
                                                />
                                            ) : (
                                                <UserIcon size={48} />
                                            )}

                                            {/* Reason badge */}
                                            <div style={{
                                                position: 'absolute', bottom: '0', left: '0', right: '0',
                                                background: 'linear-gradient(transparent, rgba(0,0,0,0.85))',
                                                padding: '24px 12px 10px',
                                            }}>
                                                <span style={{
                                                    background: 'var(--warning)',
                                                    color: 'black',
                                                    padding: '4px 10px',
                                                    borderRadius: 'var(--radius-sm)',
                                                    fontSize: '0.7rem',
                                                    fontWeight: 700,
                                                }}>
                                                    💡 {sug.reason}
                                                </span>
                                            </div>

                                            {/* Pose bucket */}
                                            <span style={{
                                                position: 'absolute', top: '8px', left: '8px',
                                                background: `${poseBucketColor(sug.pose_bucket)}dd`,
                                                color: 'white', padding: '3px 8px', borderRadius: 'var(--radius-sm)',
                                                fontSize: '0.65rem', fontWeight: 700, textTransform: 'uppercase',
                                            }}>
                                                {sug.pose_bucket || 'frontal'}
                                            </span>

                                            {/* Similarity */}
                                            <span style={{
                                                position: 'absolute', top: '8px', right: '8px',
                                                background: 'rgba(0,0,0,0.7)', color: 'var(--success)',
                                                padding: '3px 8px', borderRadius: 'var(--radius-sm)',
                                                fontSize: '0.65rem', fontWeight: 700, fontFamily: "'JetBrains Mono'",
                                            }}>
                                                {((sug.similarity || 0) * 100).toFixed(0)}% match
                                            </span>
                                        </div>

                                        {/* Metadata */}
                                        <div style={{ padding: '12px 14px' }}>
                                            {/* Quality */}
                                            <div style={{ marginBottom: '10px' }}>
                                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
                                                    <span>Quality Score</span>
                                                    <span style={{ color: q.color, fontFamily: "'JetBrains Mono'", fontWeight: 600 }}>{q.pct}%</span>
                                                </div>
                                                <div style={{ height: '4px', background: 'var(--bg-tertiary)', borderRadius: '2px', overflow: 'hidden' }}>
                                                    <div style={{ height: '100%', width: `${q.pct}%`, background: q.color, borderRadius: '2px' }} />
                                                </div>
                                            </div>

                                            {/* Pose info */}
                                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '6px', fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '10px' }}>
                                                <div>Yaw: <span style={{ color: 'var(--text-secondary)', fontFamily: "'JetBrains Mono'" }}>{sug.yaw?.toFixed(1) ?? '—'}°</span></div>
                                                <div>Pitch: <span style={{ color: 'var(--text-secondary)', fontFamily: "'JetBrains Mono'" }}>{sug.pitch?.toFixed(1) ?? '—'}°</span></div>
                                                <div>Roll: <span style={{ color: 'var(--text-secondary)', fontFamily: "'JetBrains Mono'" }}>{sug.roll?.toFixed(1) ?? '—'}°</span></div>
                                            </div>

                                            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: '12px' }}>
                                                {formatDate(sug.created_at)}
                                            </div>

                                            {/* Actions */}
                                            <div style={{ display: 'flex', gap: '8px' }}>
                                                <button
                                                    className="btn btn-primary btn-sm"
                                                    onClick={() => handleApprove(sug.id)}
                                                    style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', fontSize: '0.8rem' }}
                                                    id={`approve-sug-${sug.id}`}
                                                >
                                                    <PlusIcon size={14} /> Add to Profile
                                                </button>
                                                <button
                                                    className="btn btn-secondary btn-sm"
                                                    onClick={() => handleReject(sug.id)}
                                                    style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--danger)', fontSize: '0.8rem' }}
                                                    id={`reject-sug-${sug.id}`}
                                                >
                                                    <XIcon size={14} /> Reject
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            )}

            {/* ── Activity Timeline ──────────────────────────── */}
            {activeTab === 'timeline' && (
                <PersonTimeline person={person} onClose={() => setActiveTab('encodings')} />
            )}

            {/* ── Replacement Modal ──────────────────────────── */}
            {replaceModal && (
                <div className="modal-overlay" onClick={() => setReplaceModal(null)}>
                    <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: '600px' }}>
                        <div className="modal-header">
                            <h3 className="modal-title">Encoding Limit Reached</h3>
                            <button className="modal-close" onClick={() => setReplaceModal(null)}>
                                <XIcon size={18} />
                            </button>
                        </div>

                        <div style={{ padding: '0 20px 8px', color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                            This person has <strong>{replaceModal.current_count}/{replaceModal.cap}</strong> encodings.
                            Choose one to replace with the new suggestion:
                        </div>

                        <div style={{ padding: '0 20px 20px' }}>
                            {replaceModal.replacement_candidates?.map(candidate => {
                                const h = healthLabel(candidate.health_score);
                                return (
                                    <div
                                        key={candidate.encoding_id}
                                        style={{
                                            display: 'flex', alignItems: 'center', gap: '12px',
                                            padding: '12px', borderRadius: 'var(--radius-md)',
                                            border: '1px solid var(--border-color)',
                                            marginBottom: '8px',
                                            cursor: 'pointer',
                                            transition: 'var(--transition-fast)',
                                        }}
                                        onClick={() => handleApproveWithReplace(replaceModal.suggestionId, candidate.encoding_id)}
                                        onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--warning)'}
                                        onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border-color)'}
                                    >
                                        <div style={{ flex: 1 }}>
                                            <div style={{ fontSize: '0.8rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                                                {candidate.pose_bucket?.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())}
                                            </div>
                                            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                                                Quality: {((candidate.quality || 0) * 100).toFixed(0)}% · Uses: {candidate.usage_count} · Matches: {candidate.success_count}
                                            </div>
                                        </div>
                                        <span style={{ color: h.color, fontSize: '0.7rem', fontWeight: 700 }}>
                                            ♥ {h.text} ({(candidate.health_score * 100).toFixed(0)}%)
                                        </span>
                                        <span style={{ color: 'var(--warning)', fontSize: '0.75rem' }}>Replace →</span>
                                    </div>
                                );
                            })}
                        </div>

                        <div style={{ padding: '12px 20px', borderTop: '1px solid var(--border-color)', display: 'flex', justifyContent: 'flex-end' }}>
                            <button className="btn btn-secondary" onClick={() => setReplaceModal(null)}>Cancel</button>
                        </div>
                    </div>
                </div>
            )}
        </AppShell>
    );
}
