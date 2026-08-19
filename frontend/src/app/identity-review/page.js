'use client';

/**
 * AI Identity Review — Background Learning System
 *
 * Three-tab layout:
 *   1. Dashboard: stats overview + pending review cards
 *   2. Clusters: unknown face clusters with actions
 *   3. Activity: audit trail of all review actions
 *
 * Review workspace modal (opens from candidate card):
 *   LEFT: Known person gallery poses
 *   CENTER: Unknown cluster snapshots + timeline
 *   RIGHT: AI analysis scores + reasoning
 */

import { useState, useEffect, useCallback } from 'react';
import AppShell from '@/components/AppShell';
import api, { API_BASE } from '@/lib/api';

// ── Inline Icons (self-contained) ──────────────────────────────

function BrainIcon({ size = 18 }) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2a4 4 0 0 1 4 4c0 1.95-1.4 3.57-3.25 3.92L12 10v2" />
            <path d="M12 16v2" />
            <circle cx="12" cy="14" r="2" />
            <path d="M8 6a4 4 0 0 0-4 4v1a2 2 0 0 0 2 2" />
            <path d="M16 6a4 4 0 0 1 4 4v1a2 2 0 0 1-2 2" />
            <path d="M9 20a6 6 0 0 1 6 0" />
            <path d="M6 20a9 9 0 0 1 12 0" />
        </svg>
    );
}

function CheckIcon({ size = 16 }) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12" />
        </svg>
    );
}

function XIcon({ size = 16 }) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
        </svg>
    );
}

function SkipIcon({ size = 16 }) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="5 4 15 12 5 20 5 4" /><line x1="19" y1="5" x2="19" y2="19" />
        </svg>
    );
}

function ClockIcon({ size = 14 }) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" />
        </svg>
    );
}

// ── Helpers ─────────────────────────────────────────────────────

function snapshotUrl(path) {
    if (!path) return null;
    if (path.startsWith('http')) return path;
    return `${API_BASE}/snapshots/${path.replace(/^\/?snapshots\//, '')}`;
}

function formatTimeAgo(isoStr) {
    if (!isoStr) return '';
    const diff = (Date.now() - new Date(isoStr).getTime()) / 1000;
    if (diff < 60) return `${Math.round(diff)}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
}

function ScoreBadge({ score, label }) {
    const pct = Math.round((score || 0) * 100);
    const color = pct >= 70 ? 'var(--success)' : pct >= 40 ? 'var(--warning)' : 'var(--danger)';
    return (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '4px 0' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{label}</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <div style={{
                    width: '50px', height: '4px', borderRadius: '2px',
                    background: 'var(--bg-tertiary)', overflow: 'hidden',
                }}>
                    <div style={{
                        width: `${pct}%`, height: '100%', borderRadius: '2px',
                        background: color, transition: 'width 0.3s',
                    }} />
                </div>
                <span style={{ fontSize: '0.688rem', fontWeight: 700, color, minWidth: '32px', textAlign: 'right' }}>
                    {pct}%
                </span>
            </div>
        </div>
    );
}

// ── Main Page Component ─────────────────────────────────────────

export default function IdentityReviewPage() {
    const [tab, setTab] = useState('dashboard');
    const [stats, setStats] = useState(null);
    const [candidates, setCandidates] = useState([]);
    const [clusters, setClusters] = useState([]);
    const [activity, setActivity] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedCandidate, setSelectedCandidate] = useState(null);
    const [candidateDetail, setCandidateDetail] = useState(null);
    const [detailLoading, setDetailLoading] = useState(false);
    const [actionLoading, setActionLoading] = useState(null);
    const [candidateFilter, setCandidateFilter] = useState('pending');

    const loadDashboard = useCallback(async () => {
        try {
            const [dashStats, cands, clusts, acts] = await Promise.all([
                api.getIdentityReviewDashboard(),
                api.getIdentityReviewCandidates('pending', 20),
                api.getIdentityReviewClusters('active', 10),
                api.getIdentityReviewActivity(20),
            ]);
            setStats(dashStats);
            setCandidates(cands?.items || []);
            setClusters(clusts?.items || []);
            setActivity(acts?.items || []);
        } catch (e) {
            console.error('Failed to load identity review data:', e);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadDashboard();
        const interval = setInterval(loadDashboard, 30000);
        return () => clearInterval(interval);
    }, [loadDashboard]);

    const loadCandidates = async (status) => {
        setCandidateFilter(status);
        try {
            const result = await api.getIdentityReviewCandidates(status, 50);
            setCandidates(result?.items || []);
        } catch (e) { console.error(e); }
    };

    const openDetail = async (candidateId) => {
        setSelectedCandidate(candidateId);
        setDetailLoading(true);
        try {
            const detail = await api.getIdentityReviewCandidate(candidateId);
            setCandidateDetail(detail);
        } catch (e) {
            console.error('Failed to load candidate detail:', e);
        } finally {
            setDetailLoading(false);
        }
    };

    const handleAction = async (candidateId, action) => {
        setActionLoading(candidateId);
        try {
            if (action === 'approve') await api.approveCandidate(candidateId);
            else if (action === 'reject') await api.rejectCandidate(candidateId);
            else if (action === 'skip') await api.skipCandidate(candidateId);
            // Refresh
            await loadDashboard();
            if (selectedCandidate === candidateId) {
                setSelectedCandidate(null);
                setCandidateDetail(null);
            }
        } catch (e) {
            console.error(`Action ${action} failed:`, e);
        } finally {
            setActionLoading(null);
        }
    };

    const handleIgnoreCluster = async (clusterId) => {
        try {
            await api.ignoreCluster(clusterId);
            await loadDashboard();
        } catch (e) { console.error(e); }
    };

    // ── Render ───────────────────────────────────────────────

    return (
        <AppShell>
            <div className="page-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={{
                        width: '40px', height: '40px', borderRadius: '12px',
                        background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(59, 130, 246, 0.15))',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        color: 'var(--primary)',
                    }}>
                        <BrainIcon size={22} />
                    </div>
                    <div>
                        <h1>AI Identity Review</h1>
                        <p>Background learning system — review AI-proposed identity matches</p>
                    </div>
                </div>
            </div>

            {/* Tab Navigation */}
            <div style={{
                display: 'flex', gap: '4px', padding: '4px',
                background: 'var(--bg-secondary)', borderRadius: '10px',
                marginBottom: '20px', width: 'fit-content',
            }}>
                {['dashboard', 'candidates', 'clusters', 'activity'].map(t => (
                    <button
                        key={t}
                        onClick={() => { setTab(t); if (t === 'candidates') loadCandidates(candidateFilter); }}
                        style={{
                            padding: '8px 16px', borderRadius: '8px', border: 'none',
                            background: tab === t ? 'var(--bg-primary)' : 'transparent',
                            color: tab === t ? 'var(--text-primary)' : 'var(--text-muted)',
                            fontWeight: tab === t ? 600 : 400,
                            fontSize: '0.8125rem', cursor: 'pointer',
                            boxShadow: tab === t ? '0 1px 3px rgba(0,0,0,0.1)' : 'none',
                            transition: 'all 0.15s',
                        }}
                    >
                        {t.charAt(0).toUpperCase() + t.slice(1)}
                        {t === 'candidates' && stats?.pending_reviews > 0 && (
                            <span style={{
                                marginLeft: '6px', padding: '1px 6px',
                                borderRadius: '8px', background: 'var(--primary)',
                                color: 'white', fontSize: '0.625rem', fontWeight: 700,
                            }}>{stats.pending_reviews}</span>
                        )}
                    </button>
                ))}
            </div>

            {loading ? (
                <div style={{ padding: '60px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    Loading AI Identity Review...
                </div>
            ) : tab === 'dashboard' ? (
                <DashboardTab
                    stats={stats}
                    candidates={candidates}
                    onOpenDetail={openDetail}
                    onAction={handleAction}
                    actionLoading={actionLoading}
                />
            ) : tab === 'candidates' ? (
                <CandidatesTab
                    candidates={candidates}
                    filter={candidateFilter}
                    onFilterChange={loadCandidates}
                    onOpenDetail={openDetail}
                    onAction={handleAction}
                    actionLoading={actionLoading}
                />
            ) : tab === 'clusters' ? (
                <ClustersTab
                    clusters={clusters}
                    onIgnore={handleIgnoreCluster}
                />
            ) : (
                <ActivityTab activity={activity} />
            )}

            {/* Detail Modal */}
            {selectedCandidate && (
                <DetailModal
                    detail={candidateDetail}
                    loading={detailLoading}
                    onClose={() => { setSelectedCandidate(null); setCandidateDetail(null); }}
                    onAction={handleAction}
                    actionLoading={actionLoading}
                />
            )}
        </AppShell>
    );
}

// ── Dashboard Tab ───────────────────────────────────────────────

function DashboardTab({ stats, candidates, onOpenDetail, onAction, actionLoading }) {
    const statCards = [
        { label: 'Unknown Faces', value: stats?.unknown_faces || 0, color: 'var(--warning)' },
        { label: 'Active Clusters', value: stats?.active_clusters || 0, color: 'var(--primary)' },
        { label: 'Pending Reviews', value: stats?.pending_reviews || 0, color: 'var(--info, #3b82f6)' },
        { label: 'Approved Today', value: stats?.approved_today || 0, color: 'var(--success)' },
        { label: 'Rejected Today', value: stats?.rejected_today || 0, color: 'var(--danger)' },
        { label: 'Gallery Improvements', value: stats?.gallery_improvements || 0, color: 'var(--success)' },
    ];

    return (
        <>
            {/* Stats Grid */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
                gap: '12px', marginBottom: '24px',
            }}>
                {statCards.map(s => (
                    <div key={s.label} className="card" style={{ padding: '16px' }}>
                        <div style={{ fontSize: '0.688rem', color: 'var(--text-muted)', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                            {s.label}
                        </div>
                        <div style={{ fontSize: '1.5rem', fontWeight: 700, color: s.color, marginTop: '4px' }}>
                            {s.value}
                        </div>
                    </div>
                ))}
            </div>

            {/* Pending Review Cards */}
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">Pending Reviews</h3>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        {candidates.length} candidates
                    </span>
                </div>

                {candidates.length === 0 ? (
                    <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                        <BrainIcon size={32} />
                        <div style={{ marginTop: '12px', fontWeight: 500 }}>No pending reviews</div>
                        <div style={{ fontSize: '0.75rem', marginTop: '4px' }}>
                            The AI will generate review candidates as unknowns are detected.
                        </div>
                    </div>
                ) : (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '12px' }}>
                        {candidates.slice(0, 8).map(c => (
                            <CandidateCard
                                key={c.id}
                                candidate={c}
                                onOpenDetail={onOpenDetail}
                                onAction={onAction}
                                actionLoading={actionLoading}
                            />
                        ))}
                    </div>
                )}
            </div>
        </>
    );
}

// ── Candidate Card ──────────────────────────────────────────────

function CandidateCard({ candidate: c, onOpenDetail, onAction, actionLoading }) {
    const pct = Math.round((c.overall_score || 0) * 100);
    const scoreColor = pct >= 70 ? 'var(--success)' : pct >= 40 ? 'var(--warning)' : 'var(--danger)';

    return (
        <div
            className="card"
            style={{
                padding: '14px', cursor: 'pointer',
                border: '1px solid var(--border-color)',
                transition: 'border-color 0.15s, box-shadow 0.15s',
            }}
            onClick={() => onOpenDetail(c.id)}
            onMouseOver={(e) => { e.currentTarget.style.borderColor = 'var(--primary)'; e.currentTarget.style.boxShadow = '0 0 0 1px var(--primary)'; }}
            onMouseOut={(e) => { e.currentTarget.style.borderColor = 'var(--border-color)'; e.currentTarget.style.boxShadow = 'none'; }}
        >
            {/* Images Row */}
            <div style={{ display: 'flex', gap: '10px', marginBottom: '12px', alignItems: 'center' }}>
                <div style={{
                    width: '56px', height: '56px', borderRadius: '10px', overflow: 'hidden',
                    border: '2px solid var(--warning)', flexShrink: 0, background: '#000',
                }}>
                    {c.cluster_snapshot ? (
                        <img src={snapshotUrl(c.cluster_snapshot)} alt="Unknown" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    ) : (
                        <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.625rem' }}>?</div>
                    )}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', color: 'var(--text-muted)', fontSize: '1.2rem', padding: '0 4px' }}>→</div>
                <div style={{
                    width: '56px', height: '56px', borderRadius: '10px', overflow: 'hidden',
                    border: '2px solid var(--success)', flexShrink: 0, background: '#000',
                }}>
                    {c.person_avatar ? (
                        <img src={snapshotUrl(c.person_avatar)} alt={c.person_name} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    ) : (
                        <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                            {(c.person_name || '?')[0]}
                        </div>
                    )}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                        {c.person_name || 'Unknown'}
                    </div>
                    <div style={{ fontSize: '0.688rem', color: 'var(--text-muted)' }}>
                        {c.cluster_member_count} sightings in cluster
                    </div>
                </div>
            </div>

            {/* Score Bar */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                <div style={{
                    flex: 1, height: '6px', borderRadius: '3px',
                    background: 'var(--bg-tertiary)', overflow: 'hidden',
                }}>
                    <div style={{
                        width: `${pct}%`, height: '100%', borderRadius: '3px',
                        background: `linear-gradient(90deg, ${scoreColor}, ${scoreColor}cc)`,
                        transition: 'width 0.5s',
                    }} />
                </div>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: scoreColor, minWidth: '36px' }}>
                    {pct}%
                </span>
            </div>

            {/* Action Row */}
            <div style={{ display: 'flex', gap: '6px' }} onClick={e => e.stopPropagation()}>
                {c.recommended_action === 'approve' && (
                    <button
                        className="btn btn-sm"
                        style={{ flex: 1, background: 'rgba(16, 185, 129, 0.1)', color: 'var(--success)', border: '1px solid rgba(16, 185, 129, 0.3)' }}
                        onClick={() => onAction(c.id, 'approve')}
                        disabled={actionLoading === c.id}
                    >
                        <CheckIcon size={12} /> Approve
                    </button>
                )}
                <button
                    className="btn btn-sm"
                    style={{ flex: 1, background: 'rgba(239, 68, 68, 0.08)', color: 'var(--danger)', border: '1px solid rgba(239, 68, 68, 0.2)' }}
                    onClick={() => onAction(c.id, 'reject')}
                    disabled={actionLoading === c.id}
                >
                    <XIcon size={12} /> Reject
                </button>
                <button
                    className="btn btn-sm"
                    style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)', border: '1px solid var(--border-color)' }}
                    onClick={() => onAction(c.id, 'skip')}
                    disabled={actionLoading === c.id}
                >
                    <SkipIcon size={12} />
                </button>
            </div>
        </div>
    );
}

// ── Candidates Tab ──────────────────────────────────────────────

function CandidatesTab({ candidates, filter, onFilterChange, onOpenDetail, onAction, actionLoading }) {
    return (
        <>
            <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
                {['pending', 'approved', 'rejected', 'skipped'].map(s => (
                    <button
                        key={s}
                        className={`btn btn-sm ${filter === s ? 'btn-primary' : ''}`}
                        onClick={() => onFilterChange(s)}
                        style={{
                            textTransform: 'capitalize',
                            ...(filter !== s ? { background: 'var(--bg-secondary)', color: 'var(--text-secondary)', border: '1px solid var(--border-color)' } : {}),
                        }}
                    >
                        {s}
                    </button>
                ))}
            </div>

            {candidates.length === 0 ? (
                <div className="card" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    No {filter} candidates found.
                </div>
            ) : (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '12px' }}>
                    {candidates.map(c => (
                        <CandidateCard
                            key={c.id}
                            candidate={c}
                            onOpenDetail={onOpenDetail}
                            onAction={onAction}
                            actionLoading={actionLoading}
                        />
                    ))}
                </div>
            )}
        </>
    );
}

// ── Clusters Tab ────────────────────────────────────────────────

function ClustersTab({ clusters, onIgnore }) {
    if (clusters.length === 0) {
        return (
            <div className="card" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
                <BrainIcon size={32} />
                <div style={{ marginTop: '12px', fontWeight: 500 }}>No active clusters</div>
                <div style={{ fontSize: '0.75rem', marginTop: '4px' }}>
                    Clusters will form automatically as unknown faces are detected across multiple frames.
                </div>
            </div>
        );
    }

    return (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '12px' }}>
            {clusters.map(c => (
                <div key={c.id} className="card" style={{ padding: '14px' }}>
                    <div style={{ display: 'flex', gap: '12px', marginBottom: '10px' }}>
                        <div style={{
                            width: '64px', height: '64px', borderRadius: '12px', overflow: 'hidden',
                            background: '#000', border: '2px solid var(--border-color)', flexShrink: 0,
                        }}>
                            {c.representative_snapshot ? (
                                <img src={snapshotUrl(c.representative_snapshot)} alt="Cluster" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                            ) : (
                                <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.75rem' }}>?</div>
                            )}
                        </div>
                        <div>
                            <div style={{ fontWeight: 600, fontSize: '0.875rem', color: 'var(--text-primary)' }}>
                                Cluster ({c.member_count} faces)
                            </div>
                            <div style={{ fontSize: '0.688rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                                First seen: {formatTimeAgo(c.first_seen)}
                            </div>
                            <div style={{ fontSize: '0.688rem', color: 'var(--text-muted)' }}>
                                Last seen: {formatTimeAgo(c.last_seen)}
                            </div>
                        </div>
                    </div>
                    <div style={{ display: 'flex', gap: '6px' }}>
                        <span className={`badge ${c.status}`} style={{ fontSize: '0.625rem' }}>{c.status}</span>
                        {c.status === 'active' && (
                            <button
                                className="btn btn-sm"
                                style={{ marginLeft: 'auto', fontSize: '0.688rem', background: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}
                                onClick={() => onIgnore(c.id)}
                            >
                                Ignore
                            </button>
                        )}
                    </div>
                </div>
            ))}
        </div>
    );
}

// ── Activity Tab ────────────────────────────────────────────────

function ActivityTab({ activity }) {
    const actionIcons = {
        approved: { color: 'var(--success)', icon: '✅' },
        rejected: { color: 'var(--danger)', icon: '❌' },
        skipped: { color: 'var(--text-muted)', icon: '⏭️' },
        created_person: { color: 'var(--primary)', icon: '👤' },
        ignored_forever: { color: 'var(--text-muted)', icon: '🚫' },
        merged_cluster: { color: 'var(--info, #3b82f6)', icon: '🔗' },
    };

    if (activity.length === 0) {
        return (
            <div className="card" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
                No review activity yet.
            </div>
        );
    }

    return (
        <div className="card">
            <div className="card-header">
                <h3 className="card-title">Review Activity Timeline</h3>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column' }}>
                {activity.map((a, i) => {
                    const info = actionIcons[a.action] || { color: 'var(--text-muted)', icon: '📋' };
                    return (
                        <div
                            key={a.id}
                            style={{
                                display: 'flex', alignItems: 'center', gap: '12px',
                                padding: '10px 14px',
                                borderBottom: i < activity.length - 1 ? '1px solid var(--border-color)' : 'none',
                            }}
                        >
                            <span style={{ fontSize: '1rem' }}>{info.icon}</span>
                            <div style={{ flex: 1 }}>
                                <div style={{ fontSize: '0.8125rem', fontWeight: 500, color: info.color }}>
                                    {a.action.replace(/_/g, ' ')}
                                </div>
                                {a.details?.person_name && (
                                    <div style={{ fontSize: '0.688rem', color: 'var(--text-muted)' }}>
                                        → {a.details.person_name}
                                    </div>
                                )}
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--text-muted)', fontSize: '0.688rem' }}>
                                <ClockIcon size={12} />
                                {formatTimeAgo(a.created_at)}
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

// ── Detail Modal ────────────────────────────────────────────────

function DetailModal({ detail, loading, onClose, onAction, actionLoading }) {
    if (loading) {
        return (
            <div className="ir-modal-backdrop" onClick={onClose}>
                <div className="ir-modal-container" onClick={e => e.stopPropagation()}>
                    <div style={{ padding: '60px', textAlign: 'center', color: 'var(--text-muted)' }}>
                        Loading candidate details...
                    </div>
                </div>
            </div>
        );
    }

    if (!detail) return null;

    const d = detail;
    const cluster = d.cluster || {};
    const person = d.person || {};
    const pct = Math.round((d.overall_score || 0) * 100);

    return (
        <div className="ir-modal-backdrop" onClick={onClose}>
            <div className="ir-modal-container" onClick={e => e.stopPropagation()}>
                {/* Header */}
                <div style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    padding: '16px 20px', borderBottom: '1px solid var(--border-color)',
                }}>
                    <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 700 }}>
                        Review: {person.name || 'Unknown'} Match
                    </h3>
                    <button className="btn btn-sm" onClick={onClose} style={{ background: 'none', border: 'none', padding: '4px' }}>
                        <XIcon size={18} />
                    </button>
                </div>

                {/* Three-column layout */}
                <div style={{
                    display: 'grid', gridTemplateColumns: '1fr 1.2fr 1fr',
                    gap: '1px', background: 'var(--border-color)',
                    minHeight: '400px',
                }}>
                    {/* LEFT: Person Gallery */}
                    <div style={{ background: 'var(--bg-primary)', padding: '16px' }}>
                        <h4 style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--success)', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                            Registered Gallery
                        </h4>
                        <div style={{
                            width: '80px', height: '80px', borderRadius: '12px', overflow: 'hidden',
                            border: '2px solid var(--success)', marginBottom: '10px', background: '#000',
                        }}>
                            {person.avatar_url ? (
                                <img src={snapshotUrl(person.avatar_url)} alt={person.name} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                            ) : (
                                <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '1.5rem' }}>
                                    {(person.name || '?')[0]}
                                </div>
                            )}
                        </div>
                        <div style={{ fontWeight: 600, marginBottom: '4px' }}>{person.name}</div>
                        <div style={{ fontSize: '0.688rem', color: 'var(--text-muted)' }}>{person.role} {person.department ? `• ${person.department}` : ''}</div>

                        {/* Gallery poses */}
                        <div style={{ marginTop: '12px' }}>
                            <div style={{ fontSize: '0.688rem', color: 'var(--text-muted)', marginBottom: '6px' }}>
                                Gallery Encodings ({(person.gallery_images || []).length})
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '4px' }}>
                                {(person.gallery_images || []).slice(0, 9).map(img => (
                                    <div key={img.id} style={{
                                        aspectRatio: '1', borderRadius: '6px', overflow: 'hidden',
                                        background: '#000', border: '1px solid var(--border-color)',
                                    }}>
                                        {img.source_image ? (
                                            <img src={snapshotUrl(img.source_image)} alt="Gallery" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                                        ) : (
                                            <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.5rem', color: 'var(--text-muted)' }}>
                                                {img.pose_bucket || 'enc'}
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* CENTER: Cluster Snapshots */}
                    <div style={{ background: 'var(--bg-primary)', padding: '16px' }}>
                        <h4 style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--warning)', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                            Unknown Cluster ({cluster.member_count || 0} sightings)
                        </h4>

                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '6px', marginBottom: '12px' }}>
                            {(cluster.members || []).slice(0, 12).map(m => (
                                <div key={m.id} style={{
                                    aspectRatio: '1', borderRadius: '8px', overflow: 'hidden',
                                    background: '#000', border: '1px solid var(--border-color)',
                                    position: 'relative',
                                }}>
                                    {m.snapshot_url ? (
                                        <img src={snapshotUrl(m.snapshot_url)} alt="Unknown" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                                    ) : (
                                        <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.625rem' }}>?</div>
                                    )}
                                    <div style={{
                                        position: 'absolute', bottom: '2px', right: '2px',
                                        background: 'rgba(0,0,0,0.7)', borderRadius: '3px',
                                        padding: '1px 4px', fontSize: '0.5rem', color: 'white',
                                    }}>
                                        ×{m.occurrence}
                                    </div>
                                </div>
                            ))}
                        </div>

                        {cluster.first_seen && (
                            <div style={{ fontSize: '0.688rem', color: 'var(--text-muted)' }}>
                                First seen: {formatTimeAgo(cluster.first_seen)}<br />
                                Last seen: {formatTimeAgo(cluster.last_seen)}
                            </div>
                        )}
                    </div>

                    {/* RIGHT: AI Analysis */}
                    <div style={{ background: 'var(--bg-primary)', padding: '16px' }}>
                        <h4 style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--primary)', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                            AI Analysis
                        </h4>

                        <div style={{
                            textAlign: 'center', padding: '12px', borderRadius: '10px',
                            background: 'var(--bg-secondary)', marginBottom: '12px',
                        }}>
                            <div style={{ fontSize: '2rem', fontWeight: 800, color: pct >= 70 ? 'var(--success)' : pct >= 40 ? 'var(--warning)' : 'var(--danger)' }}>
                                {pct}%
                            </div>
                            <div style={{ fontSize: '0.688rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                Overall Confidence
                            </div>
                        </div>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', marginBottom: '14px' }}>
                            <ScoreBadge score={d.embedding_similarity} label="Embedding" />
                            {d.verifier_similarity != null && <ScoreBadge score={d.verifier_similarity} label="Verifier" />}
                            {d.vlm_confidence != null && <ScoreBadge score={d.vlm_confidence} label="VLM" />}
                            <ScoreBadge score={d.image_quality_score} label="Quality" />
                            {d.cluster_consistency != null && <ScoreBadge score={d.cluster_consistency} label="Cluster" />}
                        </div>

                        {d.vlm_explanation && (
                            <div style={{
                                padding: '10px', borderRadius: '8px', background: 'var(--bg-secondary)',
                                fontSize: '0.75rem', color: 'var(--text-secondary)',
                                lineHeight: 1.5, marginBottom: '14px',
                            }}>
                                <div style={{ fontWeight: 600, marginBottom: '4px', fontSize: '0.688rem', color: 'var(--primary)' }}>
                                    AI Reasoning
                                </div>
                                {d.vlm_explanation}
                            </div>
                        )}

                        <div style={{ fontSize: '0.688rem', color: 'var(--text-muted)', marginBottom: '12px' }}>
                            Recommended: <strong style={{ color: d.recommended_action === 'approve' ? 'var(--success)' : d.recommended_action === 'reject' ? 'var(--danger)' : 'var(--text-secondary)' }}>
                                {d.recommended_action}
                            </strong>
                        </div>
                    </div>
                </div>

                {/* Action Footer */}
                {d.status === 'pending' && (
                    <div style={{
                        display: 'flex', gap: '8px', padding: '14px 20px',
                        borderTop: '1px solid var(--border-color)',
                        justifyContent: 'flex-end',
                    }}>
                        <button
                            className="btn"
                            style={{ background: 'rgba(16, 185, 129, 0.12)', color: 'var(--success)', border: '1px solid rgba(16, 185, 129, 0.3)' }}
                            onClick={() => onAction(d.id, 'approve')}
                            disabled={actionLoading === d.id}
                        >
                            <CheckIcon size={14} /> Approve
                        </button>
                        <button
                            className="btn"
                            style={{ background: 'rgba(239, 68, 68, 0.08)', color: 'var(--danger)', border: '1px solid rgba(239, 68, 68, 0.2)' }}
                            onClick={() => onAction(d.id, 'reject')}
                            disabled={actionLoading === d.id}
                        >
                            <XIcon size={14} /> Reject
                        </button>
                        <button
                            className="btn"
                            style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)', border: '1px solid var(--border-color)' }}
                            onClick={() => onAction(d.id, 'skip')}
                            disabled={actionLoading === d.id}
                        >
                            <SkipIcon size={14} /> Skip
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}
