'use client';

/**
 * System Health Dashboard — Real-time observability for SentinelAI.
 *
 * Displays:
 *   - Recognition accuracy (known/unknown/gray-zone rates)
 *   - Pipeline timing (FPS, SCRFD, ArcFace, JPEG)
 *   - Gallery stats (embeddings, pose coverage, suggestions)
 *   - VLM / Gallery Optimizer status
 *   - Tracking & identity recovery metrics
 *   - Unknown face statistics
 */

import { useState, useEffect, useCallback } from 'react';
import api from '@/lib/api';

// ── Micro Icons ──────────────────────────────────────────────────

function PulseIcon({ color = 'var(--success)', size = 10 }) {
    return (
        <span style={{
            display: 'inline-block', width: size, height: size,
            borderRadius: '50%', background: color,
            boxShadow: `0 0 6px ${color}`,
            animation: 'pulse 2s infinite',
        }} />
    );
}

function TrendArrow({ value, suffix = '' }) {
    if (!value) return null;
    const up = value > 0;
    return (
        <span style={{
            fontSize: '0.625rem', fontWeight: 600, marginLeft: '4px',
            color: up ? 'var(--success)' : 'var(--danger)',
        }}>
            {up ? '↑' : '↓'} {Math.abs(value)}{suffix}
        </span>
    );
}

// ── Stat Card ────────────────────────────────────────────────────

function StatCard({ label, value, unit = '', color, subtitle, icon }) {
    return (
        <div style={{
            background: 'var(--bg-secondary)', borderRadius: '12px',
            padding: '16px 18px', border: '1px solid var(--border-color)',
            display: 'flex', flexDirection: 'column', gap: '4px',
            transition: 'all 0.2s',
            minWidth: 0,
        }}>
            <div style={{
                fontSize: '0.6875rem', color: 'var(--text-muted)',
                fontWeight: 500, letterSpacing: '0.03em',
                textTransform: 'uppercase', display: 'flex',
                alignItems: 'center', gap: '6px',
            }}>
                {icon && <span style={{ fontSize: '0.875rem' }}>{icon}</span>}
                {label}
            </div>
            <div style={{
                fontSize: '1.5rem', fontWeight: 700,
                color: color || 'var(--text-primary)',
                lineHeight: 1.1,
            }}>
                {value}{unit && <span style={{ fontSize: '0.75rem', fontWeight: 500, marginLeft: '2px', color: 'var(--text-muted)' }}>{unit}</span>}
            </div>
            {subtitle && (
                <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                    {subtitle}
                </div>
            )}
        </div>
    );
}

// ── Progress Bar ─────────────────────────────────────────────────

function ProgressBar({ value, max = 100, color = 'var(--primary)', label, height = 6 }) {
    const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
    return (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {label && <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', minWidth: '60px' }}>{label}</span>}
            <div style={{
                flex: 1, height, borderRadius: height / 2,
                background: 'var(--bg-tertiary)', overflow: 'hidden',
            }}>
                <div style={{
                    width: `${pct}%`, height: '100%', borderRadius: height / 2,
                    background: `linear-gradient(90deg, ${color}, ${color}cc)`,
                    transition: 'width 0.5s ease',
                }} />
            </div>
            <span style={{ fontSize: '0.6875rem', fontWeight: 600, color, minWidth: '32px', textAlign: 'right' }}>
                {Math.round(pct)}%
            </span>
        </div>
    );
}

// ── Section Header ───────────────────────────────────────────────

function SectionHeader({ title, icon, status }) {
    return (
        <div style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            marginBottom: '12px', paddingBottom: '8px',
            borderBottom: '1px solid var(--border-color)',
        }}>
            <span style={{ fontSize: '1rem' }}>{icon}</span>
            <h3 style={{
                margin: 0, fontSize: '0.875rem', fontWeight: 600,
                color: 'var(--text-primary)',
            }}>{title}</h3>
            {status && (
                <span style={{
                    marginLeft: 'auto', fontSize: '0.625rem',
                    padding: '2px 8px', borderRadius: '6px',
                    background: status === 'healthy' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                    color: status === 'healthy' ? 'var(--success)' : 'var(--danger)',
                    fontWeight: 600,
                }}>
                    {status === 'healthy' ? '● Healthy' : '● Issue'}
                </span>
            )}
        </div>
    );
}

// ── Main Page ────────────────────────────────────────────────────

export default function SystemHealthPage() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [lastUpdate, setLastUpdate] = useState(null);

    const loadData = useCallback(async () => {
        try {
            const result = await api.getSystemHealth();
            setData(result);
            setLastUpdate(new Date());
        } catch (e) {
            console.error('Failed to load system health:', e);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadData();
        const interval = setInterval(loadData, 5000); // Refresh every 5s
        return () => clearInterval(interval);
    }, [loadData]);

    if (loading) {
        return (
            <div style={{ padding: '60px', textAlign: 'center', color: 'var(--text-muted)' }}>
                Loading System Health...
            </div>
        );
    }

    const rec = data?.recognition || {};
    const pipe = data?.pipeline || {};
    const gallery = data?.gallery || {};
    const vlm = data?.vlm || {};
    const tracking = data?.tracking || {};
    const unknowns = data?.unknowns || {};
    const snapWorker = data?.snapshot_worker || {};
    const shadow = data?.shadow_gallery || {};

    const fpsColor = pipe.fps >= 20 ? 'var(--success)' : pipe.fps >= 10 ? 'var(--warning)' : 'var(--danger)';
    const workerColor = vlm.worker_running ? 'var(--success)' : 'var(--danger)';

    return (
        <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 8px' }}>
            {/* Header */}
            <div style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                marginBottom: '20px', flexWrap: 'wrap', gap: '8px',
            }}>
                <div>
                    <h1 style={{
                        margin: 0, fontSize: '1.25rem', fontWeight: 700,
                        color: 'var(--text-primary)',
                        display: 'flex', alignItems: 'center', gap: '8px',
                    }}>
                        🛡️ System Health
                    </h1>
                    <p style={{
                        margin: '4px 0 0', fontSize: '0.75rem',
                        color: 'var(--text-muted)',
                    }}>
                        Real-time pipeline observability · Auto-refresh 5s
                        {lastUpdate && (
                            <span> · Last: {lastUpdate.toLocaleTimeString()}</span>
                        )}
                    </p>
                </div>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <PulseIcon color={fpsColor} />
                    <span style={{ fontSize: '0.75rem', fontWeight: 600, color: fpsColor }}>
                        {pipe.fps || 0} FPS
                    </span>
                </div>
            </div>

            {/* ── Top Stats Row ─────────────────────────────────── */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
                gap: '10px', marginBottom: '20px',
            }}>
                <StatCard label="FPS" value={pipe.fps || 0} color={fpsColor} icon="⚡" subtitle={`SCRFD ${pipe.scrfd_ms}ms · ArcFace ${pipe.arcface_ms}ms`} />
                <StatCard label="Cache Hit Rate" value={rec.cache_hit_pct || 0} unit="%" color="var(--primary)" icon="💾" subtitle={`${rec.cache_hits || 0} hits · ${rec.cache_saves || 0} calls`} />
                <StatCard label="Gallery" value={gallery.total_embeddings || 0} icon="🖼️" subtitle={`${gallery.active_persons || 0} persons · ${gallery.avg_per_person || 0}/person`} />
                <StatCard label="VLM Queue" value={vlm.queue_length || 0} color={vlm.queue_length > 10 ? 'var(--warning)' : 'var(--text-primary)'} icon="🤖" subtitle={vlm.worker_running ? `Cycle #${vlm.worker_cycles}` : 'Worker stopped'} />
                <StatCard label="Unknowns / hr" value={unknowns.created_last_hour || 0} icon="👤" subtitle={`${unknowns.total_pending || 0} pending total`} />
                <StatCard label="Active Tracks" value={tracking.active_tracks || 0} icon="📍" subtitle={`${tracking.total_entries || 0} entries`} />
            </div>

            {/* ── Two-Column Layout ────────────────────────────── */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
                gap: '16px',
            }}>
                {/* ── Recognition ─────────────────────────────── */}
                <div style={{
                    background: 'var(--bg-secondary)', borderRadius: '14px',
                    padding: '18px', border: '1px solid var(--border-color)',
                }}>
                    <SectionHeader title="Recognition" icon="🎯" status="healthy" />
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        <ProgressBar value={rec.known_pct || 0} color="var(--success)" label="Known" />
                        <ProgressBar value={rec.unknown_pct || 0} color="var(--danger)" label="Unknown" />
                        <ProgressBar value={rec.gray_zone_pct || 0} color="var(--warning)" label="Gray Zone" />
                        <div style={{
                            display: 'grid', gridTemplateColumns: '1fr 1fr',
                            gap: '8px', marginTop: '6px',
                        }}>
                            <div style={{
                                padding: '8px 10px', borderRadius: '8px',
                                background: 'var(--bg-primary)',
                                border: '1px solid var(--border-color)',
                            }}>
                                <div style={{ fontSize: '0.625rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Avg Latency</div>
                                <div style={{ fontSize: '1rem', fontWeight: 700 }}>{rec.latency_ms || 0}<span style={{ fontSize: '0.625rem', color: 'var(--text-muted)' }}>ms</span></div>
                            </div>
                            <div style={{
                                padding: '8px 10px', borderRadius: '8px',
                                background: 'var(--bg-primary)',
                                border: '1px solid var(--border-color)',
                            }}>
                                <div style={{ fontSize: '0.625rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Total</div>
                                <div style={{ fontSize: '1rem', fontWeight: 700 }}>{(rec.total_recognitions || 0).toLocaleString()}</div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* ── Pipeline Timing ────────────────────────── */}
                <div style={{
                    background: 'var(--bg-secondary)', borderRadius: '14px',
                    padding: '18px', border: '1px solid var(--border-color)',
                }}>
                    <SectionHeader title="Pipeline Timing" icon="⏱️" />
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        {[
                            { label: 'SCRFD (Detection)', value: pipe.scrfd_ms, max: 60, color: '#8b5cf6' },
                            { label: 'ArcFace (Recognition)', value: pipe.arcface_ms, max: 10, color: '#06b6d4' },
                            { label: 'JPEG Encode', value: pipe.jpeg_ms, max: 30, color: '#f59e0b' },
                            { label: 'Snapshot Save', value: pipe.snapshot_ms, max: 50, color: '#ec4899' },
                        ].map(item => (
                            <div key={item.label} style={{
                                display: 'flex', alignItems: 'center', gap: '8px',
                            }}>
                                <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', minWidth: '120px' }}>{item.label}</span>
                                <div style={{
                                    flex: 1, height: '8px', borderRadius: '4px',
                                    background: 'var(--bg-tertiary)', overflow: 'hidden',
                                }}>
                                    <div style={{
                                        width: `${Math.min(100, ((item.value || 0) / item.max) * 100)}%`,
                                        height: '100%', borderRadius: '4px',
                                        background: `linear-gradient(90deg, ${item.color}, ${item.color}88)`,
                                        transition: 'width 0.5s',
                                    }} />
                                </div>
                                <span style={{
                                    fontSize: '0.75rem', fontWeight: 700,
                                    color: item.color, minWidth: '42px', textAlign: 'right',
                                }}>
                                    {(item.value || 0).toFixed(1)}ms
                                </span>
                            </div>
                        ))}
                        {pipe.gpu && pipe.gpu.used_mb > 0 && (
                            <div style={{
                                marginTop: '8px', padding: '8px 10px', borderRadius: '8px',
                                background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                                display: 'flex', alignItems: 'center', gap: '8px',
                            }}>
                                <span style={{ fontSize: '0.75rem' }}>🎮</span>
                                <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)' }}>GPU</span>
                                <span style={{ fontSize: '0.75rem', fontWeight: 600 }}>
                                    {pipe.gpu.used_mb}MB / {pipe.gpu.total_mb}MB
                                </span>
                                <span style={{ fontSize: '0.625rem', color: 'var(--text-muted)' }}>
                                    ({pipe.gpu.utilization}%)
                                </span>
                            </div>
                        )}
                    </div>
                </div>

                {/* ── Gallery ────────────────────────────────── */}
                <div style={{
                    background: 'var(--bg-secondary)', borderRadius: '14px',
                    padding: '18px', border: '1px solid var(--border-color)',
                }}>
                    <SectionHeader title="Gallery Health" icon="🖼️" />
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        <div style={{
                            display: 'grid', gridTemplateColumns: '1fr 1fr 1fr',
                            gap: '8px',
                        }}>
                            <div style={{
                                padding: '8px', borderRadius: '8px', textAlign: 'center',
                                background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                            }}>
                                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--primary)' }}>{gallery.total_embeddings || 0}</div>
                                <div style={{ fontSize: '0.5625rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Embeddings</div>
                            </div>
                            <div style={{
                                padding: '8px', borderRadius: '8px', textAlign: 'center',
                                background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                            }}>
                                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--success)' }}>{gallery.active_persons || 0}</div>
                                <div style={{ fontSize: '0.5625rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Persons</div>
                            </div>
                            <div style={{
                                padding: '8px', borderRadius: '8px', textAlign: 'center',
                                background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                            }}>
                                <div style={{ fontSize: '1.25rem', fontWeight: 700 }}>{gallery.avg_per_person || 0}</div>
                                <div style={{ fontSize: '0.5625rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Avg/Person</div>
                            </div>
                        </div>

                        {/* Pose Histogram */}
                        {gallery.pose_histogram && Object.keys(gallery.pose_histogram).length > 0 && (
                            <div>
                                <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginBottom: '6px' }}>
                                    Pose Distribution
                                </div>
                                <div style={{ display: 'flex', gap: '4px', height: '40px', alignItems: 'flex-end' }}>
                                    {Object.entries(gallery.pose_histogram).map(([pose, count]) => {
                                        const maxCount = Math.max(1, ...Object.values(gallery.pose_histogram));
                                        const pct = (count / maxCount) * 100;
                                        const colors = {
                                            frontal: '#10b981', left: '#06b6d4', right: '#8b5cf6',
                                            up: '#f59e0b', down: '#ec4899', extreme: '#ef4444',
                                        };
                                        return (
                                            <div key={pose} style={{
                                                flex: 1, display: 'flex', flexDirection: 'column',
                                                alignItems: 'center', gap: '2px',
                                            }}>
                                                <div style={{
                                                    width: '100%', borderRadius: '3px 3px 0 0',
                                                    background: colors[pose] || 'var(--text-muted)',
                                                    height: `${Math.max(4, pct)}%`,
                                                    transition: 'height 0.5s',
                                                    minHeight: '4px',
                                                }} />
                                                <span style={{
                                                    fontSize: '0.5rem', color: 'var(--text-muted)',
                                                    textTransform: 'capitalize',
                                                }}>{pose.slice(0, 3)}</span>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}

                        <div style={{ display: 'flex', gap: '8px' }}>
                            <div style={{
                                flex: 1, padding: '6px 10px', borderRadius: '8px',
                                background: 'rgba(251, 191, 36, 0.08)',
                                border: '1px solid rgba(251, 191, 36, 0.2)',
                                fontSize: '0.6875rem',
                            }}>
                                <span style={{ color: 'var(--warning)', fontWeight: 600 }}>{gallery.pending_suggestions || 0}</span>
                                <span style={{ color: 'var(--text-muted)' }}> pending suggestions</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* ── VLM / Gallery Optimizer ─────────────────── */}
                <div style={{
                    background: 'var(--bg-secondary)', borderRadius: '14px',
                    padding: '18px', border: '1px solid var(--border-color)',
                }}>
                    <SectionHeader
                        title="Gallery Optimizer"
                        icon="🤖"
                        status={vlm.worker_running ? 'healthy' : undefined}
                    />
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        <div style={{
                            display: 'flex', alignItems: 'center', gap: '8px',
                            padding: '8px 10px', borderRadius: '8px',
                            background: vlm.worker_running
                                ? 'rgba(16, 185, 129, 0.06)'
                                : 'rgba(239, 68, 68, 0.06)',
                            border: `1px solid ${vlm.worker_running ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)'}`,
                        }}>
                            <PulseIcon color={workerColor} />
                            <span style={{
                                fontSize: '0.75rem', fontWeight: 600,
                                color: workerColor,
                            }}>
                                Worker {vlm.worker_running ? 'Running' : 'Stopped'}
                            </span>
                            <span style={{
                                fontSize: '0.625rem', color: 'var(--text-muted)',
                                marginLeft: 'auto',
                            }}>
                                Cycle #{vlm.worker_cycles || 0}
                                {vlm.worker_failures > 0 && (
                                    <span style={{ color: 'var(--danger)', marginLeft: '6px' }}>
                                        ({vlm.worker_failures} failures)
                                    </span>
                                )}
                            </span>
                        </div>

                        <div style={{
                            display: 'grid', gridTemplateColumns: '1fr 1fr 1fr',
                            gap: '6px',
                        }}>
                            <div style={{
                                padding: '8px', borderRadius: '8px', textAlign: 'center',
                                background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                            }}>
                                <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--warning)' }}>{vlm.queue_length || 0}</div>
                                <div style={{ fontSize: '0.5rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Queue</div>
                            </div>
                            <div style={{
                                padding: '8px', borderRadius: '8px', textAlign: 'center',
                                background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                            }}>
                                <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--success)' }}>{vlm.approved || 0}</div>
                                <div style={{ fontSize: '0.5rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Approved</div>
                            </div>
                            <div style={{
                                padding: '8px', borderRadius: '8px', textAlign: 'center',
                                background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                            }}>
                                <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--danger)' }}>{vlm.rejected || 0}</div>
                                <div style={{ fontSize: '0.5rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Rejected</div>
                            </div>
                        </div>

                        {vlm.reviewed_total > 0 && (
                            <ProgressBar
                                value={vlm.merge_rate_pct || 0}
                                color="var(--success)"
                                label="Merge Rate"
                            />
                        )}
                    </div>
                </div>

                {/* ── Identity Recovery ───────────────────────── */}
                <div style={{
                    background: 'var(--bg-secondary)', borderRadius: '14px',
                    padding: '18px', border: '1px solid var(--border-color)',
                }}>
                    <SectionHeader title="Identity Recovery" icon="🔄" />
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        <div style={{
                            display: 'grid', gridTemplateColumns: '1fr 1fr 1fr',
                            gap: '6px',
                        }}>
                            <div style={{
                                padding: '8px', borderRadius: '8px', textAlign: 'center',
                                background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                            }}>
                                <div style={{ fontSize: '1rem', fontWeight: 700 }}>{tracking.recovery_attempted || 0}</div>
                                <div style={{ fontSize: '0.5rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Attempted</div>
                            </div>
                            <div style={{
                                padding: '8px', borderRadius: '8px', textAlign: 'center',
                                background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                            }}>
                                <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--success)' }}>{tracking.recovery_verified || 0}</div>
                                <div style={{ fontSize: '0.5rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Verified</div>
                            </div>
                            <div style={{
                                padding: '8px', borderRadius: '8px', textAlign: 'center',
                                background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                            }}>
                                <div style={{ fontSize: '1rem', fontWeight: 700, color: tracking.identity_swaps > 0 ? 'var(--danger)' : 'var(--success)' }}>{tracking.identity_swaps || 0}</div>
                                <div style={{ fontSize: '0.5rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>ID Swaps</div>
                            </div>
                        </div>

                        {tracking.recovery_attempted > 0 && (
                            <ProgressBar
                                value={tracking.carryover_pct || 0}
                                color="var(--success)"
                                label="Success Rate"
                            />
                        )}

                        {/* Rejection Breakdown */}
                        {tracking.carryover_rejections && Object.keys(tracking.carryover_rejections).length > 0 && (
                            <div style={{
                                padding: '8px 10px', borderRadius: '8px',
                                background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                            }}>
                                <div style={{ fontSize: '0.625rem', color: 'var(--text-muted)', marginBottom: '4px', textTransform: 'uppercase' }}>
                                    Rejection Reasons
                                </div>
                                {Object.entries(tracking.carryover_rejections)
                                    .sort((a, b) => b[1] - a[1])
                                    .map(([reason, count]) => (
                                        <div key={reason} style={{
                                            display: 'flex', justifyContent: 'space-between',
                                            fontSize: '0.6875rem', padding: '2px 0',
                                        }}>
                                            <span style={{ color: 'var(--text-muted)' }}>{reason.replace(/_/g, ' ')}</span>
                                            <span style={{ fontWeight: 600, color: 'var(--danger)' }}>{count}</span>
                                        </div>
                                    ))}
                            </div>
                        )}
                    </div>
                </div>

                {/* ── Unknown Faces ───────────────────────────── */}
                <div style={{
                    background: 'var(--bg-secondary)', borderRadius: '14px',
                    padding: '18px', border: '1px solid var(--border-color)',
                }}>
                    <SectionHeader title="Unknown Faces" icon="👤" />
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        <div style={{
                            display: 'grid', gridTemplateColumns: '1fr 1fr',
                            gap: '6px',
                        }}>
                            <div style={{
                                padding: '10px', borderRadius: '8px', textAlign: 'center',
                                background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                            }}>
                                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--warning)' }}>{unknowns.created_last_hour || 0}</div>
                                <div style={{ fontSize: '0.5625rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Last Hour</div>
                            </div>
                            <div style={{
                                padding: '10px', borderRadius: '8px', textAlign: 'center',
                                background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                            }}>
                                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--success)' }}>{unknowns.matched_last_24h || 0}</div>
                                <div style={{ fontSize: '0.5625rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Matched 24h</div>
                            </div>
                        </div>
                        <div style={{
                            display: 'grid', gridTemplateColumns: '1fr 1fr 1fr',
                            gap: '6px',
                        }}>
                            <div style={{
                                padding: '6px', borderRadius: '6px', textAlign: 'center',
                                background: 'var(--bg-primary)',
                            }}>
                                <div style={{ fontSize: '0.875rem', fontWeight: 600 }}>{unknowns.total_pending || 0}</div>
                                <div style={{ fontSize: '0.5rem', color: 'var(--text-muted)' }}>Pending</div>
                            </div>
                            <div style={{
                                padding: '6px', borderRadius: '6px', textAlign: 'center',
                                background: 'var(--bg-primary)',
                            }}>
                                <div style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--success)' }}>{unknowns.total_enrolled || 0}</div>
                                <div style={{ fontSize: '0.5rem', color: 'var(--text-muted)' }}>Enrolled</div>
                            </div>
                            <div style={{
                                padding: '6px', borderRadius: '6px', textAlign: 'center',
                                background: 'var(--bg-primary)',
                            }}>
                                <div style={{ fontSize: '0.875rem', fontWeight: 600 }}>{unknowns.total || 0}</div>
                                <div style={{ fontSize: '0.5rem', color: 'var(--text-muted)' }}>Total</div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* ── Snapshot Worker ─────────────────────── */}
                <div style={{
                    background: 'var(--bg-secondary)', borderRadius: '14px',
                    padding: '18px', border: '1px solid var(--border-color)',
                }}>
                    <SectionHeader title="Snapshot Worker" icon="📸" status={snapWorker.running ? 'healthy' : undefined} />
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        <div style={{
                            display: 'grid', gridTemplateColumns: '1fr 1fr 1fr',
                            gap: '6px',
                        }}>
                            <div style={{
                                padding: '8px', borderRadius: '8px', textAlign: 'center',
                                background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                            }}>
                                <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--success)' }}>{snapWorker.processed || 0}</div>
                                <div style={{ fontSize: '0.5rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Written</div>
                            </div>
                            <div style={{
                                padding: '8px', borderRadius: '8px', textAlign: 'center',
                                background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                            }}>
                                <div style={{ fontSize: '1rem', fontWeight: 700, color: snapWorker.fill_pct > 50 ? 'var(--warning)' : 'var(--text-primary)' }}>{snapWorker.queue_size || 0}/{snapWorker.max_queue || 200}</div>
                                <div style={{ fontSize: '0.5rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Queue</div>
                            </div>
                            <div style={{
                                padding: '8px', borderRadius: '8px', textAlign: 'center',
                                background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                            }}>
                                <div style={{ fontSize: '1rem', fontWeight: 700 }}>{snapWorker.avg_write_ms || 0}<span style={{ fontSize: '0.5rem', color: 'var(--text-muted)' }}>ms</span></div>
                                <div style={{ fontSize: '0.5rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Avg Write</div>
                            </div>
                        </div>
                        {(snapWorker.dropped_normal > 0 || snapWorker.dropped_debug > 0) && (
                            <div style={{
                                padding: '6px 10px', borderRadius: '8px',
                                background: 'rgba(251, 191, 36, 0.08)',
                                border: '1px solid rgba(251, 191, 36, 0.2)',
                                fontSize: '0.6875rem',
                            }}>
                                Dropped: <span style={{ color: 'var(--warning)', fontWeight: 600 }}>{snapWorker.dropped_normal || 0}</span> normal, <span style={{ color: 'var(--text-muted)' }}>{snapWorker.dropped_debug || 0}</span> debug
                            </div>
                        )}
                    </div>
                </div>

                {/* ── Shadow Gallery ──────────────────────── */}
                <div style={{
                    background: 'var(--bg-secondary)', borderRadius: '14px',
                    padding: '18px', border: '1px solid var(--border-color)',
                }}>
                    <SectionHeader title="Shadow Gallery" icon="🔮" status={shadow.has_manifest ? (shadow.running ? 'healthy' : undefined) : undefined} />
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        {!shadow.has_manifest ? (
                            <div style={{
                                padding: '12px', borderRadius: '8px', textAlign: 'center',
                                background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                                fontSize: '0.75rem', color: 'var(--text-muted)',
                            }}>
                                No validation set configured.
                                <br />
                                <span style={{ fontSize: '0.625rem' }}>Use the API to initialize: POST /api/system-health/shadow-gallery/init</span>
                            </div>
                        ) : (
                            <>
                                <div style={{
                                    display: 'flex', alignItems: 'center', gap: '8px',
                                    padding: '8px 10px', borderRadius: '8px',
                                    background: shadow.running ? 'rgba(16, 185, 129, 0.06)' : 'rgba(251, 191, 36, 0.06)',
                                    border: `1px solid ${shadow.running ? 'rgba(16, 185, 129, 0.2)' : 'rgba(251, 191, 36, 0.2)'}`,
                                }}>
                                    <PulseIcon color={shadow.running ? 'var(--success)' : 'var(--warning)'} />
                                    <span style={{ fontSize: '0.75rem', fontWeight: 600, color: shadow.running ? 'var(--success)' : 'var(--warning)' }}>
                                        {shadow.running ? 'Validating' : 'Idle'}
                                    </span>
                                    <span style={{ fontSize: '0.625rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>
                                        Every {shadow.interval_minutes || 30}min · {shadow.total_validations || 0} runs
                                    </span>
                                </div>
                                {shadow.latest && (
                                    <div style={{
                                        display: 'grid', gridTemplateColumns: '1fr 1fr 1fr',
                                        gap: '6px',
                                    }}>
                                        <div style={{
                                            padding: '8px', borderRadius: '8px', textAlign: 'center',
                                            background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                                        }}>
                                            <div style={{
                                                fontSize: '1.25rem', fontWeight: 700,
                                                color: shadow.latest.accuracy >= 80 ? 'var(--success)' : shadow.latest.accuracy >= 60 ? 'var(--warning)' : 'var(--danger)',
                                            }}>
                                                {shadow.latest.accuracy || 0}%
                                                {shadow.accuracy_trend !== null && shadow.accuracy_trend !== undefined && (
                                                    <TrendArrow value={shadow.accuracy_trend} suffix="%" />
                                                )}
                                            </div>
                                            <div style={{ fontSize: '0.5rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Accuracy</div>
                                        </div>
                                        <div style={{
                                            padding: '8px', borderRadius: '8px', textAlign: 'center',
                                            background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                                        }}>
                                            <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--success)' }}>{shadow.latest.true_positives || 0}</div>
                                            <div style={{ fontSize: '0.5rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Correct</div>
                                        </div>
                                        <div style={{
                                            padding: '8px', borderRadius: '8px', textAlign: 'center',
                                            background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                                        }}>
                                            <div style={{ fontSize: '1rem', fontWeight: 700, color: (shadow.latest.false_positives || shadow.latest.misses) ? 'var(--danger)' : 'var(--success)' }}>
                                                {(shadow.latest.false_positives || 0) + (shadow.latest.misses || 0)}
                                            </div>
                                            <div style={{ fontSize: '0.5rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>FP + Miss</div>
                                        </div>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                </div>
            </div>

            {/* Pulse animation keyframe */}
            <style jsx>{`
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.4; }
                }
            `}</style>
        </div>
    );
}
