'use client';

/**
 * Incident evidence pack.
 *
 * Shows what the system genuinely recorded: the annotated alert frame, images
 * captured on the same camera around it, and a timeline of surrounding
 * events. There is no video — this system has no recorder — so the page never
 * implies one exists.
 */

import { useState, useEffect, useCallback } from 'react';
import { useParams, useRouter } from 'next/navigation';
import AppShell from '@/components/AppShell';
import EvidenceGrid from '@/components/EvidenceGrid';
import IncidentStatusControl from '@/components/IncidentStatusControl';
import api from '@/lib/api';
import {
    AlertIcon, CameraIcon, ClockIcon, RefreshIcon, SecurityIcon,
} from '@/components/Icons';

const SEVERITY_TONE = {
    high: 'var(--danger-light)',
    medium: 'var(--warning-light)',
    low: 'var(--accent-light)',
};

function fullWhen(iso) {
    if (!iso) return '—';
    try {
        return new Date(iso).toLocaleString('en-GB', {
            weekday: 'short', day: 'numeric', month: 'short', year: 'numeric',
            hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
        });
    } catch { return '—'; }
}

function clockOf(iso) {
    if (!iso) return '—';
    try {
        return new Date(iso).toLocaleTimeString([], {
            hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
        });
    } catch { return '—'; }
}

function Field({ label, value }) {
    return (
        <div>
            <div style={{
                fontSize: '0.625rem', fontWeight: 700, letterSpacing: '0.07em',
                textTransform: 'uppercase', color: 'var(--text-muted)',
            }}>
                {label}
            </div>
            <div style={{ fontSize: '0.9375rem', fontWeight: 600, marginTop: '3px' }}>
                {value}
            </div>
        </div>
    );
}

export default function IncidentDetailPage() {
    const { id } = useParams();
    const router = useRouter();
    const [pack, setPack] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const load = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            setPack(await api.getIncident(id));
        } catch (e) {
            setError(e.message || 'Could not load this incident.');
            setPack(null);
        } finally {
            setLoading(false);
        }
    }, [id]);

    useEffect(() => { load(); }, [load]);

    if (loading) {
        return (
            <AppShell>
                <div className="card" style={{ padding: '80px 20px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    <div style={{
                        width: '38px', height: '38px', margin: '0 auto 14px',
                        border: '3px solid var(--border-color)', borderTopColor: 'var(--danger)',
                        borderRadius: '50%', animation: 'spin 1s linear infinite',
                    }} />
                    Loading evidence…
                </div>
            </AppShell>
        );
    }

    if (error || !pack) {
        return (
            <AppShell>
                <div className="card" style={{ padding: '60px 20px', textAlign: 'center' }}>
                    <div style={{ opacity: 0.3, marginBottom: '12px' }}><AlertIcon size={48} /></div>
                    <div style={{ fontWeight: 600, fontSize: '1.125rem', marginBottom: '8px' }}>
                        Incident Unavailable
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: '20px' }}>
                        {error}
                    </div>
                    <div style={{ display: 'flex', gap: '8px', justifyContent: 'center' }}>
                        <button className="btn btn-secondary" onClick={() => router.push('/incidents')}>
                            Back to incidents
                        </button>
                        <button className="btn btn-primary" onClick={load}>Retry</button>
                    </div>
                </div>
            </AppShell>
        );
    }

    const inc = pack.incident;
    const tone = SEVERITY_TONE[inc.severity] || 'var(--text-muted)';
    const meta = inc.metadata || {};

    return (
        <AppShell>
            <div className="page-header">
                <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => router.push('/incidents')}
                    style={{ marginBottom: '12px' }}
                >
                    ← All incidents
                </button>

                <div className="toolbar-responsive" style={{
                    display: 'flex', alignItems: 'flex-start',
                    justifyContent: 'space-between', gap: '16px',
                }}>
                    <div>
                        <div style={{
                            fontFamily: "'JetBrains Mono', monospace",
                            fontSize: '0.75rem', color: 'var(--text-muted)',
                            letterSpacing: '0.05em',
                        }}>
                            INCIDENT {inc.seq ? `#${inc.seq}` : `· ${String(inc.event_id).slice(0, 8)}`}
                        </div>
                        <h1 style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                            <span style={{ color: tone, display: 'flex' }}><SecurityIcon size={24} /></span>
                            {inc.label}
                        </h1>
                        <p style={{ color: tone, fontWeight: 600, textTransform: 'uppercase', fontSize: '0.75rem', letterSpacing: '0.06em' }}>
                            {inc.severity} severity
                        </p>
                    </div>
                    <button className="btn btn-secondary btn-sm" onClick={load}>
                        <RefreshIcon size={14} />
                    </button>
                </div>
            </div>

            {/* Facts */}
            <div className="card">
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                    gap: '18px',
                }}>
                    <Field
                        label="Camera"
                        value={
                            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                                <CameraIcon size={14} />
                                {inc.zone || inc.camera_name}
                            </span>
                        }
                    />
                    <Field label="Location" value={inc.camera_location || '—'} />
                    <Field
                        label="Time"
                        value={
                            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                                <ClockIcon size={14} />
                                {fullWhen(inc.created_at)}
                            </span>
                        }
                    />
                    {meta.duration_sec !== undefined && (
                        <Field label="Duration" value={`${Math.round(meta.duration_sec)}s`} />
                    )}
                    {meta.person_count !== undefined && (
                        <Field label="People involved" value={meta.person_count} />
                    )}
                    {meta.threat_class && <Field label="Detected" value={meta.threat_class} />}
                </div>

                <div style={{
                    marginTop: '18px', paddingTop: '16px',
                    borderTop: '1px solid var(--border-color)',
                    fontSize: '0.9375rem', color: 'var(--text-secondary)',
                }}>
                    {inc.summary}
                </div>
            </div>

            {/* Status */}
            <div className="card" style={{ marginTop: '20px' }}>
                <div className="card-header">
                    <div>
                        <h3 className="card-title">Status</h3>
                        <div className="card-subtitle">
                            {pack.review?.updated_at
                                ? `Last updated ${fullWhen(pack.review.updated_at)}`
                                : 'Not yet reviewed'}
                        </div>
                    </div>
                </div>
                <IncidentStatusControl
                    eventId={inc.event_id}
                    status={pack.review?.status || 'open'}
                    onChange={(r) => setPack((p) => ({ ...p, review: { ...p.review, ...r } }))}
                />
            </div>

            {/* Evidence */}
            <div className="card" style={{ marginTop: '20px' }}>
                <div className="card-header">
                    <div>
                        <h3 className="card-title">Evidence</h3>
                        <div className="card-subtitle">Images captured around this alert</div>
                    </div>
                </div>
                <EvidenceGrid evidence={pack.evidence} windowSec={pack.evidence_window_sec} />
            </div>

            {/* Timeline */}
            <div className="card" style={{ marginTop: '20px' }}>
                <div className="card-header">
                    <div>
                        <h3 className="card-title">Timeline</h3>
                        <div className="card-subtitle">
                            What else this camera saw around the alert
                        </div>
                    </div>
                </div>

                {pack.timeline?.length === 0 ? (
                    <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
                        No other events recorded in this window.
                    </div>
                ) : (
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                        {pack.timeline.map((t) => (
                            <div
                                key={t.id}
                                style={{
                                    display: 'flex', gap: '12px', padding: '7px 10px',
                                    alignItems: 'baseline', borderRadius: '6px',
                                    background: t.is_alert ? 'rgba(239, 68, 68, 0.08)' : 'transparent',
                                    borderLeft: `2px solid ${t.is_alert ? tone : 'transparent'}`,
                                }}
                            >
                                <span style={{
                                    fontFamily: "'JetBrains Mono', monospace",
                                    fontSize: '0.75rem', color: 'var(--text-muted)',
                                    flexShrink: 0, minWidth: '66px',
                                }}>
                                    {clockOf(t.at)}
                                </span>
                                <span style={{
                                    fontSize: '0.8125rem',
                                    fontWeight: t.is_alert ? 700 : 400,
                                    color: t.is_alert ? tone : 'var(--text-secondary)',
                                }}>
                                    {t.text}
                                </span>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            <div style={{
                marginTop: '16px', fontSize: '0.75rem',
                color: 'var(--text-muted)', lineHeight: 1.6,
            }}>
                This system records images, not video. Evidence is limited to frames
                captured at the time of the alert and detections logged on the same camera.
            </div>
        </AppShell>
    );
}
