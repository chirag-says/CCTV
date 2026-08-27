'use client';

/**
 * Daily Campus Summary — one day of campus activity for administration.
 *
 * Every figure is composed from data the system already records. Two carry
 * caveats that the UI is responsible for preserving:
 *   - peak occupancy is RECONSTRUCTED, so it is labelled "estimated"
 *   - camera availability is null when nothing ran, and shows a dash rather
 *     than claiming 100% uptime for a system that was switched off
 *
 * The narrative is templated on the backend from these same numbers — no
 * model involved, so it is instant and never contradicts the metrics above it.
 */

import { useState, useEffect, useCallback } from 'react';
import AppShell from '@/components/AppShell';
import StatCard from '@/components/StatCard';
import HourlyChart from '@/components/HourlyChart';
import DatePicker, { todayKey } from '@/components/DatePicker';
import DepartmentCard from '@/components/DepartmentCard';
import api from '@/lib/api';
import {
    UsersIcon, AlertIcon, RefreshIcon, CampusIcon, UnknownFaceIcon,
    VideoCameraIcon, TrafficIcon, SecurityIcon, ClockIcon,
} from '@/components/Icons';

const DASH = '—';

function hourLabel(h) {
    return h === null || h === undefined ? DASH : `${String(h).padStart(2, '0')}:00`;
}

function timeLabel(iso) {
    if (!iso) return DASH;
    try {
        return new Date(iso).toLocaleTimeString([], {
            hour: '2-digit', minute: '2-digit', hour12: false,
        });
    } catch { return DASH; }
}

export default function DailySummaryPage() {
    const [date, setDate] = useState(todayKey());
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const load = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            setData(await api.getDailySummary(date));
        } catch (e) {
            setError(e.message || 'Could not load the summary.');
            setData(null);
        } finally {
            setLoading(false);
        }
    }, [date]);

    useEffect(() => { load(); }, [load]);

    const m = data?.metrics;
    const noData = data?.no_data;

    return (
        <AppShell>
            <div className="page-header">
                <div className="toolbar-responsive" style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px',
                }}>
                    <div>
                        <h1>{data?.is_today ? "Today's Campus" : 'Campus Summary'}</h1>
                        <p>Activity across every camera, for one day</p>
                    </div>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                        <DatePicker value={date} onChange={setDate} id="summary-date" />
                        <button className="btn btn-secondary btn-sm" onClick={load} id="summary-refresh">
                            <RefreshIcon size={14} />
                        </button>
                    </div>
                </div>
            </div>

            {loading && !data ? (
                <div className="card" style={{ padding: '80px 20px', textAlign: 'center' }}>
                    <div style={{
                        width: '40px', height: '40px', margin: '0 auto 16px',
                        border: '3px solid var(--border-color)', borderTopColor: 'var(--primary)',
                        borderRadius: '50%', animation: 'spin 1s linear infinite',
                    }} />
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                        Building summary…
                    </div>
                </div>
            ) : error ? (
                <div className="card" style={{ padding: '60px 20px', textAlign: 'center' }}>
                    <div style={{ opacity: 0.3, marginBottom: '12px' }}><AlertIcon size={48} /></div>
                    <div style={{ fontWeight: 600, fontSize: '1.125rem', marginBottom: '8px' }}>
                        Summary Unavailable
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: '20px' }}>
                        {error}
                    </div>
                    <button className="btn btn-primary" onClick={load}>Retry</button>
                </div>
            ) : noData ? (
                <div className="card">
                    <div className="empty-state">
                        <div className="empty-state-icon"><CampusIcon size={32} /></div>
                        <div className="empty-state-title">No activity recorded on this day</div>
                        <div className="empty-state-text">
                            No camera reported anything on {date}. That is not the same as an
                            empty campus — most likely no cameras were running.
                        </div>
                    </div>
                </div>
            ) : (
                <>
                    {/* Narrative — derived from the numbers below, never beyond them */}
                    {data.narrative?.length > 0 && (
                        <div className="card" style={{ marginBottom: '20px' }}>
                            <div style={{ display: 'flex', gap: '12px' }}>
                                <span style={{ color: 'var(--primary)', display: 'flex', marginTop: '2px' }}>
                                    <CampusIcon size={20} />
                                </span>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                                    {data.narrative.map((line, i) => (
                                        <div key={i} style={{
                                            fontSize: '0.9375rem',
                                            color: i === 0 ? 'var(--text-primary)' : 'var(--text-secondary)',
                                            fontWeight: i === 0 ? 600 : 400,
                                            lineHeight: 1.5,
                                        }}>
                                            {line}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Headline metrics */}
                    <div className="stats-grid">
                        <StatCard
                            label="People Detected"
                            value={m.people_detected.toLocaleString()}
                            icon={<UsersIcon size={20} />}
                            variant="primary"
                        />
                        <StatCard
                            label="Faculty Attendance"
                            value={m.faculty_total ? `${m.faculty_pct}%` : DASH}
                            suffix={m.faculty_total ? `${m.faculty_present}/${m.faculty_total}` : ''}
                            icon={<ClockIcon size={20} />}
                            variant={
                                m.faculty_pct >= 85 ? 'success'
                                    : m.faculty_pct >= 70 ? 'warning' : 'danger'
                            }
                        />
                        <StatCard
                            label="Peak Occupancy"
                            value={m.peak_occupancy}
                            suffix="est."
                            icon={<CampusIcon size={20} />}
                            variant="accent"
                        />
                        <StatCard
                            label="Vehicles"
                            value={m.vehicles.toLocaleString()}
                            icon={<TrafficIcon size={20} />}
                            variant="accent"
                        />
                        <StatCard
                            label="Security Incidents"
                            value={m.security_incidents}
                            icon={<SecurityIcon size={20} />}
                            variant={m.security_incidents > 0 ? 'danger' : 'success'}
                        />
                        <StatCard
                            label="Unknown Persons"
                            value={m.unknown_persons.toLocaleString()}
                            icon={<UnknownFaceIcon size={20} />}
                            variant="warning"
                        />
                        <StatCard
                            label="Camera Availability"
                            value={
                                m.camera_availability_pct === null
                                    ? DASH
                                    : `${m.camera_availability_pct}%`
                            }
                            suffix={m.cameras_active ? `${m.cameras_active} active` : 'none running'}
                            icon={<VideoCameraIcon size={20} />}
                            variant={
                                m.camera_availability_pct === null ? 'accent'
                                    : m.camera_availability_pct >= 98 ? 'success' : 'warning'
                            }
                        />
                        {data.is_today && m.on_campus_now !== null && (
                            <StatCard
                                label="On Campus Now"
                                value={m.on_campus_now}
                                icon={<UsersIcon size={20} />}
                                variant="success"
                            />
                        )}
                    </div>

                    {/* Arrival / departure rhythm */}
                    <div className="dashboard-grid" style={{ marginTop: '20px' }}>
                        <div className="card">
                            <div className="card-header">
                                <div>
                                    <h3 className="card-title">Arrivals</h3>
                                    <div className="card-subtitle">
                                        Busiest at {hourLabel(m.peak_arrival_hour)} · campus local time
                                    </div>
                                </div>
                            </div>
                            <HourlyChart data={data.hourly.arrivals} color="var(--success)" />
                        </div>
                        <div className="card">
                            <div className="card-header">
                                <div>
                                    <h3 className="card-title">Departures</h3>
                                    <div className="card-subtitle">
                                        Busiest at {hourLabel(m.peak_departure_hour)} · campus local time
                                    </div>
                                </div>
                            </div>
                            <HourlyChart data={data.hourly.departures} color="var(--accent)" />
                        </div>
                    </div>

                    {/* Attendance by department */}
                    {data.attendance_by_department?.length > 0 && (
                        <div style={{ marginTop: '20px' }}>
                            <h3 className="card-title" style={{ marginBottom: '12px' }}>
                                Attendance by Department
                            </h3>
                            <div style={{
                                display: 'grid',
                                gridTemplateColumns: 'repeat(auto-fill, minmax(230px, 1fr))',
                                gap: '12px',
                            }}>
                                {data.attendance_by_department.map((d) => (
                                    <DepartmentCard key={d.name} dept={d} />
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Detail rows */}
                    <div className="card" style={{ marginTop: '20px' }}>
                        <div className="card-header"><h3 className="card-title">Detail</h3></div>
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            {[
                                ['Known people detected', m.known_detected],
                                ['Unknown people detected', m.unknown_persons],
                                ['Unfamiliar faces captured', m.unknown_faces_captured,
                                    'Faces snapshotted but never confirmed as a presence — always at least the number above'],
                                ['Faculty seen', m.faculty_detected],
                                ['Students seen', m.students_detected],
                                ['Peak occupancy reached at', timeLabel(m.peak_occupancy_at),
                                    'Reconstructed from presence intervals; occupancy is never stored'],
                                ['Camera downtime', m.camera_downtime_sec
                                    ? `${Math.round(m.camera_downtime_sec / 60)} min` : 'none'],
                                ['Total events recorded', m.total_events.toLocaleString()],
                            ].map(([label, value, hint]) => (
                                <div key={label} style={{
                                    display: 'flex', justifyContent: 'space-between',
                                    alignItems: 'baseline', gap: '16px',
                                    padding: '10px 0', borderBottom: '1px solid var(--border-color)',
                                }}>
                                    <div>
                                        <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                                            {label}
                                        </div>
                                        {hint && (
                                            <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginTop: '1px' }}>
                                                {hint}
                                            </div>
                                        )}
                                    </div>
                                    <div style={{
                                        fontWeight: 600, fontFamily: "'JetBrains Mono', monospace",
                                        whiteSpace: 'nowrap',
                                    }}>
                                        {value}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div style={{
                        marginTop: '16px', fontSize: '0.75rem',
                        color: 'var(--text-muted)', lineHeight: 1.6,
                    }}>
                        Peak occupancy is an estimate reconstructed from detection intervals —
                        the system does not record occupancy directly. People out of camera
                        view are not counted.
                    </div>
                </>
            )}
        </AppShell>
    );
}
