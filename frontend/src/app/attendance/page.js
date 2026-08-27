'use client';

/**
 * Faculty Attendance — department rollup for a campus day, with drill-down.
 *
 * All data comes from /api/attendance/*, which derives attendance from
 * committed detection events. Nothing here is recorded or editable: this is
 * a report, not a register.
 *
 * Labelling rule (COLLEGE_FEATURES_SPEC §1.3): the system observes cameras,
 * not the campus boundary. Columns say "First detected" / "Last detected",
 * never "Arrived" / "Departed".
 */

import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import AppShell from '@/components/AppShell';
import StatCard from '@/components/StatCard';
import DatePicker, { todayKey } from '@/components/DatePicker';
import DepartmentCard from '@/components/DepartmentCard';
import AttendanceStatusBadge from '@/components/AttendanceStatusBadge';
import api from '@/lib/api';
import { useToast } from '@/lib/ToastContext';
import {
    UsersIcon, CheckCircleIcon, ClockIcon, AlertIcon,
    DownloadIcon, RefreshIcon,
} from '@/components/Icons';

const EXPORT_FORMATS = [
    { fmt: 'csv', label: 'CSV' },
    { fmt: 'xlsx', label: 'Excel' },
];

const STATUS_FILTERS = [
    { value: '', label: 'All' },
    { value: 'present', label: 'Present' },
    { value: 'late', label: 'Late' },
    { value: 'absent', label: 'Absent' },
];

/** Render a stored UTC timestamp as campus-local HH:MM. */
function timeOf(iso) {
    if (!iso) return '—';
    try {
        return new Date(iso).toLocaleTimeString([], {
            hour: '2-digit', minute: '2-digit', hour12: false,
        });
    } catch { return '—'; }
}

function durationOf(minutes) {
    if (!minutes) return '—';
    const h = Math.floor(minutes / 60);
    const m = minutes % 60;
    return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

export default function AttendancePage() {
    const [date, setDate] = useState(todayKey());
    const [summary, setSummary] = useState(null);
    const [day, setDay] = useState(null);
    const [department, setDepartment] = useState(null);
    const [status, setStatus] = useState('');
    const [loading, setLoading] = useState(true);
    const [exporting, setExporting] = useState(false);
    const [error, setError] = useState(null);
    const { showToast } = useToast();
    const router = useRouter();

    const load = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const [s, d] = await Promise.all([
                api.getAttendanceSummary(date),
                api.getAttendanceDay({ date, department, status }),
            ]);
            setSummary(s);
            setDay(d);
        } catch (e) {
            setError(e.message || 'Could not load attendance. Make sure the backend is running.');
            setSummary(null);
            setDay(null);
        } finally {
            setLoading(false);
        }
    }, [date, department, status]);

    useEffect(() => { load(); }, [load]);

    async function handleExport(fmt) {
        setExporting(true);
        try {
            const filename = await api.downloadAttendanceExport({ date, format: fmt, department, status });
            showToast({ type: 'success', message: `Downloaded ${filename}` });
        } catch (e) {
            showToast({ type: 'error', message: e.message || 'Export failed' });
        } finally {
            setExporting(false);
        }
    }

    const totals = summary?.totals;
    // A day with no events at all is a holiday or a camera outage, not a day
    // when everybody was absent. Reporting 0% would be a confident lie.
    const noData = summary?.no_data;

    return (
        <AppShell>
            <div className="page-header">
                <div className="toolbar-responsive" style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px',
                }}>
                    <div>
                        <h1>Faculty Attendance</h1>
                        <p>Derived from camera detections · {summary?.rule?.timezone || 'campus time'}</p>
                    </div>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
                        <DatePicker value={date} onChange={setDate} id="attendance-date" />
                        <button className="btn btn-secondary btn-sm" onClick={load} id="attendance-refresh">
                            <RefreshIcon size={14} />
                        </button>
                        {EXPORT_FORMATS.map(({ fmt, label }) => (
                            <button
                                key={fmt}
                                className="btn btn-secondary btn-sm"
                                disabled={exporting || loading}
                                onClick={() => handleExport(fmt)}
                                id={`attendance-export-${fmt}`}
                                style={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}
                            >
                                <DownloadIcon size={13} /> {exporting ? '…' : label}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {loading && !summary ? (
                <div className="card" style={{ padding: '80px 20px', textAlign: 'center' }}>
                    <div style={{
                        width: '40px', height: '40px', margin: '0 auto 16px',
                        border: '3px solid var(--border-color)', borderTopColor: 'var(--primary)',
                        borderRadius: '50%', animation: 'spin 1s linear infinite',
                    }} />
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                        Loading attendance…
                    </div>
                </div>
            ) : error ? (
                <div className="card" style={{ padding: '60px 20px', textAlign: 'center' }}>
                    <div style={{ opacity: 0.3, marginBottom: '12px' }}><AlertIcon size={48} /></div>
                    <div style={{ fontWeight: 600, fontSize: '1.125rem', marginBottom: '8px' }}>
                        Attendance Unavailable
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: '20px' }}>
                        {error}
                    </div>
                    <button className="btn btn-primary" onClick={load}>Retry</button>
                </div>
            ) : (
                <>
                    {noData && (
                        <div className="card" style={{
                            marginBottom: '20px',
                            borderColor: 'rgba(245, 158, 11, 0.3)',
                            background: 'rgba(245, 158, 11, 0.05)',
                            display: 'flex', alignItems: 'center', gap: '12px',
                        }}>
                            <span style={{ color: 'var(--warning-light)', display: 'flex' }}>
                                <AlertIcon size={22} />
                            </span>
                            <div>
                                <div style={{ fontWeight: 600, color: 'var(--warning-light)' }}>
                                    No detections recorded on this day
                                </div>
                                <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                                    No cameras reported any entry or exit. This is not the same as
                                    everyone being absent, so the attendance percentage is not shown.
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Campus totals */}
                    <div className="stats-grid">
                        <StatCard
                            label="Attendance"
                            value={noData ? '—' : `${totals?.pct ?? 0}%`}
                            icon={<UsersIcon size={20} />}
                            variant={
                                noData ? 'accent'
                                    : (totals?.pct ?? 0) >= 85 ? 'success'
                                        : (totals?.pct ?? 0) >= 70 ? 'warning' : 'danger'
                            }
                        />
                        <StatCard
                            label="Present"
                            value={`${totals?.attending ?? 0} / ${totals?.total ?? 0}`}
                            icon={<CheckCircleIcon size={20} />}
                            variant="success"
                        />
                        <StatCard
                            label="Late"
                            value={totals?.late ?? 0}
                            icon={<ClockIcon size={20} />}
                            variant="warning"
                        />
                        <StatCard
                            label="Absent"
                            value={totals?.absent ?? 0}
                            icon={<AlertIcon size={20} />}
                            variant="danger"
                        />
                    </div>

                    {/* Enrolment gap — these people can never be recognised */}
                    {totals?.no_encodings > 0 && (
                        <div className="card" style={{
                            marginTop: '20px', padding: '12px 16px',
                            fontSize: '0.8125rem', color: 'var(--text-muted)',
                            display: 'flex', alignItems: 'center', gap: '8px',
                        }}>
                            <AlertIcon size={16} />
                            <span>
                                <strong style={{ color: 'var(--text-secondary)' }}>
                                    {totals.no_encodings}
                                </strong>{' '}
                                faculty {totals.no_encodings === 1 ? 'has' : 'have'} no enrolled face
                                and cannot be recognised — they will show as absent every day until
                                a photo is added.
                            </span>
                        </div>
                    )}

                    {/* Departments */}
                    {summary?.departments?.length > 0 && (
                        <div style={{ marginTop: '20px' }}>
                            <div style={{
                                display: 'flex', alignItems: 'center',
                                justifyContent: 'space-between', marginBottom: '12px',
                            }}>
                                <h3 className="card-title">By Department</h3>
                                {department && (
                                    <button className="btn btn-secondary btn-sm" onClick={() => setDepartment(null)}>
                                        Clear filter
                                    </button>
                                )}
                            </div>
                            <div style={{
                                display: 'grid',
                                gridTemplateColumns: 'repeat(auto-fill, minmax(230px, 1fr))',
                                gap: '12px',
                            }}>
                                {summary.departments.map((d) => (
                                    <DepartmentCard
                                        key={d.name}
                                        dept={d}
                                        selected={department === d.name}
                                        onSelect={setDepartment}
                                    />
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Person table */}
                    <div className="card" style={{ marginTop: '20px' }}>
                        <div className="card-header">
                            <div>
                                <h3 className="card-title">
                                    {department ? `${department} Faculty` : 'All Faculty'}
                                    {' '}({day?.records?.length ?? 0})
                                </h3>
                                <div className="card-subtitle">{summary?.rule?.summary}</div>
                            </div>
                            <div style={{ display: 'flex', gap: '6px' }}>
                                {STATUS_FILTERS.map((f) => (
                                    <button
                                        key={f.value || 'all'}
                                        className={`btn btn-sm ${status === f.value ? 'btn-primary' : 'btn-secondary'}`}
                                        onClick={() => setStatus(f.value)}
                                        id={`attendance-filter-${f.value || 'all'}`}
                                    >
                                        {f.label}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {!day?.records?.length ? (
                            <div className="empty-state">
                                <div className="empty-state-icon"><UsersIcon size={32} /></div>
                                <div className="empty-state-title">No faculty to show</div>
                                <div className="empty-state-text">
                                    {status || department
                                        ? 'No one matches the current filters.'
                                        : 'No one is marked as Faculty yet. Set a person’s Type to Faculty on the Persons page to include them here.'}
                                </div>
                            </div>
                        ) : (
                            <div className="table-responsive-wrapper">
                                <table className="data-table">
                                    <thead>
                                        <tr>
                                            <th>Status</th>
                                            <th>Name</th>
                                            <th>Department</th>
                                            <th>First detected</th>
                                            <th>Last detected</th>
                                            <th>Observed</th>
                                            <th title="Independent detections, and how many distinct cameras saw them">
                                                Evidence
                                            </th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {day.records.map((r) => (
                                            <tr
                                                key={r.person_id}
                                                onClick={() => router.push(`/persons/${r.person_id}`)}
                                                style={{ cursor: 'pointer' }}
                                            >
                                                <td>
                                                    <AttendanceStatusBadge
                                                        status={r.status}
                                                        sightings={r.sightings}
                                                        hasEncodings={r.has_encodings}
                                                    />
                                                </td>
                                                <td>
                                                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                                        <div className="avatar sm">{r.name.charAt(0)}</div>
                                                        <div>
                                                            <div style={{ fontWeight: 600, color: 'var(--primary)' }}>
                                                                {r.name}
                                                            </div>
                                                            <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)' }}>
                                                                {(r.role || '').replace(/_/g, ' ')}
                                                            </div>
                                                        </div>
                                                    </div>
                                                </td>
                                                <td>{r.department}</td>
                                                <td style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                                                    {timeOf(r.first_seen)}
                                                </td>
                                                <td style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                                                    {timeOf(r.last_seen)}
                                                </td>
                                                <td style={{ color: 'var(--text-muted)' }}>
                                                    {durationOf(r.minutes_observed)}
                                                </td>
                                                <td style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                                    {r.sightings} sighting{r.sightings === 1 ? '' : 's'}
                                                    {r.camera_count > 0 && ` · ${r.camera_count} camera${r.camera_count === 1 ? '' : 's'}`}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}
                    </div>

                    {/* Honesty footer — what these numbers do and don't mean */}
                    <div style={{
                        marginTop: '16px', fontSize: '0.75rem',
                        color: 'var(--text-muted)', lineHeight: 1.6,
                    }}>
                        Times are the first and last camera detection of the day, not gate
                        crossings. A person out of camera view is not detected, so
                        &ldquo;observed&rdquo; is a lower bound on time spent on campus.
                    </div>
                </>
            )}
        </AppShell>
    );
}
