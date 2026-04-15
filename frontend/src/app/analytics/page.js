'use client';

/**
 * Analytics Page — Insights and reporting.
 * All data fetched from backend API — no mock/dummy data.
 */

import { useState, useEffect } from 'react';
import AppShell from '@/components/AppShell';
import StatCard from '@/components/StatCard';
import HourlyChart from '@/components/HourlyChart';
import OccupancyRing from '@/components/OccupancyRing';
import ActivityHeatmap from '@/components/ActivityHeatmap';
import api, { API_BASE } from '@/lib/api';
import { useToast } from '@/lib/ToastContext';
import { EntryIcon, ExitIcon, UserIcon, ClockIcon, AlertIcon } from '@/components/Icons';

export default function AnalyticsPage() {
    const [report, setReport] = useState(null);
    const [peakTimes, setPeakTimes] = useState(null);
    const [reportType, setReportType] = useState('daily');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [exporting, setExporting] = useState(false);
    const { showToast } = useToast();

    useEffect(() => {
        loadAnalytics();
    }, [reportType]);

    async function loadAnalytics() {
        setLoading(true);
        setError(null);
        try {
            const [reportData, peakData] = await Promise.all([
                api.getReport(reportType),
                api.getPeakTimes(reportType === 'daily' ? 1 : reportType === 'weekly' ? 7 : 30),
            ]);
            setReport(reportData);
            setPeakTimes(peakData);
        } catch (e) {
            setError('Could not load analytics. Make sure the backend is running.');
        } finally {
            setLoading(false);
        }
    }

    const fmtDur = (s) => {
        if (!s) return '0m';
        const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
        return h > 0 ? `${h}h ${m}m` : `${m}m`;
    };

    return (
        <AppShell>
            <div className="page-header">
                <div className="toolbar-responsive" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div><h1>Analytics</h1><p>Insights and reporting</p></div>
                    <div style={{ display: 'flex', gap: '8px' }}>
                        {['daily', 'weekly', 'monthly'].map(t => (
                            <button key={t} className={`btn btn-sm ${reportType === t ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setReportType(t)}>
                                {t.charAt(0).toUpperCase() + t.slice(1)}
                            </button>
                        ))}
                        <div style={{ display: 'flex', gap: '6px', marginLeft: '8px', borderLeft: '1px solid var(--border-color)', paddingLeft: '8px' }}>
                            {['csv', 'json'].map(fmt => (
                                <button
                                    key={fmt}
                                    className="btn btn-sm btn-secondary"
                                    disabled={exporting}
                                    onClick={async () => {
                                        setExporting(true);
                                        try {
                                            const token = api.getToken();
                                            const url = `${API_BASE}/api/analytics/export?format=${fmt}&type=events&limit=5000`;
                                            const resp = await fetch(url, {
                                                headers: token ? { Authorization: `Bearer ${token}` } : {},
                                            });
                                            if (!resp.ok) throw new Error('Export failed');
                                            const blob = await resp.blob();
                                            const a = document.createElement('a');
                                            a.href = URL.createObjectURL(blob);
                                            a.download = `sentinel_analytics.${fmt}`;
                                            a.click();
                                            URL.revokeObjectURL(a.href);
                                            showToast({ type: 'success', message: `Exported as ${fmt.toUpperCase()}` });
                                        } catch (e) {
                                            showToast({ type: 'error', message: e.message || 'Export failed' });
                                        } finally {
                                            setExporting(false);
                                        }
                                    }}
                                >
                                    {exporting ? '...' : `⬇ ${fmt.toUpperCase()}`}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {loading ? (
                <div className="card" style={{ padding: '80px 20px', textAlign: 'center' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Loading analytics...</div>
                </div>
            ) : error ? (
                <div className="card" style={{ padding: '60px 20px', textAlign: 'center' }}>
                    <div style={{ fontSize: '2rem', marginBottom: '12px', opacity: 0.3 }}>
                        <AlertIcon size={48} />
                    </div>
                    <div style={{ fontWeight: 600, fontSize: '1.125rem', marginBottom: '8px' }}>
                        Analytics Unavailable
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: '20px' }}>
                        {error}
                    </div>
                    <button className="btn btn-primary" onClick={loadAnalytics}>Retry</button>
                </div>
            ) : (
                <>
                    <div className="stats-grid">
                        <StatCard label="Total Entries" value={report?.total_entries || 0} icon={<EntryIcon size={20} />} variant="success" />
                        <StatCard label="Total Exits" value={report?.total_exits || 0} icon={<ExitIcon size={20} />} variant="accent" />
                        <StatCard label="Unique Persons" value={report?.unique_persons || 0} icon={<UserIcon size={20} />} variant="primary" />
                        <StatCard label="Avg Duration" value={fmtDur(report?.avg_duration_sec)} icon={<ClockIcon size={20} />} variant="warning" />
                    </div>
                    <div className="dashboard-grid">
                        <div className="card">
                            <div className="card-header">
                                <div>
                                    <h3 className="card-title">Entry Distribution</h3>
                                    <div className="card-subtitle">
                                        {peakTimes ? `Peak: ${peakTimes.peak_entry_hour}:00` : 'No data'}
                                    </div>
                                </div>
                            </div>
                            {peakTimes?.hourly_distribution ? (
                                <HourlyChart data={peakTimes.hourly_distribution} color="var(--primary)" />
                            ) : (
                                <div style={{ padding: '40px 20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                                    No distribution data available for this period
                                </div>
                            )}
                        </div>
                        <div className="card">
                            <div className="card-header"><h3 className="card-title">Period Summary</h3></div>
                            {report ? (
                                <>
                                    <OccupancyRing current={report.unique_persons || 0} max={report.total_entries || 1} label="Unique/Total" />
                                    <div style={{ marginTop: '20px' }}>
                                        {[
                                            { l: 'Sessions', v: report.total_sessions || 0 },
                                            { l: 'Avg Duration', v: fmtDur(report.avg_duration_sec) },
                                            { l: 'Net Flow', v: `${(report.total_entries || 0) - (report.total_exits || 0) > 0 ? '+' : ''}${(report.total_entries || 0) - (report.total_exits || 0)}` }
                                        ].map(i => (
                                            <div key={i.l} style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 0', borderBottom: '1px solid var(--border-color)', fontSize: '0.875rem' }}>
                                                <span style={{ color: 'var(--text-muted)' }}>{i.l}</span>
                                                <span style={{ fontWeight: 600, fontFamily: "'JetBrains Mono',monospace" }}>{i.v}</span>
                                            </div>
                                        ))}
                                    </div>
                                </>
                            ) : (
                                <div style={{ padding: '40px 20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                                    No report data available
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Activity Heatmap */}
                    <div style={{ marginTop: '20px' }}>
                        <ActivityHeatmap />
                    </div>
                </>
            )}
        </AppShell>
    );
}
