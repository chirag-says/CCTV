'use client';

/**
 * Security Alerts Page — Real-time security event monitoring.
 * Displays crowd gathering, loitering, and hazard detection alerts
 * with severity indicators, filtering, and live WebSocket updates.
 */

import { useState, useEffect, useRef } from 'react';
import AppShell from '@/components/AppShell';
import api from '@/lib/api';
import {
    SecurityIcon,
    CrowdIcon,
    LoiterIcon,
    HazardIcon,
    FireIcon,
    AlertIcon,
    RefreshIcon,
    ClockIcon,
    CameraIcon,
} from '@/components/Icons';

const SUBTYPE_CONFIG = {
    gathering: {
        label: 'Crowd Gathering',
        icon: CrowdIcon,
        severity: 'medium',
        severityLabel: 'Medium',
        color: 'var(--warning-light)',
    },
    loitering: {
        label: 'Loitering Detected',
        icon: LoiterIcon,
        severity: 'low',
        severityLabel: 'Low',
        color: 'var(--accent-light)',
    },
    hazard: {
        label: 'Hazard / Threat',
        icon: HazardIcon,
        severity: 'high',
        severityLabel: 'High',
        color: 'var(--danger-light)',
    },
};

export default function SecurityAlertsPage() {
    const [alerts, setAlerts] = useState([]);
    const [total, setTotal] = useState(0);
    const [subtypeFilter, setSubtypeFilter] = useState('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const wsRef = useRef(null);

    useEffect(() => {
        loadAlerts();

        // Live updates via WebSocket
        try {
            wsRef.current = api.connectLiveEvents((event) => {
                if (event.event_type === 'security_alert') {
                    setAlerts((prev) => [event, ...prev].slice(0, 100));
                }
            });
        } catch (e) {
            console.warn('WebSocket not available:', e.message);
        }

        const interval = setInterval(loadAlerts, 30000);
        return () => {
            clearInterval(interval);
            if (wsRef.current) wsRef.current.close();
        };
    }, [subtypeFilter]);

    async function loadAlerts() {
        setLoading(true);
        setError(null);
        try {
            const params = { limit: 50 };
            if (subtypeFilter) params.subtype = subtypeFilter;
            const result = await api.getSecurityAlerts(params);
            setAlerts(result.data || []);
            setTotal(result.total || 0);
        } catch (e) {
            setError('Could not load security alerts. Make sure the backend is running.');
            setAlerts([]);
            setTotal(0);
        } finally {
            setLoading(false);
        }
    }

    const formatDate = (dateStr) => {
        try {
            return new Date(dateStr).toLocaleString('en-US', {
                month: 'short', day: 'numeric',
                hour: '2-digit', minute: '2-digit', second: '2-digit',
                hour12: false,
            });
        } catch { return '—'; }
    };

    const getRelativeTime = (dateStr) => {
        try {
            const diff = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000);
            if (diff < 60) return `${diff}s ago`;
            if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
            if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
            return `${Math.floor(diff / 86400)}d ago`;
        } catch { return '—'; }
    };

    // Compute stats from loaded alerts
    const stats = {
        total: alerts.length,
        hazard: alerts.filter(a => (a.subtype || a.metadata?.subtype) === 'hazard').length,
        gathering: alerts.filter(a => (a.subtype || a.metadata?.subtype) === 'gathering').length,
        loitering: alerts.filter(a => (a.subtype || a.metadata?.subtype) === 'loitering').length,
    };

    const getAlertConfig = (alert) => {
        const subtype = alert.subtype || alert.metadata?.subtype || 'hazard';
        return SUBTYPE_CONFIG[subtype] || SUBTYPE_CONFIG.hazard;
    };

    const getAlertDescription = (alert) => {
        const subtype = alert.subtype || alert.metadata?.subtype || '';
        const meta = alert.metadata || {};
        switch (subtype) {
            case 'gathering':
                return `${meta.person_count || '?'} people detected gathering in close proximity`;
            case 'loitering':
                const dur = meta.duration_sec ? `${Math.round(meta.duration_sec)}s` : 'extended period';
                return `${meta.person_name || 'Unknown individual'} stationary for ${dur}`;
            case 'hazard':
                return `${meta.threat_class || 'Threat'} detected with ${((alert.confidence || meta.confidence || 0) * 100).toFixed(0)}% confidence`;
            default:
                return 'Security event detected';
        }
    };

    return (
        <AppShell>
            <div className="page-header">
                <h1>Security Alerts</h1>
                <p>
                    AI-powered threat detection and safety monitoring
                    <span style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '6px',
                        marginLeft: '12px',
                    }}>
                        <span style={{
                            width: '8px', height: '8px', borderRadius: '50%',
                            background: 'var(--danger)',
                            animation: 'pulse-dot 2s infinite',
                        }}></span>
                        <span style={{
                            fontSize: '0.75rem',
                            color: 'var(--danger-light)',
                            fontWeight: 600,
                        }}>MONITORING</span>
                    </span>
                </p>
            </div>

            {/* Stats Bar */}
            <div className="security-stats">
                <div className="security-stat total">
                    <div className="stat-number">{stats.total}</div>
                    <div className="stat-label">Total Alerts</div>
                </div>
                <div className="security-stat danger">
                    <div className="stat-number">{stats.hazard}</div>
                    <div className="stat-label">Hazards</div>
                </div>
                <div className="security-stat warning">
                    <div className="stat-number">{stats.gathering}</div>
                    <div className="stat-label">Gatherings</div>
                </div>
                <div className="security-stat info">
                    <div className="stat-number">{stats.loitering}</div>
                    <div className="stat-label">Loitering</div>
                </div>
            </div>

            {/* Filters */}
            <div className="card" style={{ marginBottom: '20px' }}>
                <div className="toolbar-responsive" style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
                    {['', 'gathering', 'loitering', 'hazard'].map((type) => (
                        <button
                            key={type || 'all'}
                            id={`filter-${type || 'all'}`}
                            className={`btn btn-sm ${subtypeFilter === type ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={() => setSubtypeFilter(type)}
                        >
                            {type ? (
                                <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                                    {type === 'gathering' && <CrowdIcon size={14} />}
                                    {type === 'loitering' && <LoiterIcon size={14} />}
                                    {type === 'hazard' && <HazardIcon size={14} />}
                                    {type.charAt(0).toUpperCase() + type.slice(1)}
                                </span>
                            ) : 'All Alerts'}
                        </button>
                    ))}
                    <span style={{ marginLeft: 'auto', fontSize: '0.813rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center' }}>
                        {total} alerts
                    </span>
                    <button className="btn btn-secondary btn-sm" onClick={loadAlerts} id="refresh-alerts">
                        <RefreshIcon size={14} />
                        Refresh
                    </button>
                </div>
            </div>

            {/* Alert Cards */}
            {loading ? (
                <div className="card" style={{ padding: '60px 20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                    <div style={{
                        width: '40px', height: '40px',
                        border: '3px solid var(--border-color)', borderTopColor: 'var(--danger)',
                        borderRadius: '50%', animation: 'spin 1s linear infinite',
                        margin: '0 auto 16px',
                    }} />
                    Loading security alerts...
                </div>
            ) : error ? (
                <div className="card" style={{ padding: '60px 20px', textAlign: 'center' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: '16px' }}>{error}</div>
                    <button className="btn btn-primary btn-sm" onClick={loadAlerts}>Retry</button>
                </div>
            ) : alerts.length === 0 ? (
                <div className="card" style={{ padding: '80px 20px', textAlign: 'center' }}>
                    <div style={{ fontSize: '2rem', marginBottom: '12px', opacity: 0.15 }}>
                        <SecurityIcon size={64} />
                    </div>
                    <div style={{ fontWeight: 600, fontSize: '1.125rem', marginBottom: '8px', color: 'var(--text-secondary)' }}>
                        No Security Alerts
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', maxWidth: '400px', margin: '0 auto', lineHeight: 1.6 }}>
                        {subtypeFilter
                            ? `No "${subtypeFilter}" alerts found. Try removing the filter.`
                            : 'The system is actively monitoring for threats. Alerts for crowd gatherings, loitering, and hazards will appear here when detected.'}
                    </div>
                    <div style={{
                        display: 'flex', gap: '24px', justifyContent: 'center',
                        marginTop: '24px', fontSize: '0.75rem', color: 'var(--text-muted)',
                    }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <CrowdIcon size={14} /> Crowd
                        </span>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <LoiterIcon size={14} /> Loitering
                        </span>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <HazardIcon size={14} /> Hazards
                        </span>
                    </div>
                </div>
            ) : (
                <div className="security-grid">
                    {alerts.map((alert, index) => {
                        const config = getAlertConfig(alert);
                        const IconComponent = config.icon;
                        const subtype = alert.subtype || alert.metadata?.subtype || '';
                        const meta = alert.metadata || {};

                        return (
                            <div
                                key={alert.id || index}
                                className={`security-alert-card severity-${config.severity} animate-in`}
                                style={{ animationDelay: `${index * 30}ms` }}
                            >
                                <div className="alert-header">
                                    <div className="alert-icon">
                                        <IconComponent size={20} />
                                    </div>
                                    <div style={{ flex: 1 }}>
                                        <div className="alert-title">{config.label}</div>
                                        <div className="alert-subtitle">{getAlertDescription(alert)}</div>
                                    </div>
                                    <span className={`badge security-alert ${subtype}`} style={{ fontSize: '0.625rem' }}>
                                        {config.severityLabel}
                                    </span>
                                </div>

                                <div className="alert-details">
                                    <span className="alert-detail-chip">
                                        <CameraIcon size={10} />
                                        {alert.camera_name || alert.camera_id || 'Unknown'}
                                    </span>
                                    {meta.person_name && (
                                        <span className="alert-detail-chip">
                                            Person: {meta.person_name}
                                        </span>
                                    )}
                                    {meta.person_count && (
                                        <span className="alert-detail-chip">
                                            {meta.person_count} people
                                        </span>
                                    )}
                                    {meta.threat_class && (
                                        <span className="alert-detail-chip" style={{
                                            background: 'rgba(239, 68, 68, 0.1)',
                                            borderColor: 'rgba(239, 68, 68, 0.2)',
                                            color: 'var(--danger-light)',
                                        }}>
                                            {meta.threat_class}
                                        </span>
                                    )}
                                    {meta.duration_sec && (
                                        <span className="alert-detail-chip">
                                            <ClockIcon size={10} />
                                            {Math.round(meta.duration_sec)}s
                                        </span>
                                    )}
                                </div>

                                <div className="alert-time">
                                    <ClockIcon size={11} style={{ verticalAlign: 'middle', marginRight: '4px' }} />
                                    {formatDate(alert.created_at || alert.timestamp)}
                                    <span style={{ marginLeft: '8px', opacity: 0.6 }}>
                                        ({getRelativeTime(alert.created_at || alert.timestamp)})
                                    </span>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </AppShell>
    );
}
