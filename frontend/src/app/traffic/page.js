'use client';

/**
 * Traffic Monitor Page — Vehicle detection, ANPR, and safety alerts.
 * Premium UI matching the Dashboard's StatCard design system.
 * Features: live WebSocket updates, stat cards, filter toolbar,
 * polished vehicle + proximity cards, and responsive layout.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import AppShell from '@/components/AppShell';
import StatCard from '@/components/StatCard';
import api from '@/lib/api';
import {
    VehicleIcon,
    PlateIcon,
    TrafficIcon,
    ProximityIcon,
    CameraIcon,
    ClockIcon,
    RefreshIcon,
    AlertIcon,
    ShieldIcon,
} from '@/components/Icons';

export default function TrafficPage() {
    const [activeTab, setActiveTab] = useState('all');
    const [vehicleEvents, setVehicleEvents] = useState([]);
    const [proximityAlerts, setProximityAlerts] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [lastUpdate, setLastUpdate] = useState(null);
    const [isLive, setIsLive] = useState(false);
    const wsRef = useRef(null);

    const loadData = useCallback(async () => {
        try {
            const [vehicles, alerts] = await Promise.all([
                api.getVehicleEvents({ limit: 50 }).catch(() => ({ data: [] })),
                api.getTrafficAlerts({ limit: 50 }).catch(() => ({ data: [] })),
            ]);
            setVehicleEvents(vehicles.data || []);
            setProximityAlerts(alerts.data || []);
            setLastUpdate(new Date());
            setError(null);
        } catch (e) {
            setError('Could not load traffic data. Ensure backend is running with PIPELINE_MODE=traffic or hybrid.');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadData();

        // WebSocket for live updates
        try {
            wsRef.current = api.connectLiveEvents((event) => {
                setIsLive(true);
                if (event.event_type === 'vehicle_entry') {
                    setVehicleEvents((prev) => [event, ...prev].slice(0, 100));
                    setLastUpdate(new Date());
                } else if (event.event_type === 'security_alert' && event.subtype === 'vehicle_proximity') {
                    setProximityAlerts((prev) => [event, ...prev].slice(0, 100));
                    setLastUpdate(new Date());
                }
            });
        } catch (e) {
            console.warn('WebSocket not available:', e.message);
        }

        // Fallback polling every 15s
        const interval = setInterval(loadData, 15000);
        return () => {
            clearInterval(interval);
            if (wsRef.current) wsRef.current.close();
        };
    }, [loadData]);

    /* ── Helpers ───────────────────────────────────────────── */
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
            if (diff < 5) return 'just now';
            if (diff < 60) return `${diff}s ago`;
            if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
            if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
            return `${Math.floor(diff / 86400)}d ago`;
        } catch { return '—'; }
    };

    /* ── Stats ────────────────────────────────────────────── */
    const vehicleTypes = {};
    vehicleEvents.forEach(e => {
        const t = e.metadata?.vehicle_type || 'unknown';
        vehicleTypes[t] = (vehicleTypes[t] || 0) + 1;
    });
    const uniquePlates = new Set(vehicleEvents.map(e => e.metadata?.plate).filter(Boolean)).size;
    const highSeverity = proximityAlerts.filter(a => (a.metadata?.severity === 'high' || (a.metadata?.overlap_iou || 0) > 0)).length;

    /* ── Filtered items ───────────────────────────────────── */
    const allItems = [
        ...vehicleEvents.map(e => ({ ...e, _type: 'vehicle' })),
        ...proximityAlerts.map(a => ({ ...a, _type: 'alert' })),
    ].sort((a, b) => {
        const tA = new Date(a.created_at || a.timestamp || 0).getTime();
        const tB = new Date(b.created_at || b.timestamp || 0).getTime();
        return tB - tA;
    });

    const displayItems = activeTab === 'all' ? allItems
        : activeTab === 'vehicles' ? allItems.filter(i => i._type === 'vehicle')
            : allItems.filter(i => i._type === 'alert');

    const tabCounts = {
        all: allItems.length,
        vehicles: vehicleEvents.length,
        safety: proximityAlerts.length,
    };

    return (
        <AppShell>
            {/* ── Page Header ──────────────────────────────── */}
            <div className="page-header">
                <h1>Traffic Monitor</h1>
                <p>
                    Vehicle detection, license plate recognition &amp; proximity safety
                    <span style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '6px',
                        marginLeft: '12px',
                    }}>
                        <span style={{
                            width: '8px', height: '8px', borderRadius: '50%',
                            background: isLive ? 'var(--success)' : '#3b82f6',
                            animation: 'pulse-dot 2s infinite',
                        }}></span>
                        <span style={{
                            fontSize: '0.75rem',
                            color: isLive ? 'var(--success-light)' : '#60a5fa',
                            fontWeight: 600,
                        }}>{isLive ? 'LIVE' : 'MONITORING'}</span>
                    </span>
                </p>
            </div>

            {/* ── Stats Grid — same as Dashboard ─────────── */}
            <div className="stats-grid">
                <StatCard
                    label="Total Vehicles"
                    value={vehicleEvents.length}
                    icon={<VehicleIcon size={20} />}
                    variant="accent"
                />
                <StatCard
                    label="Unique Plates"
                    value={uniquePlates}
                    icon={<PlateIcon size={20} />}
                    variant="success"
                />
                <StatCard
                    label="Safety Alerts"
                    value={proximityAlerts.length}
                    icon={<AlertIcon size={20} />}
                    variant="danger"
                />
                <StatCard
                    label="Critical"
                    value={highSeverity}
                    icon={<ShieldIcon size={20} />}
                    variant="warning"
                />
            </div>

            {/* ── Filter Toolbar ─────────────────────────── */}
            <div className="card" style={{ marginBottom: '20px' }}>
                <div className="toolbar-responsive" style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
                    {[
                        { key: 'all', label: 'All Events', icon: null },
                        { key: 'vehicles', label: 'Vehicles & ANPR', icon: <PlateIcon size={14} /> },
                        { key: 'safety', label: 'Safety Alerts', icon: <ProximityIcon size={14} /> },
                    ].map(({ key, label, icon }) => (
                        <button
                            key={key}
                            id={`tab-${key}`}
                            className={`btn btn-sm ${activeTab === key ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={() => setActiveTab(key)}
                            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
                        >
                            {icon}
                            {label}
                            {tabCounts[key] > 0 && (
                                <span style={{
                                    background: activeTab === key
                                        ? 'var(--primary-glow)'
                                        : 'var(--hover-surface-strong)',
                                    padding: '1px 7px',
                                    borderRadius: '10px',
                                    fontSize: '0.625rem',
                                    fontWeight: 700,
                                    minWidth: '18px',
                                    textAlign: 'center',
                                }}>{tabCounts[key]}</span>
                            )}
                        </button>
                    ))}
                    <span style={{
                        marginLeft: 'auto',
                        fontSize: '0.75rem',
                        color: 'var(--text-muted)',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                    }}>
                        {lastUpdate && (
                            <>
                                <ClockIcon size={12} />
                                Updated {getRelativeTime(lastUpdate.toISOString())}
                            </>
                        )}
                    </span>
                    <button
                        className="btn btn-secondary btn-sm"
                        onClick={() => { setLoading(true); loadData(); }}
                        id="refresh-traffic"
                        style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
                    >
                        <RefreshIcon size={14} />
                        Refresh
                    </button>
                </div>
            </div>

            {/* ── Content ─────────────────────────────────── */}
            {loading ? (
                <div className="card" style={{ padding: '80px 20px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    <div style={{
                        width: '44px', height: '44px',
                        border: '3px solid var(--border-color)', borderTopColor: '#3b82f6',
                        borderRadius: '50%', animation: 'spin 1s linear infinite',
                        margin: '0 auto 16px',
                    }} />
                    <div style={{ fontSize: '0.875rem', fontWeight: 500 }}>Loading traffic data...</div>
                    <div style={{ fontSize: '0.75rem', marginTop: '6px', opacity: 0.6 }}>
                        Fetching vehicle entries and safety alerts
                    </div>
                </div>
            ) : error ? (
                <div className="card" style={{ padding: '80px 20px', textAlign: 'center' }}>
                    <div style={{ fontSize: '2rem', marginBottom: '12px', opacity: 0.15 }}>
                        <AlertIcon size={48} />
                    </div>
                    <div style={{
                        fontWeight: 600, fontSize: '1rem', marginBottom: '8px',
                        color: 'var(--text-secondary)',
                    }}>Connection Error</div>
                    <div style={{
                        color: 'var(--text-muted)', fontSize: '0.875rem',
                        marginBottom: '20px', lineHeight: 1.6, maxWidth: '420px', margin: '0 auto 20px',
                    }}>{error}</div>
                    <button className="btn btn-primary btn-sm" onClick={() => { setLoading(true); loadData(); }}>
                        Retry
                    </button>
                </div>
            ) : displayItems.length === 0 ? (
                <EmptyState tab={activeTab} />
            ) : (
                <div className="traffic-grid">
                    {displayItems.map((item, index) =>
                        item._type === 'vehicle'
                            ? <VehicleCard key={item.id || `v-${index}`} event={item} index={index}
                                formatDate={formatDate} getRelativeTime={getRelativeTime} />
                            : <ProximityCard key={item.id || `a-${index}`} alert={item} index={index}
                                formatDate={formatDate} getRelativeTime={getRelativeTime} />
                    )}
                </div>
            )}
        </AppShell>
    );
}

/* ════════════════════════════════════════════════════════════════
   Sub-components
   ════════════════════════════════════════════════════════════════ */

function EmptyState({ tab }) {
    const configs = {
        all: {
            icon: <TrafficIcon size={56} />,
            title: 'No Traffic Events Yet',
            message: 'Vehicle entries and proximity alerts will appear here when detected. Ensure PIPELINE_MODE is set to traffic or hybrid in your .env file.',
        },
        vehicles: {
            icon: <VehicleIcon size={56} />,
            title: 'No Vehicle Entries',
            message: 'License plates will be recognized and logged automatically when vehicles appear in the camera feed.',
        },
        safety: {
            icon: <ProximityIcon size={56} />,
            title: 'No Proximity Alerts',
            message: 'Alerts appear when a pedestrian is detected within dangerous proximity of a vehicle. The system monitors all camera feeds in real time.',
        },
    };
    const cfg = configs[tab] || configs.all;

    return (
        <div className="card" style={{ padding: '80px 20px', textAlign: 'center' }}>
            <div style={{ marginBottom: '16px', opacity: 0.12, display: 'flex', justifyContent: 'center' }}>
                {cfg.icon}
            </div>
            <div style={{
                fontWeight: 600, fontSize: '1.125rem', marginBottom: '8px',
                color: 'var(--text-secondary)',
            }}>{cfg.title}</div>
            <div style={{
                color: 'var(--text-muted)', fontSize: '0.875rem',
                maxWidth: '440px', margin: '0 auto', lineHeight: 1.7,
            }}>{cfg.message}</div>
            <div style={{
                display: 'flex', gap: '20px', justifyContent: 'center',
                marginTop: '28px', fontSize: '0.75rem', color: 'var(--text-muted)',
            }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px', opacity: 0.7 }}>
                    <VehicleIcon size={14} /> Vehicles
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px', opacity: 0.7 }}>
                    <PlateIcon size={14} /> ANPR
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px', opacity: 0.7 }}>
                    <ProximityIcon size={14} /> Proximity
                </span>
            </div>
        </div>
    );
}

function VehicleCard({ event, index, formatDate, getRelativeTime }) {
    const meta = event.metadata || {};
    const plate = meta.plate || '???';
    const confidence = meta.confidence ? (meta.confidence * 100).toFixed(0) : null;
    const vehicleConf = meta.vehicle_confidence ? (meta.vehicle_confidence * 100).toFixed(0) : null;

    return (
        <div
            className="vehicle-card animate-in"
            style={{ animationDelay: `${Math.min(index, 10) * 40}ms` }}
        >
            {/* Header: plate + badge */}
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '12px' }}>
                <div style={{ flex: 1 }}>
                    <div className="plate-number">{plate}</div>
                    <div style={{
                        marginTop: '10px', fontSize: '0.813rem',
                        color: 'var(--text-secondary)',
                        display: 'flex', alignItems: 'center', gap: '6px',
                    }}>
                        <VehicleIcon size={13} />
                        <span style={{ textTransform: 'capitalize', fontWeight: 600 }}>
                            {meta.vehicle_type || 'Vehicle'}
                        </span>
                        {vehicleConf && (
                            <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>
                                • {vehicleConf}%
                            </span>
                        )}
                    </div>
                </div>
                <span className="badge vehicle-entry" style={{
                    flexShrink: 0, display: 'flex', alignItems: 'center', gap: '4px',
                }}>
                    <PlateIcon size={11} />
                    ANPR
                </span>
            </div>

            {/* Meta chips */}
            <div className="vehicle-meta">
                <span className="vehicle-chip">
                    <CameraIcon size={10} />
                    {event.camera_name || event.camera_id || 'Camera'}
                </span>
                {confidence && (
                    <span className="vehicle-chip">
                        <span style={{
                            width: '6px', height: '6px', borderRadius: '50%',
                            background: Number(confidence) > 80 ? '#22c55e' : Number(confidence) > 50 ? '#fbbf24' : '#f87171',
                            flexShrink: 0,
                        }} />
                        OCR {confidence}%
                    </span>
                )}
                <span className="vehicle-chip">
                    <ClockIcon size={10} />
                    {getRelativeTime(event.created_at || event.timestamp)}
                </span>
            </div>

            {/* Timestamp */}
            <div style={{
                marginTop: '12px', fontSize: '0.688rem', color: 'var(--text-muted)',
                fontFamily: "'JetBrains Mono', monospace",
                display: 'flex', alignItems: 'center', gap: '4px',
            }}>
                <ClockIcon size={10} />
                {formatDate(event.created_at || event.timestamp)}
            </div>
        </div>
    );
}

function ProximityCard({ alert, index, formatDate, getRelativeTime }) {
    const meta = alert.metadata || {};
    const isOverlap = (meta.overlap_iou || 0) > 0;
    const severity = meta.severity || (isOverlap ? 'high' : 'medium');
    const distance = meta.min_distance_px ? Math.round(meta.min_distance_px) : null;

    return (
        <div
            className={`security-alert-card severity-${severity} animate-in`}
            style={{ animationDelay: `${Math.min(index, 10) * 40}ms` }}
        >
            {/* Header */}
            <div className="alert-header">
                <div className="alert-icon">
                    <ProximityIcon size={20} />
                </div>
                <div style={{ flex: 1 }}>
                    <div className="alert-title">Person-Vehicle Proximity</div>
                    <div className="alert-subtitle">
                        {isOverlap
                            ? `Bounding boxes OVERLAPPING (IoU: ${((meta.overlap_iou || 0) * 100).toFixed(1)}%)`
                            : distance !== null
                                ? `Pedestrian within ${distance}px of vehicle`
                                : 'Dangerous proximity detected'}
                    </div>
                </div>
                <span className={`badge security-alert ${isOverlap ? 'hazard' : ''}`} style={{ fontSize: '0.625rem' }}>
                    {severity === 'high' ? 'CRITICAL' : 'WARNING'}
                </span>
            </div>

            {/* Distance indicator */}
            <div style={{
                padding: '10px 14px',
                borderRadius: 'var(--radius-md)',
                background: isOverlap ? 'rgba(239, 68, 68, 0.08)' : 'rgba(245, 158, 11, 0.08)',
                border: `1px solid ${isOverlap ? 'rgba(239, 68, 68, 0.15)' : 'rgba(245, 158, 11, 0.15)'}`,
                fontSize: '0.875rem', fontWeight: 700,
                color: isOverlap ? '#f87171' : '#fbbf24',
                display: 'flex', alignItems: 'center', gap: '8px',
            }}>
                <span style={{ fontSize: '1.1em' }}>⚠</span>
                {isOverlap
                    ? `OVERLAP — IoU ${((meta.overlap_iou || 0) * 100).toFixed(1)}%`
                    : `${distance || '?'}px away`}
            </div>

            {/* Detail chips */}
            <div className="alert-details">
                <span className="alert-detail-chip">
                    <VehicleIcon size={10} />
                    {meta.vehicle_type || 'Vehicle'}
                    {meta.vehicle_confidence && ` (${(meta.vehicle_confidence * 100).toFixed(0)}%)`}
                </span>
                <span className="alert-detail-chip">
                    <CameraIcon size={10} />
                    {alert.camera_name || alert.camera_id || 'Camera'}
                </span>
                <span className="alert-detail-chip" style={{
                    background: severity === 'high' ? 'rgba(239, 68, 68, 0.1)' : 'rgba(245, 158, 11, 0.1)',
                    borderColor: severity === 'high' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(245, 158, 11, 0.2)',
                    color: severity === 'high' ? '#f87171' : '#fbbf24',
                }}>
                    {severity === 'high' ? '● High Severity' : '● Medium Severity'}
                </span>
            </div>

            {/* Time */}
            <div className="alert-time">
                <ClockIcon size={11} style={{ verticalAlign: 'middle', marginRight: '4px' }} />
                {formatDate(alert.created_at || alert.timestamp)}
                <span style={{ marginLeft: '8px', opacity: 0.6 }}>
                    ({getRelativeTime(alert.created_at || alert.timestamp)})
                </span>
            </div>
        </div>
    );
}
