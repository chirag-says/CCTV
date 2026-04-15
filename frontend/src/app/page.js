'use client';

/**
 * Dashboard — Main page showing system overview.
 * Displays live stats, occupancy, recent events, and peak time charts.
 * All data is fetched from the backend API — no mock/dummy data.
 */

import { useState, useEffect, useRef } from 'react';
import AppShell from '@/components/AppShell';
import StatCard from '@/components/StatCard';
import ActivityFeed from '@/components/ActivityFeed';
import OccupancyRing from '@/components/OccupancyRing';
import HourlyChart from '@/components/HourlyChart';
import api from '@/lib/api';
import CameraHealth from '@/components/CameraHealth';
import {
  UsersIcon,
  EntryIcon,
  ExitIcon,
  UnknownFaceIcon,
  VideoCameraIcon,
  FolderIcon,
  AlertIcon,
  ArrowRightIcon,
} from '@/components/Icons';

export default function DashboardPage() {
  const [dashboard, setDashboard] = useState(null);
  const [events, setEvents] = useState([]);
  const [peakTimes, setPeakTimes] = useState(null);
  const [activePersons, setActivePersons] = useState([]);
  const [isLive, setIsLive] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
    loadData();

    // Connect WebSocket for live events
    try {
      wsRef.current = api.connectLiveEvents((event) => {
        setEvents((prev) => [event, ...prev].slice(0, 50));
        setIsLive(true);
      });
    } catch (e) {
      console.warn('WebSocket not available:', e.message);
    }

    // Periodic refresh
    const interval = setInterval(loadData, 30000);

    return () => {
      clearInterval(interval);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  async function loadData() {
    try {
      const [dashData, peakData, eventsData, activeData] = await Promise.allSettled([
        api.getDashboard(),
        api.getPeakTimes(7),
        api.getEvents({ limit: 20 }),
        api.getActivePersons(),
      ]);

      if (dashData.status === 'fulfilled') setDashboard(dashData.value);
      if (peakData.status === 'fulfilled') setPeakTimes(peakData.value);
      if (eventsData.status === 'fulfilled' && eventsData.value?.data) {
        setEvents((prev) => prev.length === 0 ? eventsData.value.data : prev);
      }
      if (activeData.status === 'fulfilled' && activeData.value?.persons) {
        setActivePersons(activeData.value.persons);
      }

      setIsLive(dashData.status === 'fulfilled');
      setError(dashData.status === 'rejected' ? 'Could not connect to backend' : null);
    } catch (e) {
      setError('Could not connect to backend');
      setIsLive(false);
    } finally {
      setLoading(false);
    }
  }

  const formatDuration = (seconds) => {
    if (!seconds) return '0s';
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  };

  // Loading state
  if (loading) {
    return (
      <AppShell>
        <div className="page-header">
          <h1>Dashboard</h1>
          <p>Loading system data...</p>
        </div>
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          padding: '80px 20px',
          color: 'var(--text-muted)',
          fontSize: '0.875rem',
        }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{
              width: '40px',
              height: '40px',
              border: '3px solid var(--border-color)',
              borderTopColor: 'var(--primary)',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
              margin: '0 auto 16px',
            }} />
            Connecting to backend...
          </div>
        </div>
      </AppShell>
    );
  }

  // Error / no data state
  if (error && !dashboard) {
    return (
      <AppShell>
        <div className="page-header">
          <h1>Dashboard</h1>
          <p>Real-time surveillance overview</p>
        </div>
        <div className="card" style={{ padding: '60px 20px', textAlign: 'center' }}>
          <div style={{ fontSize: '2rem', marginBottom: '12px', opacity: 0.3 }}>
            <AlertIcon size={48} />
          </div>
          <div style={{ fontWeight: 600, fontSize: '1.125rem', marginBottom: '8px' }}>
            Backend Unavailable
          </div>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: '20px' }}>
            {error}. Make sure the backend server is running and accessible.
          </div>
          <button className="btn btn-primary" onClick={() => { setLoading(true); loadData(); }}>
            Retry Connection
          </button>
        </div>
      </AppShell>
    );
  }

  const d = dashboard || {
    current_occupancy: 0,
    today_entries: 0,
    today_exits: 0,
    today_unknown: 0,
    total_persons: 0,
    pending_unknowns: 0,
    active_cameras: 0,
  };

  return (
    <AppShell unknownCount={d.pending_unknowns}>
      {/* Header */}
      <div className="page-header">
        <h1>Dashboard</h1>
        <p>
          Real-time surveillance overview
          <span style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '6px',
            marginLeft: '12px'
          }}>
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: isLive ? 'var(--success)' : 'var(--danger)',
              animation: isLive ? 'pulse-dot 2s infinite' : 'none'
            }}></span>
            <span style={{
              fontSize: '0.75rem',
              color: isLive ? 'var(--success)' : 'var(--danger)',
              fontWeight: 600
            }}>
              {isLive ? 'LIVE' : 'OFFLINE'}
            </span>
          </span>
        </p>
      </div>

      {/* Stats Grid */}
      <div className="stats-grid">
        <StatCard
          label="Current Occupancy"
          value={d.current_occupancy}
          icon={<UsersIcon size={20} />}
          variant="primary"
          suffix="people"
        />
        <StatCard
          label="Today's Entries"
          value={d.today_entries}
          icon={<EntryIcon size={20} />}
          variant="success"
        />
        <StatCard
          label="Today's Exits"
          value={d.today_exits}
          icon={<ExitIcon size={20} />}
          variant="accent"
        />
        <StatCard
          label="Unknown Detected"
          value={d.today_unknown}
          icon={<UnknownFaceIcon size={20} />}
          variant="warning"
        />
        <StatCard
          label="Active Cameras"
          value={d.active_cameras}
          icon={<VideoCameraIcon size={20} />}
          variant="accent"
        />
        <StatCard
          label="Known Persons"
          value={d.total_persons}
          icon={<FolderIcon size={20} />}
          variant="primary"
        />
      </div>

      {/* Main Grid */}
      <div className="dashboard-grid">
        {/* Activity Feed */}
        <div className="card">
          <div className="card-header">
            <div>
              <h3 className="card-title">Recent Activity</h3>
              <div className="card-subtitle">Real-time detection events</div>
            </div>
            {isLive && (
              <span className="badge active">
                <span className="badge-dot"></span>
                Live
              </span>
            )}
          </div>
          <ActivityFeed events={events} />
        </div>

        {/* Right Column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Occupancy Ring */}
          <div className="card">
            <div className="card-header">
              <div>
                <h3 className="card-title">Occupancy</h3>
                <div className="card-subtitle">Current building occupancy</div>
              </div>
            </div>
            <OccupancyRing
              current={d.current_occupancy}
              max={100}
              label="Present"
            />
            <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.813rem' }}>
              Capacity: 100 people
            </div>
          </div>

          {/* Camera Health */}
          <CameraHealth />

          {/* Active Persons */}
          <div className="card">
            <div className="card-header">
              <div>
                <h3 className="card-title">Currently Present</h3>
                <div className="card-subtitle">{activePersons.length} persons</div>
              </div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              {activePersons.length > 0 ? (
                activePersons.slice(0, 6).map((person) => (
                  <div key={person.person_id} className="activity-item">
                    <div className="avatar sm">
                      {person.person_name?.charAt(0) || '?'}
                    </div>
                    <div className="activity-content">
                      <div className="activity-title" style={{ fontSize: '0.813rem' }}>
                        {person.person_name}
                      </div>
                      <div className="activity-subtitle">
                        {formatDuration(person.duration_sec)} present
                      </div>
                    </div>
                    <span className="badge active" style={{ fontSize: '0.625rem' }}>
                      <span className="badge-dot"></span>
                      Active
                    </span>
                  </div>
                ))
              ) : (
                <div style={{
                  padding: '20px',
                  textAlign: 'center',
                  color: 'var(--text-muted)',
                  fontSize: '0.813rem',
                }}>
                  No persons currently detected
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Peak Times Chart */}
      {peakTimes && peakTimes.hourly_distribution && (
        <div className="card" style={{ marginTop: '20px' }}>
          <div className="card-header">
            <div>
              <h3 className="card-title">Peak Entry Times</h3>
              <div className="card-subtitle">
                Last 7 days &bull; Peak hour: {peakTimes.peak_entry_hour}:00 &bull;
                Total: {peakTimes.total_entries} entries
              </div>
            </div>
          </div>
          <HourlyChart
            data={peakTimes.hourly_distribution}
            color="var(--primary)"
          />
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '24px',
            marginTop: '8px',
            fontSize: '0.75rem',
            color: 'var(--text-muted)'
          }}>
            <span>
              <span style={{
                display: 'inline-block',
                width: '12px',
                height: '12px',
                borderRadius: '3px',
                background: 'var(--primary)',
                marginRight: '6px',
                verticalAlign: 'middle'
              }}></span>
              Work Hours (8-18)
            </span>
            <span>
              <span style={{
                display: 'inline-block',
                width: '12px',
                height: '12px',
                borderRadius: '3px',
                background: 'var(--text-muted)',
                marginRight: '6px',
                verticalAlign: 'middle'
              }}></span>
              Off Hours
            </span>
          </div>
        </div>
      )}

      {/* Pending Unknown Faces Alert */}
      {d.pending_unknowns > 0 && (
        <div className="card" style={{
          marginTop: '20px',
          borderColor: 'rgba(245, 158, 11, 0.3)',
          background: 'rgba(245, 158, 11, 0.05)',
        }}>
          <div className="toolbar-responsive" style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span style={{ color: 'var(--warning-light)', display: 'flex' }}>
                <AlertIcon size={24} />
              </span>
              <div>
                <div style={{ fontWeight: 600, color: 'var(--warning-light)' }}>
                  {d.pending_unknowns} Unknown Faces Pending Review
                </div>
                <div style={{ fontSize: '0.813rem', color: 'var(--text-muted)' }}>
                  Unrecognized persons need admin approval or dismissal
                </div>
              </div>
            </div>
            <a href="/unknown-faces" className="btn btn-secondary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              Review <ArrowRightIcon size={14} />
            </a>
          </div>
        </div>
      )}
    </AppShell>
  );
}
