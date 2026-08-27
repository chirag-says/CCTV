'use client';

/**
 * JourneyPanel — "Where was this person today?"
 *
 * Metrics strip + vertical journey rail + a date navigator. Self-contained
 * so it can be dropped into the person detail page and anywhere else a
 * journey is useful.
 */

import { useState, useEffect, useCallback } from 'react';
import api from '@/lib/api';
import DatePicker, { todayKey } from './DatePicker';
import JourneyMetrics from './JourneyMetrics';
import JourneyTimeline from './JourneyTimeline';
import { AlertIcon, RefreshIcon, CampusIcon } from './Icons';

export default function JourneyPanel({ personId, initialDate }) {
    const [date, setDate] = useState(initialDate || todayKey());
    const [journey, setJourney] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const load = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            setJourney(await api.getPersonJourney(personId, date));
        } catch (e) {
            setError(e.message || 'Could not load this journey.');
            setJourney(null);
        } finally {
            setLoading(false);
        }
    }, [personId, date]);

    useEffect(() => { load(); }, [load]);

    return (
        <div>
            <div className="toolbar-responsive" style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                gap: '12px', marginBottom: '16px', flexWrap: 'wrap',
            }}>
                <div>
                    <h3 className="card-title">Journey</h3>
                    <div className="card-subtitle">
                        Places this person was detected, in order
                    </div>
                </div>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <DatePicker value={date} onChange={setDate} id="journey-date" />
                    <button className="btn btn-secondary btn-sm" onClick={load} id="journey-refresh">
                        <RefreshIcon size={14} />
                    </button>
                </div>
            </div>

            {loading ? (
                <div className="card" style={{ padding: '60px 20px', textAlign: 'center' }}>
                    <div style={{
                        width: '36px', height: '36px', margin: '0 auto 14px',
                        border: '3px solid var(--border-color)', borderTopColor: 'var(--primary)',
                        borderRadius: '50%', animation: 'spin 1s linear infinite',
                    }} />
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                        Loading journey…
                    </div>
                </div>
            ) : error ? (
                <div className="card" style={{ padding: '40px 20px', textAlign: 'center' }}>
                    <div style={{ opacity: 0.3, marginBottom: '10px' }}><AlertIcon size={40} /></div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: '16px' }}>
                        {error}
                    </div>
                    <button className="btn btn-primary btn-sm" onClick={load}>Retry</button>
                </div>
            ) : journey?.no_data ? (
                <div className="card">
                    <div className="empty-state">
                        <div className="empty-state-icon"><CampusIcon size={32} /></div>
                        <div className="empty-state-title">No detections on this day</div>
                        <div className="empty-state-text">
                            This person was not recognised by any camera on {date}.
                            That is not the same as them being absent — they may have been
                            out of camera view, or no cameras were running.
                        </div>
                    </div>
                </div>
            ) : (
                <>
                    <JourneyMetrics journey={journey} />
                    <div className="card" style={{ marginTop: '16px', padding: '20px' }}>
                        <JourneyTimeline
                            stops={journey.stops}
                            transitions={journey.transitions}
                        />
                    </div>
                    <div style={{
                        marginTop: '12px', fontSize: '0.75rem',
                        color: 'var(--text-muted)', lineHeight: 1.6,
                    }}>
                        Built from {journey.event_count} detection
                        {journey.event_count === 1 ? '' : 's'} across{' '}
                        {journey.camera_count} camera{journey.camera_count === 1 ? '' : 's'}.
                        Repeated detections at one place are shown as a single stop.
                    </div>
                </>
            )}
        </div>
    );
}
