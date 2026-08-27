'use client';

/**
 * JourneyTimeline — the vertical rail of places a person was seen.
 *
 *   08:42  ●  🚪  Main Gate
 *          │      First detected
 *          │      5 min
 *   08:47  ●  🏢  Academic Block
 *
 * Dwell time sits ON the connector between two stops, because it describes
 * the time spent at the stop above it. Time BETWEEN stops is shown as
 * "not detected", never folded into a dwell — claiming someone was somewhere
 * the cameras did not see them would be inventing data.
 *
 * Pure CSS: an absolutely-positioned rail behind relatively-positioned dots.
 * No charting library, consistent with HourlyChart and OccupancyRing.
 */

import { hhmm, humanDuration } from './JourneyMetrics';

// Zone kinds come from cameras.config.zone_icon. Kept as words rather than
// raw emoji in the DB so the icon set can change without a data migration.
const ZONE_GLYPH = {
    gate: '🚪',
    building: '🏢',
    lab: '💻',
    library: '📚',
    cafeteria: '🍴',
    parking: '🅿️',
    corridor: '🚶',
    camera: '📍',
};

const RAIL_LEFT = 52;   // px from the left edge to the rail centre
const DOT = 13;

export default function JourneyTimeline({ stops = [], transitions = [] }) {
    if (!stops.length) return null;

    const transitionAfter = (index) =>
        transitions.find((t) => t.from_index === index) || null;

    return (
        <div style={{ position: 'relative', paddingTop: '4px' }}>
            {stops.map((stop, i) => {
                const isFirst = i === 0;
                const isLast = i === stops.length - 1;
                const trans = transitionAfter(stop.index);
                const glyph = ZONE_GLYPH[stop.zone_icon] || ZONE_GLYPH.camera;

                return (
                    <div key={`${stop.index}-${stop.arrived_at}`}>
                        {/* ── Stop row ── */}
                        <div style={{ display: 'flex', alignItems: 'flex-start', position: 'relative' }}>
                            {/* Time gutter */}
                            <div style={{
                                width: `${RAIL_LEFT - DOT}px`,
                                textAlign: 'right',
                                paddingRight: '12px',
                                paddingTop: '1px',
                                fontFamily: "'JetBrains Mono', monospace",
                                fontSize: '0.8125rem',
                                fontWeight: 600,
                                color: 'var(--text-secondary)',
                                flexShrink: 0,
                            }}>
                                {hhmm(stop.arrived_at)}
                            </div>

                            {/* Dot */}
                            <div style={{
                                width: `${DOT}px`, height: `${DOT}px`, borderRadius: '50%',
                                background: 'var(--bg-secondary, #1a1b2e)',
                                border: `3px solid ${isFirst || isLast ? 'var(--primary)' : 'var(--text-muted)'}`,
                                flexShrink: 0, marginTop: '4px', zIndex: 1,
                                boxShadow: isFirst || isLast ? '0 0 0 4px var(--primary-glow)' : 'none',
                            }} />

                            {/* Place */}
                            <div style={{ paddingLeft: '14px', paddingBottom: '4px', flex: 1, minWidth: 0 }}>
                                <div style={{
                                    display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap',
                                }}>
                                    <span aria-hidden="true" style={{ fontSize: '1rem' }}>{glyph}</span>
                                    <span style={{
                                        fontWeight: 700, fontSize: '0.9375rem', color: 'var(--text-primary)',
                                    }}>
                                        {stop.zone}
                                    </span>
                                    {stop.is_gate && (
                                        <span className="badge pending" style={{ fontSize: '0.5625rem' }}>
                                            GATE
                                        </span>
                                    )}
                                </div>

                                <div style={{
                                    fontSize: '0.6875rem', color: 'var(--text-muted)', marginTop: '2px',
                                }}>
                                    {isFirst && 'First detected'}
                                    {isLast && !isFirst && 'Last detected'}
                                    {!isFirst && !isLast && `${stop.event_count} detection${stop.event_count === 1 ? '' : 's'}`}
                                    {stop.camera_name && stop.zone !== stop.camera_name && (
                                        <span> · {stop.camera_name}</span>
                                    )}
                                    {stop.dwell_sec > 0 && (
                                        <span> · seen until {hhmm(stop.left_at)}</span>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* ── Connector ── */}
                        {!isLast && (
                            <div style={{ display: 'flex', alignItems: 'stretch', position: 'relative' }}>
                                <div style={{ width: `${RAIL_LEFT - DOT}px`, flexShrink: 0 }} />
                                <div style={{
                                    width: `${DOT}px`, display: 'flex', justifyContent: 'center', flexShrink: 0,
                                }}>
                                    <div style={{
                                        width: '2px',
                                        background: trans?.same_place
                                            ? 'repeating-linear-gradient(var(--border-color) 0 4px, transparent 4px 8px)'
                                            : 'var(--border-color)',
                                        minHeight: '34px',
                                    }} />
                                </div>
                                <div style={{
                                    paddingLeft: '14px', paddingTop: '6px', paddingBottom: '10px',
                                    fontSize: '0.6875rem', color: 'var(--text-muted)',
                                }}>
                                    {stop.dwell_sec > 0 && (
                                        <div style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>
                                            {humanDuration(stop.dwell_sec)} here
                                        </div>
                                    )}
                                    {trans && (
                                        <div>
                                            {/* Same place = a spell out of view, not travel.
                                                On a single-camera site every transition is
                                                this, and "X → X" would read as broken. */}
                                            {trans.same_place
                                                ? `Not detected for ${humanDuration(trans.gap_sec)}`
                                                : `${humanDuration(trans.gap_sec)} to ${trans.to_zone}`}
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                );
            })}
        </div>
    );
}
